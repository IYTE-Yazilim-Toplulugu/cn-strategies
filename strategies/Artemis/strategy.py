"""
Contest model — 3-coin hybrid strategy with PER-COIN routing.

Architecture:
  Both engines run EVERY tick on ALL 3 coins. Each coin is independently
  routed to defensive or aggressive based on its own predicted Sharpe.
  At any moment, coin A may be defensive while coins B and C are
  aggressive — engines work in harmony, neither is "off".

Components:
  - DefensiveEngine: 4-layer engine from defensiv.py (multi-coin Layer 4
    with live correlation + vol-regime sizing). Per coin routed defensive,
    max effective exposure ~0.9x.

  - AggressiveEngine: ATTACK BOT v3 from agresif.py, MAXIMIZED for contest:
      • max_lev still 10 (low-vol periods)
      • per-coin base alloc 0.33 (was 0.31)
      • total alloc cap 1.00 (was 0.95) — full capital usage
      • VO boost goes to 0.33 (was 0.31)
      • Guard recovery: 1 bar (essentially instant, like original)
    Plus three structural improvements:
      • Smoothed multi-component entry signal (EMA20/EMA50 + 5-bar momentum
        agreement) instead of broken 1-bar tick
      • Stop-loss state: 4 consecutive losing bars → cut allocation × 0.7
      • Asymmetric guards (loose RSI extremes 95/5, fast cooldown)

  - MyStrategy: PER-COIN routing.
    For each coin: RF predicts forward Sharpe → EMA-smoothed → 5-bar dwell
    hysteresis → per-coin engine selection. Final portfolio cap keeps
    total allocation ≤ 1.0 when defensive + aggressive coins coexist.
"""

from cnlib.base_strategy import BaseStrategy
from cnlib import backtest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

COINS = ["kapcoin-usd_train", "metucoin-usd_train", "tamcoin-usd_train"]


# ================================================================
# DEFENSIVE ENGINE — from defensiv.py, multi-coin Layer 4
# ================================================================
class DefensiveEngine:
    """4-layer defensive: dual-mode scoring, regime filter, smart skip, adaptive sizing."""

    def __init__(self):
        self.max_lev = {coin: 3 for coin in COINS}
        self.atr_danger = {coin: 0.15 for coin in COINS}

    def predict(self, data, active_coins=None):
        """
        Compute defensive decisions for all coins.

        active_coins: list of coins this engine actually OWNS this tick.
                      Coins not in this list get zeroed out BEFORE Layer 4,
                      so Layer 4's budget (target_total) is distributed
                      only across owned coins. None = own all coins (legacy).
        """
        if active_coins is None:
            active_coins = list(COINS)

        candidates = []
        for coin in COINS:
            df = data[coin]
            if len(df) < 60:
                candidates.append(self._warmup(coin))
                continue

            mode = self._layer3_mode(df, coin)
            if mode == "skip":
                candidates.append(self._flat(coin))
                continue

            if mode == "mean_reversion":
                score = self._compute_score_meanrev(df)
            else:
                score = self._compute_score_momentum(df)

            regime = self._get_regime_v2(df)
            score = self._apply_regime_v2(score, regime)

            candidates.append(self._score_to_action(score, coin, df, regime))

        # Zero out coins NOT owned by defensive — Layer 4 will only
        # distribute its budget among the truly owned coins. This prevents
        # the "wasted slice" problem where defensive sized coin A as
        # 1/3-share when actually it's the only defensive coin.
        for c in candidates:
            if c["coin"] not in active_coins:
                c["signal"] = 0
                c["allocation"] = 0.0
                c["max_alloc"] = 0.0
                c["conviction"] = 0

        self._layer4_v2(candidates, data)
        return candidates

    # ---------- Layer 1: scoring ----------
    def _compute_score_momentum(self, df):
        close = df["Close"]
        vol = df["Volume"]

        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-9)
        rsi_series = 100 - 100 / (1 + rs)
        rsi = rsi_series.iloc[-1]
        rsi_prev = rsi_series.iloc[-2]

        bb_mid = close.rolling(20).mean().iloc[-1]
        bb_std = close.rolling(20).std().iloc[-1]
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        c = close.iloc[-1]
        c_prev = close.iloc[-2]

        vol_avg = vol.rolling(20).mean().iloc[-1]
        vol_ratio = vol.iloc[-1] / vol_avg if vol_avg > 0 else 1.0
        ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]

        score = 0
        if rsi_prev < 50 and rsi >= 50: score += 3
        elif rsi_prev > 50 and rsi <= 50: score -= 3
        elif 50 <= rsi <= 70: score += 2
        elif 30 <= rsi < 50: score -= 2
        elif rsi > 80: score += 1
        elif rsi < 20: score -= 1

        if c > bb_upper and c_prev <= bb_upper: score += 2
        elif c < bb_lower and c_prev >= bb_lower: score -= 2
        elif c > bb_upper: score += 1
        elif c < bb_lower: score -= 1
        elif c > bb_mid: score += 1
        elif c < bb_mid: score -= 1

        if c > ema50: score += 1
        else: score -= 1

        if len(close) >= 5:
            mom5 = (c - close.iloc[-5]) / close.iloc[-5]
            if mom5 > 0.05: score += 1
            elif mom5 < -0.05: score -= 1

        if vol_ratio > 1.5 and score != 0:
            score = int(score * 1.3)
        elif vol_ratio < 0.5:
            score = int(score * 0.5)

        return score

    def _compute_score_meanrev(self, df):
        close = df["Close"]
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-9)
        rsi = (100 - 100 / (1 + rs)).iloc[-1]

        bb_mid = close.rolling(20).mean().iloc[-1]
        bb_std = close.rolling(20).std().iloc[-1]
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        c = close.iloc[-1]

        ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
        ema_dist = (c - ema50) / ema50

        score = 0
        if rsi < 25 and ema_dist < -0.10:    score += 4
        elif rsi < 30:                       score += 2
        elif rsi > 75 and ema_dist > 0.10:   score -= 4
        elif rsi > 70:                       score -= 2
        if c < bb_lower:    score += 2
        elif c > bb_upper:  score -= 2
        return score

    # ---------- Layer 2: regime ----------
    def _get_regime_v2(self, df):
        close = df["Close"]
        if len(close) < 50:
            return "neutral"
        ema50_s = close.ewm(span=50, adjust=False).mean()
        ema50 = ema50_s.iloc[-1]
        ema50_prev = ema50_s.iloc[-10]
        price = close.iloc[-1]
        if len(close) >= 200:
            ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1]
            if price > ema50 > ema200 and ema50 > ema50_prev:
                return "bull"
            elif price < ema50 < ema200 and ema50 < ema50_prev:
                return "bear"
            return "neutral"
        if price > ema50 * 1.02: return "bull"
        elif price < ema50 * 0.98: return "bear"
        return "neutral"

    def _apply_regime_v2(self, score, regime):
        if regime == "bull":
            if score < 0:
                return 0 if score > -7 else int(score * 0.5)
        elif regime == "bear":
            if score > 0:
                return 0 if score < 7 else int(score * 0.5)
        else:
            if abs(score) < 3: return 0
            return int(score * 0.8)
        return score

    # ---------- Layer 3: mode selection ----------
    def _layer3_mode(self, df, coin):
        close = df["Close"]; high = df["High"]; low = df["Low"]; vol = df["Volume"]
        tr = pd.concat([high - low, (high - close.shift()).abs(),
                        (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        atr_long = tr.rolling(100).mean().iloc[-1]
        atr_pct = atr / close.iloc[-1]

        if atr_pct > self.atr_danger[coin]:
            return "skip"
        if atr_long > 0 and atr > 3.0 * atr_long:
            return "skip"
        vol_avg = vol.rolling(50).mean().iloc[-1]
        if vol_avg > 0 and vol.iloc[-1] < 0.2 * vol_avg:
            return "skip"
        if atr_pct > 0.08:
            return "mean_reversion"
        return "momentum"

    # ---------- Action mapping (3-coin sizing, original defensiv.py) ----------
    def _score_to_action(self, score, coin, df, regime):
        max_lev = self.max_lev[coin]
        if score >= 7:    sig, lev, alloc = 1, min(3, max_lev), 0.30
        elif score >= 5:  sig, lev, alloc = 1, min(2, max_lev), 0.25
        elif score >= 3:  sig, lev, alloc = 1, min(2, max_lev), 0.20
        elif score >= 2:  sig, lev, alloc = 1, 1, 0.15
        elif score <= -7: sig, lev, alloc = -1, min(2, max_lev), 0.25
        elif score <= -4: sig, lev, alloc = -1, min(2, max_lev), 0.20
        elif score <= -2: sig, lev, alloc = -1, 1, 0.15
        else:             sig, lev, alloc = 0, 1, 0.0

        return {
            "coin": coin, "signal": sig, "allocation": alloc, "leverage": lev,
            "conviction": self._compute_conviction(df, sig, score),
            "max_alloc": alloc, "score": score, "regime": regime
        }

    def _compute_conviction(self, df, signal, score):
        if signal == 0: return 0
        close = df["Close"]; vol = df["Volume"]
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-9)
        rsi = (100 - 100 / (1 + rs)).iloc[-1]
        vol_avg = vol.rolling(20).mean().iloc[-1]
        vol_ratio = vol.iloc[-1] / vol_avg if vol_avg > 0 else 1.0
        conviction = 1.0 + abs(score) * 0.1
        if signal == 1:
            if 40 <= rsi <= 65: conviction *= 1.3
            if rsi > 78: conviction *= 0.6
        else:
            if 35 <= rsi <= 60: conviction *= 1.3
            if rsi < 22: conviction *= 0.6
        if 1.0 <= vol_ratio <= 2.0: conviction *= 1.15
        elif vol_ratio < 0.5: conviction *= 0.7
        return conviction

    # ---------- Layer 4: multi-coin sizing with correlation ----------
    def _layer4_v2(self, candidates, data):
        active = [c for c in candidates
                  if c.get("signal", 0) != 0 and c.get("allocation", 0) > 0]
        if not active:
            return

        avg_corr = self._measure_correlation(data)
        vol_regime = self._measure_vol_regime(data)

        if avg_corr > 0.7:
            target_total = 0.65; concentration_power = 2.0
        elif avg_corr < 0.3:
            target_total = 0.85; concentration_power = 1.0
        else:
            target_total = 0.75; concentration_power = 1.5

        if vol_regime == "high":
            target_total *= 0.75
        elif vol_regime == "low":
            target_total = min(target_total * 1.10, 0.90)

        for c in active:
            if c.get("regime") in ("bull", "bear"):
                if (c["regime"] == "bull" and c["signal"] == 1) or \
                   (c["regime"] == "bear" and c["signal"] == -1):
                    c["conviction"] *= 1.30
                    break

        max_allocs = {c["coin"]: c["max_alloc"] for c in active}
        weighted = [c["conviction"] ** concentration_power for c in active]
        total_conv = sum(weighted)
        if total_conv == 0:
            return

        for c, w in zip(active, weighted):
            ideal = (w / total_conv) * target_total
            c["allocation"] = min(ideal, max_allocs[c["coin"]])

        total = sum(c["allocation"] for c in active)
        if total < target_total:
            leftover = target_total - total
            headroom = [c for c in active
                        if c["allocation"] < max_allocs[c["coin"]] - 0.01]
            if headroom:
                per_coin = leftover / len(headroom)
                for c in headroom:
                    c["allocation"] = min(c["allocation"] + per_coin,
                                          max_allocs[c["coin"]])

        total = sum(c["allocation"] for c in active)
        if total > target_total:
            factor = target_total / total
            for c in active:
                c["allocation"] *= factor

    def _measure_correlation(self, data):
        try:
            rets = pd.DataFrame({
                c: data[c]["Close"].pct_change().tail(60) for c in COINS
            }).dropna()
            if len(rets) < 30:
                return 0.5
            return (rets.corr().values.sum() - 3) / 6
        except Exception:
            return 0.5

    def _measure_vol_regime(self, data):
        try:
            vol_ratios = []
            for coin in COINS:
                df = data[coin]
                if len(df) < 60: continue
                close = df["Close"]; high = df["High"]; low = df["Low"]
                tr = pd.concat([high - low, (high - close.shift()).abs(),
                                (low - close.shift()).abs()], axis=1).max(axis=1)
                atr_pct_series = (tr.rolling(14).mean() / close).dropna()
                if len(atr_pct_series) < 30: continue
                current = atr_pct_series.iloc[-1]
                median = atr_pct_series.tail(60).median()
                if median > 0:
                    vol_ratios.append(current / median)
            if not vol_ratios:
                return "normal"
            avg = sum(vol_ratios) / len(vol_ratios)
            if avg > 1.30: return "high"
            elif avg < 0.80: return "low"
            return "normal"
        except Exception:
            return "normal"

    def _flat(self, coin):
        return {"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1,
                "conviction": 0, "max_alloc": 0, "score": 0, "regime": "neutral"}

    def _warmup(self, coin):
        return {"coin": coin, "signal": 1, "allocation": 0.20, "leverage": 2,
                "conviction": 1.0, "max_alloc": 0.20, "score": 2, "regime": "neutral"}


# ================================================================
# AGGRESSIVE ENGINE — contest-maxed with new improvements
# ================================================================
class AggressiveEngine:
    """
    Contest-tuned aggressive engine.

    More aggressive than original agresif.py:
      - Total cap 1.00 (was 0.95)
      - Per-coin base 0.33 (was 0.31)
      - VO boost goes to full 0.33 (was 0.31)
      - Guard recovery: 1 bar (immediate)

    Plus structural improvements over agresif.py:
      - Multi-component smoothed entry signal (EMA20/EMA50 + 5-bar momentum
        agreement) — eliminates random flips on 1-bar noise
      - Stop-loss: 4 consecutive losing bars → cut allocation × 0.7
      - Cooldown system (1 bar) tracks guard events properly
    """

    def __init__(self):
        self.lookback = 60

        # Per-coin allocations (3 coins splitting capital)
        self.base_alloc      = 0.33   # was 0.31 in agresif
        self.guard_alloc_rsi = 0.22
        self.guard_alloc_vol = 0.18
        self.guard_alloc_ext = 0.13
        self.vo_boost_alloc  = 0.33   # was 0.31

        # Total allocation cap (sum across coins) — kept at 0.95, NOT 1.0,
        # to stay below the 0.99-1.00 simulator rejection boundary.
        # This is the same cap as the original agresif.py for the same reason.
        self.total_alloc_cap = 0.95

        # Adaptive max leverage table — original aggressive
        self.lev_table = [(0.06, 10), (0.09, 10), (0.12, 5), (0.18, 3), (1.0, 2)]

        # Recovery / sensitivity
        self.guard_recovery_bars = 1     # immediate
        self.rsi_extreme_high_pct = 95
        self.rsi_extreme_low_pct  = 5
        self.signal_threshold = 0.003    # 0.3% momentum is enough for contest

        # Stop-loss
        self.stoploss_threshold_bars = 4
        self.stoploss_alloc_scale = 0.7

        # Persistent per-coin state
        self.last_signals = {}
        self.last_closes = {}
        self.loss_streaks = {}
        self.guard_cooldown = {}

    # ============================================================
    def predict(self, data, active_coins=None):
        """
        Compute aggressive decisions for all coins.

        active_coins: list of coins this engine actually OWNS this tick.
                      Allocations for owned coins get SCALED UP when fewer
                      coins are owned (since the engine's 1.0 budget is
                      now concentrated on fewer positions). Coins not in
                      this list are zeroed out for output, but state
                      (last_signals, loss_streaks, guard_cooldown) is
                      kept consistent — internally we track all coins.
        """
        if active_coins is None:
            active_coins = list(COINS)

        positions = []
        for coin in COINS:
            df = data[coin]
            self._update_loss_streak(coin, df["Close"])
            positions.append(self._compute_position(coin, df))

        # VO boost — recover guard-reduced positions when cooldown expired
        for pos in positions:
            if pos.get("guard_triggered") and pos.get("vo_boost") and \
               self.guard_cooldown.get(pos["coin"], 0) <= 0:
                pos["leverage"] = pos["max_lev"]
                pos["allocation"] = self.vo_boost_alloc

        # ============================================================
        # State updates use REAL signals BEFORE we zero out inactive coins.
        # This keeps loss-streak / cooldown / last_signal tracking accurate
        # across the full coin set, even for coins routed to defensive in
        # MyStrategy. When a coin flips back to aggressive, its state is
        # already up to date.
        # ============================================================
        for p in positions:
            self.last_signals[p["coin"]] = p["signal"]
            self.last_closes[p["coin"]] = data[p["coin"]]["Close"].iloc[-1]

        for coin in COINS:
            if self.guard_cooldown.get(coin, 0) > 0:
                self.guard_cooldown[coin] -= 1

        # ============================================================
        # SCALE allocations for owned coins. The engine's total budget is
        # 1.0; per-coin base alloc was tuned for 3-coin split (0.33).
        # When fewer coins are owned, redistribute the unused budget.
        #
        # Scale factor: 3.0 / n_owned, capped at 2.0 to limit single-coin
        # concentration. Per-coin cap at 0.66 for safety even under scaling.
        # ============================================================
        n_owned_signaled = sum(
            1 for p in positions
            if p["coin"] in active_coins and p["signal"] != 0
        )

        if n_owned_signaled > 0:
            scale = min(3.0 / n_owned_signaled, 2.0)
            for p in positions:
                if p["coin"] in active_coins:
                    # Per-coin cap at 0.50 (was 0.66) — prevents single-coin
                    # concentration from pushing total alloc near the
                    # 0.99-1.00 simulator rejection boundary.
                    p["allocation"] = min(p["allocation"] * scale, 0.50)
                else:
                    # Output as inactive — state was already saved above
                    p["signal"] = 0
                    p["allocation"] = 0.0
        else:
            # No owned coins have signal — just zero out non-owned for output
            for p in positions:
                if p["coin"] not in active_coins:
                    p["signal"] = 0
                    p["allocation"] = 0.0

        # Total allocation hard cap (only across owned & signaled coins)
        active_pos = [p for p in positions
                      if p["signal"] != 0 and p["allocation"] > 0]
        total = sum(p["allocation"] for p in active_pos)
        if total > self.total_alloc_cap:
            factor = self.total_alloc_cap / total
            for p in active_pos:
                p["allocation"] *= factor

        return positions

    # ============================================================
    def _update_loss_streak(self, coin, close):
        last_sig = self.last_signals.get(coin, 0)
        last_close = self.last_closes.get(coin, None)
        if last_sig == 0 or last_close is None:
            self.loss_streaks[coin] = 0
            return
        realized_ret = close.iloc[-1] / last_close - 1
        pnl = last_sig * realized_ret
        if pnl < 0:
            self.loss_streaks[coin] = self.loss_streaks.get(coin, 0) + 1
        else:
            self.loss_streaks[coin] = 0

    # ============================================================
    def _compute_position(self, coin, df):
        if len(df) < self.lookback + 5:
            return {
                "coin": coin, "signal": 1, "allocation": self.base_alloc,
                "leverage": 2, "guard_triggered": False, "vo_boost": False,
                "max_lev": 5
            }

        close = df["Close"]
        vol = df["Volume"]

        # Smoothed entry signal
        signal = self._smoothed_signal(close)
        max_lev = self._adaptive_max_leverage(close)

        # Indicators
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-9)
        rsi_series = 100 - 100 / (1 + rs)
        rsi = rsi_series.iloc[-1]
        vol_avg = vol.rolling(20).mean().iloc[-1]
        vol_ratio = vol.iloc[-1] / vol_avg if vol_avg > 0 else 1.0
        vol_fast = vol.rolling(5).mean().iloc[-1]
        vo = ((vol_fast - vol_avg) / vol_avg) * 100 if vol_avg > 0 else 0
        rsi_high, rsi_low = self._adaptive_rsi_thresholds(rsi_series)
        vol_spike_threshold = self._adaptive_volume_threshold(vol)

        # Default sizing
        lev = max_lev
        alloc = self.base_alloc
        guard_triggered = False

        # Cooldown carry-over
        if self.guard_cooldown.get(coin, 0) > 0:
            alloc = min(alloc, self.guard_alloc_rsi)
            lev = min(lev, max(2, max_lev // 2))

        # Guard 1: RSI counter-zone
        if signal == 1 and rsi > rsi_high:
            lev = max(2, max_lev // 2); alloc = self.guard_alloc_rsi
            guard_triggered = True
        elif signal == -1 and rsi < rsi_low:
            lev = max(2, max_lev // 2); alloc = self.guard_alloc_rsi
            guard_triggered = True

        # Guard 2: Volume spike
        if vol_ratio > vol_spike_threshold:
            lev = min(lev, 3)
            alloc = min(alloc, self.guard_alloc_vol)
            guard_triggered = True

        # Guard 3: Extreme combo
        rsi_ext_high = self._percentile(rsi_series, self.rsi_extreme_high_pct)
        rsi_ext_low  = self._percentile(rsi_series, self.rsi_extreme_low_pct)
        if (rsi > rsi_ext_high or rsi < rsi_ext_low) and \
           vol_ratio > vol_spike_threshold * 0.85:
            lev = 2; alloc = self.guard_alloc_ext
            guard_triggered = True

        if guard_triggered:
            self.guard_cooldown[coin] = self.guard_recovery_bars

        # Stop-loss de-escalation
        streak = self.loss_streaks.get(coin, 0)
        if streak >= self.stoploss_threshold_bars:
            alloc *= self.stoploss_alloc_scale
            lev = min(lev, max(1, max_lev // 2))
        elif streak == self.stoploss_threshold_bars - 1:
            alloc *= 0.85

        if signal == 0:
            alloc = 0.0

        # VO boost eligibility
        vo_boost = (
            vo > 15 and
            rsi > rsi_low + 5 and
            rsi < rsi_high - 5 and
            vol_ratio < vol_spike_threshold * 0.85 and
            streak < self.stoploss_threshold_bars
        )

        return {
            "coin": coin, "signal": signal,
            "allocation": alloc, "leverage": lev,
            "guard_triggered": guard_triggered,
            "vo_boost": vo_boost, "max_lev": max_lev,
        }

    # ============================================================
    def _smoothed_signal(self, close):
        c = close.iloc[-1]
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()

        if c > ema20.iloc[-1] and ema20.iloc[-1] > ema50.iloc[-1]:
            ema_dir = 1
        elif c < ema20.iloc[-1] and ema20.iloc[-1] < ema50.iloc[-1]:
            ema_dir = -1
        else:
            ema_dir = 0

        if len(close) >= 5:
            mom5 = (c - close.iloc[-5]) / close.iloc[-5]
            if mom5 > self.signal_threshold:    mom_dir = 1
            elif mom5 < -self.signal_threshold: mom_dir = -1
            else:                               mom_dir = 0
        else:
            mom_dir = 0

        bar_dir = 1 if c > close.iloc[-2] else -1

        if ema_dir != 0 and ema_dir == mom_dir:
            return ema_dir
        if ema_dir == 0 and mom_dir != 0 and mom_dir == bar_dir:
            return mom_dir
        return 0

    def _adaptive_max_leverage(self, close):
        recent = close.tail(self.lookback)
        rets = recent.pct_change().dropna()
        if len(rets) < 20:
            return self.lev_table[1][1]
        worst_move = max(abs(rets.min()), abs(rets.max()))
        for cutoff, lev in self.lev_table:
            if worst_move < cutoff:
                return lev
        return self.lev_table[-1][1]

    def _adaptive_rsi_thresholds(self, rsi_series):
        recent = rsi_series.tail(self.lookback).dropna()
        if len(recent) < 20:
            return 78, 22
        rsi_high = max(self._percentile(recent, 85), 70)
        rsi_low = min(self._percentile(recent, 15), 30)
        return rsi_high, rsi_low

    def _adaptive_volume_threshold(self, vol_series):
        recent = vol_series.tail(self.lookback).dropna()
        if len(recent) < 20:
            return 3.0
        median = recent.median()
        if median <= 0:
            return 3.0
        ratios = recent / median
        return max(2.0, min(4.0, self._percentile(ratios, 92)))

    def _percentile(self, series, pct):
        try:
            return float(series.quantile(pct / 100))
        except Exception:
            return float(series.median())


# ================================================================
# MAIN STRATEGY: PER-COIN routing — both engines work in harmony
# ================================================================
class MyStrategy(BaseStrategy):
    """
    Hybrid strategy with PER-COIN routing.

    Both engines run every tick on all 3 coins. Each coin is INDEPENDENTLY
    routed to defensive or aggressive based on its own predicted forward
    Sharpe ratio. The route just selects which engine's decision is used
    for that coin — both engines stay active in parallel.

    Result: at any given moment, e.g. coin A might be defensively held
    while coin B is aggressively long and coin C is aggressively short.
    Engines work in harmony — neither is "off". A final portfolio cap
    keeps total allocation ≤ 1.0.

    Per-coin hysteresis: each coin has its own EMA-smoothed pred_sharpe
    and dwell counter, so coins flip routes independently and only after
    sustained agreement.
    """

    def __init__(self,
                 risk_threshold=0.5,
                 smoothing_alpha=0.3,
                 hysteresis_bars=5):
        super().__init__()
        self.model = None
        self.is_trained = False
        self.min_training_data = 150
        self.coins = COINS

        self.defensive = DefensiveEngine()
        self.aggressive = AggressiveEngine()

        self.risk_threshold = risk_threshold
        self.smoothing_alpha = smoothing_alpha
        self.hysteresis_bars = hysteresis_bars

        # PER-COIN hysteresis state
        self.coin_smoothed = {c: None for c in COINS}      # EMA'd pred_sharpe
        self.coin_routes   = {c: "defensive" for c in COINS}  # current engine per coin
        self.coin_dwell    = {c: 0 for c in COINS}         # bars wanting switch

    # ---------- Feature engineering ----------
    def extract_features(self, df):
        log_ret = np.log(df['Close'] / df['Close'].shift(1))
        features = pd.DataFrame({
            'volatility': log_ret.rolling(20).std(),
            'momentum': df['Close'].pct_change(10),
            'rel_dist_ma': (df['Close'] / df['Close'].rolling(50).mean()) - 1
        })
        return features.dropna()

    def train_multi_coin_model(self, data_dict):
        all_X, all_y = [], []
        for coin in self.coins:
            df = data_dict[coin].copy()
            if len(df) < self.min_training_data:
                continue
            features = self.extract_features(df)
            fwd_ret = np.log(df['Close'] / df['Close'].shift(1)).shift(-10).rolling(10)
            target = (fwd_ret.mean() / fwd_ret.std()).dropna()
            common_idx = features.index.intersection(target.index)
            all_X.append(features.loc[common_idx])
            all_y.append(target.loc[common_idx])

        if all_X:
            X = pd.concat(all_X)
            y = pd.concat(all_y)
            self.model = RandomForestRegressor(
                n_estimators=100, max_depth=6, random_state=42
            )
            self.model.fit(X, y)
            self.is_trained = True
            print(f"Multi-coin model trained on {len(X)} samples.")

    # ---------- Per-coin routing with smoothing + hysteresis ----------
    def _route_coin(self, coin, df):
        """Decide engine for a single coin. State is per-coin."""
        feat = self.extract_features(df).tail(1)
        if feat.empty:
            return self.coin_routes[coin]

        raw_pred = float(self.model.predict(feat)[0])

        # EMA smoothing per coin
        if self.coin_smoothed[coin] is None:
            self.coin_smoothed[coin] = raw_pred
        else:
            a = self.smoothing_alpha
            self.coin_smoothed[coin] = (
                a * raw_pred + (1 - a) * self.coin_smoothed[coin]
            )

        wants = ("defensive" if self.coin_smoothed[coin] < self.risk_threshold
                 else "aggressive")

        # Hysteresis per coin
        if wants != self.coin_routes[coin]:
            self.coin_dwell[coin] += 1
            if self.coin_dwell[coin] >= self.hysteresis_bars:
                self.coin_routes[coin] = wants
                self.coin_dwell[coin] = 0
        else:
            self.coin_dwell[coin] = 0

        return self.coin_routes[coin]

    # ---------- Main predict ----------
    def predict(self, data):
        # Lazy training
        if not self.is_trained and len(data[self.coins[0]]) > self.min_training_data + 20:
            self.train_multi_coin_model(data)

        # Pre-training: all coins defensive (safer warmup)
        if not self.is_trained:
            try:
                return [
                    {k: v for k, v in p.items()
                     if k in ("coin", "signal", "allocation", "leverage")}
                    for p in self.defensive.predict(data)
                ]
            except Exception:
                return [{"coin": c, "signal": 0, "allocation": 0.0, "leverage": 1}
                        for c in COINS]

        # ============================================================
        # Step 1: Decide routes per coin FIRST.
        # Smoothing + hysteresis happens here. Each coin lands in either
        # defensive_coins or aggressive_coins.
        # ============================================================
        defensive_coins = []
        aggressive_coins = []
        for coin in COINS:
            route = self._route_coin(coin, data[coin])
            if route == "defensive":
                defensive_coins.append(coin)
            else:
                aggressive_coins.append(coin)

        # ============================================================
        # Step 2: Run BOTH engines, each told which coins it OWNS.
        #
        # Each engine internally still runs on all 3 coins — that keeps
        # state consistent — but its OUTPUT is sized only for its owned
        # coins. Defensive's Layer 4 distributes its budget across only
        # its owned coins. Aggressive scales per-coin allocations up
        # (capped at 0.66/coin) to fill its 1.0 budget on fewer coins.
        # ============================================================
        try:
            def_out = {
                p["coin"]: p for p in
                self.defensive.predict(data, active_coins=defensive_coins)
            }
        except Exception:
            def_out = {}
        try:
            agg_out = {
                p["coin"]: p for p in
                self.aggressive.predict(data, active_coins=aggressive_coins)
            }
        except Exception:
            agg_out = {}

        # ============================================================
        # Step 3: Pick each coin's decision from its assigned engine.
        # ============================================================
        final = []
        for coin in COINS:
            if coin in defensive_coins:
                chosen = def_out.get(coin)
            else:
                chosen = agg_out.get(coin)

            if chosen is None:
                chosen = {"coin": coin, "signal": 0,
                          "allocation": 0.0, "leverage": 1}

            final.append({
                "coin": coin,
                "signal": int(chosen.get("signal", 0)),
                "allocation": float(chosen.get("allocation", 0.0)),
                "leverage": int(chosen.get("leverage", 1)),
            })

        # ============================================================
        # Step 4: Final portfolio cap — total allocation ≤ 1.0.
        # Defensive can produce up to ~0.85 across its coins; aggressive
        # caps at 1.0 across its coins. When both have positions the sum
        # can exceed 1.0. Proportional scale-down preserves the relative
        # weighting between the two engines' contributions.
        # ============================================================
        # CRITICAL: cap at 0.93, NOT 1.0.
        #
        # The contest simulator rejects opens when total allocation
        # approaches the 0.99-1.00 boundary. Each rejection is a missed
        # compounding opportunity. Going from 39% rejection rate to ~0%
        # makes orders-of-magnitude difference in compounded returns.
        #
        # 0.93 = aggressive's 0.95 internal cap minus 0.02 safety margin
        # for floating-point rounding when defensive + aggressive sums.
        # ============================================================
        TOTAL_CAP = 0.93
        active = [p for p in final if p["signal"] != 0 and p["allocation"] > 0]
        total = sum(p["allocation"] for p in active)
        if total > TOTAL_CAP:
            scale = TOTAL_CAP / total
            for p in active:
                p["allocation"] *= scale

        return final

if __name__ == "__main__":
    strategy = MyStrategy()
    strategy.get_data()   # önce data yükle
    result = backtest.run(strategy=strategy, initial_capital=3000.0)
    result.print_summary()


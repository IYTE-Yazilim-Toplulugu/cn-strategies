"""
Paribu Code Night — Hybrid Momentum + ML Strategy
Safe (no liquidations) + Maximum return
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from cnlib.base_strategy import BaseStrategy
from cnlib import backtest


class HybridStrategy(BaseStrategy):

    def __init__(self):
        super().__init__()
        self.model: GradientBoostingClassifier | None = None
        self._indicator_cache: dict[str, dict[str, np.ndarray]] = {}
        self.decisions_log: list[dict] = []  # per-candle predict() output
        self._prev_autocorr: float = 0.0     # confirmation: require 2 consecutive candles above threshold

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def egit(self):
        """Train ML model on ALL coin data. Call after get_data()."""
        # Use _full_data (coin_data is limited by candle_index)
        full_data = self._full_data
        self._precompute_all_indicators(full_data)

        X, y = [], []
        for coin, df in full_data.items():
            closes = df["Close"].values
            ind = self._indicator_cache[coin]
            n = len(closes)
            for i in range(50, n - 1):
                feat = self._feature_vector_from_cache(ind, i)
                if feat is not None:
                    X.append(feat)
                    next_ret = (closes[i + 1] - closes[i]) / closes[i]
                    y.append(1 if next_ret > 0 else 0)

        self.model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, random_state=42
        )
        self.model.fit(np.array(X), np.array(y))
        print(f"  Model trained on {len(X)} samples.")

    # ------------------------------------------------------------------
    # Indicator pre-computation (vectorized, fast)
    # ------------------------------------------------------------------

    def _precompute_all_indicators(self, data: dict | None = None):
        source = data or self._full_data
        for coin, df in source.items():
            c = df["Close"].values.astype(float)
            h = df["High"].values.astype(float)
            l = df["Low"].values.astype(float)
            v = df["Volume"].values.astype(float)
            self._indicator_cache[coin] = self._compute_indicators(c, h, l, v)

    @staticmethod
    def _compute_indicators(c, h, l, v) -> dict[str, np.ndarray]:
        N = len(c)
        ind: dict[str, np.ndarray] = {}

        # --- Momentum returns ---
        for lb in [1, 2, 3, 5, 10, 20, 30]:
            mom = np.zeros(N)
            mom[lb:] = (c[lb:] - c[:-lb]) / c[:-lb]
            ind[f"mom_{lb}d"] = mom

        # --- MA ratios ---
        for w in [5, 10, 20, 50]:
            ma = pd.Series(c).rolling(w).mean().values
            ind[f"ma{w}_ratio"] = np.where(ma != 0, c / ma - 1, 0.0)

        # --- EMA ratios ---
        for span in [8, 13, 21]:
            ema = pd.Series(c).ewm(span=span).mean().values
            ind[f"ema{span}_ratio"] = np.where(ema != 0, c / ema - 1, 0.0)

        # --- Volatility ---
        rets = pd.Series(c).pct_change().values
        ind["vol_std"] = pd.Series(rets).rolling(20).std().values
        ind["vol_mean"] = pd.Series(rets).rolling(20).mean().values

        # --- Volume ratio ---
        vol_ma = pd.Series(v).rolling(20).mean().values
        ind["vol_ratio"] = np.where(vol_ma != 0, v / vol_ma - 1, 0.0)

        # --- Range ---
        high5 = pd.Series(c).rolling(5).max().values
        low5 = pd.Series(c).rolling(5).min().values
        ind["range_5d"] = np.where(c != 0, (high5 - low5) / c, 0.0)

        # --- RSI(14) ---
        delta = np.diff(c, prepend=c[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = pd.Series(gain).rolling(14).mean().values
        avg_loss = pd.Series(loss).rolling(14).mean().values
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
        ind["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

        # --- MACD ---
        ema12 = pd.Series(c).ewm(span=12).mean().values
        ema26 = pd.Series(c).ewm(span=26).mean().values
        macd_line = ema12 - ema26
        signal_line = pd.Series(macd_line).ewm(span=9).mean().values
        ind["macd_hist"] = np.where(c != 0, (macd_line - signal_line) / c, 0.0)
        ind["macd_line"] = np.where(c != 0, macd_line / c, 0.0)

        # --- Bollinger Bands ---
        bb_ma = pd.Series(c).rolling(20).mean().values
        bb_std = pd.Series(c).rolling(20).std().values
        bb_upper = bb_ma + 2 * bb_std
        bb_lower = bb_ma - 2 * bb_std
        bb_range = bb_upper - bb_lower
        ind["bb_pctb"] = np.where(bb_range != 0, (c - bb_lower) / bb_range, 0.5)
        ind["bb_width"] = np.where(bb_ma != 0, bb_range / bb_ma, 0.0)

        # --- Stochastic ---
        high14 = pd.Series(h).rolling(14).max().values
        low14 = pd.Series(l).rolling(14).min().values
        stoch_range = high14 - low14
        stoch_k = np.where(stoch_range != 0, 100 * (c - low14) / stoch_range, 50.0)
        ind["stoch_k"] = stoch_k
        ind["stoch_d"] = pd.Series(stoch_k).rolling(3).mean().values

        # --- CCI ---
        tp = (h + l + c) / 3.0
        tp_ma = pd.Series(tp).rolling(20).mean().values
        tp_md = pd.Series(tp).rolling(20).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        ).values
        ind["cci"] = np.where(tp_md != 0, (tp - tp_ma) / (0.015 * tp_md), 0.0)

        # --- Williams %R ---
        ind["williams_r"] = np.where(
            stoch_range != 0, -100 * (high14 - c) / stoch_range, -50.0
        )

        # --- ATR normalized ---
        tr = np.maximum(
            h[1:] - l[1:],
            np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])),
        )
        atr = pd.Series(tr).rolling(14).mean().values
        atr_full = np.concatenate([[np.nan], atr])
        ind["atr_norm"] = np.where(c != 0, atr_full / c, 0.0)

        # --- MFI ---
        mf = tp * v
        tp_diff = np.diff(tp, prepend=tp[0])
        pos_mf = np.where(tp_diff > 0, mf, 0.0)
        neg_mf = np.where(tp_diff < 0, mf, 0.0)
        pos_sum = pd.Series(pos_mf).rolling(14).sum().values
        neg_sum = pd.Series(neg_mf).rolling(14).sum().values
        with np.errstate(divide="ignore", invalid="ignore"):
            ind["mfi"] = np.where(
                neg_sum > 0, 100.0 - 100.0 / (1.0 + pos_sum / neg_sum), 50.0
            )

        # --- OBV momentum ---
        price_diff = np.diff(c, prepend=c[0])
        obv = np.cumsum(
            np.where(price_diff > 0, v, np.where(price_diff < 0, -v, 0.0))
        )
        obv_mom = np.zeros(N)
        for i in range(10, N):
            obv_mom[i] = (obv[i] - obv[i - 10]) / (abs(obv[i - 10]) + 1)
        ind["obv_mom"] = obv_mom

        # --- ADX direction ---
        plus_dm = np.maximum(np.diff(h, prepend=h[0]), 0.0)
        minus_dm = np.maximum(-np.diff(l, prepend=l[0]), 0.0)
        mask_p = plus_dm > minus_dm
        plus_dm_clean = np.where(mask_p, plus_dm, 0.0)
        minus_dm_clean = np.where(~mask_p, minus_dm, 0.0)
        tr_full = np.maximum(
            h - np.roll(l, 1),
            np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))),
        )
        tr_full[0] = h[0] - l[0]
        atr_adx = pd.Series(tr_full).rolling(14).mean().values
        plus_di = np.where(
            atr_adx > 0,
            100 * pd.Series(plus_dm_clean).rolling(14).mean().values / atr_adx,
            0.0,
        )
        minus_di = np.where(
            atr_adx > 0,
            100 * pd.Series(minus_dm_clean).rolling(14).mean().values / atr_adx,
            0.0,
        )
        ind["adx_dir"] = plus_di - minus_di

        return ind

    # ------------------------------------------------------------------
    # Feature names (order matters — must match training)
    # ------------------------------------------------------------------

    FEATURE_KEYS = [
        "mom_1d", "mom_2d", "mom_3d", "mom_5d", "mom_10d", "mom_20d", "mom_30d",
        "ma5_ratio", "ma10_ratio", "ma20_ratio", "ma50_ratio",
        "ema8_ratio", "ema13_ratio", "ema21_ratio",
        "vol_std", "vol_mean", "vol_ratio", "range_5d",
        "rsi_14", "macd_hist", "macd_line",
        "bb_pctb", "bb_width",
        "stoch_k", "stoch_d",
        "cci", "williams_r", "atr_norm", "mfi",
        "obv_mom", "adx_dir",
    ]

    @classmethod
    def _feature_vector_from_cache(cls, ind: dict, idx: int):
        feat = [float(ind[k][idx]) for k in cls.FEATURE_KEYS]
        if any(np.isnan(feat)):
            return None
        return np.array(feat)

    # ------------------------------------------------------------------
    # Live feature computation (for predict — only current data available)
    # ------------------------------------------------------------------

    def _live_features(self, df: pd.DataFrame):
        """Compute features from the DataFrame slice given in predict()."""
        if len(df) < 51:
            return None
        c = df["Close"].values.astype(float)
        h = df["High"].values.astype(float)
        l = df["Low"].values.astype(float)
        v = df["Volume"].values.astype(float)
        ind = self._compute_indicators(c, h, l, v)
        return self._feature_vector_from_cache(ind, len(c) - 1)

    # ------------------------------------------------------------------
    # predict() — called every candle
    # ------------------------------------------------------------------

    def predict(self, data: dict) -> list[dict]:
        coins = list(data.keys())
        decisions: list[dict] = []

        # --- Regime detection: autocorrelation on last 100 returns ---
        autocorrs = []
        volatilities = []
        for coin in coins:
            df = data[coin]
            closes = df["Close"].values

            # Autocorrelation (100-candle window)
            if len(closes) >= 102:
                window = closes[-102:]
                rets = np.diff(window) / window[:-1]
                r1 = rets[:-1] - np.mean(rets[:-1])
                r2 = rets[1:] - np.mean(rets[1:])
                denom = np.sqrt(np.sum(r1**2) * np.sum(r2**2))
                if denom > 0:
                    autocorrs.append(np.sum(r1 * r2) / denom)

            # Volatility (30-day rolling std of returns)
            if len(closes) >= 31:
                recent_rets = np.diff(closes[-31:]) / closes[-31:-1]
                volatilities.append(np.std(recent_rets))

        # Use MAX autocorr: if any coin is trending, open the regime gate for it.
        # Synthetic data: all coins trend together → max ≈ avg (no change).
        # Real data: coins trend independently → avg drags the signal down unfairly.
        avg_autocorr = np.max(autocorrs) if autocorrs else 0.0
        avg_vol = np.mean(volatilities) if volatilities else 0.02

        # --- Regime → leverage & allocation caps ---
        # Entry threshold 0.115: eliminates borderline random-walk spikes (0.10-0.114)
        # that lack true momentum. Confirmed = both current AND prev candle above threshold.
        _ENTRY = 0.115
        _FULL  = 0.30
        confirmed = self._prev_autocorr >= _ENTRY and avg_autocorr >= _ENTRY

        if not confirmed or avg_autocorr < _ENTRY:
            # Random walk / mean-reversion → DO NOT TRADE
            # No strategy beats cash in a random walk
            regime_max_lev = 1
            regime_alloc = 0.0
            min_score_threshold = 999.0  # effectively no trades
        elif avg_autocorr < _FULL:
            # Transition zone: linear interpolation _ENTRY→_FULL
            # alloc: 0.15 → 1.0, lev: 2 → 5, thresh: 0.04 → 0.005
            t = (avg_autocorr - _ENTRY) / (_FULL - _ENTRY)  # 0.0 to 1.0
            regime_alloc = 0.15 + t * 0.85
            regime_max_lev = int(2 + t * 3)  # 2 → 5
            min_score_threshold = 0.04 - t * 0.035  # 0.04 → 0.005
        else:
            # Strong momentum (synthetic-like) → full aggression
            regime_max_lev = 5
            regime_alloc = 1.0
            min_score_threshold = 0.0

        # --- Volatility filter: adaptive baseline ---
        # Scale down if current vol is extreme relative to recent norm
        if len(volatilities) >= 1 and avg_vol > 0:
            vol_ratio = avg_vol / 0.025  # ratio vs typical 2.5% daily vol
            if vol_ratio > 3.0:
                vol_scale = 0.2   # extreme vol → heavy cut
            elif vol_ratio > 2.0:
                vol_scale = 0.4
            elif vol_ratio > 1.5:
                vol_scale = 0.7
            else:
                vol_scale = 1.0
        else:
            vol_scale = 1.0

        final_alloc = regime_alloc * vol_scale

        best_coin = None
        best_score = -999.0
        best_dir = 0
        best_lev = 1

        for coin in coins:
            df = data[coin]
            if len(df) < 51:
                continue

            closes = df["Close"].values
            # --- Momentum signal ---
            past_ret = (closes[-1] - closes[-6]) / closes[-6]
            mom_dir = 1 if past_ret > 0 else -1

            # --- ML confidence ---
            feat = self._live_features(df)
            if feat is None or self.model is None:
                continue

            proba = self.model.predict_proba([feat])[0]
            ml_dir = 1 if proba[1] > 0.5 else -1
            ml_conf = max(proba) - 0.5  # 0.0 to 0.5

            # --- Agreement score ---
            if mom_dir == ml_dir:
                score = abs(past_ret) * (1 + ml_conf * 2)
            else:
                score = -1.0  # disagreement → skip

            if score > best_score:
                best_score = score
                best_coin = coin
                best_dir = mom_dir
                # Dynamic leverage (capped by regime)
                if score > 0.05:
                    best_lev = min(5, regime_max_lev)
                elif score > 0.02:
                    best_lev = min(3, regime_max_lev)
                else:
                    best_lev = min(2, regime_max_lev)

        # --- Build decisions (apply score threshold) ---
        for coin in coins:
            if (best_coin and coin == best_coin
                    and best_score > min_score_threshold):
                decisions.append({
                    "coin": coin,
                    "signal": best_dir,
                    "allocation": round(min(final_alloc, 1.0), 2),
                    "leverage": best_lev,
                })
            else:
                decisions.append({
                    "coin": coin,
                    "signal": 0,
                    "allocation": 0.0,
                    "leverage": 1,
                })

        self._prev_autocorr = avg_autocorr

        # Record candle date + every coin's decision for post-run inspection
        sample_df = next(iter(data.values()))
        candle_date = str(sample_df["Date"].iloc[-1]) if "Date" in sample_df.columns else ""
        for d in decisions:
            self.decisions_log.append({
                "date": candle_date,
                "coin": d["coin"],
                "signal": d["signal"],
                "allocation": d["allocation"],
                "leverage": d["leverage"],
                "regime_autocorr": round(avg_autocorr, 4),
                "avg_vol": round(avg_vol, 5),
            })

        return decisions


# ======================================================================
# Run backtest
# ======================================================================
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="CNLIB HybridStrategy backtest runner")
    parser.add_argument(
        "--test-data-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Path to directory with jury parquet files "
            "(kapcoin-usd_train.parquet, metucoin-usd_train.parquet, tamcoin-usd_train.parquet). "
            "If omitted, backtests on the bundled training data."
        ),
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=3000.0,
        metavar="FLOAT",
        help="Initial capital in USD (default: 3000.0)",
    )
    args = parser.parse_args()

    strategy = HybridStrategy()
    strategy.get_data()   # always train on bundled 4-year data
    strategy.egit()

    result = backtest.run(
        strategy=strategy,
        initial_capital=args.capital,
        data_dir=args.test_data_dir,  # None → bundled data, else jury data
    )
    result.print_summary()

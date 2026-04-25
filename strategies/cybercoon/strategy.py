from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from cnlib.base_strategy import BaseStrategy
from cnlib.backtest import run
from config import COINS, INITIAL_CAPITAL, StrategyConfig, DEFAULT_CONFIG


@dataclass
class CoinSignal:
    coin: str
    signal: int
    strength: float
    leverage: int


class NFIStrategy(BaseStrategy):
    """NFI-inspired 1D long/short strategy for 3 synthetic coins."""

    def __init__(self, cfg: StrategyConfig = DEFAULT_CONFIG):
        super().__init__()
        self.cfg = cfg
        self.prev_signal = {c: 0 for c in COINS}
        self.hold_days = {c: 0 for c in COINS}

    def _ema(self, s: pd.Series, span: int) -> pd.Series:
        return s.ewm(span=span, adjust=False).mean()

    def _rsi(self, close: pd.Series) -> pd.Series:
        period = self.cfg.rsi_period
        delta = close.diff()
        gain = pd.Series(np.where(delta > 0, delta, 0.0), index=close.index)
        loss = pd.Series(np.where(delta < 0, -delta, 0.0), index=close.index)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / (avg_loss + 1e-12)
        return 100 - (100 / (1 + rs))

    def _atr(self, df: pd.DataFrame) -> pd.Series:
        period = self.cfg.atr_period
        high, low, close = df["High"], df["Low"], df["Close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        return tr.rolling(period).mean()

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        x = df.copy()
        x["ema_fast"] = self._ema(x["Close"], cfg.ema_fast)
        x["ema_mid"] = self._ema(x["Close"], cfg.ema_mid)
        x["ema_slow"] = self._ema(x["Close"], cfg.ema_slow)
        x["rsi"] = self._rsi(x["Close"])
        x["atr"] = self._atr(x)
        x["atr_pct"] = x["atr"] / (x["Close"] + 1e-12)
        x["ret_1d"] = x["Close"].pct_change(1)
        x["ret_3d"] = x["Close"].pct_change(3)
        x["ret_7d"] = x["Close"].pct_change(7)
        x["hh"] = x["High"].rolling(cfg.lookback).max()
        x["ll"] = x["Low"].rolling(cfg.lookback).min()
        return x

    def _regime(self, r: pd.Series) -> str:
        if r["Close"] > r["ema_slow"] and r["ema_mid"] > r["ema_slow"]:
            return "bull"
        if r["Close"] < r["ema_slow"] and r["ema_mid"] < r["ema_slow"]:
            return "bear"
        return "sideways"

    def _score_and_signal(self, coin: str, r: pd.Series) -> CoinSignal:
        cfg = self.cfg

        long_score = 0
        long_score += int(r["Close"] > r["ema_fast"])
        long_score += int(r["ema_fast"] > r["ema_mid"])
        long_score += int(cfg.rsi_long_low <= r["rsi"] <= cfg.rsi_long_high)
        long_score += int(r["Close"] >= cfg.breakout_pct * r["hh"])
        long_score += int(bool(r["ret_3d"] > 0) and bool(r["ret_7d"] > 0))

        short_score = 0
        short_score += int(r["Close"] < r["ema_fast"])
        short_score += int(r["ema_fast"] < r["ema_mid"])
        short_score += int(cfg.rsi_short_low <= r["rsi"] <= cfg.rsi_short_high)
        short_score += int(r["Close"] <= cfg.breakdown_pct * r["ll"])
        short_score += int(bool(r["ret_3d"] < 0) and bool(r["ret_7d"] < 0))

        regime = self._regime(r)
        atr_pct = float(r["atr_pct"])
        atr_thresh = cfg.atr_high[coin]

        if long_score >= cfg.score_threshold and long_score > short_score:
            signal = 1
            strength = float(long_score)
        elif short_score >= cfg.score_threshold and short_score > long_score:
            signal = -1
            strength = float(short_score)
        else:
            signal = 0
            strength = 0.0

        # Leverage policy
        if signal == 0:
            lev = cfg.lev_weak
        elif atr_pct > atr_thresh:
            lev = cfg.lev_weak
        elif regime in ("bull", "bear") and atr_pct < atr_thresh * cfg.atr_low_multiplier:
            lev = cfg.lev_strong
        else:
            lev = cfg.lev_normal

        # Hard risk-off
        if self.prev_signal[coin] == 1 and r["ret_1d"] < -cfg.shock_exit_pct:
            signal, strength, lev = 0, 0.0, cfg.lev_weak
        if self.prev_signal[coin] == -1 and r["ret_1d"] > cfg.shock_exit_pct:
            signal, strength, lev = 0, 0.0, cfg.lev_weak

        # Trend-break exit
        if self.prev_signal[coin] == 1 and r["Close"] < r["ema_mid"] and r["ret_3d"] < 0:
            signal, strength, lev = 0, 0.0, cfg.lev_weak
        if self.prev_signal[coin] == -1 and r["Close"] > r["ema_mid"] and r["ret_3d"] > 0:
            signal, strength, lev = 0, 0.0, cfg.lev_weak

        # Time-stop
        if self.hold_days[coin] >= cfg.time_stop_days and abs(r["ret_3d"]) < cfg.time_stop_momentum:
            signal, strength, lev = 0, 0.0, cfg.lev_weak

        return CoinSignal(coin=coin, signal=signal, strength=strength, leverage=lev)

    def _allocations(self, sigs: List[CoinSignal]) -> Dict[str, float]:
        cfg = self.cfg
        active = sorted(
            [s for s in sigs if s.signal != 0],
            key=lambda x: x.strength,
            reverse=True,
        )
        alloc = {c: 0.0 for c in COINS}

        if not active:
            return alloc

        if len(active) == 1:
            alloc[active[0].coin] = cfg.alloc_single
        elif len(active) == 2:
            alloc[active[0].coin] = cfg.alloc_first
            alloc[active[1].coin] = cfg.alloc_second
        else:
            alloc[active[0].coin] = cfg.alloc_first
            alloc[active[1].coin] = cfg.alloc_second_of_three
            alloc[active[2].coin] = cfg.alloc_third

        total = sum(alloc.values())
        if total > 1.0:
            for c in alloc:
                alloc[c] /= total

        return alloc

    def predict(self, data: Dict[str, pd.DataFrame]) -> List[Dict]:
        for c in COINS:
            if len(data[c]) < self.cfg.min_bars:
                return [
                    {"coin": c0, "signal": 0, "allocation": 0.0, "leverage": 1}
                    for c0 in COINS
                ]

        sigs: List[CoinSignal] = []
        for c in COINS:
            df = self._features(data[c])
            r = df.iloc[-1]
            sigs.append(self._score_and_signal(c, r))

        alloc = self._allocations(sigs)

        out = []
        for s in sigs:
            a = float(alloc[s.coin]) if s.signal != 0 else 0.0
            out.append({
                "coin": s.coin,
                "signal": int(s.signal),
                "allocation": float(a),
                "leverage": int(s.leverage if s.signal != 0 else 1),
            })
            self.prev_signal[s.coin] = int(s.signal)
            self.hold_days[s.coin] = self.hold_days[s.coin] + 1 if s.signal != 0 else 0

        return out


def main() -> None:
    strategy = NFIStrategy()
    result = run(strategy, initial_capital=INITIAL_CAPITAL, silent=True)

    result.print_summary()


if __name__ == "__main__":
    main()

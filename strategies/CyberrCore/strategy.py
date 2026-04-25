

from cnlib.base_strategy import BaseStrategy
from cnlib import backtest
import numpy as np
import pandas as pd

COINS      = ["kapcoin-usd_train", "metucoin-usd_train", "tamcoin-usd_train"]
MAX_ALLOC  = 0.90
WARMUP     = 35
AC_WINDOW  = 30
AC_MOM     = +0.20   # bu ustunde: momentum
AC_MRV     = -0.20   # bu altinda: mean-reversion


class RegimeSwitchStrategy(BaseStrategy):

    def predict(self, data: dict) -> list[dict]:
        if len(next(iter(data.values()))) < WARMUP:
            return self._flat()

        signals:   dict[str, int]   = {}
        vols:      dict[str, float] = {}
        leverages: dict[str, int]   = {}

        for coin in COINS:
            df     = data[coin]
            closes = df["Close"]
            ret    = closes.pct_change().dropna()

            last_ret = float(ret.iloc[-1])
            vol20    = float(ret.rolling(20).std().iloc[-1])

            # Adaptif kaldirac
            if len(ret.dropna()) >= 60:
                vol_median = float(ret.rolling(60).std().dropna().median())
            else:
                vol_median = vol20

            vols[coin]      = vol20 if vol20 > 1e-9 else 0.01
            leverages[coin] = 10 if vol20 < vol_median else 5

            # Rolling AC ile rejim tespiti
            if len(ret) >= AC_WINDOW + 1:
                rolling_ac = float(ret.iloc[-AC_WINDOW:].autocorr(1))
            else:
                rolling_ac = 0.65  # yeterli veri yok, varsayilan momentum

            # Rejim secimi
            if rolling_ac > AC_MOM:
                # MOMENTUM: lag-1 yonunde git
                signals[coin] = 1 if last_ret > 0 else -1

            elif rolling_ac < AC_MRV:
                # MEAN-REVERSION: lag-1 tersine git
                signals[coin] = -1 if last_ret > 0 else 1

            else:
                # FLAT: belirsiz rejim, risk alma
                signals[coin]   = 0
                leverages[coin] = 1

        # Inverse-vol allocation
        active = [c for c in COINS if signals[c] != 0]

        if not active:
            return self._flat()

        inv_vols  = {c: 1.0 / vols[c] for c in active}
        total_inv = sum(inv_vols.values())
        allocs    = {
            c: round((inv_vols[c] / total_inv) * MAX_ALLOC, 6)
            for c in active
        }

        return [
            {
                "coin":       coin,
                "signal":     signals[coin],
                "allocation": allocs.get(coin, 0.0) if signals[coin] != 0 else 0.0,
                "leverage":   leverages[coin] if signals[coin] != 0 else 1,
            }
            for coin in COINS
        ]

    def _flat(self) -> list[dict]:
        return [
            {"coin": c, "signal": 0, "allocation": 0.0, "leverage": 1}
            for c in COINS
        ]


if __name__ == "__main__":
    strategy = RegimeSwitchStrategy()
    result   = backtest.run(strategy, initial_capital=3000.0)

    result.print_summary()

    df = result.portfolio_dataframe()
    print(f"\nPortfoy minimum : ${df['portfolio_value'].min():>20,.2f}")
    print(f"Portfoy maksimum: ${df['portfolio_value'].max():>20,.2f}")
    print(f"Islem sayisi    : {len(result.trade_history)}")

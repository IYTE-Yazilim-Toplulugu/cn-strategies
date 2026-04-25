"""
Rebalancing periyodu ve allocation optimizasyonu.
"""
import pandas as pd
from cnlib.base_strategy import BaseStrategy
from cnlib import backtest


class RobustStrategy(BaseStrategy):
    FAST, SLOW, TREND_SPAN = 5, 20, 50
    CONFIRM_FAST, CONFIRM_SLOW = 5, 15
    MIN_CANDLES = 55
    LEVERAGE = {"kapcoin-usd_train": 3, "metucoin-usd_train": 3, "tamcoin-usd_train": 5}

    def __init__(self, alloc=0.28, rebalance=20):
        super().__init__()
        self.ALLOC = alloc
        self.REBALANCE_EVERY = rebalance

    def _ema_signal(self, closes):
        if len(closes) < self.MIN_CANDLES:
            return 0
        ef = closes.ewm(span=self.FAST, adjust=False).mean().iloc[-1]
        es = closes.ewm(span=self.SLOW, adjust=False).mean().iloc[-1]
        et = closes.ewm(span=self.TREND_SPAN, adjust=False).mean().iloc[-1]
        cf = closes.ewm(span=self.CONFIRM_FAST, adjust=False).mean().iloc[-1]
        cs = closes.ewm(span=self.CONFIRM_SLOW, adjust=False).mean().iloc[-1]
        if ef > es > et and cf > cs:
            return 1
        if ef < es < et and cf < cs:
            return -1
        return 0

    def predict(self, data):
        if self.candle_index > 0 and self.candle_index % self.REBALANCE_EVERY == 0:
            return [{"coin": c, "signal": 0, "allocation": 0.0, "leverage": 1} for c in data]
        signals = {c: self._ema_signal(df["Close"]) for c, df in data.items()}
        decisions = []
        for coin in sorted(data, key=lambda c: (0 if signals[c] == 0 else 1)):
            sig = signals[coin]
            decisions.append({
                "coin": coin, "signal": sig,
                "allocation": self.ALLOC if sig != 0 else 0.0,
                "leverage": self.LEVERAGE[coin] if sig != 0 else 1,
            })
        return decisions


results = []
for rebalance in [5, 10, 15, 20, 30, 50]:
    for alloc in [0.25, 0.28, 0.30, 0.33]:
        s = RobustStrategy(alloc=alloc, rebalance=rebalance)
        r = backtest.run(s, initial_capital=3000.0, silent=True)
        results.append({
            "rebalance": rebalance, "alloc": alloc,
            "return_pct": r.return_pct,
            "final": r.final_portfolio_value,
            "trades": r.total_trades,
            "failed": getattr(r, "failed_opens", "?"),
            "liq": r.total_liquidations,
        })

results.sort(key=lambda x: x["return_pct"], reverse=True)
print(f"{'reb':>4} {'alloc':>6} {'return%':>12} {'final$':>15} {'trades':>7} {'liq':>4}")
print("-" * 55)
for r in results[:16]:
    print(f"{r['rebalance']:>4} {r['alloc']:>6.2f} {r['return_pct']:>12.1f} "
          f"{r['final']:>15,.0f} {r['trades']:>7} {r['liq']:>4}")

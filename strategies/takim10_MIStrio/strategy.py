"""
FINAL SUBMISSION STRATEGY - Paribu CodeNight
=============================================
Iki mekanizmanin birlesmesi:

1. Trailing Stop (Kar Koruma):
   Trend boyunca en yüksek noktadan %15 dususte pozisyon kapatilir.
   trailed_out bayragi ile ayni trend dalgasinda tekrar girilmez
   (stop-cascade önlemi). Bayrak ancak EMA nötre dönünce sifirlanir.

2. Rebalancing (cash/pv uyumsuzlugu duzeltmesi):
   portfolio.py allocation'i "mevcut cash" degil "portfolio_value"
   üzerinden hesaplar. Pozisyonlar 3x leverage ile buyuyunce
   cash sabit kalirken pv >> cash olur → Failed Open.
   Her 20 candle'da bir tüm pozisyonlar kapatilir, cash = pv'ye esitlenir,
   sonraki candle'da pozisyonlar guncel pv ile yeniden acilir.
   Not: yarismada transaction cost/slippage yok, bu mekanik bir duzeltme.

Iki mekanizma birbirini bozmaz:
- Trailing stop ateşlenince trailed_out=1, rebalancing bunu sifirlamaz.
- Bayrak sadece EMA nötre dönünce sifirlanir.
- Rebalancing sonrasi yeniden aciluslar trailing stop durumuna göre yapilir.
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from cnlib.base_strategy import BaseStrategy
from cnlib import backtest


class FinalStrategy(BaseStrategy):

    FAST        = 5
    SLOW        = 20
    TREND       = 50
    CONFIRM_F   = 5
    CONFIRM_S   = 15
    MIN_CANDLES = 55

    LEVERAGE        = 3     # Tüm coinler için 3x: %33 düşüşe kadar likidite yok
    ALLOC           = 0.252 # 3 * 0.252 = 0.756 → %24 nakit tamponu, 0 Failed Open
    TRAILING_PCT    = 0.15  # Zirveden %15 düşüşte trailing stop
    REBALANCE_EVERY = 20    # Her 20 candle'da cash=pv sıfırlama

    def __init__(self):
        super().__init__()
        # 0=serbest  1=long trendinde stoplandı  -1=short trendinde stoplandı
        # defaultdict: test ortamında coin isimleri _test olabilir, KeyError önlenir
        self.trailed_out = defaultdict(int)

    def _ema_signals(self, closes: pd.Series):
        ef = closes.ewm(span=self.FAST,    adjust=False).mean()
        es = closes.ewm(span=self.SLOW,    adjust=False).mean()
        et = closes.ewm(span=self.TREND,   adjust=False).mean()
        cf = closes.ewm(span=self.CONFIRM_F, adjust=False).mean()
        cs = closes.ewm(span=self.CONFIRM_S, adjust=False).mean()
        is_long  = (ef > es) & (es > et) & (cf > cs)
        is_short = (ef < es) & (es < et) & (cf < cs)
        return is_long.values, is_short.values

    def _trend_duration(self, arr: np.ndarray) -> int:
        i, dur = len(arr) - 1, 0
        while i >= 0 and arr[i]:
            dur += 1
            i -= 1
        return dur

    def _coin_signal(self, coin: str, df: pd.DataFrame) -> int:
        closes = df["Close"]
        highs  = df["High"]
        lows   = df["Low"]

        if len(closes) < self.MIN_CANDLES:
            return 0

        is_long, is_short = self._ema_signals(closes)
        price = closes.iloc[-1]

        if is_long[-1]:
            if self.trailed_out[coin] == 1:
                return 0  # Bu dalga bitti, bir sonraki trende kadar bekle
            dur = self._trend_duration(is_long)
            hh  = highs.iloc[-dur:].max() if dur > 0 else price
            if price <= hh * (1.0 - self.TRAILING_PCT):
                self.trailed_out[coin] = 1
                return 0
            return 1

        elif is_short[-1]:
            if self.trailed_out[coin] == -1:
                return 0
            dur = self._trend_duration(is_short)
            ll  = lows.iloc[-dur:].min() if dur > 0 else price
            if price >= ll * (1.0 + self.TRAILING_PCT):
                self.trailed_out[coin] = -1
                return 0
            return -1

        else:
            # Nötr: trend dalgasi bitti, bayrak sifirla
            self.trailed_out[coin] = 0
            return 0

    def predict(self, data: dict) -> list[dict]:
        # Rebalancing: pozisyonlari kapat, cash = pv olsun.
        # trailed_out bayragi korunur — trend durumu degismedi.
        if self.candle_index > 0 and self.candle_index % self.REBALANCE_EVERY == 0:
            return [
                {"coin": c, "signal": 0, "allocation": 0.0, "leverage": 1}
                for c in data
            ]

        signals = {c: self._coin_signal(c, df) for c, df in data.items()}

        # Kapatmalar once islenir → cash acilir, yeni pozisyonlar basarir
        decisions = []
        for coin in sorted(data, key=lambda c: (0 if signals[c] == 0 else 1)):
            sig = signals[coin]
            if sig != 0:
                decisions.append({
                    "coin":       coin,
                    "signal":     sig,
                    "allocation": self.ALLOC,
                    "leverage":   self.LEVERAGE,
                })
            else:
                decisions.append({
                    "coin":       coin,
                    "signal":     0,
                    "allocation": 0.0,
                    "leverage":   1,
                })
        return decisions


if __name__ == "__main__":
    strategy = FinalStrategy()
    result = backtest.run(strategy=strategy, initial_capital=3000.0)
    result.print_summary()

    pdf = result.portfolio_dataframe()
    q = len(pdf) // 4
    print("\nQUARTERLY BREAKDOWN:")
    for i in range(4):
        s, e = i * q, min((i + 1) * q - 1, len(pdf) - 1)
        sv = pdf.iloc[s]["portfolio_value"]
        ev = pdf.iloc[e]["portfolio_value"]
        print(f"  Q{i+1}: ${sv:>12,.0f} -> ${ev:>12,.0f}  ({(ev/sv-1)*100:>+8.1f}%)")

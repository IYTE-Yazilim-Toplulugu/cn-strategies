"""
Overfit testi:
1. Parametre hassasiyeti — kucuk degisimler sonucu cok etkiliyor mu?
2. Walk-forward — her 4 ceyrek ayri backteste sokulursa sonuclar tutarli mi?
3. Rebalancing periyodu hassasiyeti — 20 gun optimal mi yoksa sadece bu dataya mi uymakta?
"""
import pandas as pd
import numpy as np
from cnlib.base_strategy import BaseStrategy
from cnlib import backtest
from pathlib import Path

DATA_DIR = Path("C:/Users/yusuf/AppData/Local/Programs/Python/Python312/Lib/site-packages/cnlib/data")


def make_strategy(fast=5, slow=20, trend=50, cf=5, cs=15,
                  trailing=0.15, alloc=0.25, rebal=20):
    class S(BaseStrategy):
        FAST=fast; SLOW=slow; TREND=trend
        CONFIRM_F=cf; CONFIRM_S=cs; MIN_CANDLES=trend+5
        LEVERAGE=3; ALLOC=alloc; TRAILING_PCT=trailing; REBALANCE_EVERY=rebal

        def __init__(self):
            super().__init__()
            self.trailed_out = {c: 0 for c in
                ["kapcoin-usd_train","metucoin-usd_train","tamcoin-usd_train"]}

        def _ema_signals(self, closes):
            ef = closes.ewm(span=self.FAST,    adjust=False).mean()
            es = closes.ewm(span=self.SLOW,    adjust=False).mean()
            et = closes.ewm(span=self.TREND,   adjust=False).mean()
            cf2 = closes.ewm(span=self.CONFIRM_F, adjust=False).mean()
            cs2 = closes.ewm(span=self.CONFIRM_S, adjust=False).mean()
            return ((ef>es)&(es>et)&(cf2>cs2)).values, ((ef<es)&(es<et)&(cf2<cs2)).values

        def _dur(self, arr):
            i, d = len(arr)-1, 0
            while i>=0 and arr[i]: d+=1; i-=1
            return d

        def _coin_signal(self, coin, df):
            closes=df["Close"]; highs=df["High"]; lows=df["Low"]
            if len(closes)<self.MIN_CANDLES: return 0
            il, is_ = self._ema_signals(closes)
            p = closes.iloc[-1]
            if il[-1]:
                if self.trailed_out[coin]==1: return 0
                d=self._dur(il); hh=highs.iloc[-d:].max() if d>0 else p
                if p<=hh*(1-self.TRAILING_PCT): self.trailed_out[coin]=1; return 0
                return 1
            elif is_[-1]:
                if self.trailed_out[coin]==-1: return 0
                d=self._dur(is_); ll=lows.iloc[-d:].min() if d>0 else p
                if p>=ll*(1+self.TRAILING_PCT): self.trailed_out[coin]=-1; return 0
                return -1
            else:
                self.trailed_out[coin]=0; return 0

        def predict(self, data):
            if self.candle_index>0 and self.candle_index%self.REBALANCE_EVERY==0:
                return [{"coin":c,"signal":0,"allocation":0.0,"leverage":1} for c in data]
            sigs={c:self._coin_signal(c,df) for c,df in data.items()}
            dec=[]
            for coin in sorted(data, key=lambda c:(0 if sigs[c]==0 else 1)):
                s=sigs[coin]
                dec.append({"coin":coin,"signal":s,
                             "allocation":self.ALLOC if s!=0 else 0.0,
                             "leverage":self.LEVERAGE if s!=0 else 1})
            return dec
    return S()


# ----------------------------------------------------------------
# TEST 1: Rebalancing periyodu hassasiyeti
# ----------------------------------------------------------------
print("=" * 55)
print("TEST 1: REBALANCING HASSASIYETI (overfit gostergesi)")
print("=" * 55)
print(f"{'Periyot':>8}  {'Return%':>10}  {'Final$':>14}")
print("-" * 38)
for r in [5, 10, 15, 20, 25, 30, 40, 50]:
    s = make_strategy(rebal=r)
    res = backtest.run(s, initial_capital=3000.0, silent=True)
    print(f"{r:>8}  {res.return_pct:>10.1f}  {res.final_portfolio_value:>14,.0f}")

# ----------------------------------------------------------------
# TEST 2: Trailing stop hassasiyeti
# ----------------------------------------------------------------
print()
print("=" * 55)
print("TEST 2: TRAILING STOP HASSASIYETI")
print("=" * 55)
print(f"{'Trailing%':>9}  {'Return%':>10}  {'Final$':>14}")
print("-" * 38)
for t in [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]:
    s = make_strategy(trailing=t)
    res = backtest.run(s, initial_capital=3000.0, silent=True)
    print(f"{t*100:>8.0f}%  {res.return_pct:>10.1f}  {res.final_portfolio_value:>14,.0f}")

# ----------------------------------------------------------------
# TEST 3: Walk-forward — her ceyrek icin ayri backtest
# ----------------------------------------------------------------
print()
print("=" * 55)
print("TEST 3: WALK-FORWARD (ceyreklik tutarlilik)")
print("=" * 55)

# Tam veri uzunlugu
df0 = pd.read_parquet(DATA_DIR / "tamcoin-usd_train.parquet")
total = len(df0)
q = total // 4

print(f"{'Donem':>6}  {'Candle':>12}  {'Return%':>10}  {'Final$':>12}")
print("-" * 46)

for i in range(4):
    start = i * q
    s = make_strategy()
    res = backtest.run(s, initial_capital=3000.0, start_candle=start, silent=True)
    # start'tan itibaren sadece kalan kismini degerlendiriyoruz
    pdf = res.portfolio_dataframe()
    if len(pdf) == 0:
        continue
    sv = 3000.0
    ev = pdf.iloc[-1]["portfolio_value"]
    ret = (ev / sv - 1) * 100
    print(f"  Q{i+1}    candle {start:>4}-{total:>4}  {ret:>10.1f}  {ev:>12,.0f}")

# ----------------------------------------------------------------
# TEST 4: EMA parametre hassasiyeti
# ----------------------------------------------------------------
print()
print("=" * 55)
print("TEST 4: EMA PARAMETRE HASSASIYETI")
print("=" * 55)
print(f"{'EMA':>14}  {'Return%':>10}  {'Final$':>14}")
print("-" * 44)
for (f, s, t) in [(3,10,40),(5,20,50),(8,25,80),(10,30,100),(13,50,150)]:
    st = make_strategy(fast=f, slow=s, trend=t)
    res = backtest.run(st, initial_capital=3000.0, silent=True)
    print(f"  ({f}/{s}/{t}):   {res.return_pct:>10.1f}  {res.final_portfolio_value:>14,.0f}")

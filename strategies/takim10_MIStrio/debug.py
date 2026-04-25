"""
Stratejinin derinlemesine debugu:
1. Look-ahead bias var mi?
2. Leverage matemati dogru mu?
3. Short pozisyonlar ne kadar drag yapiyor?
4. Failed Opens neden oluyor?
5. Kaldirali getiri matematiksel olarak mantikli mi?
"""
import pandas as pd
import numpy as np
from pathlib import Path
from cnlib.base_strategy import BaseStrategy
from cnlib import backtest

DATA_DIR = Path("C:/Users/yusuf/AppData/Local/Programs/Python/Python312/Lib/site-packages/cnlib/data")

# ---------------------------------------------------------------
# TEST 1: Look-ahead bias kontrolu
# backtest.py'de data = strategy._candle_data(i) → df.iloc[:i+1]
# Yani candle i kapandiginda, i'nin Close'u da data icinde.
# Sinyal i'nin Close'uyla hesaplanip AYNI candle'in fiyatiyla islem gerceklesiyor.
# Bu "same-bar execution" — gercek piyasada yapilabilir mi?
# ---------------------------------------------------------------
print("=" * 60)
print("TEST 1: SAME-BAR EXECUTION (look-ahead degil ama dikkat)")
print("=" * 60)
print("backtest.py satir 104-131:")
print("  data    = _candle_data(i)   → iloc[:i+1] yani i dahil")
print("  prices  = current_prices(i) → Close[i]")
print("  predict(data) cagrilir      → EMA hesabinda Close[i] var")
print("  update_positions(prices)    → Close[i] ile islem acilir")
print()
print("SONUC: Look-ahead bias YOK. Ama Close[i] hem sinyali")
print("  belirliyor hem de islem fiyati — gercek piyasada ancak")
print("  'market on close' emriyle mumkun. Simulasyon icin kabul")
print("  edilebilir bir varsayim.")
print()

# ---------------------------------------------------------------
# TEST 2: Leverage matematigi — elle hesap
# ---------------------------------------------------------------
print("=" * 60)
print("TEST 2: LEVERAGE MATEMATIK DOGRULAMA")
print("=" * 60)

# Basit senaryo: $3000, allocation=0.9, leverage=5
# tamcoin: 291.89 -> 7474.80 (25.6x)
start_price = 291.89
end_price = 7474.80
capital = 3000.0
alloc = 0.9
lev = 5

# Tek pozisyon acip hic kapamayalim (teorik maksimum, gercekte EMA cikis yapar)
pos_capital = capital * alloc  # 2700
pnl = pos_capital * lev * (end_price - start_price) / start_price
final = capital + pnl
print(f"Teorik: $3000 ile tamcoin'e {alloc*100:.0f}% alloc, {lev}x leverage")
print(f"  Pozisyon sermayesi: ${pos_capital:,.0f}")
print(f"  Fiyat degisimi: {(end_price/start_price - 1)*100:.1f}%")
print(f"  PnL: ${pnl:,.0f}")
print(f"  Final: ${final:,.0f}")
print()

# GERCEK COMPOUNDING: Her kapanip acilindiginda portfoy buyugunden yeni pozisyon daha buyuk
print("Compounding etkisi: portfoy buyudukce yeni pozisyon da buyuyor")
print("Ornek: $3000 → $10000 oldugunda ayni %90 alloc → $9000 pozisyon")
print("  Leverage ile bu geometrik buyume katlanir")
print()

# ---------------------------------------------------------------
# TEST 3: Short pozisyonlarin etkisi (long-only vs long+short)
# ---------------------------------------------------------------
print("=" * 60)
print("TEST 3: SHORT POZISYONLARIN ETKISI")
print("=" * 60)

class LongOnlyStrategy(BaseStrategy):
    def predict(self, data):
        decisions = []
        sigs = {}
        for coin, df in data.items():
            closes = df["Close"]
            if len(closes) < 55:
                sigs[coin] = 0
                continue
            ef = closes.ewm(span=5, adjust=False).mean().iloc[-1]
            es = closes.ewm(span=20, adjust=False).mean().iloc[-1]
            et = closes.ewm(span=50, adjust=False).mean().iloc[-1]
            cf = closes.ewm(span=5, adjust=False).mean().iloc[-1]
            cs = closes.ewm(span=15, adjust=False).mean().iloc[-1]
            if ef > es > et and cf > cs:
                sigs[coin] = 1
            elif ef > es > et:
                sigs[coin] = 1
            else:
                sigs[coin] = 0
        active = [c for c, s in sigs.items() if s == 1]
        alloc = round(0.70 / len(active), 4) if active else 0.0
        for coin in data:
            s = sigs.get(coin, 0)
            decisions.append({"coin": coin, "signal": s,
                               "allocation": alloc if s == 1 else 0.0, "leverage": 3})
        return decisions

class LongShortStrategy(BaseStrategy):
    def predict(self, data):
        decisions = []
        sigs = {}
        for coin, df in data.items():
            closes = df["Close"]
            if len(closes) < 55:
                sigs[coin] = 0
                continue
            ef = closes.ewm(span=5, adjust=False).mean().iloc[-1]
            es = closes.ewm(span=20, adjust=False).mean().iloc[-1]
            et = closes.ewm(span=50, adjust=False).mean().iloc[-1]
            cf = closes.ewm(span=5, adjust=False).mean().iloc[-1]
            cs = closes.ewm(span=15, adjust=False).mean().iloc[-1]
            if ef > es > et and cf > cs:
                sigs[coin] = 1
            elif ef < es < et and cf < cs:
                sigs[coin] = -1
            else:
                sigs[coin] = 0
        active = [c for c, s in sigs.items() if s != 0]
        alloc = round(0.70 / len(active), 4) if active else 0.0
        coins_sorted = sorted(data.keys(), key=lambda c: (0 if sigs.get(c, 0) == 0 else 1))
        for coin in coins_sorted:
            s = sigs.get(coin, 0)
            decisions.append({"coin": coin, "signal": s,
                               "allocation": alloc if s != 0 else 0.0, "leverage": 3})
        return decisions

r_long = backtest.run(LongOnlyStrategy(), initial_capital=3000.0, silent=True)
r_ls   = backtest.run(LongShortStrategy(), initial_capital=3000.0, silent=True)

print(f"  Long-only  → ${r_long.final_portfolio_value:>12,.0f}  ({r_long.return_pct:>+.1f}%)")
print(f"  Long+Short → ${r_ls.final_portfolio_value:>12,.0f}  ({r_ls.return_pct:>+.1f}%)")
print(f"  Short drag: ${r_long.final_portfolio_value - r_ls.final_portfolio_value:>+,.0f}")
print()

# ---------------------------------------------------------------
# TEST 4: Failed Opens analizi
# ---------------------------------------------------------------
print("=" * 60)
print("TEST 4: FAILED OPENS ANALIZI (283 BASARISIZ ISLEM)")
print("=" * 60)
print("portfolio.py satir 131-137:")
print("  allocated_capital = portfolio_value * allocation")
print("  if allocated_capital > self.cash:")
print("      return error  ← Failed Open buradan geliyor")
print()
print("NEDEN OLUYOR?")
print("  Portfoy degeri = nakit + acik pozisyonlarin degeri")
print("  Ama 'acik pozisyon degeri' nakit degil — cekiverilemsz.")
print("  Ornek: portfoy=$10K, nakit=$1K, pozisyonlarda=$9K")
print("  allocation=0.233 → 0.233 * $10K = $2.3K isteniyor")
print("  Ama kasada sadece $1K var → FAILED OPEN")
print()

# Gorselleştir: ne zaman oluyor?
print("COZUM: allocation'i portfoy_value degil, cash uzerinden hesapla")
print("  veya toplam allokayon daha dusuk tut (ornegin %50)")
print()

# ---------------------------------------------------------------
# TEST 5: $31M sonucunun matematiksel kontrolu
# ---------------------------------------------------------------
print("=" * 60)
print("TEST 5: $31M SONUCUNUN MATEMATIKSEL DOGRULAMA")
print("=" * 60)

class SimpleEMA3Strategy(BaseStrategy):
    def predict(self, data):
        sigs = {}
        for coin, df in data.items():
            closes = df["Close"]
            if len(closes) < 45:
                sigs[coin] = 0
                continue
            ef = closes.ewm(span=3, adjust=False).mean().iloc[-1]
            es = closes.ewm(span=10, adjust=False).mean().iloc[-1]
            et = closes.ewm(span=40, adjust=False).mean().iloc[-1]
            sigs[coin] = 1 if ef > es > et else 0
        active = [c for c, s in sigs.items() if s == 1]
        alloc = round(0.9 / len(active), 4) if active else 0.0
        lev = {"kapcoin-usd_train": 3, "metucoin-usd_train": 3, "tamcoin-usd_train": 5}
        return [{"coin": c, "signal": sigs[c],
                 "allocation": alloc if sigs[c] == 1 else 0.0,
                 "leverage": lev[c] if sigs[c] == 1 else 1} for c in data]

r_ema3 = backtest.run(SimpleEMA3Strategy(), initial_capital=3000.0, silent=True)
print(f"  EMA(3/10/40) + (3/3/5)x → ${r_ema3.final_portfolio_value:>15,.0f}")
print(f"  Return: {r_ema3.return_pct:>+.1f}%")
print(f"  Toplam trade: {r_ema3.total_trades}")
print(f"  Liquidation: {r_ema3.total_liquidations}")
print()

# Tamcoin'in EMA3 ile kac candle long pozisyonda oldugu
df_tam = pd.read_parquet(DATA_DIR / "tamcoin-usd_train.parquet")
closes = df_tam["Close"]
ef = closes.ewm(span=3, adjust=False).mean()
es = closes.ewm(span=10, adjust=False).mean()
et = closes.ewm(span=40, adjust=False).mean()
in_long = (ef > es) & (es > et)
print(f"  tamcoin EMA3 long durumunda kac gun: {in_long.sum()} / {len(in_long)}")
print(f"  Bu surede tamcoin fiyati: "
      f"{closes[in_long].iloc[0]:.1f} → {closes[in_long].iloc[-1]:.1f}")

# ---------------------------------------------------------------
# TEST 6: Out-of-sample robustness — veriyi yariya bol
# ---------------------------------------------------------------
print()
print("=" * 60)
print("TEST 6: OVERFITTING KONTROLU — VERIYI IKIYE BOL")
print("=" * 60)
print("Ilk 785 candle (yarim) uzerinde EMA(3/10/40) performansi:")

class SimpleEMA3StrategyHalf(BaseStrategy):
    def predict(self, data):
        sigs = {}
        for coin, df in data.items():
            closes = df["Close"]
            if len(closes) < 45:
                sigs[coin] = 0
                continue
            ef = closes.ewm(span=3, adjust=False).mean().iloc[-1]
            es = closes.ewm(span=10, adjust=False).mean().iloc[-1]
            et = closes.ewm(span=40, adjust=False).mean().iloc[-1]
            sigs[coin] = 1 if ef > es > et else 0
        active = [c for c, s in sigs.items() if s == 1]
        alloc = round(0.9 / len(active), 4) if active else 0.0
        lev = {"kapcoin-usd_train": 3, "metucoin-usd_train": 3, "tamcoin-usd_train": 5}
        return [{"coin": c, "signal": sigs[c],
                 "allocation": alloc if sigs[c] == 1 else 0.0,
                 "leverage": lev[c] if sigs[c] == 1 else 1} for c in data]

# Strateji tam veriyle calisacak ama biz sadece ilk yarinin sonucuna bakacagiz
r_full = backtest.run(SimpleEMA3StrategyHalf(), initial_capital=3000.0, silent=True)
pdf = r_full.portfolio_dataframe()

half = len(pdf) // 2
val_half = pdf.iloc[half]["portfolio_value"]
val_full = pdf.iloc[-1]["portfolio_value"]
ret_first = (val_half / 3000 - 1) * 100
ret_second = (val_full / val_half - 1) * 100

print(f"  Ilk yari (785 candle):  $3,000 → ${val_half:>12,.0f}  ({ret_first:>+.1f}%)")
print(f"  Ikinci yari (785 candle): ${val_half:,.0f} → ${val_full:>12,.0f}  ({ret_second:>+.1f}%)")
print()
print("YORUM: Her iki yaride de guclu getiri gozlemleniyorsa")
print("  strateji TRENDLERE duyarli, veri icin overfitted degil.")
print("  Ancak parametreler (EMA 3/10/40) IN-SAMPLE optimize edildi,")
print("  farkli coin davranislarina hassastir.")

# Kafein - Mimari Yapisi

Bu submission'in ana dosyasi `strategy.py` dosyasidir. Strateji `HybridStrategy` sinifi uzerinden calisir ve CNLIB `BaseStrategy` yapisina uygun olarak her candle kapanisinda kapcoin, metucoin ve tamcoin icin karar uretir.

## Genel Fikir

Strateji tek bir sinyale guvenmez. ML tahmini, momentum onayi, piyasa rejimi ve volatilite filtresini birlikte degerlendirir. Uygun ortam yoksa islem acmamak da bilincli bir karardir.

```text
ML sinyali + momentum onayi + rejim filtresi + volatilite kontrolu = final trade karari
```

## Bilesenler

- **Veri yukleme:** `get_data()` ile CNLIB train verisi okunur.
- **Feature uretimi:** Return, momentum, hareketli ortalama, volatilite, RSI, MACD, Bollinger ve hacim tabanli teknik feature'lar hesaplanir.
- **Model:** `GradientBoostingClassifier` kullanilir. Model, bir sonraki fiyat yonunu tahmin eden yardimci sinyal uretir.
- **Momentum onayi:** Kisa vadeli fiyat davranisi ML sinyaliyle ayni yondeyse trade adayi guclenir.
- **Rejim filtresi:** Son 100 gunluk return autocorrelation izlenir. Momentum karakteri zayifsa strateji daha korumaci davranir.
- **Volatilite kontrolu:** Piyasa cok oynakken pozisyon boyutu dusurulur.
- **Risk ayari:** Allocation ve leverage sinyal gucu ile piyasa kosuluna gore belirlenir; toplam pozisyon riski sinirli tutulur.

## Calisma Akisi

1. `HybridStrategy` olusturulur.
2. `get_data()` ile coin verileri yuklenir.
3. `egit()` ile feature'lar uretilir ve model egitilir.
4. Backtest sirasinda `predict()` her candle icin karar verir.
5. Cikti olarak her coin icin long, short veya flat yonunde aksiyon uretilir.

## Dosya Yapisi

```text
strategies/
  Kafein/
    strategy.py              # zorunlu ana strateji dosyasi
    MIMARI_YAPISI.md         # juri icin mimari ozeti
    MODEL_EGITIM_NOTU.md     # model egitimi ve calistirma notu
```

Strateji disaridan ek CSV, UI dosyasi veya kayitli model dosyasi gerektirmez. Gerekli model runtime'da egitilir; bunun icin `get_data()` sonrasi `egit()` akisi calistirilmalidir.

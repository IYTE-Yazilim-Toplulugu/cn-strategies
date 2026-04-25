# Model Egitim Notu

Bu submission icinde ayrica `model.pkl` veya dis model dosyasi yoktur. Ana dosya `strategy.py` icindeki `HybridStrategy` sinifidir.

## Otomatik Egitim Davranisi

`strategy.py` dosyasi dogrudan calistirildiginda model otomatik egitilir. Dosyanin en altindaki `__main__` akisi su sirayi izler:

```text
HybridStrategy olusturulur
get_data() ile CNLIB verisi yuklenir
egit() ile model egitilir
backtest.run(...) calistirilir
```

Yani juri dosyayi script olarak calistirirse ek bir egitim komutuna gerek yoktur.

## Import Edilerek Calistirma Notu

Eger degerlendirme sistemi `strategy.py` dosyasini import edip sadece `HybridStrategy()` nesnesi olusturur ve `egit()` cagirmadan `predict()` calistirirsa, mevcut kodda model egitilmis olmaz. Bu durumda `self.model is None` kalir ve ML tabanli sinyaller devreye girmez.

Bu nedenle en guvenli calistirma akisi sudur:

```python
strategy = HybridStrategy()
strategy.get_data()
strategy.egit()
backtest.run(strategy=strategy, ...)
```

Bu dosya sadece aciklama notudur; strateji mantiginda herhangi bir degisiklik yapilmaz.

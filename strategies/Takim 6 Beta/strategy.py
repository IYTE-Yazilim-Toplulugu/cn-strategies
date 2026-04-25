"""
codenight — cnlib hackathon stratejisi (3 hayali coin).

Yarisma kurallari (ozet)
------------------------
- Egitim: ~4 yillik train verisi; degerlendirme train *sonrasi* 1 yil (o veri
  stratejiye verilmez; `predict()` yalnizca `data` ile gelen gecmise bakar).
- Look-ahead yasak: `fit` / `egit` **predict() icinde yok**; sadece `predict_proba`.
- Degerlendirme metrigi: final portfoy degeri (yarisma sihhi).

Bench: ``python -m bench --algorithms algorithms.karslioglu``

MLStrateji: `get_data()` yukler, `egit(train_end)` ile **tek sefer** model kurar.
Uzun parquet (4y+1y birlesik) varsayiminda son `_EVAL_WINDOW_RESERVE` gun egitime
alınmaz. Paket dagitimindaki ~1570 gunluk dosya tipik olarak sadece train → tamami
egitime girer.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from cnlib import backtest
from cnlib.base_strategy import BaseStrategy, COINS
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier


@dataclass(frozen=True)
class ParametreSeti:
    long_threshold: float
    short_threshold: float
    leverage_strength: float
    max_positions: int
    max_allocation_per_position: float
    # Muhafazakâr mod: kısa devre / likidite riskini azaltmak için
    allow_short: bool = True
    allow_leverage2: bool = True
    min_prob_edge: float = 0.0
    # Son N mumun günlük getiri std'si bu değeri aşarsa o coinde yeni sinyal yok (None = filtre yok)
    max_daily_vol: float | None = None
    # Seçilen en iyi adayın (|p-0.5|) gücü; altındayse tüm portföy nakit
    min_strength: float = 0.0
    # Üç coinin ort. vol'u aşılırsa o mumda hiç pozisyon açma (piyasa genel stresi)
    market_mean_vol_max: float | None = None
    # True ise long sadece MA(5) > MA(20) (son mum)
    long_requires_uptrend: bool = False
    # Iki long aday varsa P(↑) farki; altinda belirsizlik -> nakit
    min_prob_gap_second: float = 0.0
    # 0..1 risk skoru; uzeri coinde pozisyon yok (None = filtre yok)
    max_risk_skoru: float | None = None
    # Tahsis carpanı: max(0.2, 1 - risk * ceza); 0 = riskle kucultme yok
    risk_alloc_penalty: float = 0.0
    # Long acilista stop_loss fiyatı: giris * (1 - bu oran); None = yok
    stop_loss_pct_long: float | None = None
    # P(yukari) bu kadar yuksekse MA(5)>MA(20) sarti aranmaz (2018 rebound vb.)
    long_uptrend_bypass_prob: float | None = None
    # Short: guclu piyasa (cok coin MA5>MA20) iken short yok; coin bazinda dusus trendi sart
    short_requires_downtrend: bool = True
    short_disable_if_uptrend_coins_ge: int | None = 2
    stop_loss_pct_short: float | None = None
    # 2x: strength >= leverage_strength (long icin up ~ 0.5+strength). None = vol sarti yok
    leverage_max_vol_frac: float | None = None
    # En az N coin MA5>MA20 ve piyasa vol sakin ise long olasilik esigi bu kadar dusur (tavan ~0.58)
    bull_regime_long_relief: float | None = None
    bull_regime_vol_frac: float | None = None  # relief sadece ort vol < market_mean_vol_max * bu
    # Az coin uptrend (stres/ayi) iken long esigini yukselt — chop'ta asiri islem azalir
    stress_long_elev: float | None = None
    stress_max_uptrend_coins: int | None = None  # uptrend sayisi <= bu ise stress
    # Short: up_prob < short_ust - margin (marjinal shortlari kes)
    short_prob_margin: float = 0.0
    # Bu altinda islem acma (kucuk pozisyon / slip / gereksiz risk)
    min_effective_allocation: float | None = None
    # Tahsisi vol ile olcekle: alloc *= min(1, vol_alloc_ref / gunluk_vol_10); None = kapali
    vol_alloc_ref: float | None = None


def varsayilan_yarismaci_parametreleri() -> ParametreSeti:
    """
    Bilinmeyen 1Y testi için: tek dönemde şişirilmiş getiri yerine
    düşük MDD, gereksiz işlem yok, oynak/stres döneminde nakit.
    """
    # Denge + zarar farkindaligi: asagidaki min_eff / vol_alloc / loss-weight egitim ile desteklenir.
    return ParametreSeti(
        long_threshold=0.652,
        short_threshold=0.265,
        leverage_strength=0.37,
        max_positions=1,
        max_allocation_per_position=0.148,
        allow_short=True,
        allow_leverage2=True,
        min_prob_edge=0.038,
        max_daily_vol=0.0405,
        min_strength=0.084,
        market_mean_vol_max=0.037,
        long_requires_uptrend=True,
        min_prob_gap_second=0.0185,
        max_risk_skoru=0.828,
        risk_alloc_penalty=0.44,
        stop_loss_pct_long=0.05,
        long_uptrend_bypass_prob=0.742,
        short_requires_downtrend=True,
        short_disable_if_uptrend_coins_ge=2,
        stop_loss_pct_short=0.052,
        leverage_max_vol_frac=0.88,
        bull_regime_long_relief=0.034,
        bull_regime_vol_frac=0.96,
        stress_long_elev=0.026,
        stress_max_uptrend_coins=0,
        short_prob_margin=0.028,
        min_effective_allocation=0.023,
        vol_alloc_ref=0.027,
    )


class MLStrateji(BaseStrategy):
    """
    RF + (yeterli veride) HistGradientBoosting; risk kurallari + stop.
    Hackathon: egitim yalnizca `egit()`; `predict()` sadece tahmin + kurallar.
    """

    # Tek parquet ~4y train ise tamami kullanilir; daha uzunsa son yil egitim disi.
    _PARQUET_TYPICAL_TRAIN_ONLY_MAX = 1650
    _EVAL_WINDOW_RESERVE = 365

    _ENSEMBLE_RF_AGIRLIK = 0.52
    # Etiket: yarin getirisi bu esigi gecerse 1 (biraz dusuk = daha dengeli sinif)
    _LABEL_MIN_FWD_RET = 0.00028
    # Son mumlara hafif agirlik (rejim kaymasi)
    _RECENCY_WEIGHT_SPAN = 0.24

    def __init__(
        self,
        params: ParametreSeti | None = None,
        trade_start: int | None = None,
        trade_horizon: int | None = None,
    ) -> None:
        super().__init__()
        if params is None:
            params = varsayilan_yarismaci_parametreleri()
        self.params = params
        self.trade_start = trade_start
        self.trade_horizon = trade_horizon
        self.modeller: dict[str, dict[str, Any]] = {}
        self._initial_train_end: int = 0

    def get_data(self, data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
        out = super().get_data(data_dir)
        fd = getattr(self, "_full_data", None)
        if fd:
            self._full_data = {k: v.copy() for k, v in fd.items()}
        elif self.coin_data:
            self._full_data = {c: df.copy() for c, df in self.coin_data.items()}
        else:
            self._full_data = {}
        lens = [len(self._full_data[c]) for c in COINS if c in self._full_data]
        n = max(lens) if lens else 0
        self._fit_son = n
        if n <= MLStrateji._PARQUET_TYPICAL_TRAIN_ONLY_MAX:
            self._initial_train_end = n
        else:
            self._initial_train_end = max(150, n - MLStrateji._EVAL_WINDOW_RESERVE)
        self.egit(self._initial_train_end)
        return out

    def _ozellik_ve_etiket(self, df):
        closes = df["Close"].astype(float).replace([np.inf, -np.inf], np.nan)
        if (closes <= 0).any() or closes.isna().any():
            closes = closes.where(closes > 0, np.nan).ffill().bfill()

        high = df["High"].astype(float).replace([np.inf, -np.inf], np.nan)
        low = df["Low"].astype(float).replace([np.inf, -np.inf], np.nan)
        high = high.ffill().bfill()
        low = low.ffill().bfill()
        hl_range = (high - low) / closes.replace(0, np.nan)
        hl_range = hl_range.replace([np.inf, -np.inf], np.nan)

        if "Volume" in df.columns:
            vol = df["Volume"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        else:
            vol = pd.Series(0.0, index=df.index)
        vma = vol.rolling(20, min_periods=5).mean()
        vol_ratio = vol / vma.replace(0, np.nan)
        vol_ratio = vol_ratio.replace([np.inf, -np.inf], np.nan)

        returns_1 = closes.pct_change()
        returns_5 = closes.pct_change(5)
        returns_10 = closes.pct_change(10)
        returns_20 = closes.pct_change(20)
        ma5 = closes.rolling(5).mean()
        ma20 = closes.rolling(20).mean()
        ma60 = closes.rolling(60, min_periods=20).mean()
        volatility_10 = returns_1.rolling(10).std()
        m20 = ma20.replace(0, np.nan)
        ma_ratio = (ma5 / m20) - 1.0
        ma_ratio = ma_ratio.replace([np.inf, -np.inf], np.nan)
        m60 = ma60.replace(0, np.nan)
        close_vs_ma60 = (closes / m60) - 1.0
        close_vs_ma60 = close_vs_ma60.replace([np.inf, -np.inf], np.nan)

        delta = closes.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        ag = gain.rolling(14, min_periods=10).mean()
        al = loss.rolling(14, min_periods=10).mean()
        rs = ag / (al + 1e-12)
        rsi = (100.0 - (100.0 / (1.0 + rs))) / 100.0
        rsi = rsi.replace([np.inf, -np.inf], np.nan)

        X = np.column_stack(
            [
                returns_1,
                returns_5,
                returns_10,
                returns_20,
                ma_ratio,
                close_vs_ma60,
                volatility_10,
                hl_range,
                vol_ratio,
                rsi,
            ]
        )
        fwd_ret = closes.shift(-1) / closes - 1.0
        y = (fwd_ret > MLStrateji._LABEL_MIN_FWD_RET).astype(int).to_numpy()

        valid = np.isfinite(X).all(axis=1) & np.isfinite(fwd_ret.to_numpy())
        if not valid.any():
            return np.empty((0, 10)), np.empty((0,), dtype=int)
        valid = valid.copy()
        valid[-1] = False  # son satirin etiketi yok (gelecek candle bilinmez)
        return X[valid], y[valid]

    def _son_gunluk_vol_10(self, df) -> float:
        """Son ~10 mumun kapanış getirisi std (NaN: yeterli veri yok)."""
        c = df["Close"].astype(float).replace([np.inf, -np.inf], np.nan)
        c = c.where(c > 0, np.nan).ffill().bfill()
        r = c.pct_change()
        tail = r.tail(10).dropna()
        if len(tail) < 4:
            return float("nan")
        with np.errstate(invalid="ignore", divide="ignore"):
            s = float(tail.std())
        if not np.isfinite(s):
            return float("nan")
        return s

    def _yeni_rf(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=480,
            max_depth=7,
            min_samples_leaf=6,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=1,
        )

    def _yeni_hgb(self) -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            max_iter=320,
            learning_rate=0.048,
            max_depth=6,
            min_samples_leaf=26,
            l2_regularization=0.78,
            early_stopping=True,
            validation_fraction=0.11,
            n_iter_no_change=16,
            class_weight="balanced",
            random_state=42,
        )

    def _recency_sample_weight(self, n: int) -> np.ndarray:
        if n <= 0:
            return np.array([], dtype=float)
        span = MLStrateji._RECENCY_WEIGHT_SPAN
        t = np.linspace(-span, 0.0, n, dtype=float)
        w = np.exp(t)
        w *= n / np.sum(w)
        return w.astype(float)

    def _fit_coin_models(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        sw = self._recency_sample_weight(len(X))
        rf = self._yeni_rf()
        rf.fit(X, y, sample_weight=sw)
        out: dict[str, Any] = {"rf": rf}
        if len(np.unique(y)) < 2 or len(X) < 96:
            return out
        hgb = self._yeni_hgb()
        try:
            hgb.fit(X, y, sample_weight=sw)
        except (ValueError, TypeError):
            try:
                hgb.fit(X, y)
            except ValueError:
                return out
        out["hgb"] = hgb
        return out

    @staticmethod
    def _ensemble_up_prob(models: dict[str, Any], x_row: np.ndarray) -> float:
        rf = models["rf"]
        pr = float(rf.predict_proba([x_row])[0, 1])
        hgb = models.get("hgb")
        if hgb is None:
            return pr
        ph = float(hgb.predict_proba([x_row])[0, 1])
        w = MLStrateji._ENSEMBLE_RF_AGIRLIK
        return w * pr + (1.0 - w) * ph

    def _long_trend_up(self, dfc) -> bool:
        """Kısa ort > uzun ort (son kapanış); aksi düşen bıçaklarda long yok."""
        c = dfc["Close"].astype(float)
        if len(c) < 22:
            return True
        ma5 = c.rolling(5).mean()
        ma20 = c.rolling(20).mean()
        a, b = ma5.iloc[-1], ma20.iloc[-1]
        if np.isnan(a) or np.isnan(b):
            return True
        return bool(a > b)

    def _coin_risk_skoru(self, dfc) -> float:
        """
        0 = sakin, 1 = cok riskli (son ~15 gun getiri oynakligi + kisa vadeli drawdown).
        Hangi coinin 'zit' oldugunu ayirt etmek icin; yuksek skor = daha az para / girme.
        """
        c = dfc["Close"].astype(float).replace([np.inf, -np.inf], np.nan)
        c = c.where(c > 0, np.nan).ffill().bfill()
        if len(c) < 12:
            return 0.35
        r = c.pct_change().tail(15).dropna()
        if len(r) < 4:
            return 0.35
        vol = float(r.std())
        tail = c.tail(21)
        peak = tail.cummax()
        dd_ser = (tail - peak) / peak.replace(0, np.nan)
        dd = float(dd_ser.min()) if dd_ser.notna().any() else 0.0
        dd = max(0.0, -dd)
        vol_n = min(1.0, vol / 0.055)
        dd_n = min(1.0, dd * 3.5)
        return float(np.clip(0.55 * vol_n + 0.45 * dd_n, 0.0, 1.0))

    def egit(self, train_end: int) -> None:
        src: dict = getattr(self, "_full_data", None) or getattr(self, "coin_data", {}) or {}
        for coin in COINS:
            if coin not in src or train_end <= 0:
                continue
            df = src[coin].iloc[:train_end].copy()
            X, y = self._ozellik_ve_etiket(df)
            if len(X) < 100:
                continue

            self.modeller[coin] = self._fit_coin_models(X, y)

    def predict(self, data: dict) -> list[dict]:
        decisions = {
            coin: {"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1}
            for coin in COINS
        }
        current_idx = len(data[COINS[0]]) - 1
        if self.trade_start is not None:
            if current_idx < self.trade_start:
                return [decisions[coin] for coin in COINS]
            if self.trade_horizon is not None and current_idx >= (self.trade_start + self.trade_horizon):
                return [decisions[coin] for coin in COINS]

        candidates: list[dict] = []

        p = self.params
        long_eşik = max(p.long_threshold, 0.5 + p.min_prob_edge)
        short_üst = min(p.short_threshold, 0.5 - p.min_prob_edge) if p.allow_short else 0.0
        short_kesim = (
            (short_üst - float(p.short_prob_margin)) if p.allow_short else 1.0
        )

        vlist: list[float] = []
        for cn in COINS:
            v = self._son_gunluk_vol_10(data[cn])
            if not np.isnan(v):
                vlist.append(v)
        if (
            p.market_mean_vol_max is not None
            and len(vlist) == 3
            and (sum(vlist) / 3.0) > p.market_mean_vol_max
        ):
            return [decisions[coin] for coin in COINS]

        n_uptrend = sum(1 for cn in COINS if self._long_trend_up(data[cn]))
        mean_vol = (sum(vlist) / len(vlist)) if len(vlist) == 3 else float("nan")
        long_eşik_eff = float(long_eşik)
        if (
            p.bull_regime_long_relief is not None
            and p.bull_regime_long_relief > 0.0
            and n_uptrend >= 2
            and len(vlist) == 3
            and np.isfinite(mean_vol)
        ):
            cap_ref = (
                float(p.market_mean_vol_max)
                if p.market_mean_vol_max is not None
                else 0.045
            )
            cap = cap_ref * float(p.bull_regime_vol_frac or 1.0)
            if mean_vol <= cap:
                long_eşik_eff = max(0.58, long_eşik_eff - float(p.bull_regime_long_relief))
        if (
            p.stress_long_elev is not None
            and p.stress_long_elev > 0.0
            and p.stress_max_uptrend_coins is not None
            and n_uptrend <= p.stress_max_uptrend_coins
        ):
            long_eşik_eff = min(0.78, long_eşik_eff + float(p.stress_long_elev))
        short_piyasa_kapali = (
            p.allow_short
            and p.short_disable_if_uptrend_coins_ge is not None
            and n_uptrend >= p.short_disable_if_uptrend_coins_ge
        )

        for coin in COINS:
            dfc = data[coin]
            if coin not in self.modeller:
                continue

            if p.max_daily_vol is not None:
                vol = self._son_gunluk_vol_10(dfc)
                if not np.isnan(vol) and vol > p.max_daily_vol:
                    continue

            X, _ = self._ozellik_ve_etiket(dfc)
            if len(X) == 0:
                continue

            up_prob = self._ensemble_up_prob(self.modeller[coin], X[-1])
            risk_s = self._coin_risk_skoru(dfc)
            if p.max_risk_skoru is not None and risk_s > p.max_risk_skoru:
                continue

            if up_prob > long_eşik_eff:
                need_up = p.long_requires_uptrend
                if (
                    need_up
                    and p.long_uptrend_bypass_prob is not None
                    and up_prob >= float(p.long_uptrend_bypass_prob)
                ):
                    need_up = False
                if need_up and not self._long_trend_up(dfc):
                    continue
                candidates.append(
                    {
                        "coin": coin,
                        "strength": float(up_prob - 0.5),
                        "signal": 1,
                        "up_prob": float(up_prob),
                        "risk": float(risk_s),
                        "entry_px": float(dfc["Close"].iloc[-1]),
                    }
                )
            elif (
                p.allow_short
                and not short_piyasa_kapali
                and up_prob < short_kesim
                and (not p.short_requires_downtrend or not self._long_trend_up(dfc))
            ):
                candidates.append(
                    {
                        "coin": coin,
                        "strength": float(0.5 - up_prob),
                        "signal": -1,
                        "up_prob": float(up_prob),
                        "risk": float(risk_s),
                        "entry_px": float(dfc["Close"].iloc[-1]),
                    }
                )

        longs = [c for c in candidates if c["signal"] == 1]
        if len(longs) >= 2 and p.min_prob_gap_second > 0.0:
            longs_sorted = sorted(longs, key=lambda x: x["up_prob"], reverse=True)
            if longs_sorted[0]["up_prob"] - longs_sorted[1]["up_prob"] < p.min_prob_gap_second:
                return [decisions[coin] for coin in COINS]

        candidates.sort(key=lambda x: x["strength"], reverse=True)
        if candidates and p.min_strength > 0.0 and candidates[0]["strength"] < p.min_strength:
            return [decisions[coin] for coin in COINS]
        selected = candidates[: p.max_positions]
        total_strength = sum(float(c["strength"]) for c in selected)

        for c in selected:
            coin = c["coin"]
            strength = float(c["strength"])
            signal = int(c["signal"])
            risk_s = float(c["risk"])
            entry_px = float(c["entry_px"])
            if total_strength > 0:
                base_alloc = min(
                    p.max_allocation_per_position,
                    max(0.0, strength / total_strength),
                )
            else:
                base_alloc = 0.0
            if p.risk_alloc_penalty > 0.0:
                scale = max(0.2, 1.0 - risk_s * p.risk_alloc_penalty)
                base_alloc *= scale
            allocation = float(base_alloc)
            if p.vol_alloc_ref is not None:
                v10p = self._son_gunluk_vol_10(data[coin])
                if not np.isnan(v10p) and v10p > 1e-9:
                    allocation *= min(1.0, float(p.vol_alloc_ref) / v10p)
            if (
                p.min_effective_allocation is not None
                and allocation < float(p.min_effective_allocation)
            ):
                continue

            want_lev2 = p.allow_leverage2 and strength >= p.leverage_strength
            if (
                want_lev2
                and p.leverage_max_vol_frac is not None
                and p.max_daily_vol is not None
            ):
                v10 = self._son_gunluk_vol_10(data[coin])
                if np.isnan(v10) or v10 > float(p.max_daily_vol) * float(p.leverage_max_vol_frac):
                    want_lev2 = False
            leverage = 2 if want_lev2 else 1

            sl: float | None = None
            if signal == 1 and p.stop_loss_pct_long is not None and p.stop_loss_pct_long > 0:
                sl = entry_px * (1.0 - float(p.stop_loss_pct_long))
            elif (
                signal == -1
                and p.stop_loss_pct_short is not None
                and p.stop_loss_pct_short > 0
            ):
                sl = entry_px * (1.0 + float(p.stop_loss_pct_short))

            decisions[coin] = {
                "coin": coin,
                "signal": signal,
                "allocation": allocation,
                "leverage": leverage,
                "stop_loss": sl,
            }

        out_list = [decisions[coin] for coin in COINS]
        tot_al = sum(float(d["allocation"]) for d in out_list if int(d["signal"]) != 0)
        if tot_al > 1.0 + 1e-9:
            sc = 1.0 / tot_al
            for d in out_list:
                if int(d["signal"]) != 0:
                    d["allocation"] = float(d["allocation"]) * sc
        return out_list


def veri_ozeti_yazdir(strateji: MLStrateji) -> None:
    print("\nVeri Ozeti")
    print("-" * 55)
    for coin in COINS:
        df = strateji.coin_data[coin]
        start_date = df["Date"].iloc[0]
        end_date = df["Date"].iloc[-1]
        rows = len(df)
        last_closes = ", ".join(f"{v:.2f}" for v in df["Close"].tail(3))
        print(
            f"{coin:20} | rows: {rows:4d} | "
            f"date: {start_date} -> {end_date} | last closes: [{last_closes}]"
        )


def backtest_calistir(
    params: ParametreSeti,
    start_candle: int = 0,
    trade_start: int | None = None,
    trade_horizon: int | None = None,
):
    strateji = MLStrateji(params=params, trade_start=trade_start, trade_horizon=trade_horizon)
    strateji.get_data()

    # cnlib 0.1.4 ile modeli predict() icinde artan veriyle online guncelliyoruz.
    sonuc = backtest.run(
        strategy=strateji,
        initial_capital=3000.0,
        start_candle=start_candle,
    )
    return strateji, sonuc, int(sonuc.total_candles)


def en_iyi_parametreleri_bul() -> ParametreSeti:
    adaylar = [
        ParametreSeti(
            long_threshold=l,
            short_threshold=s,
            leverage_strength=lev,
            max_positions=mxp,
            max_allocation_per_position=alloc,
        )
        for l, s, lev in product(
            [0.63, 0.66],
            [0.37, 0.34],
            [0.16, 0.20],
        )
        for mxp, alloc in [(1, 0.45), (2, 0.40)]
    ]

    en_iyi = adaylar[0]
    en_iyi_getiri = float("-inf")
    sirali_sonuclar: list[tuple[float, ParametreSeti]] = []

    print("\nParametre optimizasyonu basladi...")
    for idx, params in enumerate(adaylar, start=1):
        _, sonuc, _ = backtest_calistir(params=params, start_candle=0)
        getiri = float(sonuc.return_pct)
        sirali_sonuclar.append((getiri, params))
        print(
            f"[{idx:02d}/{len(adaylar)}] return: {getiri:+9.2f}% | "
            f"long>{params.long_threshold:.2f} short<{params.short_threshold:.2f} "
            f"lev2@{params.leverage_strength:.2f} pos={params.max_positions}"
        )
        if getiri > en_iyi_getiri:
            en_iyi_getiri = getiri
            en_iyi = params

    sirali_sonuclar.sort(key=lambda x: x[0], reverse=True)
    print("\nTop 3 parametre seti:")
    for rank, (ret, prm) in enumerate(sirali_sonuclar[:3], start=1):
        print(
            f"{rank}. {ret:+.2f}% | long>{prm.long_threshold:.2f} "
            f"short<{prm.short_threshold:.2f} lev2@{prm.leverage_strength:.2f} "
            f"pos={prm.max_positions}"
        )

    return en_iyi


def run_backtest() -> None:
    """Varsayılan parametrelerle kısa cnlib backtest (proje kökünden)."""
    s = MLStrateji()
    s.get_data()
    sonuc = backtest.run(strategy=s, initial_capital=3000.0, start_candle=0)
    sonuc.print_summary()
    fo = getattr(sonuc, "failed_opens", None)
    ex = f" | açılamayan: {fo}" if fo is not None else ""
    print(
        f"\n  karslioglu: dönüş % {sonuc.return_pct:+.2f} | "
        f"işlem: {sonuc.total_trades} | validasyon: {sonuc.validation_errors}{ex}"
    )


def main() -> None:
    optimize_params = False
    en_iyi_params = varsayilan_yarismaci_parametreleri()
    if optimize_params:
        en_iyi_params = en_iyi_parametreleri_bul()
    strateji, sonuc, total_candles = backtest_calistir(params=en_iyi_params, start_candle=0)

    print("\nSecilen en iyi parametreler:")
    print(
        f"long>{en_iyi_params.long_threshold:.2f} "
        f"short<{en_iyi_params.short_threshold:.2f} "
        f"lev2@{en_iyi_params.leverage_strength:.2f} "
        f"max_pos={en_iyi_params.max_positions}"
    )
    veri_ozeti_yazdir(strateji)
    print(
        "Not: cnlib 0.1.4 ile model online guncelleniyor."
        f" Veri gorunur mum sayisi: {total_candles}"
    )
    sonuc.print_summary()


if __name__ == "__main__":
    main()

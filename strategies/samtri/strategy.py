"""
XGBoost tabanlı strateji - cnlib

Tasarım prensipleri (overfit'e karşı):
1. Feature engineering tamamen rolling/causal — look-ahead yok
2. Eğitim sadece train kısmında (__init__ değil, egit() metodunda)
3. Konservatif position sizing — model "kesin" dese bile full allocation yok
4. Probability threshold — modelin emin olmadığı durumda flat (signal=0)
5. Üç sınıflı label: aşağı / yatay / yukarı (binary'den daha az gürültülü)
6. Sığ ağaçlar (max_depth=4) — overfit'e karşı
7. Feature sayısı kasten az tutuldu, korelasyonlu olanlar elendi
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from cnlib.base_strategy import BaseStrategy
from cnlib import backtest


# ============================================================
# Feature engineering — hepsi causal (sadece geçmişe bakar)
# ============================================================
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = df["Close"]
    volume = df["Volume"]

    # --- Trend ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    out["macd_hist"] = (macd - macd_signal) / close

    # --- Mean reversion / momentum ---
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    out["rsi"] = 100 - 100 / (1 + rs)

    # --- Bollinger pozisyonu ---
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    out["bb_pos"] = ((close - ma20) / (2 * std20 + 1e-9)).clip(-2, 2)

    # --- Returns (multi-horizon momentum) ---
    out["ret1"] = close.pct_change(1)
    out["ret5"] = close.pct_change(5)

    # --- Volatilite rejimi ---
    ret_daily = close.pct_change()
    vol5 = ret_daily.rolling(5).std()
    vol20 = ret_daily.rolling(20).std()
    out["vol_regime"] = vol5 / (vol20 + 1e-9)
    out["volatility"] = vol20

    # --- Hacim ---
    vol_ma20 = volume.rolling(20).mean()
    out["vol_ratio"] = volume / (vol_ma20 + 1e-9)

    return out


# ============================================================
# Label: üç sınıflı (aşağı=0, yatay=1, yukarı=2)
# ============================================================
def make_labels(df: pd.DataFrame, horizon: int = 3, quantile: float = 0.33) -> pd.Series:
    fwd_ret = df["Close"].shift(-horizon) / df["Close"] - 1.0
    valid = fwd_ret.dropna()
    if len(valid) == 0:
        return pd.Series(1, index=df.index, dtype=int)

    lo = valid.quantile(quantile)
    hi = valid.quantile(1 - quantile)

    y = pd.Series(1, index=df.index, dtype=int)
    y[fwd_ret <= lo] = 0
    y[fwd_ret >= hi] = 2
    return y


# ============================================================
# Strateji
# ============================================================
class XGBStrategy(BaseStrategy):
    XGB_PARAMS = dict(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    HORIZON = 3
    LABEL_QUANTILE = 0.33
    PROB_THR = 0.45
    MIN_HISTORY = 30
    MAX_ALLOC_PER_COIN = 1

    def __init__(self):
        super().__init__()
        self.models = {}
        self.feature_cols = None
        self._train_data = {}

    def egit(self):
        """get_data() sonrası çağırın. self._train_data'yı kullanır."""
        source = self._train_data

        for coin, df in source.items():
            if len(df) < 100:
                print(f"[{coin}] yetersiz data ({len(df)} satır), atlıyorum")
                continue

            X_full = compute_features(df)
            y_full = make_labels(df, horizon=self.HORIZON, quantile=self.LABEL_QUANTILE)

            df_join = X_full.copy()
            df_join["y"] = y_full
            df_join = df_join.iloc[: -self.HORIZON]
            df_join = df_join.dropna()

            if len(df_join) < 50:
                print(f"[{coin}] dropna sonrası yetersiz ({len(df_join)} satır), atlıyorum")
                continue

            X = df_join.drop(columns=["y"]).values
            y = df_join["y"].values

            if self.feature_cols is None:
                self.feature_cols = list(X_full.columns)

            tr_classes = set(int(c) for c in np.unique(y))
            missing_tr = {0, 1, 2} - tr_classes
            if missing_tr:
                print(f"[{coin}] UYARI: train'de {missing_tr} sınıfı yok, atlıyorum")
                continue

            model = XGBClassifier(**self.XGB_PARAMS)
            model.fit(X, y, verbose=False)

            self.models[coin] = model
            unique, counts = np.unique(y, return_counts=True)
            cdict = dict(zip([int(u) for u in unique], counts.tolist()))
            dist = {
                "aşağı": cdict.get(0, 0),
                "yatay": cdict.get(1, 0),
                "yukarı": cdict.get(2, 0),
            }
            print(f"[{coin}] eğitildi | n={len(X)} | sınıflar: {dist}")

    def predict(self, data: dict) -> list[dict]:
        decisions = []
        candidates = {}

        for coin, df in data.items():
            if coin not in self.models or len(df) < self.MIN_HISTORY:
                candidates[coin] = (0, 0.0)
                continue

            feats = compute_features(df)
            last_row = feats.iloc[-1].values
            if np.isnan(last_row).any():
                candidates[coin] = (0, 0.0)
                continue

            proba = self.models[coin].predict_proba(last_row.reshape(1, -1))[0]
            p_down, p_flat, p_up = float(proba[0]), float(proba[1]), float(proba[2])

            if p_up > self.PROB_THR and p_up > p_down:
                candidates[coin] = (1, p_up)
            elif p_down > self.PROB_THR and p_down > p_up:
                candidates[coin] = (-1, p_down)
            else:
                candidates[coin] = (0, 0.0)

        active = [(c, sig, conf) for c, (sig, conf) in candidates.items() if sig != 0]
        total_conf = sum(conf for _, _, conf in active)

        for coin in data.keys():
            sig, conf = candidates[coin]
            if sig == 0 or total_conf == 0:
                decisions.append({"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1})
            else:
                raw_alloc = (conf / total_conf) * 0.9
                alloc = min(raw_alloc, self.MAX_ALLOC_PER_COIN)
                decisions.append({
                    "coin": coin,
                    "signal": sig,
                    "allocation": round(float(alloc), 3),
                    "leverage": 1,
                })

        return decisions


# ============================================================
# Çalıştır
# ============================================================
if __name__ == "__main__":
    strategy = XGBStrategy()
    strategy.get_data()

    total_candles = len(next(iter(strategy._full_data.values())))
    TRAIN_END = int(total_candles)
    print(f"[INFO] Toplam: {total_candles} | Eğitim: {TRAIN_END} | Test: {total_candles - TRAIN_END}")

    strategy._train_data = {
        coin: df.iloc[:TRAIN_END].copy()
        for coin, df in strategy._full_data.items()
    }
    strategy.egit()

    # Sadece test bölgesinde (%15) backtest yap
    result = backtest.run(strategy=strategy, initial_capital=3000.0, start_candle=TRAIN_END)
    result.print_summary()

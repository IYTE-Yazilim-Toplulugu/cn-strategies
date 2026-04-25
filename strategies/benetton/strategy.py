"""
Bu dosya hafif makine öğrenmesi destekli stratejiyi içerir.

Yöntem:
- Ortak feature set üzerinden yön tahmini yapar
- Model: RandomForestClassifier
- Model predict içinde eğitilmez
- Önce get_data(), sonra model eğitimi, sonra backtest.run()

Bu sürümde:
- gereksiz ikinci add_indicators çağrısı kaldırıldı
- sklearn warning'i çıkmaması için tahmin sırasında DataFrame veriliyor
"""

import logging
import os
import joblib
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from cnlib.base_strategy import BaseStrategy

from config import (
    ML_ALLOCATION,
    DEFAULT_LEVERAGE,
    ML_STOP_LOSS_PCT,
    ML_TAKE_PROFIT_PCT,
    ML_PROBA_THRESHOLD,
    MIN_HISTORY,
)
from indicators import add_indicators
from utils import has_nan, build_tp_sl, _flat_decision


class MLConfirmedStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.models = {}
        self.feature_columns = [
            "ret_1",
            "ma_diff_safe",
            "ma_diff_fast",
            "obv_diff",
            "bb_position",
            "volatility",
            "volume_confirmed",
            "volume_spike",
        ]

    # _flat_decision is now imported from utils
    pass

    def _feature_frame(self, enriched_df: pd.DataFrame) -> pd.DataFrame:
        """
        Buraya gelen df zaten add_indicators geçmiş veri olmalı.
        Tekrar add_indicators çağırmıyoruz.
        """
        features = pd.DataFrame(index=enriched_df.index)
        features["ret_1"] = enriched_df["RETURNS"]
        features["ma_diff_safe"] = enriched_df["MA_DIFF_SAFE"]
        features["ma_diff_fast"] = enriched_df["MA_DIFF_FAST"]
        features["obv_diff"] = enriched_df["OBV_DIFF"]
        features["bb_position"] = enriched_df["BB_POSITION"]
        features["volatility"] = enriched_df["VOLATILITY"]
        features["volume_confirmed"] = enriched_df["VOLUME_CONFIRMED"].astype(float)
        features["volume_spike"] = enriched_df["VOLUME_SPIKE"].astype(float)
        return features

    def _build_labels(self, enriched_df: pd.DataFrame) -> pd.Series:
        """
        Bir sonraki kapanış mevcut kapanıştan yüksekse 1, değilse 0.
        """
        return (enriched_df["Close"].shift(-1) > enriched_df["Close"]).astype(int)

    def prepare_models(self):
        """
        get_data() sonrası çağrılır.
        coin_data içindeki verilerle her coin için model eğitir.
        """
        self.models = {}

        for coin, df in self._full_data.items():
            enriched = add_indicators(df)

            X = self._feature_frame(enriched)
            y = self._build_labels(enriched)

            dataset = X.copy()
            dataset["target"] = y
            dataset = dataset.dropna().reset_index(drop=True)

            if len(dataset) < MIN_HISTORY:
                continue

            X_clean = dataset[self.feature_columns]
            y_clean = dataset["target"]

            model = LogisticRegression(
                class_weight="balanced", 
                random_state=42, 
                max_iter=1000
            )
            model.fit(X_clean.iloc[:-1], y_clean.iloc[:-1])
            self.models[coin] = model

    def save_models(self, path: str = "trained_models"):
        """Eğitilmiş modelleri PKL formatında diske kaydeder."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        for coin, model in self.models.items():
            model_path = save_dir / f"{coin}_model.pkl"
            joblib.dump(model, model_path)
        logging.info(f"Modeller {save_dir} klasörüne kaydedildi.")

    def load_models(self, path: str = None):
        """Diskteki PKL modellerini RAM'e yükler."""
        if path is None:
            path = os.path.dirname(os.path.abspath(__file__))
            
        load_dir = Path(path)
        coins = ["kapcoin-usd_train", "metucoin-usd_train", "tamcoin-usd_train"]
        for coin in coins:
            model_path = load_dir / f"{coin}_model.pkl"
            if model_path.exists():
                self.models[coin] = joblib.load(model_path)
            else:
                logging.error(f"Eğitilmiş model bulunamadı: {model_path} - Lütfen önce modelleri eğitin!")
        logging.info("Tüm modeller başarıyla diskten yüklendi.")

    def _build_long_decision(self, coin: str, entry: float) -> dict:
        take_profit, stop_loss = build_tp_sl(
            entry=entry,
            direction=1,
            stop_loss_pct=ML_STOP_LOSS_PCT,
            take_profit_pct=ML_TAKE_PROFIT_PCT,
        )
        return {
            "coin": coin,
            "signal": 1,
            "allocation": ML_ALLOCATION,
            "leverage": DEFAULT_LEVERAGE,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
        }

    def _build_short_decision(self, coin: str, entry: float) -> dict:
        take_profit, stop_loss = build_tp_sl(
            entry=entry,
            direction=-1,
            stop_loss_pct=ML_STOP_LOSS_PCT,
            take_profit_pct=ML_TAKE_PROFIT_PCT,
        )
        return {
            "coin": coin,
            "signal": -1,
            "allocation": ML_ALLOCATION,
            "leverage": DEFAULT_LEVERAGE,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
        }

    def predict(self, data: dict) -> list[dict]:
        decisions = []

        for coin, df in data.items():
            if coin not in self.models or len(df) < MIN_HISTORY:
                decisions.append(_flat_decision(coin))
                continue

            try:
                enriched = add_indicators(df)
                X = self._feature_frame(enriched).dropna()

                if len(X) == 0:
                    decisions.append(_flat_decision(coin))
                    continue

                last_features = X.iloc[[-1]][self.feature_columns]

                if has_nan(last_features.iloc[0].values):
                    decisions.append(_flat_decision(coin))
                    continue

                model = self.models[coin]
                proba = model.predict_proba(last_features)[0]

                # Tek sınıflı model koruması
                if len(proba) < 2:
                    proba_up = 1.0 if model.classes_[0] == 1 else 0.0
                else:
                    proba_up = proba[1]

                entry = float(enriched["Close"].iloc[-1])

                if proba_up >= ML_PROBA_THRESHOLD:
                    decisions.append(self._build_long_decision(coin, entry))
                elif proba_up <= (1 - ML_PROBA_THRESHOLD):
                    decisions.append(self._build_short_decision(coin, entry))
                else:
                    decisions.append(_flat_decision(coin))
            except Exception as e:
                logging.warning(f"Predict error for {coin}: {e}")
                decisions.append(_flat_decision(coin))

        return decisions
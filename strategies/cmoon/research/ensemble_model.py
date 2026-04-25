"""
EnsembleModel — RF + Gradient Boost soft-voting wrapper.

Ayrı bir modülde tanımlı olması gerekiyor: train_models.py __main__ olarak
çalıştırıldığında pickle sınıf yolunu 'research.ensemble_model.EnsembleModel'
olarak kaydeder. Böylece herhangi bir scriptten yüklenebilir.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class EnsembleModel:
    """
    Soft-voting ensemble: RF + Gradient Boost probability ortalaması.

    Interface (strategy.py._ml_prob() tarafından beklenen):
        .classes_         — np.array([0, 1])
        .predict_proba(X) — (n_samples, 2) float array
        .predict(X)       — binary predictions
        .feature_importances_ — ortalama importance (save_feature_importance için)
    """

    classes_ = np.array([0, 1])

    def __init__(self, rf: RandomForestClassifier, boost: object) -> None:
        self.rf = rf
        self.boost = boost

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return (self.rf.predict_proba(X) + self.boost.predict_proba(X)) / 2.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def feature_importances_(self) -> np.ndarray:
        return (
            np.asarray(self.rf.feature_importances_, dtype=float)
            + np.asarray(self.boost.feature_importances_, dtype=float)
        ) / 2.0

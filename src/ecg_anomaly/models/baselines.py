"""Unsupervised baseline scorers on tabular features. Higher score = more anomalous."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler


class BaselineScorer:
    """``kind``: 'iforest' (Isolation Forest) or 'gmm' (negative log-likelihood of a GMM).

    Features are robust-scaled (median / IQR) and clipped to +-10 so a few extreme beats do
    not dominate. Fitting uses no labels.
    """

    def __init__(self, kind: str = "iforest", seed: int = 0, n_components: int = 4,
                 n_estimators: int = 200):
        if kind not in {"iforest", "gmm"}:
            raise ValueError("kind must be 'iforest' or 'gmm'")
        self.kind, self.seed, self.n_components = kind, seed, n_components
        self.n_estimators = n_estimators

    def _prep(self, X):
        return np.clip(self.scaler_.transform(X), -10, 10)

    def fit(self, X) -> "BaselineScorer":
        X = np.asarray(X, dtype=float)
        self.scaler_ = RobustScaler().fit(X)
        Z = self._prep(X)
        if self.kind == "iforest":
            self.model_ = IsolationForest(n_estimators=self.n_estimators, random_state=self.seed,
                                           n_jobs=-1)
        else:
            self.model_ = GaussianMixture(self.n_components, covariance_type="full",
                                          reg_covar=1e-3, random_state=self.seed)
        self.model_.fit(Z)
        return self

    def score(self, X) -> np.ndarray:
        Z = self._prep(np.asarray(X, dtype=float))
        return -self.model_.score_samples(Z)  # IF: -(anomaly score); GMM: -log-likelihood

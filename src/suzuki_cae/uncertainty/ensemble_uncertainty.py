"""
Ensemble-based uncertainty quantification.

Trains N bootstrap replicas of a base estimator.  The spread across replicas
provides an empirical prediction interval.  Calibration adjusts the interval
width using held-out coverage data.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from dataclasses import dataclass


@dataclass
class PredictionInterval:
    mean:   np.ndarray
    lower:  np.ndarray
    upper:  np.ndarray
    std:    np.ndarray
    flagged: np.ndarray   # True where uncertainty exceeds threshold


class EnsembleUQ:
    """Bootstrap ensemble uncertainty estimator.

    Parameters
    ----------
    base_estimator : Sklearn-compatible regressor.
    n_estimators   : Number of bootstrap replicas.
    alpha          : Desired miscoverage rate (0.05 → 95 % interval).
    flag_threshold_std : Flag predictions with std > factor × mean(std).
    """

    def __init__(
        self,
        base_estimator=None,
        n_estimators: int = 50,
        alpha: float = 0.05,
        flag_threshold_std: float = 2.0,
        seed: int = 42,
    ) -> None:
        if base_estimator is None:
            base_estimator = RandomForestRegressor(n_estimators=100, random_state=seed)
        self.base_estimator      = base_estimator
        self.n_estimators        = n_estimators
        self.alpha               = alpha
        self.flag_threshold_std  = flag_threshold_std
        self.seed                = seed
        self._ensemble: list = []
        self._calibration_factor: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleUQ":
        rng = np.random.default_rng(self.seed)
        n = len(X)
        self._ensemble = []
        for i in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)          # bootstrap resample
            est = clone(self.base_estimator)
            est.set_params(random_state=int(rng.integers(0, 10_000)))
            est.fit(X[idx], y[idx])
            self._ensemble.append(est)
        return self

    def predict_interval(self, X: np.ndarray) -> PredictionInterval:
        preds = np.stack([est.predict(X) for est in self._ensemble], axis=1)  # (n, K)
        mean  = preds.mean(axis=1)
        std   = preds.std(axis=1) * self._calibration_factor
        z     = self._z_from_alpha(self.alpha)
        lower = mean - z * std
        upper = mean + z * std
        threshold = self.flag_threshold_std * std.mean()
        flagged   = std > threshold
        return PredictionInterval(mean=mean, lower=lower, upper=upper, std=std, flagged=flagged)

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> float:
        """Adjust interval width so coverage ≈ 1 - alpha on calibration set.

        Uses a simple multiplicative scaling of the std estimate.
        """
        pi = self.predict_interval(X_cal)
        raw_coverage = np.mean((y_cal >= pi.lower) & (y_cal <= pi.upper))
        target = 1.0 - self.alpha
        if raw_coverage < target:
            # Widen intervals
            self._calibration_factor *= (target / max(raw_coverage, 1e-6)) ** 0.5
        elif raw_coverage > target + 0.05:
            # Narrow overly conservative intervals
            self._calibration_factor *= (target / raw_coverage) ** 0.5
        return self._calibration_factor

    @staticmethod
    def _z_from_alpha(alpha: float) -> float:
        from scipy.stats import norm
        return norm.ppf(1.0 - alpha / 2.0)
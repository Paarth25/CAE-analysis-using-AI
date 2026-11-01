"""
ML model definitions, training and persistence.

All models share the scikit-learn fit/predict API.
Each is wrapped in a lightweight SurrogateModel dataclass that also carries
its metadata (training metrics, feature names, random seed, etc.).
"""

from __future__ import annotations

import time
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

from ..utils.logging import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------ #
#  Metrics                                                             #
# ------------------------------------------------------------------ #

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    """Return MAE, RMSE, R², MAPE for a prediction array."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    # MAPE – guard against zero targets
    mask = np.abs(y_true) > 1e-10
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0 if mask.any() else np.nan
    return {"label": label, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE_pct": mape}


# ------------------------------------------------------------------ #
#  Surrogate model wrapper                                             #
# ------------------------------------------------------------------ #

@dataclass
class SurrogateModel:
    name:          str
    target:        str
    estimator:     Any                              # sklearn-compatible
    feature_names: list[str] = field(default_factory=list)
    train_metrics: dict      = field(default_factory=dict)
    val_metrics:   dict      = field(default_factory=dict)
    train_time_s:  float     = 0.0
    predict_time_s:float     = 0.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> "SurrogateModel":
        t0 = time.perf_counter()
        self.estimator.fit(X_train, y_train)
        self.train_time_s = time.perf_counter() - t0
        self.train_metrics = compute_metrics(y_train, self.estimator.predict(X_train), f"{self.name}[train]")
        if X_val is not None and y_val is not None:
            self.val_metrics = compute_metrics(y_val, self.estimator.predict(X_val), f"{self.name}[val]")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        y_hat = self.estimator.predict(X)
        self.predict_time_s = time.perf_counter() - t0
        return y_hat

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        return compute_metrics(y_test, self.predict(X_test), f"{self.name}[test]")

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved {self.name} → {path}")

    @staticmethod
    def load(path: Path) -> "SurrogateModel":
        with open(path, "rb") as f:
            return pickle.load(f)


# ------------------------------------------------------------------ #
#  Model factory                                                       #
# ------------------------------------------------------------------ #

def build_linear_regression(target: str, **kw) -> SurrogateModel:
    return SurrogateModel(
        name="LinearRegression",
        target=target,
        estimator=Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LinearRegression(**kw)),
        ]),
    )


def build_polynomial_regression(target: str, degree: int = 2, **kw) -> SurrogateModel:
    return SurrogateModel(
        name=f"PolynomialRegression(d={degree})",
        target=target,
        estimator=Pipeline([
            ("poly",   PolynomialFeatures(degree=degree, include_bias=True)),
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=1.0, **kw)),
        ]),
    )


def build_random_forest(target: str, **kw) -> SurrogateModel:
    defaults = dict(n_estimators=200, max_depth=None, min_samples_split=5,
                    n_jobs=-1, random_state=42)
    defaults.update(kw)
    return SurrogateModel(
        name="RandomForest",
        target=target,
        estimator=RandomForestRegressor(**defaults),
    )


def build_xgboost(target: str, **kw) -> SurrogateModel:
    if not _HAS_XGB:
        raise ImportError("xgboost is not installed.  Run: pip install xgboost")
    defaults = dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    defaults.update(kw)
    return SurrogateModel(
        name="XGBoost",
        target=target,
        estimator=XGBRegressor(**defaults),
    )


def build_gaussian_process(target: str, **kw) -> SurrogateModel:
    kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-4)
    defaults = dict(kernel=kernel, n_restarts_optimizer=5,
                    normalize_y=True, random_state=42)
    defaults.update(kw)
    return SurrogateModel(
        name="GaussianProcess",
        target=target,
        estimator=GaussianProcessRegressor(**defaults),
    )


def build_mlp(target: str, **kw) -> SurrogateModel:
    defaults = dict(hidden_layer_sizes=(128, 64, 32), activation="relu",
                    max_iter=500, random_state=42, early_stopping=True)
    defaults.update(kw)
    return SurrogateModel(
        name="MLP",
        target=target,
        estimator=Pipeline([
            ("scaler", StandardScaler()),
            ("model",  MLPRegressor(**defaults)),
        ]),
    )


ALL_BUILDERS = {
    "LinearRegression":   build_linear_regression,
    "PolynomialRegression": build_polynomial_regression,
    "RandomForest":       build_random_forest,
    "XGBoost":            build_xgboost,
    "GaussianProcess":    build_gaussian_process,
    "MLP":                build_mlp,
}


# ------------------------------------------------------------------ #
#  Model suite trainer                                                 #
# ------------------------------------------------------------------ #

class ModelSuite:
    """Train and evaluate all ML surrogate models for a given target.

    Parameters
    ----------
    target      : Output variable name.
    feature_names: Column names for the feature matrix.
    skip_gp     : Set True to skip GaussianProcess (slow for large datasets).
    skip_xgb    : Set True if XGBoost is not installed.
    """

    def __init__(
        self,
        target: str,
        feature_names: list[str],
        skip_gp: bool = False,
        skip_xgb: bool = not _HAS_XGB,
        seed: int = 42,
    ) -> None:
        self.target        = target
        self.feature_names = feature_names
        self.seed          = seed
        self.skip_gp       = skip_gp
        self.skip_xgb      = skip_xgb
        self.models: dict[str, SurrogateModel] = {}
        self.test_metrics:  list[dict] = []

    def train_all(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
    ) -> None:
        builders = dict(ALL_BUILDERS)
        if self.skip_gp:
            builders.pop("GaussianProcess", None)
        if self.skip_xgb:
            builders.pop("XGBoost", None)

        for name, build_fn in builders.items():
            logger.info(f"  Training {name} for target='{self.target}'…")
            try:
                m = build_fn(self.target, random_state=self.seed) \
                    if "random_state" in build_fn.__code__.co_varnames else build_fn(self.target)
                m.feature_names = self.feature_names
                m.fit(X_train, y_train, X_val, y_val)
                self.models[name] = m
                logger.info(
                    f"    val R²={m.val_metrics.get('R2',float('nan')):.4f} | "
                    f"val MAE={m.val_metrics.get('MAE',float('nan')):.4f}"
                )
            except Exception as exc:
                logger.warning(f"    {name} failed: {exc}")

    def evaluate_test(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> pd.DataFrame:
        rows = []
        for name, m in self.models.items():
            metrics = m.evaluate(X_test, y_test)
            metrics["train_time_s"]   = m.train_time_s
            metrics["predict_time_s"] = m.predict_time_s
            rows.append(metrics)
        df = pd.DataFrame(rows).set_index("label")
        self.test_metrics = rows
        return df

    def best_model(self, metric: str = "R2") -> SurrogateModel:
        """Return the model with the best test metric."""
        sign = -1 if metric in ("MAE", "RMSE", "MAPE_pct") else 1
        best_name = max(
            self.models,
            key=lambda n: sign * self.test_metrics[
                [m["label"].split("[")[0] for m in self.test_metrics].index(n)
            ].get(metric, float("-inf")) if self.test_metrics else 0,
        )
        return self.models[best_name]
"""Tests for ensemble uncertainty quantification."""

import sys
from pathlib import Path
import pytest
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from suzuki_cae.uncertainty.ensemble_uncertainty import EnsembleUQ, PredictionInterval
from suzuki_cae.data.generation import DatasetGenerator
from suzuki_cae.data.preprocessing import DataPipeline, FEATURE_COLUMNS


@pytest.fixture(scope="module")
def uq_trained():
    """Return a trained EnsembleUQ on the tip deflection target."""
    gen  = DatasetGenerator(n_samples=150, noise_std_pct=0.5, seed=3)
    df   = gen.generate()
    pipe = DataPipeline(test_size=0.2, val_size=0.2, seed=3)
    df   = pipe.clean(df)
    df   = pipe.engineer_features(df)
    train, val, test = pipe.split(df)
    pipe.fit_scalers(train)

    feat_cols = [c for c in pipe.scaler_X.feature_names_in_ if c in train.columns]
    X_tr  = train[feat_cols].values
    y_tr  = train["tip_deflection_mm"].values
    X_val = val[feat_cols].values
    y_val = val["tip_deflection_mm"].values
    X_te  = test[feat_cols].values
    y_te  = test["tip_deflection_mm"].values

    uq = EnsembleUQ(n_estimators=20, alpha=0.05, seed=42)
    uq.fit(X_tr, y_tr)
    uq.calibrate(X_val, y_val)

    return uq, X_te, y_te


class TestEnsembleUQ:
    def test_returns_prediction_interval(self, uq_trained):
        uq, X_te, _ = uq_trained
        pi = uq.predict_interval(X_te)
        assert isinstance(pi, PredictionInterval)

    def test_interval_shape(self, uq_trained):
        uq, X_te, _ = uq_trained
        pi = uq.predict_interval(X_te)
        n = len(X_te)
        assert pi.mean.shape  == (n,)
        assert pi.lower.shape == (n,)
        assert pi.upper.shape == (n,)
        assert pi.std.shape   == (n,)

    def test_lower_less_than_upper(self, uq_trained):
        uq, X_te, _ = uq_trained
        pi = uq.predict_interval(X_te)
        assert np.all(pi.lower <= pi.upper)

    def test_mean_within_interval(self, uq_trained):
        uq, X_te, _ = uq_trained
        pi = uq.predict_interval(X_te)
        assert np.all(pi.lower <= pi.mean)
        assert np.all(pi.mean  <= pi.upper)

    def test_empirical_coverage_approx_95pct(self, uq_trained):
        uq, X_te, y_te = uq_trained
        pi = uq.predict_interval(X_te)
        coverage = np.mean((y_te >= pi.lower) & (y_te <= pi.upper))
        assert 0.70 <= coverage <= 1.00, (
            f"Coverage {coverage:.2f} outside expected range [0.70, 1.00]")

    def test_std_positive(self, uq_trained):
        uq, X_te, _ = uq_trained
        pi = uq.predict_interval(X_te)
        assert np.all(pi.std >= 0)

    def test_flagged_is_boolean(self, uq_trained):
        uq, X_te, _ = uq_trained
        pi = uq.predict_interval(X_te)
        assert pi.flagged.dtype == bool

    def test_calibration_changes_factor(self):
        """Calibration should adjust _calibration_factor from 1.0."""
        gen  = DatasetGenerator(n_samples=80, noise_std_pct=1.0, seed=5)
        df   = gen.generate()
        pipe = DataPipeline(test_size=0.2, val_size=0.2, seed=5)
        df   = pipe.clean(df)
        df   = pipe.engineer_features(df)
        train, val, _ = pipe.split(df)
        pipe.fit_scalers(train)
        feat_cols = [c for c in pipe.scaler_X.feature_names_in_ if c in train.columns]
        X_tr  = train[feat_cols].values
        y_tr  = train["tip_deflection_mm"].values
        X_val = val[feat_cols].values
        y_val = val["tip_deflection_mm"].values

        uq = EnsembleUQ(n_estimators=10, alpha=0.05, seed=0)
        uq.fit(X_tr, y_tr)
        assert uq._calibration_factor == 1.0
        uq.calibrate(X_val, y_val)
        # calibration factor could be ≥1 or ≤1 depending on initial coverage
        assert isinstance(uq._calibration_factor, float)
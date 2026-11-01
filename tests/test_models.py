"""Integration test: dataset → preprocessing → model training → evaluation."""

import sys
from pathlib import Path
import pytest
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from suzuki_cae.data.generation import DatasetGenerator
from suzuki_cae.data.preprocessing import DataPipeline, FEATURE_COLUMNS, TARGET_COLUMNS
from suzuki_cae.models.model_training import (
    build_linear_regression, build_random_forest, build_polynomial_regression,
    compute_metrics,
)


@pytest.fixture(scope="module")
def prepared_data():
    gen  = DatasetGenerator(n_samples=150, noise_std_pct=0.0, seed=7)
    df   = gen.generate()
    pipe = DataPipeline(test_size=0.2, val_size=0.2, seed=7)
    df   = pipe.clean(df)
    df   = pipe.engineer_features(df)
    train, val, test = pipe.split(df)
    pipe.fit_scalers(train)
    return pipe, train, val, test


def _get_xy(pipe, df, target):
    X = pipe.get_X(df)
    y = pipe.get_y(df, target)
    return X, y


class TestLinearRegression:
    def test_r2_positive(self, prepared_data):
        pipe, train, val, test = prepared_data
        target = "tip_deflection_mm"
        m = build_linear_regression(target)
        X_tr, y_tr = _get_xy(pipe, train, target)
        X_v, y_v   = _get_xy(pipe, val, target)
        X_te, y_te = _get_xy(pipe, test, target)
        m.fit(X_tr, y_tr, X_v, y_v)
        metrics = m.evaluate(X_te, y_te)
        assert metrics["R2"] > 0.80, f"Linear R²={metrics['R2']:.3f} below threshold"

    def test_predict_shape(self, prepared_data):
        pipe, train, val, test = prepared_data
        target = "tip_deflection_mm"
        m = build_linear_regression(target)
        X_tr, y_tr = _get_xy(pipe, train, target)
        m.fit(X_tr, y_tr)
        X_te, _ = _get_xy(pipe, test, target)
        preds = m.predict(X_te)
        assert preds.shape == (len(test),)


class TestRandomForest:
    def test_r2_high(self, prepared_data):
        pipe, train, val, test = prepared_data
        target = "tip_deflection_mm"
        m = build_random_forest(target, n_estimators=50)
        X_tr, y_tr = _get_xy(pipe, train, target)
        X_v, y_v   = _get_xy(pipe, val, target)
        X_te, y_te = _get_xy(pipe, test, target)
        m.fit(X_tr, y_tr, X_v, y_v)
        metrics = m.evaluate(X_te, y_te)
        assert metrics["R2"] > 0.95, f"RF R²={metrics['R2']:.3f} below threshold"


class TestMetrics:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        m = compute_metrics(y, y)
        assert m["MAE"]  == pytest.approx(0.0)
        assert m["RMSE"] == pytest.approx(0.0)
        assert m["R2"]   == pytest.approx(1.0)

    def test_mape_calculation(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 190.0])
        m = compute_metrics(y_true, y_pred)
        expected_mape = (10.0/100.0 + 10.0/200.0) / 2.0 * 100.0
        assert m["MAPE_pct"] == pytest.approx(expected_mape, rel=1e-5)


class TestEndToEnd:
    """Full pipeline: params → data → features → model → prediction."""

    def test_end_to_end_pipeline(self):
        # Generate 100 samples
        gen  = DatasetGenerator(n_samples=100, noise_std_pct=0.0, seed=0)
        df   = gen.generate()
        pipe = DataPipeline(test_size=0.15, val_size=0.15, seed=0)
        df   = pipe.clean(df)
        df   = pipe.engineer_features(df)
        train, val, test = pipe.split(df)
        pipe.fit_scalers(train)

        target = "tip_deflection_mm"
        X_tr, y_tr = _get_xy(pipe, train, target)
        X_te, y_te = _get_xy(pipe, test, target)

        model = build_random_forest(target, n_estimators=50)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)

        assert len(preds) == len(y_te)
        assert all(p > 0 for p in preds), "Deflections must be positive"
        metrics = compute_metrics(y_te, preds)
        assert metrics["R2"] > 0.90
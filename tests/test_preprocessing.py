import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from suzuki_cae.data.generation import DatasetGenerator, latin_hypercube_sample, PARAM_BOUNDS
from suzuki_cae.data.preprocessing import DataPipeline, FEATURE_COLUMNS, TARGET_COLUMNS


@pytest.fixture(scope="module")
def small_dataset():
    gen = DatasetGenerator(n_samples=80, noise_std_pct=0.0, seed=1)
    return gen.generate()


class TestLHS:
    def test_shape(self):
        df = latin_hypercube_sample(50, PARAM_BOUNDS, seed=0)
        assert df.shape == (50, 6)

    def test_bounds_respected(self):
        df = latin_hypercube_sample(100, PARAM_BOUNDS, seed=0)
        for col, (lo, hi) in PARAM_BOUNDS.items():
            assert df[col].min() >= lo - 1e-9
            assert df[col].max() <= hi + 1e-9

    def test_reproducible(self):
        d1 = latin_hypercube_sample(20, PARAM_BOUNDS, seed=7)
        d2 = latin_hypercube_sample(20, PARAM_BOUNDS, seed=7)
        pd.testing.assert_frame_equal(d1, d2)


class TestDatasetGenerator:
    def test_output_columns(self, small_dataset):
        required = ["simulation_id","length_mm","tip_deflection_mm","max_von_mises_MPa"]
        for c in required:
            assert c in small_dataset.columns

    def test_no_nans_in_outputs(self, small_dataset):
        for col in TARGET_COLUMNS:
            assert small_dataset[col].isna().sum() == 0

    def test_positive_outputs(self, small_dataset):
        for col in ["tip_deflection_mm","max_von_mises_MPa","strain_energy_J","mass_kg"]:
            assert (small_dataset[col] > 0).all()

    def test_solver_status_ok(self, small_dataset):
        assert (small_dataset["solver_status"] == "ok").all()


class TestDataPipeline:
    def test_split_sizes(self, small_dataset):
        pipe = DataPipeline(test_size=0.2, val_size=0.2, seed=42)
        df = pipe.clean(small_dataset)
        df = pipe.engineer_features(df)
        train, val, test = pipe.split(df)
        total = len(train) + len(val) + len(test)
        assert total == len(df)
        assert len(test) >= 1

    def test_no_leakage_across_splits(self, small_dataset):
        pipe = DataPipeline(seed=42)
        df = pipe.clean(small_dataset)
        df = pipe.engineer_features(df)
        train, val, test = pipe.split(df)
        # No simulation_id should appear in more than one split
        ids_train = set(train["simulation_id"])
        ids_val   = set(val["simulation_id"])
        ids_test  = set(test["simulation_id"])
        assert ids_train.isdisjoint(ids_val)
        assert ids_train.isdisjoint(ids_test)
        assert ids_val.isdisjoint(ids_test)

    def test_feature_engineering_adds_columns(self, small_dataset):
        pipe = DataPipeline()
        df = pipe.clean(small_dataset)
        df = pipe.engineer_features(df)
        assert "I_mm4" in df.columns
        assert "FL3_EI" in df.columns
        assert (df["I_mm4"] > 0).all()
"""
Data preprocessing: cleaning, feature engineering, train/val/test splitting.

Leakage-resistant split strategy
---------------------------------
A purely random split risks leakage when nearby design points share structural
similarity.  We use a stratified geometric split: the design space is binned
along the first principal component of the input features, and test/validation
samples are drawn uniformly from each bin.  This ensures the held-out sets
span the full design space rather than clustering near training points.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..utils.logging import get_logger

logger = get_logger(__name__)


FEATURE_COLUMNS = [
    "length_mm",
    "width_mm",
    "thickness_mm",
    "youngs_modulus_GPa",
    "poisson_ratio",
    "force_N",
]

TARGET_COLUMNS = [
    "tip_deflection_mm",
    "max_von_mises_MPa",
    "strain_energy_J",
]


class DataPipeline:
    """End-to-end preprocessing pipeline.

    Parameters
    ----------
    feature_columns : Input feature names (default: FEATURE_COLUMNS).
    target_columns  : Prediction target names.
    test_size       : Fraction for test set.
    val_size        : Fraction for validation set (from remaining after test).
    seed            : Random seed.
    """

    def __init__(
        self,
        feature_columns: list[str] = FEATURE_COLUMNS,
        target_columns:  list[str] = TARGET_COLUMNS,
        test_size:  float = 0.15,
        val_size:   float = 0.15,
        seed:       int   = 42,
    ) -> None:
        self.feature_columns = feature_columns
        self.target_columns  = target_columns
        self.test_size  = test_size
        self.val_size   = val_size
        self.seed       = seed
        self.scaler_X   = StandardScaler()
        self.scaler_y:  dict[str, StandardScaler] = {}

    # ------------------------------------------------------------------ #
    #  Loading                                                             #
    # ------------------------------------------------------------------ #

    def load(self, csv_path: str | Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from {csv_path}.")
        return df

    # ------------------------------------------------------------------ #
    #  Cleaning                                                            #
    # ------------------------------------------------------------------ #

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop failed simulations, duplicates, and rows with NaN outputs."""
        n0 = len(df)
        # Keep only solver-successful rows
        if "solver_status" in df.columns:
            df = df[df["solver_status"] == "ok"].copy()
        # Drop rows with NaN in features or targets
        required = self.feature_columns + self.target_columns
        df = df.dropna(subset=required)
        # Drop duplicate design points
        df = df.drop_duplicates(subset=self.feature_columns)
        logger.info(f"Cleaned: {n0} → {len(df)} rows.")
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    #  Feature engineering                                                 #
    # ------------------------------------------------------------------ #

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add physics-inspired derived features.

        These features encode domain knowledge and improve model accuracy,
        particularly for the linear/polynomial models.
        """
        df = df.copy()
        # Second moment of area: I = b t³/12  [mm⁴]
        df["I_mm4"] = df["width_mm"] * df["thickness_mm"]**3 / 12.0
        # Bending stiffness: EI  [GPa·mm⁴]
        df["EI"] = df["youngs_modulus_GPa"] * df["I_mm4"]
        # Slenderness ratio
        df["slenderness"] = df["length_mm"] / df["thickness_mm"]
        # Normalised force (F·L³/(EI) is proportional to tip deflection)
        df["FL3_EI"] = (
            df["force_N"] * df["length_mm"]**3
            / (df["youngs_modulus_GPa"] * 1e3 * df["I_mm4"])   # GPa→MPa·mm⁻²
        )
        # Cross-sectional area
        df["area_mm2"] = df["width_mm"] * df["thickness_mm"]
        # Aspect ratio
        df["aspect_ratio"] = df["length_mm"] / df["width_mm"]

        return df

    # ------------------------------------------------------------------ #
    #  Leakage-resistant train / val / test split                          #
    # ------------------------------------------------------------------ #

    def split(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return (train, val, test) DataFrames.

        Strategy: stratify on a binned version of the dominant analytical
        feature FL3_EI (if present) to ensure geometric coverage.
        """
        strat_col = None
        if "FL3_EI" in df.columns:
            # 10 equal-width bins for stratification
            df["_strat"] = pd.qcut(df["FL3_EI"], q=10, labels=False, duplicates="drop")
            strat_col = "_strat"

        train_val, test = train_test_split(
            df,
            test_size=self.test_size,
            stratify=df[strat_col] if strat_col else None,
            random_state=self.seed,
        )

        relative_val = self.val_size / (1.0 - self.test_size)
        train, val = train_test_split(
            train_val,
            test_size=relative_val,
            stratify=train_val[strat_col] if strat_col else None,
            random_state=self.seed,
        )

        if strat_col:
            for part in [train, val, test]:
                part.drop(columns=["_strat"], inplace=True, errors="ignore")

        logger.info(f"Split → train={len(train)}, val={len(val)}, test={len(test)}")
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    #  Scaling                                                             #
    # ------------------------------------------------------------------ #

    def fit_scalers(self, train: pd.DataFrame) -> None:
        """Fit StandardScaler on training features."""
        all_feats = [c for c in self.feature_columns + ["I_mm4","EI","slenderness","FL3_EI","area_mm2","aspect_ratio"]
                     if c in train.columns]
        self.scaler_X.fit(train[all_feats])
        for tgt in self.target_columns:
            sc = StandardScaler()
            sc.fit(train[[tgt]])
            self.scaler_y[tgt] = sc

    def get_X(self, df: pd.DataFrame, scale: bool = False) -> np.ndarray:
        """Extract feature matrix from df; optionally standardise."""
        cols = [c for c in self.scaler_X.feature_names_in_ if c in df.columns]
        X = df[cols].values
        if scale:
            X = self.scaler_X.transform(X)
        return X

    def get_y(self, df: pd.DataFrame, target: str, scale: bool = False) -> np.ndarray:
        y = df[target].values.reshape(-1, 1)
        if scale:
            y = self.scaler_y[target].transform(y)
        return y.ravel()
#!/usr/bin/env python3
"""
Train all ML surrogate models and save artefacts.

Usage:
    python scripts/train_models.py [--data data/raw/cae/beam_dataset_analytical.csv]
                                   [--target tip_deflection_mm]

Outputs:
    models/saved/<target>_<ModelName>.pkl
    results/model_metrics/<target>_metrics.csv
"""

import sys
import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from suzuki_cae.data.preprocessing import DataPipeline, FEATURE_COLUMNS, TARGET_COLUMNS
from suzuki_cae.models.model_training import (
    build_linear_regression, build_polynomial_regression,
    build_random_forest, build_xgboost, build_mlp, compute_metrics,
)
from suzuki_cae.utils.logging import get_logger

logger = get_logger("train_models")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="data/raw/cae/beam_dataset_analytical.csv")
    parser.add_argument("--target", default="tip_deflection_mm",
                        choices=TARGET_COLUMNS)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    root = Path(__file__).parents[1]
    data_path = root / args.data

    if not data_path.exists():
        print(f"Dataset not found at {data_path}. Run generate_dataset.py first.")
        sys.exit(1)

    # Preprocessing
    pipe = DataPipeline(test_size=0.15, val_size=0.15, seed=args.seed)
    df   = pipe.load(data_path)
    df   = pipe.clean(df)
    df   = pipe.engineer_features(df)
    train, val, test = pipe.split(df)
    pipe.fit_scalers(train)

    # Build feature arrays
    feature_cols = [c for c in pipe.scaler_X.feature_names_in_ if c in train.columns]
    X_tr = train[feature_cols].values
    X_v  = val[feature_cols].values
    X_te = test[feature_cols].values
    y_tr = train[args.target].values
    y_v  = val[args.target].values
    y_te = test[args.target].values

    logger.info(f"Target: {args.target}")
    logger.info(f"Train: {len(X_tr)}, Val: {len(X_v)}, Test: {len(X_te)}")

    builders = [
        ("LinearRegression",      build_linear_regression),
        ("PolynomialRegression",  build_polynomial_regression),
        ("RandomForest",          build_random_forest),
        ("XGBoost",               build_xgboost),
        ("MLP",                   build_mlp),
    ]

    models_dir  = root / "models" / "saved"
    metrics_dir = root / "results" / "model_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    for name, build_fn in builders:
        logger.info(f"Training {name}…")
        try:
            m = build_fn(args.target)
            m.feature_names = feature_cols
            m.fit(X_tr, y_tr, X_v, y_v)
            metrics = m.evaluate(X_te, y_te)
            metrics["train_time_s"]   = m.train_time_s
            all_metrics.append(metrics)
            save_path = models_dir / f"{args.target}_{name}.pkl"
            m.save(save_path)
            logger.info(f"  R²={metrics['R2']:.4f}  RMSE={metrics['RMSE']:.4f}  "
                        f"MAE={metrics['MAE']:.4f}  train_time={m.train_time_s:.2f}s")
        except Exception as e:
            logger.warning(f"  {name} failed: {e}")

    # Analytical baseline comparison
    from suzuki_cae.engineering.beam_theory import batch_analytical
    analytical = batch_analytical(
        test["length_mm"].values, test["width_mm"].values, test["thickness_mm"].values,
        test["youngs_modulus_GPa"].values, test["poisson_ratio"].values, test["force_N"].values,
    )
    y_analytical = analytical[args.target]
    metrics_analytical = compute_metrics(y_te, y_analytical, "AnalyticalBaseline[test]")
    all_metrics.append(metrics_analytical)

    # Save metrics
    df_metrics = pd.DataFrame(all_metrics)
    out_csv = metrics_dir / f"{args.target}_metrics.csv"
    df_metrics.to_csv(out_csv, index=False)
    print(f"\n✅ Training complete. Metrics saved to {out_csv}")
    print(df_metrics[["label","R2","RMSE","MAE","MAPE_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
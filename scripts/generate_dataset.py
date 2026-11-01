#!/usr/bin/env python3
"""
Generate the CAE training dataset using Latin Hypercube Sampling + analytical solver.

Usage:
    python scripts/generate_dataset.py [--n-samples 600] [--noise 1.5] [--seed 42]

Outputs:
    data/raw/cae/beam_dataset_analytical.csv
    data/raw/experimental/synthetic_experimental.csv
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from suzuki_cae.data.generation import DatasetGenerator, build_synthetic_experimental_dataset
from suzuki_cae.utils.logging import get_logger

logger = get_logger("generate_dataset")


def main():
    parser = argparse.ArgumentParser(description="Generate beam dataset")
    parser.add_argument("--n-samples", type=int, default=600)
    parser.add_argument("--noise",     type=float, default=1.5,
                        help="Gaussian noise % added to analytical outputs (mimics mesh error)")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    root = Path(__file__).parents[1]

    # --- CAE (analytical) dataset ---
    gen = DatasetGenerator(n_samples=args.n_samples, noise_std_pct=args.noise, seed=args.seed)
    df_cae = gen.generate()
    out_cae = root / "data" / "raw" / "cae" / "beam_dataset_analytical.csv"
    gen.save(df_cae, out_cae)
    logger.info(f"CAE dataset: {len(df_cae)} rows → {out_cae}")

    # --- Synthetic experimental dataset ---
    df_exp = build_synthetic_experimental_dataset(n_specimens=5, trials_per_load=3, seed=99)
    out_exp = root / "data" / "raw" / "experimental" / "synthetic_experimental.csv"
    out_exp.parent.mkdir(parents=True, exist_ok=True)
    df_exp.to_csv(out_exp, index=False)
    logger.info(f"Experimental dataset: {len(df_exp)} rows → {out_exp}")

    print("\n✅ Dataset generation complete.")
    print(f"   CAE rows:          {len(df_cae)}")
    print(f"   Experimental rows: {len(df_exp)}")


if __name__ == "__main__":
    main()
"""
Exploratory data analysis utilities.

Produces summary statistics, distributions, correlations and pairplots
for the generated CAE dataset.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from pathlib import Path

from ..data.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS
from ..utils.logging import get_logger

logger = get_logger(__name__)


class EDAReport:
    """Run exploratory analysis and save figures.

    Parameters
    ----------
    df         : Cleaned simulation dataset.
    output_dir : Directory to save figures.
    """

    def __init__(self, df: pd.DataFrame, output_dir: Path = Path("results/figures")) -> None:
        self.df  = df
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    def summary_statistics(self) -> pd.DataFrame:
        """Return descriptive statistics for all numeric columns."""
        return self.df[FEATURE_COLUMNS + TARGET_COLUMNS].describe().T

    def correlation_matrix(self, save: bool = True) -> pd.DataFrame:
        """Pearson correlation between inputs and outputs."""
        cols  = FEATURE_COLUMNS + TARGET_COLUMNS
        corr  = self.df[cols].corr()
        if save:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                        center=0, square=True, ax=ax, annot_kws={"size": 8})
            ax.set_title("Pearson Correlation Matrix")
            fig.tight_layout()
            fig.savefig(self.out / "correlation_matrix.png", dpi=150)
            plt.close(fig)
            logger.info("Saved correlation_matrix.png")
        return corr

    def output_distributions(self, save: bool = True) -> None:
        """Plot histograms of all target outputs."""
        fig, axes = plt.subplots(1, len(TARGET_COLUMNS), figsize=(14, 4))
        for ax, col in zip(axes, TARGET_COLUMNS):
            data = self.df[col].dropna()
            ax.hist(data, bins=40, edgecolor="white", color="#1f77b4")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            ax.set_title(f"Distribution of {col}")
        fig.tight_layout()
        if save:
            fig.savefig(self.out / "output_distributions.png", dpi=150)
            plt.close(fig)
            logger.info("Saved output_distributions.png")

    def doe_coverage(self, save: bool = True) -> None:
        """Pairplot of the design parameters to visualise LHS coverage."""
        sample = self.df[FEATURE_COLUMNS].sample(min(300, len(self.df)), random_state=0)
        fig = sns.pairplot(sample, plot_kws={"alpha": 0.3, "s": 10})
        fig.figure.suptitle("LHS Design Space Coverage", y=1.02)
        if save:
            fig.savefig(self.out / "doe_coverage.png", dpi=120)
            plt.close("all")
            logger.info("Saved doe_coverage.png")

    def run_all(self) -> dict:
        stats = self.summary_statistics()
        corr  = self.correlation_matrix()
        self.output_distributions()
        self.doe_coverage()
        return {"summary": stats, "correlation": corr}
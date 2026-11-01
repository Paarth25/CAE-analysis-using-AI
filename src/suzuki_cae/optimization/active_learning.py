"""
Active learning loop for intelligent simulation selection.

Strategy: query-by-committee / uncertainty sampling.
  1. Train a GP or ensemble surrogate on current dataset.
  2. Predict uncertainty on a large candidate pool.
  3. Select the top-k highest-uncertainty candidates.
  4. 'Run' new simulations (analytical oracle here; CalculiX in production).
  5. Add results to dataset.
  6. Retrain and track improvement vs. random sampling baseline.
"""

from __future__ import annotations

import logging
from typing import Callable
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .design_optimizer import DesignOptimizer
from ..data.generation import DatasetGenerator, latin_hypercube_sample, PARAM_BOUNDS
from ..engineering.beam_theory import batch_analytical
from ..uncertainty.ensemble_uncertainty import EnsembleUQ
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ActiveLearner:
    """Uncertainty-guided dataset enrichment loop.

    Parameters
    ----------
    oracle_fn   : Callable(params_df) → outputs_df.  In production this calls
                  the CalculiX solver; here we use the analytical model.
    target      : Output variable to optimise surrogate accuracy for.
    n_candidates: Size of candidate pool sampled per iteration.
    n_select    : Number of candidates selected per iteration.
    n_random    : Random-sampling baseline (equal budget).
    seed        : Random seed.
    """

    def __init__(
        self,
        target: str = "tip_deflection_mm",
        n_candidates: int = 2000,
        n_select: int = 10,
        seed: int = 42,
    ) -> None:
        self.target       = target
        self.n_candidates = n_candidates
        self.n_select     = n_select
        self.seed         = seed
        self._history: list[dict] = []

    @staticmethod
    def _oracle(candidate_df: pd.DataFrame) -> pd.Series:
        """Analytical oracle – replace with CalculiX calls in production."""
        out = batch_analytical(
            length_mm          = candidate_df["length_mm"].values,
            width_mm           = candidate_df["width_mm"].values,
            thickness_mm       = candidate_df["thickness_mm"].values,
            youngs_modulus_GPa = candidate_df["youngs_modulus_GPa"].values,
            poisson_ratio      = candidate_df["poisson_ratio"].values,
            force_N            = candidate_df["force_N"].values,
        )
        return pd.Series(out["tip_deflection_mm"], name="tip_deflection_mm")

    def run(
        self,
        initial_df: pd.DataFrame,
        n_iterations: int = 5,
        features: list[str] | None = None,
    ) -> pd.DataFrame:
        """Execute the active learning loop.

        Returns
        -------
        history DataFrame with columns:
            iteration, strategy, n_train, rmse_test
        """
        if features is None:
            features = ["length_mm","width_mm","thickness_mm",
                        "youngs_modulus_GPa","poisson_ratio","force_N"]

        pool_active = initial_df.copy()
        pool_random = initial_df.copy()
        history = []

        rng = np.random.default_rng(self.seed)

        # Common held-out test set
        test_doe = latin_hypercube_sample(300, PARAM_BOUNDS, seed=self.seed + 999)
        test_out = self._oracle(test_doe)
        X_test = test_doe[features].values
        y_test = test_out.values

        for iteration in range(n_iterations):
            logger.info(f"Active learning iteration {iteration+1}/{n_iterations}")

            # --- ACTIVE: train UQ model ---
            X_a = pool_active[features].values
            y_a = pool_active[self.target].values
            uq = EnsembleUQ(
                base_estimator=RandomForestRegressor(n_estimators=50, random_state=42),
                n_estimators=20,
                seed=self.seed,
            )
            uq.fit(X_a, y_a)

            # Evaluate on test set
            pi = uq.predict_interval(X_test)
            rmse_active = np.sqrt(np.mean((y_test - pi.mean)**2))

            # Sample candidates
            cand_doe = latin_hypercube_sample(self.n_candidates, PARAM_BOUNDS,
                                              seed=self.seed + iteration * 100)
            X_cand = cand_doe[features].values
            pi_cand = uq.predict_interval(X_cand)

            # Select highest-uncertainty candidates
            top_idx = np.argsort(pi_cand.std)[-self.n_select:]
            new_rows = cand_doe.iloc[top_idx].copy()
            new_rows[self.target] = self._oracle(new_rows).values
            pool_active = pd.concat([pool_active, new_rows], ignore_index=True)

            # --- RANDOM baseline ---
            X_r = pool_random[features].values
            y_r = pool_random[self.target].values
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X_r, y_r)
            rmse_random = np.sqrt(np.mean((y_test - rf.predict(X_test))**2))

            rand_idx = rng.integers(0, len(cand_doe), size=self.n_select)
            rand_rows = cand_doe.iloc[rand_idx].copy()
            rand_rows[self.target] = self._oracle(rand_rows).values
            pool_random = pd.concat([pool_random, rand_rows], ignore_index=True)

            history.append(dict(
                iteration   = iteration + 1,
                n_train     = len(pool_active),
                rmse_active = rmse_active,
                rmse_random = rmse_random,
                improvement_pct = (rmse_random - rmse_active) / rmse_random * 100,
            ))
            logger.info(f"  RMSE active={rmse_active:.5f}, random={rmse_random:.5f}, "
                        f"improvement={history[-1]['improvement_pct']:.1f}%")

        return pd.DataFrame(history)
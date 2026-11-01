"""
Mesh convergence study.

Runs the same beam geometry at multiple mesh densities and compares results
to assess discretisation error.  Richardson extrapolation is used to estimate
the grid-converged value.

Reference: Roache (1998) Verification and Validation in Computational Science.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Optional
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConvergencePoint:
    mesh_size_mm:  float
    n_elements:    int
    tip_deflection_mm:  float
    max_von_mises_MPa:  float
    error_vs_analytical_pct: float


class MeshConvergenceStudy:
    """Run mesh refinement study and compute Richardson-extrapolated values.

    Parameters
    ----------
    solver_fn : Callable(mesh_size_mm) → SimulationResult
    mesh_sizes_mm : Ordered list from coarsest to finest.
    analytical_deflection_mm : Reference value for error computation.
    """

    def __init__(
        self,
        solver_fn: Callable,
        mesh_sizes_mm: list[float],
        analytical_deflection_mm: float,
    ) -> None:
        self.solver_fn     = solver_fn
        self.mesh_sizes    = sorted(mesh_sizes_mm, reverse=True)   # coarse→fine
        self.analytical    = analytical_deflection_mm
        self.results:      list[ConvergencePoint] = []

    def run(self) -> pd.DataFrame:
        for h in self.mesh_sizes:
            logger.info(f"Running mesh size {h} mm …")
            sim = self.solver_fn(h)
            if sim.tip_deflection_mm is None:
                logger.warning(f"  Skipped h={h}: solver returned None")
                continue
            err = (sim.tip_deflection_mm - self.analytical) / self.analytical * 100
            self.results.append(ConvergencePoint(
                mesh_size_mm=h,
                n_elements=sim.n_elements,
                tip_deflection_mm=sim.tip_deflection_mm,
                max_von_mises_MPa=sim.max_von_mises_MPa or 0.0,
                error_vs_analytical_pct=err,
            ))

        return pd.DataFrame([vars(r) for r in self.results])

    def richardson_extrapolation(self) -> Optional[float]:
        """Estimate grid-converged tip deflection using Richardson extrapolation.

        Uses the two finest mesh results.
        Requires at least 2 completed results.
        """
        if len(self.results) < 2:
            return None
        fine   = sorted(self.results, key=lambda r: r.mesh_size_mm)[0]
        medium = sorted(self.results, key=lambda r: r.mesh_size_mm)[1]
        p = 2.0   # assumed convergence order for quadratic elements
        r = medium.mesh_size_mm / fine.mesh_size_mm
        f2 = medium.tip_deflection_mm
        f1 = fine.tip_deflection_mm
        return f1 + (f1 - f2) / (r**p - 1)

    def grid_convergence_index(self) -> Optional[float]:
        """Compute the Grid Convergence Index (GCI) for tip deflection."""
        extrap = self.richardson_extrapolation()
        if extrap is None:
            return None
        fine = sorted(self.results, key=lambda r: r.mesh_size_mm)[0]
        eps  = (fine.tip_deflection_mm - extrap) / extrap
        Fs   = 1.25   # safety factor for two-mesh GCI
        return Fs * abs(eps)
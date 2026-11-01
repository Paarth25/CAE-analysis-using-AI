"""
Constrained structural design optimisation using the surrogate model.

Objective:  Minimise component mass.
Constraints:
  - Max Von Mises stress < allowable_stress_MPa
  - Tip deflection < allowable_deflection_mm
  - thickness_mm >= min_thickness_mm
  - All design parameters within training domain bounds

The surrogate-optimal design is verified with the analytical solver.
For a real deployment, verification would use the CalculiX FEA solver.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint, Bounds

from ..engineering.beam_theory import batch_analytical, BeamParameters, CantileverBeam
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OptimisationResult:
    length_mm:          float
    width_mm:           float
    thickness_mm:       float
    youngs_modulus_GPa: float
    poisson_ratio:      float
    force_N:            float
    mass_kg:            float
    tip_deflection_mm:  float
    max_von_mises_MPa:  float
    surrogate_mass_kg:  float
    verification_mass_kg: float
    verification_tip_deflection_mm: float
    verification_von_mises_MPa: float
    converged:          bool
    n_evaluations:      int


class DesignOptimizer:
    """Surrogate-model-driven structural design optimiser.

    Parameters
    ----------
    surrogate_mass        : Trained SurrogateModel predicting mass_kg.
    surrogate_deflection  : Trained SurrogateModel predicting tip_deflection_mm.
    surrogate_stress      : Trained SurrogateModel predicting max_von_mises_MPa.
    allowable_stress_MPa  : Maximum Von Mises stress constraint.
    allowable_deflection_mm : Maximum tip deflection constraint.
    fixed_material        : If not None, (E_GPa, nu, density) fixed to this tuple.
    """

    def __init__(
        self,
        surrogate_mass=None,
        surrogate_deflection=None,
        surrogate_stress=None,
        allowable_stress_MPa: float = 200.0,
        allowable_deflection_mm: float = 5.0,
        min_thickness_mm: float = 2.0,
        fixed_material: tuple | None = (68.9, 0.33, 2700.0),
    ) -> None:
        self.sm_mass        = surrogate_mass
        self.sm_deflection  = surrogate_deflection
        self.sm_stress      = surrogate_stress
        self.sigma_allow    = allowable_stress_MPa
        self.delta_allow    = allowable_deflection_mm
        self.t_min          = min_thickness_mm
        self.fixed_material = fixed_material   # (E_GPa, nu, density_kg_m3)

    def _x_to_params(self, x: np.ndarray) -> dict:
        """Unpack optimisation vector → parameter dict."""
        if self.fixed_material is not None:
            E, nu, rho = self.fixed_material
            L, b, t, F = x[0], x[1], x[2], x[3]
        else:
            L, b, t, E, nu, F = x[0], x[1], x[2], x[3], x[4], x[5]
            rho = 2700.0
        return dict(length_mm=L, width_mm=b, thickness_mm=t,
                    youngs_modulus_GPa=E, poisson_ratio=nu, force_N=F,
                    density_kg_m3=rho)

    def _analytical_outputs(self, params: dict) -> dict:
        """Fast analytical evaluation (used for verification)."""
        # Clamp to bounds to handle DE polish step potentially overshooting slightly
        clamped = dict(params)
        bounds_map = {
            "length_mm":          BeamParameters.LENGTH_BOUNDS,
            "width_mm":           BeamParameters.WIDTH_BOUNDS,
            "thickness_mm":       BeamParameters.THICKNESS_BOUNDS,
            "youngs_modulus_GPa": BeamParameters.MODULUS_BOUNDS,
            "poisson_ratio":      BeamParameters.POISSON_BOUNDS,
            "force_N":            BeamParameters.FORCE_BOUNDS,
        }
        for key, (lo, hi) in bounds_map.items():
            if key in clamped:
                clamped[key] = max(lo, min(hi, clamped[key]))
        p = BeamParameters(**{k: v for k, v in clamped.items()})
        beam = CantileverBeam(p)
        r = beam.solve()
        return {"mass_kg": r.mass_kg, "tip_deflection_mm": r.tip_deflection_mm,
                "max_von_mises_MPa": r.max_von_mises_MPa}

    def _feature_vec(self, params: dict) -> np.ndarray:
        """Build feature vector expected by the surrogate models."""
        return np.array([[
            params["length_mm"],
            params["width_mm"],
            params["thickness_mm"],
            params["youngs_modulus_GPa"],
            params["poisson_ratio"],
            params["force_N"],
        ]])

    def optimise(
        self,
        force_N: float = 50.0,
        seed: int = 42,
        popsize: int = 15,
        maxiter: int = 200,
    ) -> OptimisationResult:
        """Run differential evolution optimisation.

        Parameters
        ----------
        force_N  : Applied load (fixed throughout optimisation if material is fixed).
        """
        logger.info(f"Starting optimisation (F={force_N} N, σ_allow={self.sigma_allow} MPa, "
                    f"δ_allow={self.delta_allow} mm) …")

        use_analytical_fallback = (
            self.sm_mass is None or
            self.sm_deflection is None or
            self.sm_stress is None
        )

        if use_analytical_fallback:
            logger.warning("No surrogate models provided; using analytical solver as oracle.")

        # Bounds: (L, b, t) when material is fixed
        if self.fixed_material is not None:
            bounds = [
                (80.0, 160.0),    # L
                (15.0, 40.0),     # b
                (self.t_min, 6.0),# t
                (force_N, force_N),  # F fixed
            ]
        else:
            bounds = [
                (80.0,  160.0),
                (15.0,  40.0),
                (self.t_min, 6.0),
                (65.0,  210.0),
                (0.27,  0.35),
                (force_N, force_N),
            ]

        n_evals = [0]

        def objective(x):
            n_evals[0] += 1
            params = self._x_to_params(x)
            if use_analytical_fallback:
                out = self._analytical_outputs(params)
                return out["mass_kg"]
            else:
                X = self._feature_vec(params)
                return float(self.sm_mass.predict(X)[0])

        def stress_constraint(x):
            params = self._x_to_params(x)
            if use_analytical_fallback:
                return self._analytical_outputs(params)["max_von_mises_MPa"]
            return float(self.sm_stress.predict(self._feature_vec(params))[0])

        def deflection_constraint(x):
            params = self._x_to_params(x)
            if use_analytical_fallback:
                return self._analytical_outputs(params)["tip_deflection_mm"]
            return float(self.sm_deflection.predict(self._feature_vec(params))[0])

        constraints = [
            NonlinearConstraint(stress_constraint,    -np.inf, self.sigma_allow),
            NonlinearConstraint(deflection_constraint, -np.inf, self.delta_allow),
        ]

        result = differential_evolution(
            objective,
            bounds=Bounds([b[0] for b in bounds], [b[1] for b in bounds]),
            constraints=constraints,
            seed=seed,
            popsize=popsize,
            maxiter=maxiter,
            tol=1e-5,
            mutation=(0.5, 1.5),
            recombination=0.9,
            polish=True,
            workers=1,
        )

        opt_params = self._x_to_params(result.x)
        logger.info(f"  Optimisation {'converged' if result.success else 'did not converge'}.")
        logger.info(f"  Best objective (surrogate mass): {result.fun:.6f} kg")
        logger.info(f"  Parameters: {opt_params}")

        # Verification with analytical solver (or CalculiX in production)
        verif = self._analytical_outputs(opt_params)
        logger.info(f"  Verification mass:  {verif['mass_kg']:.6f} kg")
        logger.info(f"  Verification δ_tip: {verif['tip_deflection_mm']:.4f} mm")
        logger.info(f"  Verification σ_vm:  {verif['max_von_mises_MPa']:.2f} MPa")

        surr_deflection = deflection_constraint(result.x)
        surr_stress     = stress_constraint(result.x)

        return OptimisationResult(
            length_mm=opt_params["length_mm"],
            width_mm=opt_params["width_mm"],
            thickness_mm=opt_params["thickness_mm"],
            youngs_modulus_GPa=opt_params["youngs_modulus_GPa"],
            poisson_ratio=opt_params["poisson_ratio"],
            force_N=opt_params["force_N"],
            mass_kg=verif["mass_kg"],
            tip_deflection_mm=surr_deflection,
            max_von_mises_MPa=surr_stress,
            surrogate_mass_kg=result.fun,
            verification_mass_kg=verif["mass_kg"],
            verification_tip_deflection_mm=verif["tip_deflection_mm"],
            verification_von_mises_MPa=verif["max_von_mises_MPa"],
            converged=result.success,
            n_evaluations=n_evals[0],
        )
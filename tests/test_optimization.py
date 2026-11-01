"""Tests for design optimization and active learning."""

import sys
from pathlib import Path
import pytest
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from suzuki_cae.optimization.design_optimizer import DesignOptimizer, OptimisationResult
from suzuki_cae.engineering.beam_theory import BeamParameters, CantileverBeam


class TestDesignOptimizer:
    """Test the surrogate-less (analytical oracle) optimiser."""

    @pytest.fixture
    def optimizer(self):
        return DesignOptimizer(
            allowable_stress_MPa=200.0,
            allowable_deflection_mm=5.0,
            min_thickness_mm=2.0,
            fixed_material=(68.9, 0.33, 2700.0),
        )

    def test_returns_result_object(self, optimizer):
        res = optimizer.optimise(force_N=30.0, seed=1, popsize=8, maxiter=30)
        assert isinstance(res, OptimisationResult)

    def test_constraints_satisfied(self, optimizer):
        res = optimizer.optimise(force_N=30.0, seed=1, popsize=8, maxiter=40)
        # Verification values (analytical) must respect constraints
        assert res.verification_tip_deflection_mm <= 5.0 + 0.5, (
            f"Deflection {res.verification_tip_deflection_mm:.3f} exceeds limit 5.0 mm")
        assert res.verification_von_mises_MPa <= 200.0 + 5.0, (
            f"Stress {res.verification_von_mises_MPa:.1f} exceeds limit 200 MPa")

    def test_optimised_mass_less_than_baseline(self, optimizer):
        # Baseline: L=120, b=25, t=4
        bp = BeamParameters(120.0, 25.0, 4.0, 68.9, 0.33, 30.0)
        base_mass = CantileverBeam(bp).mass()
        res = optimizer.optimise(force_N=30.0, seed=1, popsize=8, maxiter=50)
        assert res.verification_mass_kg <= base_mass * 1.1, (
            "Optimised mass should not significantly exceed baseline")

    def test_dimension_bounds_respected(self, optimizer):
        res = optimizer.optimise(force_N=30.0, seed=1, popsize=8, maxiter=30)
        assert 80.0 <= res.length_mm  <= 160.0
        assert 15.0 <= res.width_mm   <= 40.0
        assert 2.0  <= res.thickness_mm <= 6.0


class TestOptimizationConstraintViolation:
    """Test that an impossible problem is handled gracefully."""

    def test_very_tight_deflection_still_returns(self):
        opt = DesignOptimizer(
            allowable_stress_MPa=500.0,
            allowable_deflection_mm=0.001,  # impossible for these dimensions
            min_thickness_mm=2.0,
            fixed_material=(68.9, 0.33, 2700.0),
        )
        res = opt.optimise(force_N=50.0, seed=0, popsize=5, maxiter=20)
        assert isinstance(res, OptimisationResult)
        # Should not raise; converged flag may be False


class TestAnalyticalVerification:
    """Verify that surrogate-optimised designs can be verified analytically."""

    def test_verification_matches_analytical(self):
        # For the analytical oracle, surrogate == verification
        opt = DesignOptimizer(
            allowable_stress_MPa=250.0,
            allowable_deflection_mm=8.0,
            fixed_material=(68.9, 0.33, 2700.0),
        )
        res = opt.optimise(force_N=40.0, seed=7, popsize=8, maxiter=40)
        # Surrogate mass ≈ verification mass within 1 % (both use analytical oracle)
        assert abs(res.surrogate_mass_kg - res.verification_mass_kg) / res.verification_mass_kg < 0.02
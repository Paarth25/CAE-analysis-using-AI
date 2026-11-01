"""
Tests for the analytical beam theory module.

Reference values computed by hand from Euler-Bernoulli beam equations.
All tolerances reflect floating-point precision, not engineering approximations.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from suzuki_cae.engineering.beam_theory import (
    BeamParameters, CantileverBeam, BeamAnalyticalResults, batch_analytical
)


# ------------------------------------------------------------------ #
#  Reference calculation (manually verified)                           #
# ------------------------------------------------------------------ #
# Parameters:  L=100mm, b=20mm, t=4mm, E=70GPa, nu=0.33, F=50N
# I = 20 * 4³ / 12 = 106.667 mm⁴
# δ = 50 * 100³ / (3 * 70000 * 106.667 * 1e-12 * 1e-9 / 1e-6) -- done in SI:
#   L=0.1 m, I = 106.667e-12 m⁴, E = 70e9 Pa
#   δ = 50 * 0.1³ / (3 * 70e9 * 106.667e-12) = 0.05/(22.4) ≈ 0.002232 m ≈ 2.232 mm
# σ = 6*50*0.1 / (0.02 * 0.004²) = 30/(3.2e-7) = 93.75 MPa  (in SI, then MPa)

REF_PARAMS = dict(
    length_mm=100.0,
    width_mm=20.0,
    thickness_mm=4.0,
    youngs_modulus_GPa=70.0,
    poisson_ratio=0.33,
    force_N=50.0,
    density_kg_m3=2700.0,
)

def ref_beam() -> CantileverBeam:
    return CantileverBeam(BeamParameters(**REF_PARAMS))


class TestBeamParameters:
    def test_valid_construction(self):
        p = BeamParameters(**REF_PARAMS)
        assert p.length_mm == 100.0

    def test_invalid_length_raises(self):
        with pytest.raises(ValueError, match="length_mm"):
            BeamParameters(50.0, 20.0, 4.0, 70.0, 0.33, 50.0)  # L=50 < 80 mm

    def test_invalid_modulus_raises(self):
        with pytest.raises(ValueError, match="youngs_modulus_GPa"):
            BeamParameters(100.0, 20.0, 4.0, 300.0, 0.33, 50.0)

    def test_boundary_values_accepted(self):
        # Min and max boundary values should be accepted
        p_min = BeamParameters(80.0, 15.0, 2.0, 65.0, 0.27, 5.0)
        p_max = BeamParameters(160.0, 40.0, 6.0, 210.0, 0.35, 100.0)
        assert p_min.length_mm == 80.0
        assert p_max.length_mm == 160.0


class TestSecondMomentOfArea:
    def test_formula(self):
        beam = ref_beam()
        I = beam.second_moment_area()
        expected = 20.0 * 4.0**3 / 12.0   # = 106.667 mm⁴
        assert abs(I - expected) < 1e-6

    def test_units_mm4(self):
        # Result must be in mm⁴ (order of magnitude check)
        beam = ref_beam()
        I = beam.second_moment_area()
        assert 50 < I < 500   # mm⁴ for this geometry


class TestTipDeflection:
    def test_reference_value(self):
        beam = ref_beam()
        delta = beam.tip_deflection()
        # Manual: δ = 50*(0.1)³/(3*70e9*106.667e-12) m → mm
        I_m4 = 106.6667e-12
        expected_m = 50 * 0.1**3 / (3 * 70e9 * I_m4)
        expected_mm = expected_m * 1e3
        assert abs(delta - expected_mm) / expected_mm < 1e-6

    def test_increases_with_force(self):
        beam = ref_beam()
        d1 = beam.tip_deflection()
        p2 = BeamParameters(**{**REF_PARAMS, "force_N": 100.0})
        d2 = CantileverBeam(p2).tip_deflection()
        assert d2 == pytest.approx(2.0 * d1, rel=1e-6)

    def test_decreases_with_modulus(self):
        beam = ref_beam()
        d1 = beam.tip_deflection()
        p2 = BeamParameters(**{**REF_PARAMS, "youngs_modulus_GPa": 140.0})
        d2 = CantileverBeam(p2).tip_deflection()
        assert d2 == pytest.approx(0.5 * d1, rel=1e-6)

    def test_cubic_length_scaling(self):
        """δ ∝ L³: doubling L from 80 to 160 mm must produce 8× deflection."""
        p1 = BeamParameters(80.0, 20.0, 4.0, 70.0, 0.33, 50.0)
        p2 = BeamParameters(160.0, 20.0, 4.0, 70.0, 0.33, 50.0)
        d1 = CantileverBeam(p1).tip_deflection()
        d2 = CantileverBeam(p2).tip_deflection()
        assert d2 == pytest.approx(8.0 * d1, rel=1e-6)


class TestMaxBendingStress:
    def test_reference_value(self):
        beam = ref_beam()
        sigma = beam.max_bending_stress()
        # σ = 6*50*100 / (20*4²) = 30000/320 = 93.75 MPa
        assert abs(sigma - 93.75) < 0.01

    def test_linear_in_force(self):
        beam1 = ref_beam()
        s1 = beam1.max_bending_stress()
        p2 = BeamParameters(**{**REF_PARAMS, "force_N": 25.0})
        s2 = CantileverBeam(p2).max_bending_stress()
        assert s2 == pytest.approx(0.5 * s1, rel=1e-6)


class TestStrainEnergy:
    def test_twice_force_quadruple_energy(self):
        """U ∝ F²."""
        beam1 = ref_beam()
        u1 = beam1.strain_energy()
        p2 = BeamParameters(**{**REF_PARAMS, "force_N": 100.0})
        u2 = CantileverBeam(p2).strain_energy()
        assert u2 == pytest.approx(4.0 * u1, rel=1e-6)

    def test_positive(self):
        assert ref_beam().strain_energy() > 0.0


class TestMass:
    def test_formula(self):
        beam = ref_beam()
        m = beam.mass()
        expected = 2700.0 * (20e-3) * (4e-3) * (100e-3)
        assert abs(m - expected) < 1e-10

    def test_scales_with_density(self):
        beam1 = ref_beam()
        m1 = beam1.mass()
        p2 = BeamParameters(**{**REF_PARAMS, "density_kg_m3": 7850.0})
        m2 = CantileverBeam(p2).mass()
        assert m2 == pytest.approx(m1 * 7850.0 / 2700.0, rel=1e-6)


class TestSolve:
    def test_returns_all_fields(self):
        beam = ref_beam()
        r = beam.solve()
        assert isinstance(r, BeamAnalyticalResults)
        assert r.tip_deflection_mm > 0
        assert r.max_von_mises_MPa > 0
        assert r.mass_kg > 0
        assert r.strain_energy_J > 0

    def test_max_displacement_equals_tip(self):
        """For a cantilever with end load, max displacement is at tip."""
        r = ref_beam().solve()
        assert r.max_displacement_mm == r.tip_deflection_mm


class TestBatchAnalytical:
    def test_single_matches_scalar(self):
        beam = ref_beam()
        scalar_delta = beam.tip_deflection()
        batch_out = batch_analytical(
            np.array([REF_PARAMS["length_mm"]]),
            np.array([REF_PARAMS["width_mm"]]),
            np.array([REF_PARAMS["thickness_mm"]]),
            np.array([REF_PARAMS["youngs_modulus_GPa"]]),
            np.array([REF_PARAMS["poisson_ratio"]]),
            np.array([REF_PARAMS["force_N"]]),
        )
        assert abs(batch_out["tip_deflection_mm"][0] - scalar_delta) < 1e-8

    def test_array_shape(self):
        n = 50
        out = batch_analytical(
            np.full(n, 100.0), np.full(n, 20.0), np.full(n, 4.0),
            np.full(n, 70.0), np.full(n, 0.33), np.full(n, 50.0),
        )
        assert all(len(v) == n for v in out.values())
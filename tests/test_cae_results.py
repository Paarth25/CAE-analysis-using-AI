"""Tests for CAE module (solver availability check + result extraction logic)."""

import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from suzuki_cae.cae.solver import CalculixSolver, SolverStatus, SimulationResult
from suzuki_cae.cae.result_extraction import ResultExtractor


class TestCalculixSolver:
    def test_instantiates_without_error(self):
        solver = CalculixSolver(executable="ccx_does_not_exist")
        assert solver is not None

    def test_returns_not_found_when_missing(self):
        solver = CalculixSolver(executable="ccx_does_not_exist_xyz")
        result = solver.run(Path("dummy.inp"), "test_001")
        assert result.solver_status == SolverStatus.NOT_FOUND

    def test_available_property(self):
        solver = CalculixSolver(executable="ccx_does_not_exist_xyz")
        assert solver.available is False


class TestVonMisesCalculation:
    """Unit tests for stress invariant calculations (no FEA needed)."""

    def test_uniaxial_tension(self):
        """Von Mises under uniaxial tension σ₁₁ = σ must equal σ."""
        extractor = ResultExtractor()
        sigma = 100.0
        vm = extractor._von_mises(sigma, 0, 0, 0, 0, 0)
        assert abs(vm - sigma) < 1e-6

    def test_pure_shear(self):
        """Von Mises for pure shear τ must equal √3 τ."""
        import math
        extractor = ResultExtractor()
        tau = 50.0
        vm  = extractor._von_mises(0, 0, 0, tau, 0, 0)
        assert abs(vm - math.sqrt(3.0) * tau) < 1e-6

    def test_hydrostatic_zero_vm(self):
        """Hydrostatic stress σ₁=σ₂=σ₃ gives Von Mises = 0."""
        extractor = ResultExtractor()
        sigma = 200.0
        vm = extractor._von_mises(sigma, sigma, sigma, 0, 0, 0)
        assert vm < 1e-8

    def test_principal_uniaxial(self):
        """Max principal stress under uniaxial load equals that load."""
        extractor = ResultExtractor()
        sigma = 150.0
        p1 = extractor._principal_max(sigma, 0, 0, 0, 0, 0)
        assert abs(p1 - sigma) < 1e-6


class TestInputDeckWriter:
    def test_writes_file(self, tmp_path):
        from suzuki_cae.cae.input_deck import InputDeckWriter
        # Create a minimal mock mesh .inp
        mock_mesh = tmp_path / "beam.inp"
        mock_mesh.write_text("** mesh placeholder\n*NODE\n1, 0.0, 0.0, 0.0\n")

        writer = InputDeckWriter(
            mesh_inp_path=mock_mesh,
            youngs_modulus_GPa=70.0,
            poisson_ratio=0.33,
            density_kg_m3=2700.0,
            force_N=50.0,
        )
        out = writer.write(tmp_path / "beam_analysis.inp")
        assert out.exists()
        content = out.read_text()
        assert "*MATERIAL" in content
        assert "*STEP"     in content
        assert "*BOUNDARY" in content
        assert "70000.00"  in content   # E in MPa

    def test_force_in_deck(self, tmp_path):
        from suzuki_cae.cae.input_deck import InputDeckWriter
        mock = tmp_path / "m.inp"
        mock.write_text("** mock\n")
        writer = InputDeckWriter(mock, 70.0, 0.33, 2700.0, 75.0)
        out = writer.write(tmp_path / "out.inp")
        assert "75.000000" in out.read_text()
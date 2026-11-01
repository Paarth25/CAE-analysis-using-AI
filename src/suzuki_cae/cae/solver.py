"""
CalculiX solver runner and result extraction.

Calls ccx as a subprocess and parses the .frd (results) and .dat
(printed output) files to extract engineering quantities.
"""

from __future__ import annotations

import subprocess
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)


class SolverStatus(str, Enum):
    OK        = "ok"
    FAILED    = "failed"
    TIMEOUT   = "timeout"
    NOT_FOUND = "solver_not_found"


@dataclass
class SimulationResult:
    """Results from a single CalculiX simulation run."""
    simulation_id:          str
    solver_status:          SolverStatus
    tip_deflection_mm:      Optional[float] = None
    max_displacement_mm:    Optional[float] = None
    max_von_mises_MPa:      Optional[float] = None
    max_principal_stress_MPa: Optional[float] = None
    max_strain:             Optional[float] = None
    strain_energy_J:        Optional[float] = None
    mass_kg:                Optional[float] = None
    n_nodes:                int = 0
    n_elements:             int = 0
    mesh_size_mm:           float = 0.0
    solver_wall_time_s:     float = 0.0
    error_message:          str = ""
    raw_frd_path:           Optional[Path] = None


class CalculixSolver:
    """Wrap the ccx binary and extract scalar engineering results.

    Parameters
    ----------
    executable      : Name or full path of the CalculiX binary.
    timeout_seconds : Kill solver after this many seconds.
    work_dir        : Temporary directory for solver files.
    """

    def __init__(
        self,
        executable: str = "ccx",
        timeout_seconds: int = 120,
        work_dir: Path = Path("data/raw/cae/runs"),
    ) -> None:
        self.executable = executable
        self.timeout    = timeout_seconds
        self.work_dir   = Path(work_dir)
        self._available = self._check_executable()

    def _check_executable(self) -> bool:
        found = shutil.which(self.executable) is not None
        if not found:
            logger.warning(
                f"CalculiX executable '{self.executable}' not found on PATH. "
                "Install CalculiX and ensure 'ccx' is accessible, or set "
                "executable to its full path."
            )
        return found

    @property
    def available(self) -> bool:
        return self._available

    def run(self, inp_path: Path, simulation_id: str) -> SimulationResult:
        """Execute ccx on *inp_path* and return a SimulationResult.

        If CalculiX is not installed, returns a NOT_FOUND status rather
        than raising an exception.
        """
        import time

        if not self._available:
            return SimulationResult(
                simulation_id=simulation_id,
                solver_status=SolverStatus.NOT_FOUND,
                error_message=(
                    f"CalculiX binary '{self.executable}' not found. "
                    "Install CalculiX to enable FEA simulation."
                ),
            )

        inp_path   = Path(inp_path)
        run_dir    = self.work_dir / simulation_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Copy .inp to run directory (ccx writes outputs alongside the input)
        run_inp = run_dir / inp_path.name
        import shutil as _sh
        _sh.copy2(inp_path, run_inp)

        # Stem (ccx takes the filename without extension)
        stem = run_inp.stem

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [self.executable, stem],
                cwd=run_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            wall_time = time.perf_counter() - t0
        except subprocess.TimeoutExpired:
            return SimulationResult(
                simulation_id=simulation_id,
                solver_status=SolverStatus.TIMEOUT,
                solver_wall_time_s=self.timeout,
                error_message=f"Solver timed out after {self.timeout}s",
            )
        except Exception as exc:
            return SimulationResult(
                simulation_id=simulation_id,
                solver_status=SolverStatus.FAILED,
                error_message=str(exc),
            )

        if proc.returncode != 0:
            logger.warning(f"ccx returned {proc.returncode} for {simulation_id}")
            return SimulationResult(
                simulation_id=simulation_id,
                solver_status=SolverStatus.FAILED,
                solver_wall_time_s=wall_time,
                error_message=proc.stderr[:500],
            )

        frd_path = run_dir / f"{stem}.frd"
        dat_path = run_dir / f"{stem}.dat"

        from .result_extraction import ResultExtractor
        extractor = ResultExtractor()
        result    = extractor.extract(frd_path, dat_path, simulation_id)
        result.solver_wall_time_s = wall_time
        result.raw_frd_path       = frd_path

        logger.info(
            f"[{simulation_id}] ccx OK in {wall_time:.1f}s | "
            f"δ_tip={result.tip_deflection_mm:.3f}mm "
            f"σ_vm={result.max_von_mises_MPa:.1f}MPa"
        )
        return result
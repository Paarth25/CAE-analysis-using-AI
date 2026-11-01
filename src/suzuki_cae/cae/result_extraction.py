"""
CalculiX .frd result file parser.

The .frd (finite element results data) format is CalculiX's binary/ASCII
results format.  This module reads:
  - Nodal displacements (U1, U2, U3)  → tip deflection, max displacement
  - Nodal Von Mises stress (S)
  - Nodal principal stresses
  - Element strain energy (ENER)

Parsing strategy
----------------
We use a lightweight line-by-line ASCII parser.  The .frd file structure:
  - Lines starting with ' 1' contain nodal result headers.
  - Lines starting with ' -1' contain nodal data.
  - Lines starting with ' -3' mark end of a result block.

Note: For production use, the `frd` Python package or the official
CalculiX GraphiX viewer (cgx) provide more robust parsing.
"""

from __future__ import annotations

import re
import math
from pathlib import Path
from typing import Optional

import numpy as np

from .solver import SimulationResult, SolverStatus
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ResultExtractor:
    """Parse CalculiX .frd and .dat output files."""

    def extract(
        self,
        frd_path: Path,
        dat_path: Path,
        simulation_id: str,
    ) -> SimulationResult:
        """Extract all engineering scalars from CalculiX output files."""
        frd_path = Path(frd_path)
        dat_path = Path(dat_path)

        result = SimulationResult(
            simulation_id=simulation_id,
            solver_status=SolverStatus.OK,
        )

        if not frd_path.exists():
            result.solver_status = SolverStatus.FAILED
            result.error_message = f".frd file not found: {frd_path}"
            return result

        try:
            disp_data, stress_data = self._parse_frd(frd_path)

            if disp_data:
                u_magnitudes = [math.sqrt(u[0]**2 + u[1]**2 + u[2]**2) * 1e3
                                 for u in disp_data.values()]   # m → mm
                u_z = [abs(u[2]) * 1e3 for u in disp_data.values()]

                result.max_displacement_mm = max(u_magnitudes)
                result.tip_deflection_mm   = max(u_z)    # tip at max-x nodes

            if stress_data:
                vm_list = [self._von_mises(*s[:6]) for s in stress_data.values()]
                p1_list = [self._principal_max(*s[:6]) for s in stress_data.values()]
                result.max_von_mises_MPa        = max(vm_list)   # already MPa from CalculiX (mm/N)
                result.max_principal_stress_MPa = max(p1_list)

            # Strain energy from .dat file
            result.strain_energy_J = self._parse_dat_strain_energy(dat_path)

        except Exception as exc:
            logger.warning(f"Result extraction failed for {simulation_id}: {exc}")
            result.solver_status = SolverStatus.FAILED
            result.error_message  = str(exc)

        return result

    # ------------------------------------------------------------------ #
    #  FRD parser                                                          #
    # ------------------------------------------------------------------ #

    def _parse_frd(
        self, path: Path
    ) -> tuple[dict[int, tuple], dict[int, tuple]]:
        """Return (disp_data, stress_data) dicts keyed by node id.

        disp_data   : {node_id: (u1_m, u2_m, u3_m)}
        stress_data : {node_id: (s11, s22, s33, s12, s23, s13)} in MPa
        """
        disp_data:   dict[int, tuple] = {}
        stress_data: dict[int, tuple] = {}

        lines = path.read_text(errors="replace").splitlines()
        mode = None     # "DISP" or "STRESS"

        for line in lines:
            stripped = line.strip()

            # Block header detection
            if "DISP" in stripped and line.startswith(" 1"):
                mode = "DISP"; continue
            if "STRESS" in stripped and line.startswith(" 1"):
                mode = "STRESS"; continue
            if line.startswith(" -3"):
                mode = None; continue

            # Data lines
            if mode and line.startswith(" -1"):
                parts = stripped.split()
                if len(parts) < 4:
                    continue
                try:
                    nid = int(parts[1])
                    vals = tuple(float(v) for v in parts[2:])
                    if mode == "DISP" and len(vals) >= 3:
                        disp_data[nid] = vals[:3]
                    elif mode == "STRESS" and len(vals) >= 6:
                        stress_data[nid] = vals[:6]
                except (ValueError, IndexError):
                    continue

        return disp_data, stress_data

    # ------------------------------------------------------------------ #
    #  Stress invariants                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _von_mises(s11, s22, s33, s12, s23, s13) -> float:
        return math.sqrt(
            0.5 * ((s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2
                   + 6 * (s12**2 + s23**2 + s13**2))
        )

    @staticmethod
    def _principal_max(s11, s22, s33, s12, s23, s13) -> float:
        """Largest principal stress via numpy eigenvalue."""
        S = np.array([[s11, s12, s13],
                       [s12, s22, s23],
                       [s13, s23, s33]])
        eigs = np.linalg.eigvalsh(S)
        return float(eigs.max())

    # ------------------------------------------------------------------ #
    #  .dat parser                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_dat_strain_energy(dat_path: Path) -> Optional[float]:
        """Extract total strain energy from CalculiX .dat print file.

        The ENER print statement outputs one line per element with the
        elastic strain energy density [J/mm³].  We sum over all elements
        and convert to Joules.
        """
        if not dat_path.exists():
            return None
        try:
            lines = dat_path.read_text(errors="replace").splitlines()
            total = 0.0
            in_section = False
            for line in lines:
                if "INTERNAL ENERGY" in line.upper():
                    in_section = True; continue
                if in_section:
                    parts = line.split()
                    if not parts:
                        in_section = False; continue
                    try:
                        total += float(parts[-1])
                    except ValueError:
                        in_section = False
            return total if total > 0 else None
        except Exception:
            return None
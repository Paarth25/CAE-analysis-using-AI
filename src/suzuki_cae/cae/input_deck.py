"""
CalculiX .inp input-deck writer for the cantilever beam problem.

Writes a complete linear-elastic static analysis file that CalculiX (ccx)
can solve directly.  The mesh node/element data must already exist in a
Gmsh-exported .inp file which this module reads and appends to.

Structure of the generated .inp:
  *HEADING
  *NODE / *ELEMENT   (from Gmsh export)
  *MATERIAL
  *SOLID SECTION
  *BOUNDARY          (fixed end: all dof = 0)
  *STEP
  *STATIC
  *CLOAD             (distributed tip force)
  *NODE FILE / *EL FILE  (output requests)
  *END STEP
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent


class InputDeckWriter:
    """Assemble a CalculiX input deck from mesh + simulation parameters.

    Parameters
    ----------
    mesh_inp_path        : Path to Gmsh-exported Abaqus .inp file.
    youngs_modulus_GPa   : [GPa]
    poisson_ratio        : [-]
    density_kg_m3        : [kg/m³]
    force_N              : Total tip load [N] applied in -z direction.
    tip_node_set_name    : Name of the node set covering the tip face.
    fixed_node_set_name  : Name of the node set covering the fixed face.
    """

    def __init__(
        self,
        mesh_inp_path: Path,
        youngs_modulus_GPa: float,
        poisson_ratio: float,
        density_kg_m3: float,
        force_N: float,
        tip_node_set_name: str = "TIP_NODES",
        fixed_node_set_name: str = "FIXED_NODES",
    ) -> None:
        self.mesh_path   = Path(mesh_inp_path)
        self.E_MPa       = youngs_modulus_GPa * 1e3        # CalculiX uses MPa if mm/N
        self.nu          = poisson_ratio
        self.rho         = density_kg_m3 * 1e-12           # kg/m³ → t/mm³ (consistent mm/N/t)
        self.F           = force_N
        self.tip_set     = tip_node_set_name
        self.fixed_set   = fixed_node_set_name

    def write(self, output_path: Path) -> Path:
        """Write the complete .inp file to *output_path*."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        mesh_content = self.mesh_path.read_text()

        deck = dedent(f"""\
        ** ============================================================
        ** CalculiX input deck – cantilever beam linear static analysis
        ** Units: mm, N, MPa, tonne (consistent SI-mm system)
        ** ============================================================
        *HEADING
        Cantilever beam linear elastic static analysis
        **
        {mesh_content}
        **
        ** --- Material ---
        *MATERIAL, NAME=BEAM_MATERIAL
        *ELASTIC
        {self.E_MPa:.2f}, {self.nu:.4f}
        *DENSITY
        {self.rho:.6e}
        **
        ** --- Section ---
        *SOLID SECTION, ELSET=EALL, MATERIAL=BEAM_MATERIAL
        **
        ** --- Boundary conditions: fixed end (all dof = 0) ---
        *BOUNDARY
        {self.fixed_set}, 1, 6, 0.0
        **
        ** --- Step ---
        *STEP
        *STATIC
        **
        ** --- Load: distributed force on tip nodes ---
        ** (individual node forces sum to total F; actual distribution
        **  depends on node count, applied in -z direction)
        *CLOAD
        {self.tip_set}, 3, -{self.F:.6f}
        **
        ** --- Output requests ---
        *NODE FILE
        U, RF
        *EL FILE
        S, E, ENER
        *NODE PRINT, NSET={self.tip_set}
        U
        *EL PRINT
        S
        **
        *END STEP
        """)

        output_path.write_text(deck)
        return output_path
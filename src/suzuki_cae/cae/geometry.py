"""
Gmsh geometry and mesh generation for the cantilever beam.

Geometry
--------
The beam is a rectangular solid:
  - Fixed end at x = 0
  - Free (tip) end at x = L
  - Width in y-direction: b
  - Thickness in z-direction: t

Element type
------------
C3D10 (10-node tetrahedral, quadratic) via Gmsh's built-in Delaunay
mesher.  This element gives good stress accuracy even on moderately
coarse meshes.

Physical groups
---------------
  1 = fixed_face   (x=0 plane, for displacement BCs)
  2 = tip_face     (x=L plane, for force application)
  3 = volume       (all elements, for material assignment)

Usage
-----
    geom = BeamGeometry(length_mm=100, width_mm=20, thickness_mm=4)
    msh_path = geom.generate_mesh(mesh_size_mm=2.5, output_dir=Path("data/raw/cae/meshes"))
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)

try:
    import gmsh
    _HAS_GMSH = True
except ImportError:
    _HAS_GMSH = False
    logger.warning("gmsh Python API not found.  Install with: pip install gmsh")


@dataclass
class MeshInfo:
    """Basic mesh statistics returned after mesh generation."""
    path: Path
    n_nodes: int
    n_elements: int
    mesh_size_mm: float
    element_type: str = "C3D10"


class BeamGeometry:
    """Create Gmsh geometry and generate FEA mesh for a rectangular beam.

    Parameters
    ----------
    length_mm, width_mm, thickness_mm : Beam dimensions [mm].
    """

    def __init__(self, length_mm: float, width_mm: float, thickness_mm: float) -> None:
        self.L = length_mm
        self.b = width_mm
        self.t = thickness_mm

    def generate_mesh(
        self,
        mesh_size_mm: float = 2.5,
        output_dir: Path = Path("data/raw/cae/meshes"),
        sim_id: str = "beam",
    ) -> Optional[MeshInfo]:
        """Generate a C3D10 tetrahedral mesh and write a .inp file.

        Returns None if gmsh is not available.
        """
        if not _HAS_GMSH:
            logger.error("gmsh not available – cannot generate mesh.")
            return None

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        msh_path = output_dir / f"{sim_id}_mesh.msh"
        inp_path = output_dir / f"{sim_id}_mesh.inp"

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)   # suppress console spam
        gmsh.model.add("beam")

        # Geometry: rectangular box [0,L] × [0,b] × [0,t]
        box = gmsh.model.occ.addBox(0, 0, 0, self.L, self.b, self.t)
        gmsh.model.occ.synchronize()

        # Global mesh size
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size_mm * 0.5)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size_mm)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)   # Delaunay 3D

        # Physical groups
        # Fixed face: surface at x = 0
        fixed_surfs = [s for s in gmsh.model.occ.getSurfaces()
                        if abs(gmsh.model.occ.getCenterOfMass(2, s[1])[0]) < 1e-6]
        fixed_tags  = [s[1] for s in fixed_surfs]
        if fixed_tags:
            gmsh.model.addPhysicalGroup(2, fixed_tags, tag=1, name="fixed_face")

        # Tip face: surface at x = L
        tip_surfs = [s for s in gmsh.model.occ.getSurfaces()
                      if abs(gmsh.model.occ.getCenterOfMass(2, s[1])[0] - self.L) < 1e-6]
        tip_tags  = [s[1] for s in tip_surfs]
        if tip_tags:
            gmsh.model.addPhysicalGroup(2, tip_tags, tag=2, name="tip_face")

        # Volume
        gmsh.model.addPhysicalGroup(3, [box], tag=3, name="volume")

        # Generate second-order tet mesh (C3D10)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(2)

        # Save
        gmsh.write(str(msh_path))
        gmsh.write(str(inp_path))   # Abaqus .inp format for CalculiX

        # Count mesh entities
        _, el_tags, _ = gmsh.model.mesh.getElements(3)
        n_elements = sum(len(t) for t in el_tags)
        _, nd_tags, _ = gmsh.model.mesh.getNodes()
        n_nodes = len(nd_tags[0]) if nd_tags else 0

        gmsh.finalize()
        logger.info(f"Mesh generated: {n_nodes} nodes, {n_elements} elements → {inp_path}")

        return MeshInfo(
            path=inp_path,
            n_nodes=n_nodes,
            n_elements=n_elements,
            mesh_size_mm=mesh_size_mm,
        )
"""
CAE integration layer: Gmsh geometry/meshing + CalculiX FEA solver.

Requires external binaries:
  - gmsh (Python API: pip install gmsh)
  - ccx  (CalculiX, available via package manager or from calculix.de)

If these are not installed, all modules degrade gracefully and report
their unavailability rather than raising import errors at module load time.
"""
from .solver import CalculixSolver, SimulationResult, SolverStatus
from .geometry import BeamGeometry
from .input_deck import InputDeckWriter
from .result_extraction import ResultExtractor
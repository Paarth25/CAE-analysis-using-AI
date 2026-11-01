"""
Euler-Bernoulli Cantilever Beam Theory
=======================================
Analytical closed-form solutions for a rectangular cross-section cantilever beam
with a concentrated end load.  These equations serve as:
  1. Verification of the CAE solver.
  2. A physics-informed baseline for ML model comparison.

All parameters are accepted in engineering units (mm, N, GPa, MPa) and
conversions to SI are handled internally before any calculation.

Theory
------
Second moment of area (rectangular cross-section about neutral axis):
    I = b * t³ / 12

Tip deflection (Euler–Bernoulli, linear elastic):
    δ_tip = F * L³ / (3 * E * I)

Maximum bending stress at fixed root (outer fibre):
    σ_max = M_max * y_max / I  =  F * L * (t/2) / I  =  6 * F * L / (b * t²)

Shear stress at neutral axis (rectangular section):
    τ_max = 3 * F / (2 * b * t)

Von Mises stress at root (combined bending + shear, approximate):
    σ_vm  ≈  sqrt(σ_max² + 3 * τ_max²)

Strain energy (bending only):
    U = F² * L³ / (6 * E * I)

Component mass:
    m = ρ * b * t * L
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class BeamParameters:
    """Validated beam geometry, material and loading parameters.

    Attributes
    ----------
    length_mm       : Beam length [mm]
    width_mm        : Cross-section width [mm]
    thickness_mm    : Cross-section thickness [mm]
    youngs_modulus_GPa : Young's modulus [GPa]
    poisson_ratio   : Poisson's ratio [-]
    force_N         : Transverse end load [N]
    density_kg_m3   : Material density [kg/m³]  (default 2700 = aluminium)
    """

    length_mm: float
    width_mm: float
    thickness_mm: float
    youngs_modulus_GPa: float
    poisson_ratio: float
    force_N: float
    density_kg_m3: float = 2700.0

    # ------------------------------------------------------------------ #
    #  Design-space bounds (class-level constants)                         #
    # ------------------------------------------------------------------ #
    LENGTH_BOUNDS       = (80.0, 160.0)     # mm
    WIDTH_BOUNDS        = (15.0, 40.0)      # mm
    THICKNESS_BOUNDS    = (2.0, 6.0)        # mm
    MODULUS_BOUNDS      = (65.0, 210.0)     # GPa
    POISSON_BOUNDS      = (0.27, 0.35)
    FORCE_BOUNDS        = (5.0, 100.0)      # N

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Raise ValueError for any parameter outside design-space bounds."""
        checks = [
            ("length_mm",          self.length_mm,          self.LENGTH_BOUNDS),
            ("width_mm",           self.width_mm,            self.WIDTH_BOUNDS),
            ("thickness_mm",       self.thickness_mm,        self.THICKNESS_BOUNDS),
            ("youngs_modulus_GPa", self.youngs_modulus_GPa,  self.MODULUS_BOUNDS),
            ("poisson_ratio",      self.poisson_ratio,       self.POISSON_BOUNDS),
            ("force_N",            self.force_N,             self.FORCE_BOUNDS),
        ]
        for name, value, (lo, hi) in checks:
            if not (lo <= value <= hi):
                raise ValueError(
                    f"{name}={value} is outside the design-space bounds [{lo}, {hi}]."
                )
        if self.density_kg_m3 <= 0:
            raise ValueError("density_kg_m3 must be positive.")


@dataclass
class BeamAnalyticalResults:
    """Container for all analytical beam results.

    Attributes (all in engineering output units)
    -------------------------------------------
    second_moment_area_mm4    : I  [mm⁴]
    tip_deflection_mm         : δ_tip [mm]
    max_displacement_mm       : same as tip for cantilever with end load
    max_bending_stress_MPa    : σ_max at root, outer fibre [MPa]
    max_shear_stress_MPa      : τ_max at neutral axis [MPa]
    max_von_mises_MPa         : approximate combined Von Mises at root [MPa]
    max_principal_stress_MPa  : σ₁ at root, outer fibre (≈ bending stress) [MPa]
    max_strain                : ε_max = σ_max / E [-]
    strain_energy_J           : U (bending only) [J]
    mass_kg                   : component mass [kg]
    """

    second_moment_area_mm4: float
    tip_deflection_mm: float
    max_displacement_mm: float
    max_bending_stress_MPa: float
    max_shear_stress_MPa: float
    max_von_mises_MPa: float
    max_principal_stress_MPa: float
    max_strain: float
    strain_energy_J: float
    mass_kg: float

    def as_dict(self) -> dict:
        return {
            "second_moment_area_mm4":   self.second_moment_area_mm4,
            "tip_deflection_mm":        self.tip_deflection_mm,
            "max_displacement_mm":      self.max_displacement_mm,
            "max_bending_stress_MPa":   self.max_bending_stress_MPa,
            "max_shear_stress_MPa":     self.max_shear_stress_MPa,
            "max_von_mises_MPa":        self.max_von_mises_MPa,
            "max_principal_stress_MPa": self.max_principal_stress_MPa,
            "max_strain":               self.max_strain,
            "strain_energy_J":          self.strain_energy_J,
            "mass_kg":                  self.mass_kg,
        }


class CantileverBeam:
    """Euler-Bernoulli cantilever beam analytical solver.

    Parameters
    ----------
    params : BeamParameters
        Validated beam parameter set.

    Examples
    --------
    >>> p = BeamParameters(100, 20, 4, 70.0, 0.33, 50.0)
    >>> beam = CantileverBeam(p)
    >>> results = beam.solve()
    >>> print(f"Tip deflection: {results.tip_deflection_mm:.4f} mm")
    """

    def __init__(self, params: BeamParameters) -> None:
        self.p = params

    # ------------------------------------------------------------------ #
    #  Derived SI values used internally                                   #
    # ------------------------------------------------------------------ #

    @property
    def L_m(self) -> float:
        """Beam length in metres."""
        return self.p.length_mm * 1e-3

    @property
    def b_m(self) -> float:
        """Width in metres."""
        return self.p.width_mm * 1e-3

    @property
    def t_m(self) -> float:
        """Thickness in metres."""
        return self.p.thickness_mm * 1e-3

    @property
    def E_Pa(self) -> float:
        """Young's modulus in Pa."""
        return self.p.youngs_modulus_GPa * 1e9

    @property
    def F_N(self) -> float:
        """Applied force in N."""
        return self.p.force_N

    # ------------------------------------------------------------------ #
    #  Core analytical equations                                           #
    # ------------------------------------------------------------------ #

    def second_moment_area(self) -> float:
        """I = b * t³ / 12  [mm⁴]"""
        return (self.p.width_mm * self.p.thickness_mm**3) / 12.0

    def tip_deflection(self) -> float:
        """δ = F L³ / (3 E I)  [mm]

        Derivation: integrating the bending-moment equation twice along a
        simply fixed beam gives this closed form for a concentrated end load.
        """
        I_m4 = (self.b_m * self.t_m**3) / 12.0
        delta_m = (self.F_N * self.L_m**3) / (3.0 * self.E_Pa * I_m4)
        return delta_m * 1e3  # convert to mm

    def max_bending_stress(self) -> float:
        """σ_max = 6 F L / (b t²)  [MPa]

        From: M_max = F·L at fixed support, y_max = t/2, I = bt³/12
        → σ = M·y/I = F·L·(t/2) / (bt³/12) = 6FL/(bt²)
        """
        sigma_Pa = (6.0 * self.F_N * self.L_m) / (self.b_m * self.t_m**2)
        return sigma_Pa * 1e-6  # convert Pa → MPa

    def max_shear_stress(self) -> float:
        """τ_max = 3 V / (2 A)  at neutral axis of rectangular section  [MPa]

        Shear force V = F throughout the beam for a concentrated end load.
        Area A = b·t.
        """
        A_m2 = self.b_m * self.t_m
        tau_Pa = 1.5 * self.F_N / A_m2
        return tau_Pa * 1e-6

    def max_von_mises(self) -> float:
        """Approximate Von Mises stress at the fixed-end outer fibre [MPa].

        At the outer fibre the shear stress is zero; the maximum bending
        stress alone defines Von Mises there. At the neutral axis the bending
        stress is zero; τ_max gives σ_vm = √3·τ.  The absolute maximum
        Von Mises depends on the stress state along the section.  For a
        slender beam the bending-stress-dominated value at the outer fibre
        governs; we return max(σ_bend, √3·τ_max) as a conservative estimate.
        """
        sigma = self.max_bending_stress()
        tau = self.max_shear_stress()
        vm_outer = sigma  # shear = 0 at outer fibre
        vm_neutral = np.sqrt(3.0) * tau
        return float(max(vm_outer, vm_neutral))

    def max_principal_stress(self) -> float:
        """σ₁ at the fixed-end outer fibre (pure bending, τ = 0) [MPa].

        σ₁ = σ_bend/2 + sqrt((σ_bend/2)² + τ²) → σ_bend when τ = 0.
        """
        return self.max_bending_stress()

    def max_strain(self) -> float:
        """ε_max = σ_max / E  (linear elastic, uniaxial) [-]"""
        sigma_Pa = self.max_bending_stress() * 1e6
        return sigma_Pa / self.E_Pa

    def strain_energy(self) -> float:
        """U = F² L³ / (6 E I)  [J]

        Integral of M²/(2EI) dx from 0 to L with M(x) = F(L-x).
        """
        I_m4 = (self.b_m * self.t_m**3) / 12.0
        return (self.F_N**2 * self.L_m**3) / (6.0 * self.E_Pa * I_m4)

    def mass(self) -> float:
        """m = ρ · b · t · L  [kg]"""
        V_m3 = self.b_m * self.t_m * self.L_m
        return self.p.density_kg_m3 * V_m3

    # ------------------------------------------------------------------ #
    #  Solve – compute all outputs at once                                 #
    # ------------------------------------------------------------------ #

    def solve(self) -> BeamAnalyticalResults:
        """Compute and return all analytical results."""
        I_mm4 = self.second_moment_area()
        delta  = self.tip_deflection()
        return BeamAnalyticalResults(
            second_moment_area_mm4   = I_mm4,
            tip_deflection_mm        = delta,
            max_displacement_mm      = delta,       # tip governs for end load
            max_bending_stress_MPa   = self.max_bending_stress(),
            max_shear_stress_MPa     = self.max_shear_stress(),
            max_von_mises_MPa        = self.max_von_mises(),
            max_principal_stress_MPa = self.max_principal_stress(),
            max_strain               = self.max_strain(),
            strain_energy_J          = self.strain_energy(),
            mass_kg                  = self.mass(),
        )


# --------------------------------------------------------------------------- #
#  Batch analytical evaluation (vectorised over parameter arrays)              #
# --------------------------------------------------------------------------- #

def batch_analytical(
    length_mm: np.ndarray,
    width_mm: np.ndarray,
    thickness_mm: np.ndarray,
    youngs_modulus_GPa: np.ndarray,
    poisson_ratio: np.ndarray,
    force_N: np.ndarray,
    density_kg_m3: Optional[float] = 2700.0,
) -> dict[str, np.ndarray]:
    """Vectorised analytical solution for arrays of beam parameters.

    Skips the per-row BeamParameters validation intentionally for speed;
    use only with pre-validated design-of-experiments arrays.

    Returns
    -------
    dict mapping output name → numpy array of the same length as inputs.
    """
    L  = length_mm    * 1e-3
    b  = width_mm     * 1e-3
    t  = thickness_mm * 1e-3
    E  = youngs_modulus_GPa * 1e9
    F  = force_N

    I_m4   = b * t**3 / 12.0
    delta  = (F * L**3) / (3.0 * E * I_m4)           # m
    sigma  = (6.0 * F * L) / (b * t**2) * 1e-6        # MPa
    tau    = 1.5 * F / (b * t) * 1e-6                 # MPa
    vm     = np.maximum(sigma, np.sqrt(3.0) * tau)
    eps    = sigma * 1e6 / E
    U      = (F**2 * L**3) / (6.0 * E * I_m4)
    mass   = density_kg_m3 * b * t * L

    return {
        "tip_deflection_mm":        delta * 1e3,
        "max_displacement_mm":      delta * 1e3,
        "max_von_mises_MPa":        vm,
        "max_principal_stress_MPa": sigma,
        "max_strain":               eps,
        "strain_energy_J":          U,
        "mass_kg":                  mass,
        "second_moment_area_mm4":   I_m4 * 1e12,
    }
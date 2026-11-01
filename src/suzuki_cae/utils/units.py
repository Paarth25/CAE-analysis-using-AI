"""
Unit conversion utilities.

Convention: all internal calculations in SI (m, Pa, N, kg).
Engineering I/O uses mm, MPa, GPa, N, kg.
"""

from __future__ import annotations


# ---------- length ----------
def mm_to_m(v): return v * 1e-3
def m_to_mm(v): return v * 1e3

# ---------- stress / pressure ----------
def pa_to_mpa(v): return v * 1e-6
def mpa_to_pa(v): return v * 1e6
def gpa_to_pa(v): return v * 1e9
def pa_to_gpa(v): return v * 1e-9

# ---------- energy ----------
def nmm_to_j(v): return v * 1e-3  # N·mm → J

# ---------- mass ----------
def g_to_kg(v): return v * 1e-3
def kg_to_g(v): return v * 1e3


UNIT_LABELS = {
    "length_mm":          "mm",
    "width_mm":           "mm",
    "thickness_mm":       "mm",
    "youngs_modulus_GPa": "GPa",
    "poisson_ratio":      "–",
    "force_N":            "N",
    "tip_deflection_mm":  "mm",
    "max_displacement_mm":"mm",
    "max_von_mises_MPa":  "MPa",
    "max_principal_stress_MPa": "MPa",
    "max_strain":         "–",
    "strain_energy_J":    "J",
    "mass_kg":            "kg",
}
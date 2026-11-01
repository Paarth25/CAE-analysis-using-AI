"""
Validation utilities: compare CAE / ML predictions against analytical theory.
"""

from __future__ import annotations
import numpy as np


def percentage_error(predicted: float | np.ndarray, reference: float | np.ndarray) -> float | np.ndarray:
    """(predicted - reference) / |reference| * 100  [%]"""
    return (predicted - reference) / np.abs(reference) * 100.0


def relative_error(predicted: float | np.ndarray, reference: float | np.ndarray) -> float | np.ndarray:
    """|(predicted - reference)| / |reference|  [-]"""
    return np.abs(predicted - reference) / np.abs(reference)


def check_tip_deflection(
    simulated_mm: float,
    analytical_mm: float,
    tolerance_pct: float = 5.0,
) -> dict:
    """Assert that simulated tip deflection is within *tolerance_pct* of analytical value."""
    err_pct = percentage_error(simulated_mm, analytical_mm)
    return {
        "simulated_mm": simulated_mm,
        "analytical_mm": analytical_mm,
        "error_pct": err_pct,
        "within_tolerance": abs(err_pct) <= tolerance_pct,
        "tolerance_pct": tolerance_pct,
    }


def check_bending_stress(
    simulated_MPa: float,
    analytical_MPa: float,
    tolerance_pct: float = 10.0,
) -> dict:
    """Assert that simulated bending stress is within *tolerance_pct* of analytical value.

    A wider default tolerance than deflection is used because stress is more
    sensitive to mesh quality and stress-extraction region.
    """
    err_pct = percentage_error(simulated_MPa, analytical_MPa)
    return {
        "simulated_MPa": simulated_MPa,
        "analytical_MPa": analytical_MPa,
        "error_pct": err_pct,
        "within_tolerance": abs(err_pct) <= tolerance_pct,
        "tolerance_pct": tolerance_pct,
    }


def slenderness_ratio(length_mm: float, thickness_mm: float) -> float:
    """L / t – Euler–Bernoulli theory is accurate when this is large (> 10)."""
    return length_mm / thickness_mm


def shear_correction_factor(length_mm: float, thickness_mm: float) -> float:
    """Estimate the shear-deformation correction (Timoshenko) contribution.

    δ_shear / δ_bending ≈ (E / G) * (t/L)² * 3/5
    For aluminium (E/G ≈ 2.6) this is negligible for L/t > 10.
    """
    ratio = thickness_mm / length_mm
    # Generic E/G ≈ 2.6 for ν ≈ 0.33
    return 2.6 * ratio**2 * 0.6
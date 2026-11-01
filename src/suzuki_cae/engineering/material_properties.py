"""
Material property definitions and lookup utilities.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str
    youngs_modulus_GPa: float
    poisson_ratio: float
    density_kg_m3: float
    yield_strength_MPa: float
    ultimate_strength_MPa: float

    @property
    def shear_modulus_GPa(self) -> float:
        """G = E / (2(1 + ν))  [GPa]"""
        return self.youngs_modulus_GPa / (2.0 * (1.0 + self.poisson_ratio))

    @property
    def bulk_modulus_GPa(self) -> float:
        """K = E / (3(1 - 2ν))  [GPa]"""
        return self.youngs_modulus_GPa / (3.0 * (1.0 - 2.0 * self.poisson_ratio))


MATERIAL_LIBRARY: dict[str, Material] = {
    "aluminium_6061": Material(
        name="Aluminium 6061-T6",
        youngs_modulus_GPa=68.9,
        poisson_ratio=0.33,
        density_kg_m3=2700.0,
        yield_strength_MPa=276.0,
        ultimate_strength_MPa=310.0,
    ),
    "steel_mild": Material(
        name="Mild Steel S275",
        youngs_modulus_GPa=200.0,
        poisson_ratio=0.30,
        density_kg_m3=7850.0,
        yield_strength_MPa=275.0,
        ultimate_strength_MPa=430.0,
    ),
    "steel_high": Material(
        name="High-strength Steel S355",
        youngs_modulus_GPa=210.0,
        poisson_ratio=0.30,
        density_kg_m3=7850.0,
        yield_strength_MPa=355.0,
        ultimate_strength_MPa=490.0,
    ),
}


def get_material(key: str) -> Material:
    if key not in MATERIAL_LIBRARY:
        raise KeyError(f"Unknown material '{key}'. Available: {list(MATERIAL_LIBRARY.keys())}")
    return MATERIAL_LIBRARY[key]
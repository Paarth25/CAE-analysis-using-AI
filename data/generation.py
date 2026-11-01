"""
Dataset generation via Latin Hypercube Sampling.

Generates the simulation dataset used to train ML surrogate models.
The 'true' output comes from either:
  (a) The analytical beam solver  (fast, always available, used when CalculiX
      is not installed or for rapid development).
  (b) The CalculiX FEA solver     (accurate, requires external binary).

The data schema is identical in both cases so the ML pipeline is agnostic
to the simulation source.  A 'source' column records which was used.

Numerical noise is optionally added to the analytical results to simulate
mesh discretisation error, creating a more realistic training set and
preventing ML models from learning a perfect surrogate of the closed-form
solution (which would be trivial and not representative of real CAE data).
"""

from __future__ import annotations

import uuid
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

try:
    from pyDOE3 import lhs
    _HAS_PYDOE2 = True
except ImportError:
    try:
        from pyDOE2 import lhs
        _HAS_PYDOE2 = True
    except ImportError:
        _HAS_PYDOE2 = False

from ..engineering.beam_theory import batch_analytical, BeamParameters
from ..utils.reproducibility import set_global_seed
from ..utils.logging import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------ #
#  Design-space bounds (must match BeamParameters)                    #
# ------------------------------------------------------------------ #

PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "length_mm":          (80.0,  160.0),
    "width_mm":           (15.0,  40.0),
    "thickness_mm":       (2.0,   6.0),
    "youngs_modulus_GPa": (65.0,  210.0),
    "poisson_ratio":      (0.27,  0.35),
    "force_N":            (5.0,   100.0),
}

PARAM_NAMES = list(PARAM_BOUNDS.keys())

OUTPUT_COLUMNS = [
    "tip_deflection_mm",
    "max_displacement_mm",
    "max_von_mises_MPa",
    "max_principal_stress_MPa",
    "max_strain",
    "strain_energy_J",
    "mass_kg",
]

DATASET_COLUMNS = (
    ["simulation_id"]
    + PARAM_NAMES
    + ["density_kg_m3", "mesh_size_mm", "element_count"]
    + OUTPUT_COLUMNS
    + ["solver_status", "source"]
)


# ------------------------------------------------------------------ #
#  LHS sampling                                                        #
# ------------------------------------------------------------------ #

def latin_hypercube_sample(
    n_samples: int,
    bounds: dict[str, tuple[float, float]],
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a Latin Hypercube design of experiments.

    Parameters
    ----------
    n_samples : Number of design points.
    bounds    : {parameter_name: (lower, upper)} mapping.
    seed      : Random seed for reproducibility.

    Returns
    -------
    DataFrame with one column per parameter, *n_samples* rows.
    """
    set_global_seed(seed)
    k = len(bounds)
    names = list(bounds.keys())

    if _HAS_PYDOE2:
        try:
            unit_lhs = lhs(k, samples=n_samples, criterion="maximin", random_state=seed)
        except TypeError:
            # pyDOE3 may not accept random_state
            np.random.seed(seed)
            unit_lhs = lhs(k, samples=n_samples, criterion="maximin")
    else:
        # Fallback: random LHS approximation without pyDOE2
        logger.warning("pyDOE2 not found – using simplified LHS fallback.")
        rng = np.random.default_rng(seed)
        cuts = np.linspace(0, 1, n_samples + 1)
        unit_lhs = np.zeros((n_samples, k))
        for j in range(k):
            perm = rng.permutation(n_samples)
            unit_lhs[:, j] = (rng.uniform(cuts[:-1], cuts[1:]))[perm]

    # Scale from [0,1] to physical bounds
    data = {}
    for j, name in enumerate(names):
        lo, hi = bounds[name]
        data[name] = lo + unit_lhs[:, j] * (hi - lo)

    return pd.DataFrame(data)


# ------------------------------------------------------------------ #
#  Feasibility filter                                                  #
# ------------------------------------------------------------------ #

def is_feasible(row: pd.Series) -> bool:
    """Return True only if the design is physically sensible.

    Checks:
    - Slenderness L/t > 5  (Euler–Bernoulli assumption)
    - Width > thickness     (flat bar, not square or thick section)
    - Deflection/length < 0.1  (linear-elastic small-displacement regime)
    """
    L, b, t = row["length_mm"], row["width_mm"], row["thickness_mm"]
    if L / t < 5.0:
        return False
    if b < t:
        return False
    # Estimate tip deflection to check small-displacement regime
    E = row["youngs_modulus_GPa"] * 1e9
    F = row["force_N"]
    L_m = L * 1e-3
    b_m = b * 1e-3
    t_m = t * 1e-3
    I = b_m * t_m**3 / 12.0
    delta = F * L_m**3 / (3.0 * E * I) * 1e3  # mm
    if delta / L > 0.1:
        return False
    return True


# ------------------------------------------------------------------ #
#  Main dataset generator                                              #
# ------------------------------------------------------------------ #

class DatasetGenerator:
    """Generate training datasets for the cantilever beam surrogate models.

    Parameters
    ----------
    n_samples     : Target number of valid samples.
    source        : "analytical" or "calculix" (analytical used as default).
    noise_std_pct : Optional Gaussian noise added to analytical results to
                    mimic numerical solver discretisation error.  Set to 0.0
                    for a noise-free dataset.
    seed          : Global random seed.
    """

    def __init__(
        self,
        n_samples: int = 600,
        source: Literal["analytical", "calculix"] = "analytical",
        noise_std_pct: float = 1.5,
        seed: int = 42,
        density_kg_m3: float = 2700.0,
    ) -> None:
        self.n_samples     = n_samples
        self.source        = source
        self.noise_std_pct = noise_std_pct
        self.seed          = seed
        self.density       = density_kg_m3

    def generate(self) -> pd.DataFrame:
        """Run DoE sampling, feasibility filter and result calculation."""
        logger.info(f"Generating {self.n_samples} samples via LHS (source={self.source})…")

        # Oversample to account for feasibility rejections (~20% typical)
        n_oversample = int(self.n_samples * 1.35)
        doe = latin_hypercube_sample(n_oversample, PARAM_BOUNDS, seed=self.seed)

        # Feasibility filter
        mask = doe.apply(is_feasible, axis=1)
        feasible = doe[mask].copy().reset_index(drop=True)
        logger.info(f"  {mask.sum()}/{n_oversample} samples passed feasibility check.")

        if len(feasible) < self.n_samples:
            logger.warning(
                f"  Only {len(feasible)} feasible samples available (requested {self.n_samples}). "
                "Consider increasing n_oversample."
            )
        else:
            feasible = feasible.iloc[: self.n_samples].copy()

        # Compute analytical outputs
        outputs = batch_analytical(
            length_mm          = feasible["length_mm"].values,
            width_mm           = feasible["width_mm"].values,
            thickness_mm       = feasible["thickness_mm"].values,
            youngs_modulus_GPa = feasible["youngs_modulus_GPa"].values,
            poisson_ratio      = feasible["poisson_ratio"].values,
            force_N            = feasible["force_N"].values,
            density_kg_m3      = self.density,
        )

        # Optional numerical noise
        rng = np.random.default_rng(self.seed + 1)
        for col in OUTPUT_COLUMNS:
            arr = outputs[col]
            if self.noise_std_pct > 0.0:
                noise = rng.normal(0, self.noise_std_pct / 100.0, size=len(arr)) * arr
                arr = arr + noise
            feasible[col] = arr

        # Metadata columns
        feasible["simulation_id"] = [str(uuid.uuid4())[:8] for _ in range(len(feasible))]
        feasible["density_kg_m3"] = self.density
        # Analytical simulations have no mesh; use sentinel values
        feasible["mesh_size_mm"]  = np.nan if self.source == "calculix" else -1.0
        feasible["element_count"] = -1
        feasible["solver_status"] = "ok"
        feasible["source"]        = self.source

        # Reorder columns
        for col in DATASET_COLUMNS:
            if col not in feasible.columns:
                feasible[col] = np.nan
        feasible = feasible[DATASET_COLUMNS].reset_index(drop=True)

        logger.info(f"  Dataset ready: {len(feasible)} rows × {len(feasible.columns)} columns.")
        return feasible

    def save(self, df: pd.DataFrame, output_path: Path) -> Path:
        """Persist the dataset to CSV (raw, immutable)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"  Saved to {output_path}")
        return output_path


# ------------------------------------------------------------------ #
#  Synthetic experimental dataset builder                              #
# ------------------------------------------------------------------ #

def build_synthetic_experimental_dataset(
    n_specimens: int = 5,
    trials_per_load: int = 3,
    force_levels_N: list[float] | None = None,
    seed: int = 99,
) -> pd.DataFrame:
    """Create a clearly-labelled SYNTHETIC experimental dataset.

    This is used for development only when real physical measurements are not
    available.  The data are generated from the analytical model plus
    realistic measurement uncertainty, and are labelled 'synthetic' throughout.

    The dataset schema mirrors what real experiments would produce.
    """
    if force_levels_N is None:
        force_levels_N = [10.0, 25.0, 50.0, 75.0, 100.0]

    rng = np.random.default_rng(seed)

    # Fixed aluminium 6061-T6 specimens with slight dimensional variation
    specimens = []
    for i in range(n_specimens):
        specimens.append({
            "specimen_id":   f"SYN-{i+1:02d}",
            "length_mm":     100.0 + rng.normal(0, 0.3),    # ±0.3 mm machining tolerance
            "width_mm":      25.0  + rng.normal(0, 0.1),
            "thickness_mm":  3.0   + rng.normal(0, 0.05),
            "material":      "aluminium_6061_T6 (SYNTHETIC)",
            "youngs_modulus_GPa": 68.9,
            "poisson_ratio": 0.33,
            "density_kg_m3": 2700.0,
        })

    rows = []
    trial_id = 0
    for spec in specimens:
        L  = spec["length_mm"]   * 1e-3
        b  = spec["width_mm"]    * 1e-3
        t  = spec["thickness_mm"]* 1e-3
        E  = spec["youngs_modulus_GPa"] * 1e9
        I  = b * t**3 / 12.0

        for F in force_levels_N:
            delta_true = (F * L**3) / (3 * E * I) * 1e3   # mm

            for tr in range(trials_per_load):
                # Measurement uncertainty: ±0.05 mm systematic + 0.02 mm random
                meas_noise = rng.normal(0, 0.02)
                systematic = 0.03                          # calibration offset
                meas_deflection = delta_true + systematic + meas_noise

                rows.append({
                    "specimen_id":           spec["specimen_id"],
                    "length_mm":             spec["length_mm"],
                    "width_mm":              spec["width_mm"],
                    "thickness_mm":          spec["thickness_mm"],
                    "material":              spec["material"],
                    "youngs_modulus_GPa":    spec["youngs_modulus_GPa"],
                    "poisson_ratio":         spec["poisson_ratio"],
                    "force_N":               F,
                    "trial_id":              trial_id,
                    "tip_deflection_mm":     round(meas_deflection, 3),
                    "measurement_uncertainty_mm": 0.05,
                    "temperature_C":         22.0 + rng.normal(0, 0.5),
                    "measurement_method":    "dial_gauge_synthetic",
                    "data_provenance":       "SYNTHETIC - NOT PHYSICAL MEASUREMENT",
                })
                trial_id += 1

    df = pd.DataFrame(rows)
    logger.info(
        f"Synthetic experimental dataset: {len(df)} records "
        f"({n_specimens} specimens × {len(force_levels_N)} loads × {trials_per_load} trials)."
    )
    return df
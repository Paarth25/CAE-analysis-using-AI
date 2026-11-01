"""Reproducibility helpers: seed management, environment info."""

from __future__ import annotations
import os
import random
import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_environment_info() -> dict:
    import platform, sys
    return {
        "python_version": sys.version,
        "platform":       platform.platform(),
        "numpy_version":  np.__version__,
    }
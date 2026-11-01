"""Central path management."""

from __future__ import annotations
from pathlib import Path


def project_root() -> Path:
    """Return the absolute path to the repository root."""
    return Path(__file__).resolve().parents[4]


def data_raw(subdir: str = "") -> Path:
    p = project_root() / "data" / "raw" / subdir
    p.mkdir(parents=True, exist_ok=True)
    return p


def data_processed() -> Path:
    p = project_root() / "data" / "processed"
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_saved() -> Path:
    p = project_root() / "models" / "saved"
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_figures() -> Path:
    p = project_root() / "results" / "figures"
    p.mkdir(parents=True, exist_ok=True)
    return p
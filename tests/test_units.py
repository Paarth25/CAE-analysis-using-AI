import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from suzuki_cae.utils.units import mm_to_m, m_to_mm, pa_to_mpa, mpa_to_pa, gpa_to_pa


def test_mm_m_roundtrip():
    assert m_to_mm(mm_to_m(100.0)) == pytest.approx(100.0)

def test_pa_mpa():
    assert pa_to_mpa(1e6) == pytest.approx(1.0)

def test_mpa_pa():
    assert mpa_to_pa(1.0) == pytest.approx(1e6)

def test_gpa_pa():
    assert gpa_to_pa(1.0) == pytest.approx(1e9)
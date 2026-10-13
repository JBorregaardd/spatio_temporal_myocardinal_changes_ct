"""Tests for the body surface area from the DICOM metadata table."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "measurements"))

from BSA import body_surface_area, calculate_bsa  # noqa: E402


def test_du_bois_reference_value() -> None:
    assert calculate_bsa(70.0, 175.0) == pytest.approx(1.85, abs=0.01)


def test_one_row_per_patient_with_first_valid_height_and_weight() -> None:
    metadata = pd.DataFrame(
        {
            "pseudonymized_id": [
                "CGPS-10y-august-2026_0000_SERIES0001",
                "CGPS-10y-august-2026_0000_SERIES0013",
                "CGPS-10y-august-2026_0001_SERIES0002",
                "CGPS-10y-august-2026_0002_SERIES0005",
            ],
            "PatientSize": [0.0, 1.75, 1.60, np.nan],
            "PatientWeight": [70.0, 0.0, 55.0, 80.0],
        }
    )
    patients = body_surface_area(metadata).set_index("patient_id")

    assert list(patients.index) == ["0000", "0001", "0002"]
    assert patients.loc["0000", "height_cm"] == pytest.approx(175.0)
    assert patients.loc["0000", "PatientWeight"] == pytest.approx(70.0)
    assert patients.loc["0000", "BSA"] == pytest.approx(calculate_bsa(70.0, 175.0))
    assert patients.loc["0001", "BSA"] == pytest.approx(calculate_bsa(55.0, 160.0))
    assert np.isnan(patients.loc["0002", "BSA"])

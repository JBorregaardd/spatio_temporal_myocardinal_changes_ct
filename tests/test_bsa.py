"""Tests for the body surface area from the DICOM metadata table."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "measurements"))

from BSA import body_surface_area, calculate_bsa, gender_and_age  # noqa: E402


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


def test_gender_and_age_strips_prefix_and_codes_sex() -> None:
    transdata = pd.DataFrame(
        {
            "CGPS-10y-august-2026": ["CGPS-10y-august-2026_0000", "CGPS-10y-august-2026_0001", np.nan],
            "CGPS-10y-august-2026-PatientSex": ["M", "F", "M"],
            "CGPS-10y-august-2026-PatientAgeCompletedYears": ["65", "72", "50"],
        }
    )
    patients = gender_and_age(transdata, "CGPS10").set_index("patient_id")

    assert list(patients.index) == ["0000", "0001"]
    assert patients.loc["0000", "sex"] == 0
    assert patients.loc["0001", "sex"] == 1
    assert patients.loc["0001", "age"] == 72


def test_gender_and_age_keeps_binary_sex_codes() -> None:
    transdata = pd.DataFrame(
        {
            "CGPS-10y-august-2026": [
                "CGPS-10y-august-2026_0000",
                "CGPS-10y-august-2026_0001",
                "CGPS-10y-august-2026_0002",
                "CGPS-10y-august-2026_0003",
            ],
            "CGPS-10y-august-2026-PatientSex": ["0", "1", "0.0", "1.0"],
            "CGPS-10y-august-2026-PatientAgeCompletedYears": ["65", "72", "50", "80"],
        }
    )
    patients = gender_and_age(transdata, "CGPS10").set_index("patient_id")

    assert list(patients["sex"]) == [0, 1, 0, 1]


def test_gender_and_age_uses_baseline_columns_for_cgps1() -> None:
    transdata = pd.DataFrame(
        {
            "CGPS-1-august-2026": ["CGPS-1-august-2026_0000"],
            "CGPS-1-august-2026-PatientSex": ["F"],
            "CGPS-1-august-2026-PatientAgeCompletedYears": ["55"],
            "CGPS-10y-august-2026": ["CGPS-10y-august-2026_0000"],
            "CGPS-10y-august-2026-PatientSex": ["F"],
            "CGPS-10y-august-2026-PatientAgeCompletedYears": ["65"],
        }
    )
    assert gender_and_age(transdata, "CGPS1").set_index("patient_id").loc["0000", "age"] == 55
    assert gender_and_age(transdata, "CGPS10").set_index("patient_id").loc["0000", "age"] == 65


def test_gender_and_age_rejects_unknown_dataset() -> None:
    with pytest.raises(ValueError):
        gender_and_age(pd.DataFrame(), "ImageCAS_1-200")

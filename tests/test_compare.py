"""Tests of the table linking the CGPS1 and CGPS10 patient ids."""
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "comparative_analysis"))
sys.path.insert(0, str(REPO_ROOT / "LVM_thickness"))

from compare import extract_ids, link_table  # noqa: E402


def test_extract_ids_keeps_leading_zeros() -> None:
    values = pd.Series(["CGPS-10y-august-2026_0730_SERIES0004_volume_1", "/data/CGPS-10y-august-2026_0001.nii.gz", None])
    ids = extract_ids(values, "CGPS-10y-august-2026")
    assert ids.tolist()[:2] == ["0730", "0001"]
    assert pd.isna(ids[2])


def test_extract_ids_does_not_mix_datasets() -> None:
    ids = extract_ids(pd.Series(["CGPS-10y-august-2026_0730_SERIES0004"]), "CGPS-1-august-2026")
    assert ids.isna().all()


def test_link_table_extracts_ids_per_subject() -> None:
    links = pd.DataFrame(
        {
            "Follow-up NIFTI file": [
                "CGPS-10y-august-2026_0730_SERIES0004_volume_1",
                "CGPS-10y-august-2026_0002_SERIES0001_volume_1",
                None,
                "CGPS-10y-august-2026_0730_SERIES0004_volume_1",
            ],
            "Baseline NIFTI file": [
                "CGPS-1-august-2026_00001_SERIES0005",
                "CGPS-1-august-2026_00002_SERIES0003",
                "CGPS-1-august-2026_00003_SERIES0003",
                "CGPS-1-august-2026_00001_SERIES0005",
            ],
        }
    )
    table = link_table(links)
    assert list(table.columns) == ["CGPS1", "CGPS10"]
    assert table.values.tolist() == [["00001", "0730"], ["00002", "0002"]]


def test_link_table_without_column_raises() -> None:
    links = pd.DataFrame({"Baseline NIFTI file": ["CGPS-1-august-2026_00001_SERIES0005"]})
    with pytest.raises(ValueError, match="Follow-up NIFTI file"):
        link_table(links)

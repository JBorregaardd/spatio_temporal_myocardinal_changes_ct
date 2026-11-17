"""Tests for the end-systolic lookup of main_statistics_df.py and the whole-LV table of main_all_df.py."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "measurements"))
sys.path.insert(0, str(REPO_ROOT / "LVM_thickness"))

from main_all_df import ROW_LABELS, all_table  # noqa: E402
from main_statistics_df import es_segmentation_path  # noqa: E402
from pipeline import DataSource  # noqa: E402

PATTERN = "CGPS_{id}_*/heart.nii.gz"


@pytest.fixture
def data(tmp_path: Path) -> DataSource:
    return DataSource(str(tmp_path), str(tmp_path / "ED"), "{id}.nii.gz", PATTERN)


def test_es_segmentation_path_found_missing_and_unset(
    tmp_path: Path, data: DataSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    es_file = tmp_path / "ES" / "CGPS_0001_SERIES2" / "heart.nii.gz"
    es_file.parent.mkdir(parents=True)
    es_file.write_bytes(b"")
    monkeypatch.delenv("ES_SEGMENTATION_FOLDER", raising=False)
    assert es_segmentation_path(data, "0001") is None
    monkeypatch.setenv("ES_SEGMENTATION_FOLDER", str(tmp_path / "ES"))
    assert es_segmentation_path(data, "0001") == str(es_file)
    assert es_segmentation_path(data, "0002") is None


def test_es_segmentation_path_with_two_matches_is_none(
    tmp_path: Path, data: DataSource, monkeypatch: pytest.MonkeyPatch
) -> None:
    for series in ("SERIES2", "SERIES3"):
        path = tmp_path / "ES" / f"CGPS_0001_{series}" / "heart.nii.gz"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"")
    monkeypatch.setenv("ES_SEGMENTATION_FOLDER", str(tmp_path / "ES"))
    assert es_segmentation_path(data, "0001") is None


def test_all_table_gives_partial_rows_their_own_n() -> None:
    data = pd.DataFrame({"patient_id": ["1", "2", "3"], "sex": [0, 1, 1]})
    for key in ROW_LABELS:
        data[key] = [1.0, 2.0, 3.0]
    data["ejection_fraction"] = [50.0, 60.0, np.nan]
    data["lv_esv_ml"] = np.nan
    column = all_table({"All": data})["All"]
    assert column["LV ejection fraction (%)"] == "55.00 ± 7.07 (n=2)"
    assert column["LV end-systolic volume (mL)"] == ""
    assert column["Mass (g)"] == "2.00 ± 1.00"

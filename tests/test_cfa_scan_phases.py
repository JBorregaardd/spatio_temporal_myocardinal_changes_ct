"""Tests for the R-R percentage and heart rate of scans from their series descriptions."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cfa"))

from scan_phases import group_files, parse_groups, rr_interval_s, scan_phases  # noqa: E402


def test_rr_interval_s() -> None:
    assert rr_interval_s("HALF 648ms 0.90s PIQE 0.5 CE") == 0.90
    assert rr_interval_s("FULL 73% 1.00s Cardiac 0.5 CE") == 1.00
    assert np.isnan(rr_interval_s("HALF 300ms 0.00s CE"))
    assert np.isnan(rr_interval_s(None))


def test_scan_phases_matches_on_file_name(tmp_path: Path) -> None:
    filelist = tmp_path / "es.txt"
    filelist.write_text("/data/NIFTI/a_SERIES0004.nii.gz\n/data/NIFTI/b_SERIES0002.nii.gz\n\n")
    folder = tmp_path / "best"
    folder.mkdir()
    (folder / "c_81%.nii.gz").write_text("")
    metadata = pd.DataFrame(
        {
            " filename ": ["a_SERIES0004.nii.gz", "c_81%.nii.gz", "a_SERIES0004.nii.gz"],
            "patient_id": ["a", "c", "a"],
            "SeriesDescription": [" HALF 300ms 0.80s CE ", "FULL 81% 1.20s CE", "duplicate"],
        }
    )
    table = scan_phases(group_files({"ES": str(filelist)}, {"best": str(folder)}), metadata).set_index("filename")
    assert table.loc["a_SERIES0004.nii.gz", "rr_percent"] == pytest.approx(37.5)
    assert table.loc["a_SERIES0004.nii.gz", "heart_rate_bpm"] == pytest.approx(75)
    assert np.isnan(table.loc["b_SERIES0002.nii.gz", "rr_percent"])
    assert table.loc["c_81%.nii.gz", "group"] == "best"
    assert table.loc["c_81%.nii.gz", "heart_rate_bpm"] == pytest.approx(50)


def test_parse_groups_rejects_missing_path() -> None:
    assert parse_groups(["ES=/x.txt"]) == {"ES": "/x.txt"}
    with pytest.raises(SystemExit):
        parse_groups(["ES"])

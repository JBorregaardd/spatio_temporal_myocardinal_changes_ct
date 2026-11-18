"""Tests of the table linking the CGPS1 and CGPS10 patient ids."""
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "comparative_analysis"))
sys.path.insert(0, str(REPO_ROOT / "LVM_thickness"))

from compare import add_phases, extract_ids, link_table, rr_percent  # noqa: E402


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


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("HALF 648ms 0.90s PIQE 0.5 CE", 72.0),
        ("FULL 73% 1.00s Cardiac 0.5 CE", 73.0),
        ("HALF 400ms 0.00s PIQE 0.5 CE", None),
        ("Cardiac 0.5 CE", None),
        (None, None),
    ],
)
def test_rr_percent_handles_both_conventions(description: str | None, expected: float | None) -> None:
    value = rr_percent(description)
    if expected is None:
        assert pd.isna(value)
    else:
        assert value == pytest.approx(expected)


def test_add_phases_uses_series_of_linked_scan() -> None:
    links = pd.DataFrame(
        {
            "Baseline NIFTI file": ["CGPS-1-august-2026_00001_SERIES0005", "CGPS-1-august-2026_00002_SERIES0003"],
            "Follow-up NIFTI file": [
                "CGPS-10y-august-2026_0730_SERIES0004_volume_1",
                "CGPS-10y-august-2026_0002_SERIES0001_volume_1",
            ],
        }
    )
    filelists = {
        "CGPS1": pd.DataFrame(
            {
                "PseudoID": ["CGPS-1-august-2026_00001", "CGPS-1-august-2026_00001", "CGPS-1-august-2026_00002"],
                "filename": ["x_SERIES0004.nii.gz", "x_SERIES0005.nii.gz", "x_SERIES0003.nii.gz"],
                " SeriesDescription ": ["FULL 40% 1.00s CE", "FULL 75% 1.00s CE", "HALF 700ms 1.00s CE"],
            }
        ),
        "CGPS10": pd.DataFrame(
            {
                "PseudoID": ["CGPS-10y-august-2026_0730"],
                "SeriesDescription": ["HALF 648ms 0.90s PIQE 0.5 CE"],
            }
        ),
    }
    table = add_phases(link_table(links), links, filelists)
    assert table["CGPS1_rr_percent"].tolist() == pytest.approx([75.0, 70.0])
    assert table["CGPS10_rr_percent"].iloc[0] == pytest.approx(72.0)
    assert pd.isna(table["CGPS10_rr_percent"].iloc[1])
    assert table["rr_percent_difference"].iloc[0] == pytest.approx(-3.0)
    assert table["CGPS1_series_description"].iloc[0] == "FULL 75% 1.00s CE"

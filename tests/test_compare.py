"""Tests of the table linking the CGPS1 and CGPS10 patient ids."""
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "comparative_analysis"))
sys.path.insert(0, str(REPO_ROOT / "LVM_thickness"))

from compare import extract_ids, link_table, rr_percent  # noqa: E402


def links_file(**overrides: list) -> pd.DataFrame:
    """A small links file with two linked subjects, one repeated row and one row without a link."""
    columns = {
        "Baseline NIFTI file": [
            "CGPS-1-august-2026_00001_SERIES0005",
            "CGPS-1-august-2026_05070_SERIES0003",
            "CGPS-1-august-2026_00003_SERIES0003",
            "CGPS-1-august-2026_00001_SERIES0005",
        ],
        "Follow-up NIFTI file": [
            "CGPS-10y-august-2026_0730_SERIES0004_volume_1",
            "CGPS-10y-august-2026_0002_SERIES0001_volume_1",
            None,
            "CGPS-10y-august-2026_0730_SERIES0004_volume_1",
        ],
        "Baseline SeriesDescription": ["FULL 75% 1.00s CE", None, "FULL 70% 1.00s CE", "FULL 75% 1.00s CE"],
        "Follow-up SeriesDescription": [
            "HALF 648ms 0.90s PIQE 0.5 CE",
            "HALF 700ms 1.00s CE",
            None,
            "HALF 648ms 0.90s PIQE 0.5 CE",
        ],
        " Link_found (yes/no) ": ["yes", "Yes", "no", "yes"],
        "Baseline scan type": ["ED", " ed ", None, "ED"],
    }
    columns.update(overrides)
    return pd.DataFrame(columns)


def test_extract_ids_keeps_leading_zeros() -> None:
    values = pd.Series(
        ["CGPS-10y-august-2026_0730_SERIES0004_volume_1", "/data/CGPS-10y-august-2026_0001.nii.gz", None]
    )
    ids = extract_ids(values, "CGPS-10y-august-2026")
    assert ids.tolist()[:2] == ["0730", "0001"]
    assert pd.isna(ids[2])


def test_extract_ids_does_not_mix_datasets() -> None:
    ids = extract_ids(pd.Series(["CGPS-10y-august-2026_0730_SERIES0004"]), "CGPS-1-august-2026")
    assert ids.isna().all()


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


def test_link_table_uses_linked_rows_and_their_descriptions() -> None:
    table = link_table(links_file())
    assert list(table.columns) == ["CGPS1", "CGPS10", "CGPS1_rr_percent", "CGPS10_rr_percent", "rr_percent_difference"]
    assert table[["CGPS1", "CGPS10"]].values.tolist() == [["00001", "0730"], ["05070", "0002"]]
    assert table["CGPS1_rr_percent"].iloc[0] == pytest.approx(75.0)
    assert table["CGPS10_rr_percent"].tolist() == pytest.approx([72.0, 70.0])
    assert table["rr_percent_difference"].iloc[0] == pytest.approx(-3.0)


def test_link_table_blank_description_gives_no_percentage() -> None:
    table = link_table(links_file())
    assert pd.isna(table.loc[table["CGPS1"] == "05070", "CGPS1_rr_percent"]).all()
    assert pd.isna(table.loc[table["CGPS1"] == "05070", "rr_percent_difference"]).all()


def test_link_table_skips_rows_without_link() -> None:
    table = link_table(links_file())
    assert "00003" not in table["CGPS1"].tolist()


def test_link_table_keeps_only_ed_baseline_scans() -> None:
    table = link_table(links_file(**{"Baseline scan type": ["ES", "ED", None, "ES"]}))
    assert table["CGPS1"].tolist() == ["05070"]


def test_link_table_without_scan_type_raises() -> None:
    links = links_file().drop(columns=["Baseline scan type"])
    with pytest.raises(ValueError, match="Baseline scan type"):
        link_table(links)


def test_link_table_accepts_describtion_spelling() -> None:
    links = links_file().rename(
        columns={
            "Baseline SeriesDescription": "Baseline SeriesDescribtion",
            "Follow-up SeriesDescription": "Follow-up SeriesDescribtion",
        }
    )
    assert link_table(links)["CGPS1_rr_percent"].iloc[0] == pytest.approx(75.0)


def test_link_table_without_column_raises() -> None:
    links = links_file().drop(columns=["Follow-up NIFTI file"])
    with pytest.raises(ValueError, match="Follow-up NIFTI file"):
        link_table(links)


def test_link_table_without_link_found_raises() -> None:
    links = links_file().drop(columns=[" Link_found (yes/no) "])
    with pytest.raises(ValueError, match="Link_found"):
        link_table(links)

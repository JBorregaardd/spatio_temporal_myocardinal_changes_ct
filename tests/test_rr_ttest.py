"""Tests of the paired t-test of the R-R percentages of the CGPS1 and CGPS10 scans."""
import sys
from pathlib import Path

import pandas as pd
import pytest
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "repeated_measures"))
sys.path.insert(0, str(REPO_ROOT / "LVM_thickness"))

from rr_ttest import paired_rr_ttest  # noqa: E402


def test_paired_rr_ttest_uses_complete_pairs() -> None:
    links = pd.DataFrame(
        {
            "CGPS1": ["1", "2", "3", "4", "5"],
            "CGPS10": ["a", "b", "c", "d", "e"],
            "CGPS1_rr_percent": [70.0, 75.0, 72.0, None, 68.0],
            "CGPS10_rr_percent": [72.0, 74.0, 76.0, 70.0, 71.0],
        }
    )
    row = paired_rr_ttest(links).iloc[0]
    expected = stats.ttest_rel([72.0, 74.0, 76.0, 71.0], [70.0, 75.0, 72.0, 68.0])
    assert row["n"] == 4
    assert row["mean_difference"] == pytest.approx(2.0)
    assert row["t"] == pytest.approx(expected.statistic)
    assert row["p"] == pytest.approx(expected.pvalue)
    assert row["ci_low"] < 2.0 < row["ci_high"]


def test_paired_rr_ttest_without_columns_raises() -> None:
    with pytest.raises(ValueError, match="CGPS10_rr_percent"):
        paired_rr_ttest(pd.DataFrame({"CGPS1_rr_percent": [70.0]}))

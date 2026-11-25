"""Tests for the group mean and median bullseyes."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repeated_measures" / "bullseye"))

from bullseye_group_mean import subject_classes, summarise_stack  # noqa: E402


def test_summarise_stack_coverage() -> None:
    """Mean and median over subjects with data; cells with data in too few subjects are NaN."""
    stack = np.array([[1.0, 1.0, np.nan], [2.0, np.nan, np.nan], [6.0, np.nan, 5.0], [3.0, 3.0, np.nan]])
    summary = summarise_stack(stack, coverage=0.5)
    np.testing.assert_allclose(summary["mean"][:2], [3.0, 2.0])
    np.testing.assert_allclose(summary["median"][:2], [2.5, 2.0])
    assert np.isnan(summary["mean"][2]) and np.isnan(summary["median"][2])
    np.testing.assert_array_equal(summary["n"], [4, 2, 1])


def test_subject_classes() -> None:
    """Sex comes from CGPS1, else CGPS10; HT from the CGPS1 scan."""
    pairs = pd.DataFrame({"CGPS1": ["a", "b"], "CGPS10": ["x", "y"]})
    summaries = {
        "CGPS1": pd.DataFrame({"patient_id": ["a"], "sex": [1]}),
        "CGPS10": pd.DataFrame({"patient_id": ["x", "y"], "sex": [0, 0]}),
    }
    ht = pd.DataFrame({"patient_id": ["a", "b"], "HT": [True, False]})
    classes = subject_classes(pairs, summaries, ht)
    assert classes["sex"].tolist() == [1, 0]
    assert classes["HT"].tolist() == [True, False]

"""Tests for the CGPS1-CGPS10 vertex match distance per AHA segment."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repeated_measures" / "bullseye"))

from match_distance import segment_match_distance, segment_summary  # noqa: E402


def test_segment_match_distance() -> None:
    """Shares above the threshold and without a match are counted per segment."""
    distance = np.array([1.0, 4.0, 6.0, 2.0, 2.0])
    matched = np.array([True, True, False, True, True])
    segments = np.array([1, 1, 1, 2, 2])
    table = segment_match_distance(distance, matched, segments, threshold=3.0).set_index("segment")
    assert table.loc[1, "n_vertices"] == 3
    assert np.isclose(table.loc[1, "far_pct"], 200 / 3)
    assert np.isclose(table.loc[1, "unmatched_pct"], 100 / 3)
    assert np.isclose(table.loc[1, "median_mm"], 4.0)
    assert table.loc[2, "far_pct"] == 0 and table.loc[2, "unmatched_pct"] == 0


def test_segment_summary_averages_subjects() -> None:
    """The summary averages the shares over subjects and keeps the worst subject."""
    first = segment_match_distance(np.array([4.0, 1.0]), np.array([True, True]), np.array([1, 1]), 3.0)
    second = segment_match_distance(np.array([1.0, 1.0]), np.array([True, True]), np.array([1, 1]), 3.0)
    summary = segment_summary(pd.concat([first, second])).set_index("segment")
    assert summary.loc[1, "n_subjects"] == 2
    assert np.isclose(summary.loc[1, "mean_far_pct"], 25)
    assert np.isclose(summary.loc[1, "max_far_pct"], 50)

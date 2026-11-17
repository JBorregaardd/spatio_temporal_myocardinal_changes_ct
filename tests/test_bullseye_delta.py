"""Tests for the change in wall thickness per AHA segment between CGPS1 and CGPS10."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repeated_measures"))

from bullseye_delta import SEGMENTS, group_deltas, segment_deltas  # noqa: E402


def _segments(ids: list[str], offset: float) -> pd.DataFrame:
    """Per-segment table with thickness ``segment + offset`` for every patient."""
    rows = [
        {"patient_id": pid, "segment": seg, "mean_thickness": seg + offset, "median_thickness": seg + 2 * offset}
        for pid in ids
        for seg in SEGMENTS
    ]
    return pd.DataFrame(rows)


def test_segment_and_group_deltas() -> None:
    links = pd.DataFrame({"CGPS1": ["01", "02", "03"], "CGPS10": ["11", "12", "13"]})
    segments = {"CGPS1": _segments(["01", "02", "03"], 0.0), "CGPS10": _segments(["11", "12"], 1.0)}
    summaries = {
        "CGPS1": pd.DataFrame({"patient_id": ["01", "02", "03"], "sex": [1, np.nan, 0]}),
        "CGPS10": pd.DataFrame({"patient_id": ["11", "12"], "sex": [1, 0]}),
    }
    statistics = ["mean_thickness", "median_thickness"]

    deltas = segment_deltas(segments, summaries, links, statistics)
    assert len(deltas) == 2 * len(SEGMENTS)
    assert deltas["delta_mean_thickness"].tolist() == pytest.approx([1.0] * len(deltas))
    assert deltas["delta_median_thickness"].tolist() == pytest.approx([2.0] * len(deltas))
    assert deltas.groupby("CGPS1")["sex"].first().tolist() == [1, 0]

    summary = group_deltas(deltas, statistics)
    assert len(summary) == 2 * 2 * len(SEGMENTS)
    women = summary[(summary["sex"] == "female") & (summary["statistic"] == "median_thickness")]
    assert women["n"].tolist() == [1] * len(SEGMENTS)
    assert women["mean_delta"].tolist() == pytest.approx([2.0] * len(SEGMENTS))

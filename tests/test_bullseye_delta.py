"""Tests for the relative change in median wall thickness per AHA segment between CGPS1 and CGPS10."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repeated_measures"))

from bullseye_delta import PANELS, SEGMENTS, group_changes, segment_changes  # noqa: E402


def _segments(ids: list[str], factor: float) -> pd.DataFrame:
    """Per-segment table with median thickness ``10 * factor`` for every patient and segment."""
    rows = [{"patient_id": pid, "segment": seg, "median_thickness": 10 * factor} for pid in ids for seg in SEGMENTS]
    return pd.DataFrame(rows)


def test_segment_and_group_changes() -> None:
    links = pd.DataFrame({"CGPS1": ["01", "02", "03"], "CGPS10": ["11", "12", "13"]})
    segments = {"CGPS1": _segments(["01", "02", "03"], 1.0), "CGPS10": _segments(["11", "12"], 1.1)}
    summaries = {
        "CGPS1": pd.DataFrame({"patient_id": ["01", "02", "03"], "sex": [1, np.nan, 0]}),
        "CGPS10": pd.DataFrame({"patient_id": ["11", "12"], "sex": [1, 0]}),
    }
    ht = pd.DataFrame({"patient_id": ["01", "02", "03"], "HT": [True, False, True]})

    changes = segment_changes(segments, summaries, links, ht)
    assert len(changes) == 2 * len(SEGMENTS)
    assert changes["change_pct"].tolist() == pytest.approx([10.0] * len(changes))
    assert changes.groupby("CGPS1")["sex"].first().tolist() == [1, 0]
    assert changes.groupby("CGPS1")["HT"].first().tolist() == [True, False]

    summary = group_changes(changes)
    assert len(summary) == len(PANELS) * len(SEGMENTS)
    women_ht = summary[summary["HT"] & (summary["sex"] == "female")]
    assert women_ht["n"].tolist() == [1] * len(SEGMENTS)
    assert women_ht["mean_change_pct"].tolist() == pytest.approx([10.0] * len(SEGMENTS))
    assert summary[summary["HT"] & (summary["sex"] == "male")]["n"].sum() == 0

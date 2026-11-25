"""Tests for the diagnosis of the mesh vs segment method bias."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repeated_measures" / "bullseye"))

from diagnose_segment_bias import segment_diagnosis  # noqa: E402


def test_segment_diagnosis_finds_proportional_bias() -> None:
    """A mesh method that misses a fifth of the change shows a negative bias correlated with the change."""
    rng = np.random.default_rng(0)
    change = rng.normal(0.5, 1.0, 200)
    table = pd.DataFrame(
        {
            "segment": 2,
            "segment_delta_mm": change,
            "quarter_delta_mm": change + rng.normal(0, 0.1, 200),
            "mesh_delta_mm": 0.8 * change,
            "share_log2_ratio": rng.normal(0, 0.1, 200),
            "unmatched_pct": np.where(np.arange(200) < 100, 0.0, 5.0),
        }
    )
    table["difference_mm"] = table["mesh_delta_mm"] - table["segment_delta_mm"]
    row = segment_diagnosis(table).iloc[0]
    assert row["n"] == 200
    assert np.isclose(row["mean_difference_mm"], -0.2 * change.mean())
    assert row["r_difference_vs_change"] < -0.9
    assert abs(row["r_difference_vs_share_ratio"]) < 0.3

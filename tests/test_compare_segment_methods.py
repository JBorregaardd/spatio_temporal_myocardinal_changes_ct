"""Tests for the comparison of the mesh and segment methods of the CGPS10 - CGPS1 thickness change."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repeated_measures" / "bullseye"))

from compare_segment_methods import agreement, mesh_segment_deltas  # noqa: E402


def test_mesh_segment_deltas() -> None:
    """Difference of medians and median of differences per segment, leaving unmatched vertices out."""
    segments = np.array([1, 1, 1, 2, 2])
    cgps1 = np.array([8.0, 10.0, 12.0, 5.0, 7.0])
    cgps10 = np.array([9.0, 13.0, np.nan, 6.0, 8.0])
    table = mesh_segment_deltas(segments, cgps1, cgps10).set_index("segment")
    assert table.loc[1, "n_matched"] == 2
    assert np.isclose(table.loc[1, "mesh_delta_mm"], 11.0 - 9.0)
    assert np.isclose(table.loc[1, "vertex_delta_mm"], 2.0)
    assert np.isclose(table.loc[2, "mesh_delta_mm"], 1.0)


def test_agreement() -> None:
    """Bias, limits of agreement and correlation of two methods."""
    segment = pd.Series([0.0, 1.0, 2.0, 3.0])
    mesh = segment + pd.Series([0.1, -0.1, 0.1, -0.1])
    result = agreement(mesh - segment, mesh, segment)
    assert np.isclose(result["bias_mm"], 0.0)
    assert np.isclose(result["mean_abs_mm"], 0.1)
    assert result["loa_low_mm"] < 0 < result["loa_high_mm"]
    assert result["r"] > 0.99

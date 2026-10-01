"""Tests for the per-segment wall thickness statistics."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "measurements"))

from thickness import thickness_per_segment  # noqa: E402


def test_thickness_per_segment(tmp_path: Path) -> None:
    (tmp_path / "thickness.json").write_text(json.dumps({"1.0": [8.0, 9.0, 13.0], "-1.0": [20.0], "17.0": [5.0]}))
    result = thickness_per_segment(str(tmp_path))
    assert result["segment_1_median_mm"] == pytest.approx(9.0)
    assert result["segment_1_mean_mm"] == pytest.approx(10.0)
    assert result["segment_1_std_mm"] == pytest.approx(np.std([8.0, 9.0, 13.0]))
    assert result["segment_17_median_mm"] == pytest.approx(5.0)
    assert np.isnan(result["segment_2_median_mm"])
    assert len(result) == 17 * 3

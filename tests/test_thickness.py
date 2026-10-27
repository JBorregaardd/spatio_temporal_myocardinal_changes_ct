"""Tests for the per-segment wall thickness statistics."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "measurements"))

from thickness import madwt, septal_lateral_ratio, thickness_per_segment  # noqa: E402


def test_thickness_per_segment(tmp_path: Path) -> None:
    (tmp_path / "thickness.json").write_text(json.dumps({"1.0": [8.0, 9.0, 13.0], "-1.0": [20.0], "17.0": [5.0]}))
    result = thickness_per_segment(str(tmp_path))
    assert result["segment_1_median_mm"] == pytest.approx(9.0)
    assert result["segment_1_mean_mm"] == pytest.approx(10.0)
    assert result["segment_1_std_mm"] == pytest.approx(np.std([8.0, 9.0, 13.0]))
    assert result["segment_17_median_mm"] == pytest.approx(5.0)
    assert np.isnan(result["segment_2_median_mm"])
    assert len(result) == 17 * 3


def test_madwt(tmp_path: Path) -> None:
    thickness = {f"{segment}.0": [0.5, float(segment)] for segment in range(1, 17)}
    thickness["17.0"] = [100.0]
    (tmp_path / "thickness.json").write_text(json.dumps(thickness))
    assert madwt(str(tmp_path)) == pytest.approx(4.0)


def test_madwt_missing_segment(tmp_path: Path) -> None:
    (tmp_path / "thickness.json").write_text(json.dumps({f"{segment}.0": [8.0] for segment in range(1, 16)}))
    assert np.isnan(madwt(str(tmp_path)))


def test_septal_lateral_ratio(tmp_path: Path) -> None:
    thickness = {f"{segment}.0": [5.0] for segment in range(1, 18)}
    thickness["9.0"] = [6.0, 12.0]
    thickness["11.0"] = [8.0]
    thickness["14.0"] = [30.0]
    thickness["4.0"] = [30.0]
    (tmp_path / "thickness.json").write_text(json.dumps(thickness))
    assert septal_lateral_ratio(str(tmp_path)) == pytest.approx(1.5)


def test_septal_lateral_ratio_missing_wall(tmp_path: Path) -> None:
    (tmp_path / "thickness.json").write_text(json.dumps({"2.0": [8.0], "3.0": [9.0]}))
    assert np.isnan(septal_lateral_ratio(str(tmp_path)))

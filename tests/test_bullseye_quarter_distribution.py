"""Tests for the thickness distribution change per AHA segment quarter."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repeated_measures" / "bullseye"))

from bullseye_quarter_distribution import PERCENTILES, distribution_changes, quarter_labels  # noqa: E402


def test_quarter_labels() -> None:
    """Points get the drawn wedge and the quarter by radius half and angle half; the apex gets -1."""
    basal_width = 2 * np.pi / 6
    theta = np.array([np.pi / 2 - 0.1, np.pi / 2 + 0.1, np.pi / 2 + basal_width + 0.1, np.pi / 2 + 0.1, 0.0])
    radius = np.array([0.95, 0.75, 0.95, 0.5, 0.1])
    segments, quarters = quarter_labels(theta, radius)
    np.testing.assert_array_equal(segments, [1, 1, 2, 7, -1])
    np.testing.assert_array_equal(quarters, [2, 1, 3, 1, -1])


def _scan(shift: float) -> dict[str, np.ndarray]:
    """Bullseye points covering every quarter with thickness 5-15 mm plus ``shift``."""
    rng = np.random.default_rng(0)
    theta = rng.uniform(-np.pi, np.pi, 200_000)
    radius = rng.uniform(0.2, 0.99, theta.size)
    return {"theta": theta, "radius": radius, "values": np.linspace(5, 15, theta.size) + shift}


def test_shift_and_self_comparison() -> None:
    """A uniform 1 mm thickening moves every percentile by 1 mm; a scan compared with itself gives 0."""
    table = distribution_changes({"CGPS1": _scan(0.0), "CGPS10": _scan(1.0)})
    assert len(table) == 64
    for percentile in PERCENTILES:
        np.testing.assert_allclose(table[f"P{percentile}_change_mm"], 1.0)
    np.testing.assert_allclose(table["wasserstein_mm"], 1.0)
    same = distribution_changes({"CGPS1": _scan(0.0), "CGPS10": _scan(0.0)})
    np.testing.assert_allclose(same[[f"P{p}_change_mm" for p in PERCENTILES] + ["wasserstein_mm"]], 0.0)


def test_too_few_points() -> None:
    """Quarters with fewer points than the minimum have no values."""
    sparse = {key: values[:50] for key, values in _scan(0.0).items()}
    table = distribution_changes({"CGPS1": sparse, "CGPS10": sparse})
    assert table["P50_change_mm"].isna().all()

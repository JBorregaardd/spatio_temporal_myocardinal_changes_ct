"""Tests for the Gaussian smoothing of the thickness change along the mesh."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repeated_measures" / "bullseye"))

from bullseye_smooth_delta import HeatSmoother  # noqa: E402

SPACING = 0.5
SIZE = 121


def _grid() -> tuple[np.ndarray, np.ndarray]:
    """Flat square triangle mesh of SIZE x SIZE vertices SPACING mm apart."""
    x, y = np.meshgrid(np.arange(SIZE) * SPACING, np.arange(SIZE) * SPACING)
    points = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    ids = np.arange(SIZE * SIZE).reshape(SIZE, SIZE)
    a, b, c, d = ids[:-1, :-1].ravel(), ids[:-1, 1:].ravel(), ids[1:, :-1].ravel(), ids[1:, 1:].ravel()
    faces = np.concatenate([np.column_stack([a, b, d]), np.column_stack([a, d, c])])
    return points, faces


def test_impulse_spreads_with_sigma() -> None:
    """A point value spreads into a kernel whose standard deviation per axis is sigma."""
    points, faces = _grid()
    sigma = 3.0
    centre = (SIZE // 2) * SIZE + SIZE // 2
    impulse = np.zeros(len(points))
    impulse[centre] = 1.0
    kernel = HeatSmoother(points, faces, sigma, steps=40).smooth(impulse)
    weights = kernel / kernel.sum()
    offset = points[:, 0] - points[centre, 0]
    assert np.isclose(np.sqrt(np.sum(weights * offset**2)), sigma, rtol=0.1)


def test_constant_is_kept_and_small_gap_filled() -> None:
    """Normalised convolution keeps a constant field, fills a small gap and leaves a large one empty."""
    points, faces = _grid()
    smoother = HeatSmoother(points, faces, sigma=2.0)
    centre = points[(SIZE // 2) * SIZE + SIZE // 2]
    distance = np.linalg.norm(points - centre, axis=1)
    small_gap = distance > 1.0
    values = np.where(small_gap, 7.0, np.nan)
    smoothed = smoother.smooth_masked(values, small_gap)
    assert np.allclose(smoothed, 7.0)
    large_gap = distance > 15.0
    assert np.isnan(smoother.smooth_masked(np.where(large_gap, 7.0, np.nan), large_gap)[distance < 5]).all()

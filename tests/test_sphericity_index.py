"""Tests for the LV sphericity indices on synthetic cavities with known shape."""
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "measurements"))

from sphericity_index import _sphericity, sphericity_index  # noqa: E402

VOXEL_MM = 0.5
RADIUS_MM = 20.0
LENGTH_MM = 70.0


def _grid(lower_mm: tuple[float, float, float], upper_mm: tuple[float, float, float]) -> np.ndarray:
    axes = [np.arange(lo, hi + VOXEL_MM, VOXEL_MM) for lo, hi in zip(lower_mm, upper_mm)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def _half_ellipsoid(points: np.ndarray) -> np.ndarray:
    """Points inside a half-ellipsoid with its base on z = 0, its apex at z = -LENGTH_MM and base radius RADIUS_MM."""
    radial = np.hypot(points[:, 0], points[:, 1])
    return (radial**2 / RADIUS_MM**2 + points[:, 2] ** 2 / LENGTH_MM**2 <= 1) & (points[:, 2] <= 0)


def test_sphere_is_one() -> None:
    points = _grid((-RADIUS_MM - 1,) * 3, (RADIUS_MM + 1,) * 3)
    sphere = points[np.linalg.norm(points, axis=1) <= RADIUS_MM]
    result = _sphericity(sphere, np.array([0.0, 0.0, RADIUS_MM]), VOXEL_MM**3)
    assert result["long_axis_mm"] == pytest.approx(2 * RADIUS_MM, abs=VOXEL_MM)
    assert result["si_vol"] == pytest.approx(1, abs=0.03)
    assert result["si_dl_mid"] == pytest.approx(1, abs=0.03)


def test_half_ellipsoid() -> None:
    points = _grid((-RADIUS_MM - 1, -RADIUS_MM - 1, -LENGTH_MM - 1), (RADIUS_MM + 1, RADIUS_MM + 1, 0))
    result = _sphericity(points[_half_ellipsoid(points)], np.zeros(3), VOXEL_MM**3)
    assert result["long_axis_mm"] == pytest.approx(LENGTH_MM, abs=VOXEL_MM)
    assert result["edv_ml"] == pytest.approx(2 / 3 * np.pi * RADIUS_MM**2 * LENGTH_MM / 1000, rel=0.02)
    assert result["d_mid_mm"] == pytest.approx(np.sqrt(3) * RADIUS_MM, rel=0.03)
    assert result["si_vol"] == pytest.approx(4 * RADIUS_MM**2 / LENGTH_MM**2, rel=0.03)


def test_sphericity_index_on_rotated_segmentation(tmp_path: Path) -> None:
    """A half-ellipsoid LV below an LA block, tilted in image space; the indices do not depend on the orientation."""
    transform = sitk.VersorRigid3DTransform()
    transform.SetRotation((1.0, 0.5, 0.0), 0.6)
    rotation = np.array(transform.GetMatrix()).reshape(3, 3)

    size = int(2 * (LENGTH_MM + 5))
    image = sitk.Image([size] * 3, sitk.sitkUInt8)
    image.SetSpacing([1.0] * 3)
    image.SetOrigin([-(LENGTH_MM + 5)] * 3)
    index = np.stack(np.meshgrid(*[np.arange(size)] * 3, indexing="ij"), axis=-1).reshape(-1, 3).astype(float)
    lv_frame = (index + np.array(image.GetOrigin())) @ rotation
    radial = np.hypot(lv_frame[:, 0], lv_frame[:, 1])
    labels = np.zeros(len(index), dtype=np.uint8)
    labels[_half_ellipsoid(lv_frame) & (lv_frame[:, 2] < 0)] = 3
    labels[(radial <= RADIUS_MM) & (lv_frame[:, 2] >= 0) & (lv_frame[:, 2] < 30)] = 2
    segmentation = sitk.GetImageFromArray(labels.reshape(size, size, size).transpose(2, 1, 0))
    segmentation.CopyInformation(image)
    sitk.WriteImage(segmentation, str(tmp_path / "seg.nii.gz"))

    result = sphericity_index(str(tmp_path / "seg.nii.gz"))
    assert result["long_axis_mm"] == pytest.approx(LENGTH_MM, abs=1.5)
    assert result["d_mid_mm"] == pytest.approx(np.sqrt(3) * RADIUS_MM, rel=0.05)
    assert result["si_vol"] == pytest.approx(4 * RADIUS_MM**2 / LENGTH_MM**2, rel=0.05)

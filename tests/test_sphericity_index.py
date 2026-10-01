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


def _grid(extent_mm: float) -> np.ndarray:
    axis = np.arange(-extent_mm, extent_mm + VOXEL_MM, VOXEL_MM)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)


def test_sphere_is_one() -> None:
    points = _grid(RADIUS_MM + 1)
    sphere = points[np.linalg.norm(points, axis=1) <= RADIUS_MM]
    result = _sphericity(sphere, np.array([0.0, 0.0, RADIUS_MM]), VOXEL_MM**3)
    assert result["si_dl"] == pytest.approx(1, abs=0.03)
    assert result["si_vol"] == pytest.approx(1, abs=0.03)


def test_half_ellipsoid_has_ellipsoid_ratio_one() -> None:
    points = _grid(LENGTH_MM + 1)
    inside = np.hypot(points[:, 0], points[:, 1]) ** 2 / RADIUS_MM**2 + points[:, 2] ** 2 / LENGTH_MM**2 <= 1
    result = _sphericity(points[inside & (points[:, 2] <= 0)], np.zeros(3), VOXEL_MM**3)
    assert result["long_axis_mm"] == pytest.approx(LENGTH_MM, abs=VOXEL_MM)
    assert result["si_dl"] == pytest.approx(2 * RADIUS_MM / LENGTH_MM, rel=0.03)
    assert result["ellipsoid_ratio"] == pytest.approx(1, abs=0.05)


def test_cylinder_has_blunt_ellipsoid_ratio() -> None:
    points = _grid(LENGTH_MM + 1)
    inside = (np.hypot(points[:, 0], points[:, 1]) <= RADIUS_MM) & (points[:, 2] <= 0) & (points[:, 2] >= -LENGTH_MM)
    result = _sphericity(points[inside], np.zeros(3), VOXEL_MM**3)
    assert result["ellipsoid_ratio"] == pytest.approx(1.5, rel=0.05)


def test_sphericity_index_on_rotated_segmentation(tmp_path: Path) -> None:
    """A cylindrical LV below an LA block, tilted in image space; the transform maps it back onto z."""
    transform = sitk.VersorRigid3DTransform()
    transform.SetRotation((1.0, 0.5, 0.0), 0.6)
    sitk.WriteTransform(transform, str(tmp_path / "lv17_transform.txt"))

    size = int(2 * (LENGTH_MM + 5) / 1.0)
    image = sitk.Image([size] * 3, sitk.sitkUInt8)
    image.SetSpacing([1.0] * 3)
    image.SetOrigin([-(LENGTH_MM + 5)] * 3)
    index = np.stack(np.meshgrid(*[np.arange(size)] * 3, indexing="ij"), axis=-1).reshape(-1, 3).astype(float)
    image_points = index + np.array(image.GetOrigin())
    rotation = np.array(transform.GetMatrix()).reshape(3, 3)
    lv_frame = image_points @ rotation
    radial = np.hypot(lv_frame[:, 0], lv_frame[:, 1])
    labels = np.zeros(len(index), dtype=np.uint8)
    labels[(radial <= RADIUS_MM) & (lv_frame[:, 2] < 0) & (lv_frame[:, 2] >= -LENGTH_MM)] = 3
    labels[(radial <= RADIUS_MM) & (lv_frame[:, 2] >= 0) & (lv_frame[:, 2] < 30)] = 2
    array = labels.reshape(size, size, size).transpose(2, 1, 0)
    segmentation = sitk.GetImageFromArray(array)
    segmentation.CopyInformation(image)
    sitk.WriteImage(segmentation, str(tmp_path / "seg.nii.gz"))

    result = sphericity_index(str(tmp_path / "seg.nii.gz"), str(tmp_path))
    assert result["long_axis_mm"] == pytest.approx(LENGTH_MM, abs=1.5)
    assert result["d_mid_mm"] == pytest.approx(2 * RADIUS_MM, rel=0.03)

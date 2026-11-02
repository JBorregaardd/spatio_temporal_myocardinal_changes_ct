"""Tests for the valve plane extraction in side_project/valve_planes on synthetic volumes."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "side_project" / "valve_planes"))

from valve_planes import (  # noqa: E402
    AORTA_ID,
    LEFT_ATRIUM_ID,
    LEFT_VENTRICLE_ID,
    MYOCARDIUM_ID,
    VALVES,
    SliceSettings,
    adaptive_window,
    find_scans,
    locate_valve,
    orient_towards,
    process_scan,
    sample_valve_image,
    window,
)

SHAPE_ZYX = (40, 40, 40)
SPACING_XYZ = (1.0, 1.0, 2.0)


def _image(array: np.ndarray) -> sitk.Image:
    """Wrap a (z, y, x) array in an image with anisotropic spacing and a non-zero origin."""
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(SPACING_XYZ)
    image.SetOrigin((-10.0, 5.0, 100.0))
    return image


def _labels() -> np.ndarray:
    """LA on top of LV along z (mitral plane at z=20), aorta next to LV along x (aortic plane at x=30)."""
    labels = np.zeros(SHAPE_ZYX, dtype=np.uint8)
    labels[5:20, 10:30, 10:30] = LEFT_VENTRICLE_ID
    labels[20:35, 10:30, 10:30] = LEFT_ATRIUM_ID
    labels[5:20, 12:28, 30:38] = AORTA_ID
    return labels


def test_mitral_plane_normal_points_from_atrium_to_ventricle() -> None:
    labels = _labels()
    plane = locate_valve(_image(labels), labels, VALVES[1])
    assert np.allclose(plane.normal, [0.0, 0.0, -1.0], atol=1e-6)
    assert abs(plane.center_mm[2] - (100.0 + 19.5 * SPACING_XYZ[2])) < 1.0


def test_aortic_plane_normal_points_from_ventricle_to_aorta() -> None:
    labels = _labels()
    plane = locate_valve(_image(labels), labels, VALVES[0])
    assert np.allclose(plane.normal, [1.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(np.cross(plane.in_plane_u, plane.in_plane_v), plane.normal, atol=1e-6)


def test_orient_towards_points_u_at_other_valve() -> None:
    labels = _labels()
    aortic = locate_valve(_image(labels), labels, VALVES[0])
    mitral = orient_towards(locate_valve(_image(labels), labels, VALVES[1]), aortic.center_mm)
    assert np.allclose(mitral.in_plane_u, [1.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(np.cross(mitral.in_plane_u, mitral.in_plane_v), mitral.normal, atol=1e-6)


def test_parallel_views_stack_slices_along_normal() -> None:
    labels = _labels()
    ct = _image(np.where(labels == LEFT_ATRIUM_ID, 1000.0, -1000.0).astype(np.float32))
    plane = locate_valve(_image(labels), labels, VALVES[1])
    settings = SliceSettings(field_of_view_mm=10.0, resolution_mm=1.0, slab_offset_mm=4.0, views="parallel")
    image = sample_valve_image(ct, plane, settings)
    assert image.shape == (10, 10, 3)
    assert image.dtype == np.uint8
    assert (image[..., 0] == 255).all()
    assert (image[..., 2] == 0).all()


def test_process_scan_writes_images_and_metadata(tmp_path: Path) -> None:
    labels = _labels()
    sitk.WriteImage(_image(np.full(SHAPE_ZYX, 100, dtype=np.int16)), str(tmp_path / "ct.nii.gz"))
    (tmp_path / "ct.nii.gz").rename(tmp_path / "7.img.nii.gz")
    sitk.WriteImage(_image(labels), str(tmp_path / "7.heart.nii.gz"))
    assert find_scans(str(tmp_path), "{id}.img.nii.gz") == {"7": str(tmp_path / "7.img.nii.gz")}

    settings = SliceSettings(window="fixed")
    process_scan("7", str(tmp_path / "7.img.nii.gz"), str(tmp_path / "7.heart.nii.gz"), str(tmp_path), settings)
    assert (tmp_path / "7_aortic.png").is_file()
    assert (tmp_path / "7_mitral.png").is_file()
    metadata = json.loads((tmp_path / "7_valves.json").read_text())
    assert set(metadata["valves"]) == {"aortic", "mitral"}
    assert metadata["channels"] == ["en_face", "long_axis_a", "long_axis_b"]
    assert (tmp_path / "7_mitral_long_axis_a.png").is_file()


def test_orthogonal_long_axis_views_show_flow_upwards() -> None:
    labels = _labels()
    ct = _image(np.where(labels == LEFT_ATRIUM_ID, 1000.0, -1000.0).astype(np.float32))
    plane = locate_valve(_image(labels), labels, VALVES[1])
    image = sample_valve_image(ct, plane, SliceSettings(field_of_view_mm=10.0, resolution_mm=1.0))
    for channel in (1, 2):
        assert (image[:4, :, channel] == 0).all()
        assert (image[-4:, :, channel] == 255).all()


def test_adaptive_window_maps_myocardium_and_blood_to_fixed_grey_levels() -> None:
    labels = np.array([MYOCARDIUM_ID, MYOCARDIUM_ID, LEFT_VENTRICLE_ID, AORTA_ID, 0])
    for myocardium, blood in ((100.0, 400.0), (165.0, 810.0)):
        volume = np.array([myocardium, myocardium, blood, blood, -1000.0])
        settings = SliceSettings(*[0.0] * 3, *adaptive_window(volume, labels))
        grey = window(np.array([myocardium, blood]), settings)
        assert np.allclose(grey, [102, 204], atol=1)


def test_adaptive_window_rejects_scans_without_contrast() -> None:
    labels = np.array([MYOCARDIUM_ID, LEFT_VENTRICLE_ID])
    with pytest.raises(ValueError):
        adaptive_window(np.array([100.0, 50.0]), labels)

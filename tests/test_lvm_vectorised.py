"""Regression tests pinning the vectorised LVM thickness helpers to the loops they replaced."""
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import vtk
from scipy.ndimage import center_of_mass
from vtk.util.numpy_support import numpy_to_vtk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "LVM_thickness"))

from lv_generate_segments import _label_centers_of_mass  # noqa: E402
from pipeline import discover_patients  # noqa: E402
from thickness_estimation import _smooth_vectors, thickness_in_17_seg  # noqa: E402
from utils import utils  # noqa: E402


def _rotation(a: float, b: float, c: float) -> np.ndarray:
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return rx @ ry @ rz


def _polydata(points: np.ndarray, scalars: np.ndarray | None = None) -> vtk.vtkPolyData:
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float64), deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)
    if scalars is not None:
        poly.GetPointData().SetScalars(numpy_to_vtk(np.asarray(scalars, dtype=np.float64), deep=True))
    return poly


@pytest.mark.parametrize("direction", [np.eye(3), np.diag([1.0, -1.0, 1.0]), _rotation(0.3, -0.45, 0.8)])
def test_index_physical_transforms_match_simpleitk(direction: np.ndarray) -> None:
    """The vectorised affine agrees with SimpleITK's per-point transforms, including oblique geometry."""
    image = sitk.Image(40, 50, 60, sitk.sitkUInt8)
    image.SetSpacing((0.4, 0.7, 1.3))
    image.SetOrigin((-12.5, 33.25, -204.125))
    image.SetDirection(tuple(direction.flatten()))
    indices = np.random.default_rng(0).uniform(-10, 70, size=(500, 3))

    expected = np.array([image.TransformContinuousIndexToPhysicalPoint(p.tolist()) for p in indices])
    physical = utils.continuous_index_to_physical(image, indices)
    np.testing.assert_allclose(physical, expected, atol=1e-10)

    expected_back = np.array([image.TransformPhysicalPointToContinuousIndex(p.tolist()) for p in expected])
    np.testing.assert_allclose(utils.physical_to_continuous_index(image, physical), expected_back, atol=1e-10)


def test_label_centers_of_mass_is_bit_identical_to_scipy() -> None:
    """Matches scipy exactly, including negative labels and labels that are absent (NaN)."""
    labels = np.random.default_rng(1).integers(-1, 6, size=(20, 25, 30)).astype(np.int16)
    labels[labels == 4] = 0
    index = range(1, 8)

    expected = center_of_mass(np.ones_like(labels), labels, index)
    result = _label_centers_of_mass(labels, index)

    assert len(result) == len(expected)
    for got, want in zip(result, expected):
        np.testing.assert_array_equal(np.array(got), np.array(want))


def _smooth_vectors_reference(points: np.ndarray, vecs: np.ndarray, iterations: int, k: int) -> np.ndarray:
    """The original per-point VTK loop, with the in-place update made Jacobi to match the vectorised version."""
    poly = _polydata(points)
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(poly)
    locator.BuildLocator()
    vecs = vecs.copy()
    for _ in range(iterations):
        previous = vecs.copy()
        for i in range(len(points)):
            if np.linalg.norm(previous[i]) < 1e-6:
                continue
            ids = vtk.vtkIdList()
            locator.FindClosestNPoints(k, points[i], ids)
            neighbours = [ids.GetId(j) for j in range(ids.GetNumberOfIds()) if ids.GetId(j) != i]
            weights = [1.0 / n if (n := np.linalg.norm(points[j] + points[i])) > 1e-6 else 0.0 for j in neighbours]
            total = sum(weights)
            if total > 1e-6:
                vecs[i] = sum(w * previous[j] for w, j in zip(weights, neighbours)) / total
    return vecs


def test_smooth_vectors_matches_reference_loop() -> None:
    """Same neighbours, legacy weights, self-exclusion and degenerate-vector handling as the original loop."""
    rng = np.random.default_rng(2)
    points = rng.normal(size=(400, 3)) * 20 + np.array([40.0, -80.0, 150.0])
    vecs = rng.normal(size=(400, 3))
    vecs[::37] = 0.0

    expected = _smooth_vectors_reference(points, vecs, iterations=3, k=20)
    np.testing.assert_allclose(_smooth_vectors(points, vecs, 3, 20), expected, rtol=1e-10, atol=1e-12)


def test_thickness_in_17_seg_matches_reference_loop() -> None:
    """Every thickness value lands in the same segment as with the original nearest-point loop."""
    rng = np.random.default_rng(3)
    seg_points = rng.uniform(0, 50, size=(300, 3))
    seg_labels = rng.integers(-1, 18, size=300).astype(np.float64)
    thick_points = rng.uniform(0, 50, size=(1000, 3))
    thick_values = rng.uniform(0, 20, size=1000)
    mesh_17 = _polydata(seg_points, seg_labels)
    mesh_thick = _polydata(thick_points, thick_values)

    locator = vtk.vtkPointLocator()
    locator.SetDataSet(mesh_17)
    locator.BuildLocator()
    expected: dict[float, list[float]] = {}
    for i, point in enumerate(thick_points):
        segment = float(seg_labels[locator.FindClosestPoint(point)])
        expected.setdefault(segment, []).append(float(thick_values[i]))

    result = thickness_in_17_seg(mesh_thick, mesh_17)
    assert set(result) == set(expected)
    for segment, values in expected.items():
        assert result[segment] == pytest.approx(values)


def test_discover_patients_requires_image_and_segmentation(tmp_path: Path) -> None:
    """Only ids with both files are returned, sorted numerically rather than lexically."""
    image_dir = tmp_path / "data" / "1_200"
    seg_dir = tmp_path / "seg"
    image_dir.mkdir(parents=True)
    seg_dir.mkdir()
    for pid in ["1", "2", "10", "100"]:
        (image_dir / f"{pid}.img.nii.gz").touch()
        (image_dir / f"{pid}.label.nii.gz").touch()
    for pid in ["1", "10", "100", "999"]:
        (seg_dir / f"{pid}.heart.nii.gz").touch()
        (seg_dir / f"{pid}.arteries.nii.gz").touch()

    assert discover_patients(str(tmp_path), str(seg_dir)) == ["1", "10", "100"]

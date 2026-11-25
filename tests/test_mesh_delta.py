"""Tests for the per-subject wall thickness change computed on the mesh between CGPS1 and CGPS10."""
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repeated_measures" / "bullseye"))

from mesh_delta import apply_matrix, rotation_angle_deg, transfer_thickness, transform_matrix  # noqa: E402


def _rigid(angle_deg: float, translation: tuple[float, float, float]) -> sitk.VersorRigid3DTransform:
    """Rotation about z through a centre followed by a translation."""
    transform = sitk.VersorRigid3DTransform()
    transform.SetCenter((5.0, -3.0, 2.0))
    transform.SetRotation((0.0, 0.0, 1.0), np.radians(angle_deg))
    transform.SetTranslation(translation)
    return transform


def test_transform_matrix_matches_sitk() -> None:
    """The 4x4 matrix maps points like the composite SimpleITK transform it is made from."""
    composite = sitk.CompositeTransform([_rigid(30, (1.0, 2.0, 3.0)), sitk.TranslationTransform(3, (0.0, 0.0, -4.0))])
    points = np.random.default_rng(0).normal(size=(10, 3)) * 50
    expected = np.array([composite.TransformPoint(tuple(point)) for point in points])
    np.testing.assert_allclose(apply_matrix(transform_matrix(composite), points), expected, atol=1e-9)


def test_rotation_angle() -> None:
    """The rotation angle of a rigid matrix is recovered."""
    assert np.isclose(rotation_angle_deg(transform_matrix(_rigid(25, (0.0, 0.0, 0.0)))), 25)


def test_transfer_thickness_nearest_and_max_distance() -> None:
    """Each fixed vertex takes the value of its nearest moving vertex, or none when it is too far away."""
    moving = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    fixed = np.array([[1.0, 0.0, 0.0], [9.0, 0.0, 0.0], [30.0, 0.0, 0.0]])
    values, distance = transfer_thickness(fixed, moving, np.array([4.0, 7.0]), max_distance=5.0)
    np.testing.assert_allclose(values[:2], [4.0, 7.0])
    assert np.isnan(values[2])
    np.testing.assert_allclose(distance, [1.0, 1.0, 20.0])

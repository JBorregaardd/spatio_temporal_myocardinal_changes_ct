"""Tests for the GloViTa crop export in side_project/valve_planes."""
import sys
from dataclasses import asdict
from pathlib import Path

import blosc2
import numpy as np
import pytest
import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "side_project" / "valve_planes"))

from export_crops import (  # noqa: E402
    cross_validation_splits,
    intensity_reference,
    read_labels,
    sample_crop,
    save_b2nd,
    scan_hu_window,
    split_by_patient,
)
from valve_planes import LEFT_ATRIUM_ID, LEFT_VENTRICLE_ID, VALVES, adaptive_window, locate_valve  # noqa: E402


def test_intensity_reference_inverts_the_adaptive_window() -> None:
    labels = np.array([1, 1, LEFT_VENTRICLE_ID, LEFT_VENTRICLE_ID])
    volume = np.array([150.0, 150.0, 700.0, 700.0])
    myocardium, contrast = intensity_reference(list(adaptive_window(volume, labels)))
    assert myocardium == pytest.approx(150.0)
    assert contrast == pytest.approx(550.0)


def test_scan_hu_window_reads_old_and_new_metadata() -> None:
    assert scan_hu_window({"hu_window": [-300.0, 700.0]}) == [-300.0, 700.0]
    assert scan_hu_window({"settings": {"window": "adaptive", "hu_min": -314.0, "hu_max": 746.0}}) == [-314.0, 746.0]
    assert scan_hu_window({"settings": {"window": "fixed", "hu_min": -200.0, "hu_max": 600.0}}) is None


def test_crop_has_flow_along_the_last_axis() -> None:
    labels = np.zeros((40, 40, 40), dtype=np.uint8)
    labels[5:20, 10:30, 10:30] = LEFT_VENTRICLE_ID
    labels[20:35, 10:30, 10:30] = LEFT_ATRIUM_ID
    image = sitk.GetImageFromArray(labels)
    image.SetSpacing((1.0, 1.0, 2.0))
    ct = sitk.GetImageFromArray(np.where(labels == LEFT_ATRIUM_ID, 1000.0, -1000.0).astype(np.float32))
    ct.CopyInformation(image)

    plane = asdict(locate_valve(image, labels, VALVES[1]))
    crop = sample_crop(ct, plane, crop_voxels=8, resolution_mm=1.0)
    assert crop.shape == (8, 8, 8)
    assert (crop[:, :, :3] == 1000.0).all()
    assert (crop[:, :, -3:] == -1000.0).all()


def test_read_labels_skips_unclear_and_missing(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text(
        "patient_id,aortic,mitral,updated\n" "S001,closed,open,t\n" "S002,unclear,closed,t\n" "S003,open,,t\n"
    )
    labels = read_labels(str(path))
    assert labels["aortic"] == {"S001": 0, "S003": 1}
    assert labels["mitral"] == {"S001": 1, "S002": 0}


def test_split_keeps_patients_together_and_spreads_the_rare_class() -> None:
    labels = {f"S{i:03d}": int(i < 10) for i in range(50)}
    groups = {f"S{i:03d}": f"P{i // 2}" for i in range(50)}
    splits = split_by_patient(labels, groups, seed=0)

    assert sorted(s for part in splits.values() for s in part) == sorted(labels)
    patient_split = {groups[s]: name for name, part in splits.items() for s in part}
    assert all(patient_split[groups[s]] == name for name, part in splits.items() for s in part)
    assert all(any(labels[s] == 1 for s in part) for part in splits.values())


def test_cross_validation_keeps_the_test_set_and_spreads_patients_over_folds() -> None:
    labels = {f"S{i:03d}": int(i < 20) for i in range(60)}
    groups = {f"S{i:03d}": f"P{i // 2}" for i in range(60)}
    folds = cross_validation_splits(labels, groups, n_folds=5, seed=0)

    assert len(folds) == 5
    assert all(fold["test"] == split_by_patient(labels, groups, seed=0)["test"] for fold in folds)
    rest = sorted(set(labels) - set(folds[0]["test"]))
    assert sorted(scan for fold in folds for scan in fold["val"]) == rest
    for fold in folds:
        assert sorted(fold["train"] + fold["val"]) == rest
        assert {groups[s] for s in fold["train"]}.isdisjoint(groups[s] for s in fold["val"])
        assert {labels[s] for s in fold["val"]} == {0, 1}


def test_b2nd_round_trip(tmp_path: Path) -> None:
    array = np.random.default_rng(0).random((1, 8, 8, 8)).astype(np.float32)
    path = str(tmp_path / "mitral" / "images" / "S001.b2nd")
    save_b2nd(array, path)
    save_b2nd(array, path)
    loaded = blosc2.open(urlpath=path, mode="r", mmap_mode="r")[...]
    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, array)

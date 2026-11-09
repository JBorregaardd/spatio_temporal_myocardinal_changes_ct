"""Tests for the blinded ED/ES dataset in side_project/valve_planes."""
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "side_project" / "valve_planes"))

from blind_phase_dataset import assign_scan_ids, link_blinded  # noqa: E402

ROWS = [{"patient_id": f"{i:04d}", "phase": phase} for phase in ("ED", "ES") for i in range(10)]


def test_assign_scan_ids_keeps_every_scan_once_and_is_reproducible() -> None:
    key = assign_scan_ids(ROWS, seed=0)

    assert [row["scan_id"] for row in key] == [f"S{i:03d}" for i in range(1, 21)]
    assert sorted((row["patient_id"], row["phase"]) for row in key) == sorted(
        (row["patient_id"], row["phase"]) for row in ROWS
    )
    assert key == assign_scan_ids(ROWS, seed=0)
    assert [row["phase"] for row in key] != [row["phase"] for row in ROWS]


def test_assign_scan_ids_keeps_the_existing_key_and_numbers_new_scans_after_it() -> None:
    key = assign_scan_ids(ROWS, seed=0)
    new_rows = [{"patient_id": f"{i:04d}", "phase": "ES"} for i in range(10, 15)]

    extended = assign_scan_ids(ROWS + new_rows, seed=1, key=key)

    assert extended[:20] == key
    assert [row["scan_id"] for row in extended[20:]] == [f"S{i:03d}" for i in range(21, 26)]
    assert sorted(row["patient_id"] for row in extended[20:]) == [f"{i:04d}" for i in range(10, 15)]
    assert assign_scan_ids(ROWS, seed=1, key=key) == key


def test_link_blinded_links_images_and_existing_segmentations(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    for phase, pid in (("ED", "0000"), ("ES", "0000")):
        (dataset / phase).mkdir(parents=True, exist_ok=True)
        (dataset / phase / f"{pid}.nii.gz").write_text(f"{phase}-image")
    (dataset / "segmentations" / "ED").mkdir(parents=True)
    (dataset / "segmentations" / "ED" / "0000.heart.nii.gz").write_text("ED-seg")
    key = [{"scan_id": "S001", "patient_id": "0000", "phase": "ES"}, {"scan_id": "S002", "patient_id": "0000", "phase": "ED"}]

    try:
        missing = link_blinded(key, str(dataset), str(tmp_path / "blinded"))
    except OSError:
        pytest.skip("Creating symlinks is not permitted on this system")

    assert missing == 1
    assert (tmp_path / "blinded" / "images" / "S001.nii.gz").read_text() == "ES-image"
    assert (tmp_path / "blinded" / "segmentations" / "S002.heart.nii.gz").read_text() == "ED-seg"
    assert not os.path.lexists(tmp_path / "blinded" / "segmentations" / "S001.heart.nii.gz")

"""Tests for the ED/ES dataset builder in side_project/valve_planes."""
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "side_project" / "valve_planes"))

from make_phase_dataset import build_dataset, read_filelist, sample_patients  # noqa: E402

NIFTI = "/storage/Data/DTU-CGPS-10y-august-2026/NIFTI"


def test_read_filelist_keeps_one_scan_per_patient(tmp_path: Path) -> None:
    path = tmp_path / "es.txt"
    path.write_text(
        f"{NIFTI}/CGPS-10y-august-2026_0005_SERIES0013_volume_0.nii.gz\n"
        f"{NIFTI}/CGPS-10y-august-2026_0005_SERIES0012_volume_0.nii.gz\n"
        f"\n{NIFTI}/CGPS-10y-august-2026_0000_SERIES0013_volume_0.nii.gz\n"
        f"{NIFTI}/CGPS-10y-august-2026_0013_SERIES0005.nii.gz\n"
        f"{NIFTI}/CGPS-10y-august-2026_0000_SERIES0001.nii.gz\n"
    )
    scans = read_filelist(str(path))
    assert scans == {
        "0000": f"{NIFTI}/CGPS-10y-august-2026_0000_SERIES0013_volume_0.nii.gz",
        "0005": f"{NIFTI}/CGPS-10y-august-2026_0005_SERIES0012_volume_0.nii.gz",
        "0013": f"{NIFTI}/CGPS-10y-august-2026_0013_SERIES0005.nii.gz",
    }


def test_sample_patients_is_reproducible_and_checks_the_size() -> None:
    scans = {f"{i:04d}": f"{i}.nii.gz" for i in range(20)}

    drawn = sample_patients(scans, 5, random.Random(0))
    assert len(drawn) == 5
    assert drawn == sorted(drawn, key=int)
    assert drawn == sample_patients(scans, 5, random.Random(0))
    with pytest.raises(ValueError):
        sample_patients(scans, 21, random.Random(0))


def test_build_dataset_places_each_phase_in_its_own_folder(tmp_path: Path) -> None:
    sources = tmp_path / "nifti"
    sources.mkdir()
    selection = {"ED": {}, "ES": {}}
    for phase, pid, volume in (("ED", "0000", 1), ("ED", "0002", 1), ("ES", "0002", 0), ("ES", "0007", 0)):
        source = sources / f"CGPS-10y-august-2026_{pid}_SERIES0013_volume_{volume}.nii.gz"
        source.write_text(f"{pid}-{volume}")
        selection[phase][pid] = str(source)

    rows = build_dataset(selection, str(tmp_path / "out"), copy=True)

    assert [(row["patient_id"], row["phase"]) for row in rows] == [
        ("0000", "ED"), ("0002", "ED"), ("0002", "ES"), ("0007", "ES")
    ]
    assert (tmp_path / "out" / "ED" / "0002.nii.gz").read_text() == "0002-1"
    assert (tmp_path / "out" / "ES" / "0007.nii.gz").read_text() == "0007-0"

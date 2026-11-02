"""Tests for the ED/ES dataset builder in side_project/valve_planes."""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "side_project" / "valve_planes"))

from make_phase_dataset import build_dataset, read_filelist, sample_patients  # noqa: E402

NIFTI = "/storage/Data/DTU-CGPS-10y-august-2026/NIFTI"


def write_filelist(path: Path, ids: list[str], volume: int) -> None:
    """Write a file list in the CGPS layout with one scan per patient id."""
    path.write_text(
        "\n".join(f"{NIFTI}/CGPS-10y-august-2026_{pid}_SERIES0013_volume_{volume}.nii.gz" for pid in ids) + "\n"
    )


def test_read_filelist_keeps_first_scan_per_patient(tmp_path: Path) -> None:
    path = tmp_path / "es.txt"
    path.write_text(
        f"{NIFTI}/CGPS-10y-august-2026_0005_SERIES0013_volume_0.nii.gz\n"
        f"{NIFTI}/CGPS-10y-august-2026_0005_SERIES0012_volume_0.nii.gz\n"
        f"\n{NIFTI}/CGPS-10y-august-2026_0000_SERIES0013_volume_0.nii.gz\n"
    )
    scans = read_filelist(str(path))
    assert scans == {
        "0000": f"{NIFTI}/CGPS-10y-august-2026_0000_SERIES0013_volume_0.nii.gz",
        "0005": f"{NIFTI}/CGPS-10y-august-2026_0005_SERIES0012_volume_0.nii.gz",
    }


def test_sample_patients_only_draws_patients_with_both_phases(tmp_path: Path) -> None:
    write_filelist(tmp_path / "ed.txt", ["0000", "0001", "0002", "0003"], 1)
    write_filelist(tmp_path / "es.txt", ["0000", "0002", "0003", "0004"], 0)
    ed, es = read_filelist(str(tmp_path / "ed.txt")), read_filelist(str(tmp_path / "es.txt"))

    assert sample_patients(ed, es, 3, seed=0) == ["0000", "0002", "0003"]
    assert sample_patients(ed, es, 2, seed=1) == sample_patients(ed, es, 2, seed=1)
    with pytest.raises(ValueError):
        sample_patients(ed, es, 4, seed=0)


def test_build_dataset_places_each_phase_in_its_own_folder(tmp_path: Path) -> None:
    sources = tmp_path / "nifti"
    sources.mkdir()
    ed, es = {}, {}
    for pid in ["0000", "0002"]:
        for scans, volume in ((ed, 1), (es, 0)):
            source = sources / f"CGPS-10y-august-2026_{pid}_SERIES0013_volume_{volume}.nii.gz"
            source.write_text(f"{pid}-{volume}")
            scans[pid] = str(source)

    rows = build_dataset(ed, es, ["0000", "0002"], str(tmp_path / "out"), copy=True)

    assert [(row["patient_id"], row["phase"]) for row in rows] == [
        ("0000", "ED"), ("0002", "ED"), ("0000", "ES"), ("0002", "ES")
    ]
    assert (tmp_path / "out" / "ED" / "0002.nii.gz").read_text() == "0002-1"
    assert (tmp_path / "out" / "ES" / "0002.nii.gz").read_text() == "0002-0"

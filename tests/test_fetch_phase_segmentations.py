"""Tests for collecting the segmentations of the ED/ES dataset in side_project/valve_planes."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "side_project" / "valve_planes"))

from fetch_phase_segmentations import fetch_segmentations, source_segmentation  # noqa: E402


def test_source_segmentation_uses_the_phase_folder_and_scan_name() -> None:
    path = source_segmentation("/seg", "ES", "/nifti/CGPS-10y-august-2026_0002_SERIES0013_volume_0.nii.gz")
    assert Path(path) == Path(
        "/seg/Cardiac_ES_CE/CGPS-10y-august-2026_0002_SERIES0013_volume_0/segmentations/heartchambers_highres.nii.gz"
    )


def test_fetch_segmentations_copies_found_and_lists_missing(tmp_path: Path) -> None:
    root = tmp_path / "seg"
    scan = "CGPS-10y-august-2026_0000_SERIES0013_volume_1"
    found = root / "Cardiac_ED_CE" / scan / "segmentations" / "heartchambers_highres.nii.gz"
    found.parent.mkdir(parents=True)
    found.write_text("ed-0000")
    rows = [
        {"patient_id": "0000", "phase": "ED", "source_path": f"/nifti/{scan}.nii.gz"},
        {"patient_id": "0007", "phase": "ES", "source_path": "/nifti/CGPS-10y-august-2026_0007_SERIES0013_volume_0.nii.gz"},
    ]

    missing = fetch_segmentations(rows, str(root), str(tmp_path / "out"), copy=True)

    assert (tmp_path / "out" / "ED" / "0000.heart.nii.gz").read_text() == "ed-0000"
    assert [(row["patient_id"], row["phase"]) for row in missing] == [("0007", "ES")]
    assert not (tmp_path / "out" / "ES").exists()

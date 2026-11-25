"""Tests for the chamber volume curves of the multiphase CFA-2 scans."""
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cfa"))

from volume_curves import SEGMENTATION_FILE, find_scans, patient_summary, volume_curves  # noqa: E402

LV_SLICES = {0: 10, 5: 6, 10: 4, 85: 12, 90: 12}


def write_scan(
    cfa_dir: Path, image_folder: str, segmentation_folder: str, scan_id: str, lv_slices: int, spacing: float
) -> None:
    """Write an empty image and a segmentation with ``lv_slices`` 10x10 slices of LV (3) and one of myocardium (1)."""
    (cfa_dir / image_folder).mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(
        sitk.GetImageFromArray(np.zeros((2, 2, 2), dtype=np.int16)), str(cfa_dir / image_folder / f"{scan_id}.nii.gz")
    )
    array = np.zeros((20, 10, 10), dtype=np.uint8)
    array[:lv_slices] = 3
    array[19] = 1
    segmentation = sitk.GetImageFromArray(array)
    segmentation.SetSpacing((1.0, 1.0, spacing))
    path = cfa_dir / segmentation_folder / scan_id / SEGMENTATION_FILE
    path.parent.mkdir(parents=True)
    sitk.WriteImage(segmentation, str(path))


@pytest.fixture
def cfa_dir(tmp_path: Path) -> Path:
    """A CFA folder with five grid phases and an 87.5% best phase of one patient, and a stray file."""
    for phase, slices in LV_SLICES.items():
        write_scan(tmp_path, "NIFTI", "TotalSegmentator", f"CFA-2_0016_{phase:02d}", slices, 1.0)
    write_scan(tmp_path, "NIFTI_highres", "TotalSegmentator_highres", "CFA-2_0016_87.5%", 18, 0.5)
    (tmp_path / "NIFTI" / "notes.nii.gz").write_text("")
    return tmp_path


def test_find_scans_parses_patient_series_and_phase(cfa_dir: Path) -> None:
    scans = find_scans(cfa_dir)
    assert len(scans) == 6
    assert set(scans["patient_id"]) == {"CFA-2_0016"}
    assert sorted(scans.loc[scans["series"] == "grid", "phase_pct"]) == [0, 5, 10, 85, 90]
    assert scans.loc[scans["series"] == "best", "phase_pct"].item() == 87.5


def test_volume_curves_measure_each_chamber(cfa_dir: Path) -> None:
    curves = volume_curves(find_scans(cfa_dir), workers=1)
    grid = curves[curves["series"] == "grid"].set_index("phase_pct")
    assert grid.loc[0, "lv_volume_ml"] == pytest.approx(1.0)
    assert grid.loc[0, "lvm_volume_ml"] == pytest.approx(0.1)
    assert curves.loc[curves["series"] == "best", "lv_volume_ml"].item() == pytest.approx(0.9)


def test_missing_segmentation_gives_nan(cfa_dir: Path) -> None:
    (cfa_dir / "TotalSegmentator" / "CFA-2_0016_05" / SEGMENTATION_FILE).unlink()
    curves = volume_curves(find_scans(cfa_dir), workers=1)
    assert np.isnan(curves.loc[curves["scan_id"] == "CFA-2_0016_05", "lv_volume_ml"].item())
    assert patient_summary(curves)["n_grid_phases"].item() == 4


def test_patient_summary_finds_ed_es_and_interpolates_best_phase(cfa_dir: Path) -> None:
    summary = patient_summary(volume_curves(find_scans(cfa_dir), workers=1)).iloc[0]
    assert summary["edv_ml"] == pytest.approx(1.2)
    assert summary["ed_phase_pct"] == 85
    assert summary["esv_ml"] == pytest.approx(0.4)
    assert summary["es_phase_pct"] == 10
    assert summary["ef_pct"] == pytest.approx(200 / 3)
    assert summary["best_lv_volume_ml"] == pytest.approx(0.9)
    assert summary["grid_lv_volume_at_best_ml"] == pytest.approx(1.2)


def test_interpolation_wraps_around_the_r_wave() -> None:
    import pandas as pd

    curves = pd.DataFrame(
        {
            "patient_id": "p",
            "series": ["grid", "grid", "best"],
            "phase_pct": [5.0, 95.0, 0.0],
            "lv_volume_ml": [80.0, 100.0, 50.0],
        }
    )
    assert patient_summary(curves)["grid_lv_volume_at_best_ml"].item() == pytest.approx(90.0)

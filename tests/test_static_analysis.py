"""Tests for the volume, mass and BSA-indexed measures in the static analysis."""
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "LVM_thickness" / "analysis"))

from static_analysis import (  # noqa: E402
    MYOCARDIAL_DENSITY_G_PER_ML,
    bsa_dubois,
    calculate_patient_summary,
    calculate_volume_per_segment,
    label_volume_ml,
    load_demographics,
)

SPACING = (0.5, 0.5, 0.25)
VOXEL_ML = 0.5 * 0.5 * 0.25 / 1000


def _image(array: np.ndarray) -> sitk.Image:
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(SPACING)
    return image


def _synthetic_patient() -> tuple[sitk.Image, sitk.Image]:
    heart = np.zeros((10, 10, 10), dtype=np.uint8)
    heart[0:2] = 1
    heart[2:5] = 3
    heart[5:6] = 2
    lv17 = np.zeros((10, 10, 10), dtype=np.int8)
    lv17[0, :, :5] = 4
    lv17[0, :, 5:] = 17
    lv17[1] = -1
    return _image(heart), _image(lv17)


def _thickness_table() -> list[dict]:
    table = [{"segment": s, "median_thickness": 8.0} for s in range(1, 18)]
    table[2]["median_thickness"] = 14.5
    table[16]["median_thickness"] = 30.0
    table[5]["median_thickness"] = np.nan
    return table


def test_label_volume_uses_spacing() -> None:
    """Voxel count times the product of the spacing, in mL."""
    image = _image(np.ones((4, 5, 6), dtype=np.uint8))
    assert label_volume_ml(image, sitk.GetArrayViewFromImage(image) == 1) == pytest.approx(120 * VOXEL_ML)


def test_bsa_dubois_reference_value() -> None:
    """170 cm / 70 kg is ~1.81 m^2 by DuBois."""
    assert bsa_dubois(170, 70) == pytest.approx(1.8097, abs=1e-3)


def test_volume_per_segment_counts_each_label() -> None:
    """Every segment is reported; absent segments are 0 mL."""
    _, lv17 = _synthetic_patient()
    volumes = calculate_volume_per_segment(lv17)
    assert set(volumes) == set(range(1, 18))
    assert volumes[4] == pytest.approx(50 * VOXEL_ML)
    assert volumes[17] == pytest.approx(50 * VOXEL_ML)
    assert volumes[1] == 0.0


def test_patient_summary_without_demographics() -> None:
    """Imaging-only measures are filled; BSA-indexed measures are NaN."""
    heart, lv17 = _synthetic_patient()
    summary = calculate_patient_summary(heart, lv17, _thickness_table())

    assert summary["lvm_g"] == pytest.approx(200 * VOXEL_ML * MYOCARDIAL_DENSITY_G_PER_ML)
    assert summary["lvm_segments_g"] == pytest.approx(100 * VOXEL_ML * MYOCARDIAL_DENSITY_G_PER_ML)
    assert summary["lvm_unassigned_g"] == pytest.approx(100 * VOXEL_ML * MYOCARDIAL_DENSITY_G_PER_ML)
    assert summary["lvdv_ml"] == pytest.approx(300 * VOXEL_ML)
    assert summary["lav_ml"] == pytest.approx(100 * VOXEL_ML)
    assert summary["lv_mv_ratio"] == pytest.approx(summary["lvm_g"] / summary["lvdv_ml"])
    assert summary["lv_max_wall_thickness_mm"] == 14.5
    assert summary["lv_max_wall_thickness_segment"] == 3
    for key in ("bsa_m2", "lvmi", "lvdvi", "lavi"):
        assert np.isnan(summary[key])


def test_patient_summary_indexes_by_bsa(tmp_path: Path) -> None:
    """With height and weight, indexed measures are the raw measures divided by DuBois BSA."""
    csv_path = tmp_path / "demographics.csv"
    csv_path.write_text("patient_id,height_cm,weight_kg,sex,age\n7,170,70,F,55\n8,,80,M,60\n")
    demographics = load_demographics(str(csv_path))
    heart, lv17 = _synthetic_patient()

    summary = calculate_patient_summary(heart, lv17, _thickness_table(), demographics["7"])
    bsa = bsa_dubois(170, 70)
    assert summary["bsa_m2"] == pytest.approx(bsa)
    assert summary["lvmi"] == pytest.approx(summary["lvm_g"] / bsa)
    assert summary["lvdvi"] == pytest.approx(summary["lvdv_ml"] / bsa)
    assert summary["lavi"] == pytest.approx(summary["lav_ml"] / bsa)
    assert (summary["sex"], summary["age"]) == ("F", "55")

    missing_height = calculate_patient_summary(heart, lv17, _thickness_table(), demographics["8"])
    assert np.isnan(missing_height["lvmi"])


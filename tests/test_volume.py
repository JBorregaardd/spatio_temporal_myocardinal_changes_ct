"""Tests for the LV myocardial volume and mass."""
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "measurements"))

from volume import MYOCARDIAL_DENSITY_G_PER_ML, myocardium_volume  # noqa: E402


def test_myocardium_volume_counts_only_the_myocardium_label(tmp_path: Path) -> None:
    array = np.zeros((20, 20, 20), dtype=np.uint8)
    array[:10] = 1
    array[10:15] = 3
    segmentation = sitk.GetImageFromArray(array)
    segmentation.SetSpacing((0.5, 0.5, 2.0))
    path = str(tmp_path / "seg.nii.gz")
    sitk.WriteImage(segmentation, path)

    result = myocardium_volume(path)
    assert result["lvm_volume_ml"] == pytest.approx(10 * 20 * 20 * 0.5 * 0.5 * 2.0 / 1000)
    assert result["lvm_mass_g"] == pytest.approx(result["lvm_volume_ml"] * MYOCARDIAL_DENSITY_G_PER_ML)

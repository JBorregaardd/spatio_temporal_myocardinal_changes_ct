"""Left-ventricular myocardial volume and mass from the TotalSegmentator heart-chambers segmentation."""
import numpy as np
import SimpleITK as sitk

LVM_ID = 1
MYOCARDIAL_DENSITY_G_PER_ML = 1.055

" til Signe"
def myocardium_volume(segmentation_path: str) -> dict[str, float]:
    """Volume and mass of the whole LV myocardium (TotalSegmentator label 1) of one patient.

    Mass is volume times the myocardial density of 1.055 g/ml (Fuchs et al. 2016, EHJ-CVI 17:1009).
    Args:
        segmentation_path: TotalSegmentator heart-chambers segmentation.

    Returns:
        ``lvm_volume_ml`` and ``lvm_mass_g``.
    """
    segmentation = sitk.ReadImage(segmentation_path)
    voxels = int(np.count_nonzero(sitk.GetArrayViewFromImage(segmentation) == LVM_ID))
    volume_ml = voxels * float(np.prod(segmentation.GetSpacing())) / 1000
    return {"lvm_volume_ml": volume_ml, "lvm_mass_g": volume_ml * 1.055 } #1.055 g/ml is the myocardial density

def segment_volume(lv17_path: str) -> dict[str, float]:
    lv17 = sitk.ReadImage(lv17_path)
    labels = sitk.GetArrayViewFromImage(lv17)
    voxel_ml = float(np.prod(lv17.GetSpacing())) / 1000
    result = {}
    for segment in range(1, 18):
        volume_ml = np.count_nonzero(labels == segment) * voxel_ml
        result[f"segment_{segment}_volume_ml"] = volume_ml
        result[f"segment_{segment}_mass_g"] = volume_ml * 1.055 #1.055 g/ml is the myocardial density
    return result


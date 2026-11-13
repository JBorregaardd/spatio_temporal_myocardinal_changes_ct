"""Left-ventricular myocardial volume and mass and LV blood pool volume from the TotalSegmentator heart-chambers
segmentation."""
import numpy as np
import SimpleITK as sitk

LVM_ID = 1
LV_ID = 3
MYOCARDIAL_DENSITY_G_PER_ML = 1.055


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


def blood_pool_volume(segmentation_path: str) -> dict[str, float]:
    """Volume of the LV blood pool (TotalSegmentator label 3) of one patient.

    Args:
        segmentation_path: TotalSegmentator heart-chambers segmentation.

    Returns:
        ``lv_blood_pool_volume_ml``.
    """
    segmentation = sitk.ReadImage(segmentation_path)
    voxels = int(np.count_nonzero(sitk.GetArrayViewFromImage(segmentation) == LV_ID))
    volume_ml = voxels * float(np.prod(segmentation.GetSpacing())) / 1000
    return {"lv_blood_pool_volume_ml": volume_ml}


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

def ejection_fraction(end_diastole_path, end_systole_path: str) -> float:
    """Calculate the ejection fraction from the end-diastolic and end-systolic TotalSegmentator heart-chambers segmentations."""

    ed_segmentation = sitk.ReadImage(end_diastole_path)
    es_segmentation = sitk.ReadImage(end_systole_path)

    ed_voxels = int(np.count_nonzero(sitk.GetArrayViewFromImage(ed_segmentation) == LV_ID))
    es_voxels = int(np.count_nonzero(sitk.GetArrayViewFromImage(es_segmentation) == LV_ID))

    ed_volume = ed_voxels * float(np.prod(ed_segmentation.GetSpacing())) / 1000
    es_volume = es_voxels * float(np.prod(es_segmentation.GetSpacing())) / 1000

    ejection_fraction = (ed_volume - es_volume) / ed_volume

    return ejection_fraction
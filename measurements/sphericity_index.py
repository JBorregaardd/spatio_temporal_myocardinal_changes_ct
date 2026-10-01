"""Left-ventricular sphericity indices from the outputs of the LVM thickness pipeline."""
import os

import numpy as np
import SimpleITK as sitk

LA_ID = 2
LV_ID = 3
SLAB_MM = 1.0


def sphericity_index(segmentation_path: str, patient_dir: str) -> dict[str, float]:
    """Sphericity of the LV cavity (TotalSegmentator label 3) of one patient.

    Args:
        segmentation_path: TotalSegmentator heart-chambers segmentation.
        patient_dir: The patient's output folder from the LVM thickness pipeline.

    Returns:
        ``edv_ml``, ``long_axis_mm``, ``d_max_mm`` (largest D below the mitral valve), ``d_mid_mm`` (D halfway along
        L), ``si_dl`` (d_max / L), ``si_dl_mid`` (d_mid / L), ``si_vol`` (EDV over the volume of a sphere of
        diameter L) and ``ellipsoid_ratio`` (si_vol / si_dl^2: 1 for an ellipsoid, > 1 blunt apex, < 1 conical).
    """
    segmentation = sitk.ReadImage(segmentation_path)
    transform = sitk.ReadTransform(os.path.join(patient_dir, "lv17_transform.txt"))
    offset = np.array(transform.TransformPoint((0.0, 0.0, 0.0)))
    rotation = np.stack([np.array(transform.TransformPoint(tuple(axis))) - offset for axis in np.eye(3)], axis=1)

    la, lv = segmentation == LA_ID, segmentation == LV_ID
    valve = sitk.BinaryDilate(la, [2, 2, 2]) & sitk.BinaryDilate(lv, [2, 2, 2])
    cavity = (_points_mm(lv) - offset) @ rotation
    valve_centre = (_points_mm(valve).mean(axis=0) - offset) @ rotation
    return _sphericity(cavity, valve_centre, float(np.prod(segmentation.GetSpacing())))


def _points_mm(mask: sitk.Image) -> np.ndarray:
    indices = np.argwhere(sitk.GetArrayViewFromImage(mask))[:, ::-1]
    direction = np.array(mask.GetDirection()).reshape(3, 3)
    return (indices * np.array(mask.GetSpacing())) @ direction.T + np.array(mask.GetOrigin())


def _sphericity(cavity: np.ndarray, valve_centre: np.ndarray, voxel_volume_mm3: float) -> dict[str, float]:
    z = cavity[:, 2]
    long_axis = float(valve_centre[2] - z.min())
    n_slabs = int(long_axis // SLAB_MM)
    slab = ((z - z.min()) // SLAB_MM).astype(int)
    areas = np.bincount(slab[slab < n_slabs], minlength=n_slabs) * voxel_volume_mm3 / SLAB_MM
    diameters = 2 * np.sqrt(areas / np.pi)
    d_max = float(diameters.max())
    d_mid = float(np.interp(long_axis / 2, (np.arange(n_slabs) + 0.5) * SLAB_MM, diameters))
    edv_mm3 = len(cavity) * voxel_volume_mm3
    si_vol = edv_mm3 / (np.pi / 6 * long_axis**3)
    return {
        "edv_ml": edv_mm3 / 1000,
        "long_axis_mm": long_axis,
        "d_max_mm": d_max,
        "d_mid_mm": d_mid,
        "si_dl": d_max / long_axis,
        "si_dl_mid": d_mid / long_axis,
        "si_vol": si_vol,
        "ellipsoid_ratio": si_vol / (d_max / long_axis) ** 2,
    }

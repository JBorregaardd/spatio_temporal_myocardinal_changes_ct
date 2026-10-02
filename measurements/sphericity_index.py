"""Left-ventricular sphericity index from the TotalSegmentator heart-chambers segmentation."""
import numpy as np
import SimpleITK as sitk

LA_ID = 2
LV_ID = 3
SLAB_MM = 1.0


def sphericity_index(segmentation_path: str) -> dict[str, float]:
    """Sphericity of the LV cavity (TotalSegmentator label 3) of one patient.

    L is the distance from the mitral valve centre (centre of the LA/LV-cavity interface) to the endocardial apex,
    the cavity voxel farthest from it (Lang et al. 2015). The primary index is the 3D sphericity index ``si_vol``,
    EDV over the volume of a sphere of diameter L (Mannaerts et al. 2004). ``d_mid_mm`` is the area-equivalent
    diameter ``2 * sqrt(A / pi)`` of the ``SLAB_MM`` thick cavity slab perpendicular to L halfway along it. Only the
    largest connected component of the cavity is used.

    Args:
        segmentation_path: TotalSegmentator heart-chambers segmentation.

    Returns:
        ``edv_ml``, ``long_axis_mm``, ``d_mid_mm``, ``si_vol`` (EDV / (pi/6 * L^3)) and ``si_dl_mid`` (d_mid / L).
    """
    segmentation = sitk.ReadImage(segmentation_path)
    lv = sitk.RelabelComponent(sitk.ConnectedComponent(segmentation == LV_ID)) == 1
    valve = sitk.BinaryDilate(segmentation == LA_ID, [2, 2, 2]) & sitk.BinaryDilate(lv, [2, 2, 2])
    return _sphericity(_points_mm(lv), _points_mm(valve).mean(axis=0), float(np.prod(segmentation.GetSpacing())))


def _points_mm(mask: sitk.Image) -> np.ndarray:
    indices = np.argwhere(sitk.GetArrayViewFromImage(mask))[:, ::-1]
    direction = np.array(mask.GetDirection()).reshape(3, 3)
    return (indices * np.array(mask.GetSpacing())) @ direction.T + np.array(mask.GetOrigin())


def _sphericity(cavity: np.ndarray, valve_centre: np.ndarray, voxel_volume_mm3: float) -> dict[str, float]:
    distance = np.linalg.norm(cavity - valve_centre, axis=1)
    apex = cavity[distance.argmax()]
    long_axis = float(distance.max())
    height = (cavity - apex) @ ((valve_centre - apex) / long_axis)
    n_slabs = int(long_axis // SLAB_MM)
    slab = (height // SLAB_MM).astype(int)
    areas = np.bincount(slab[slab < n_slabs], minlength=n_slabs) * voxel_volume_mm3 / SLAB_MM
    centres = (np.arange(n_slabs) + 0.5) * SLAB_MM
    d_mid = float(np.interp(long_axis / 2, centres, 2 * np.sqrt(areas / np.pi)))
    edv_mm3 = len(cavity) * voxel_volume_mm3
    return {
        "edv_ml": edv_mm3 / 1000,
        "long_axis_mm": long_axis,
        "d_mid_mm": d_mid,
        "si_vol": edv_mm3 / (np.pi / 6 * long_axis**3),
        "si_dl_mid": d_mid / long_axis,
    }

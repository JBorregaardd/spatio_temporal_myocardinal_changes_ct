"""Left-ventricular wall thickness per AHA segment from the outputs of the LVM thickness pipeline."""
import json
import os

import numpy as np

SEGMENTS = range(1, 18)
WALL_SEGMENTS = range(1, 17)
SEPTAL_SEGMENTS = (2, 3, 8, 9)
LATERAL_SEGMENTS = (5, 6, 11, 12)


def load_thickness(patient_dir: str) -> dict[int, np.ndarray]:
    """Wall thickness measurements per segment of one patient.

    Args:
        patient_dir: The patient's output folder from the LVM thickness pipeline.

    Returns:
        Segment number to its thickness measurements in mm.
    """
    with open(os.path.join(patient_dir, "thickness.json")) as f:
        return {int(float(segment)): np.array(values) for segment, values in json.load(f).items()}


def thickness_per_segment(patient_dir: str) -> dict[str, float]:
    """Median, mean and standard deviation of the wall thickness in each AHA segment of one patient.


    Args:
        patient_dir: The patient's output folder from the LVM thickness pipeline.

    Returns:
        ``segment_<n>_median_mm``, ``segment_<n>_mean_mm`` and ``segment_<n>_std_mm`` for segments 1-17.
    """
    thickness = load_thickness(patient_dir)

    result = {}
    for segment in SEGMENTS:
        values = thickness.get(segment, np.array([]))
        has_values = len(values) > 0
        result[f"segment_{segment}_median_mm"] = float(np.median(values)) if has_values else np.nan
        result[f"segment_{segment}_mean_mm"] = float(np.mean(values)) if has_values else np.nan
        result[f"segment_{segment}_std_mm"] = float(np.std(values)) if has_values else np.nan
    return result


def madwt(patient_dir: str) -> float:
    """Mean absolute deviation of the maximum segmental wall thickness (MadWT) of one patient, after Hesse et al.

    In each of AHA segments 1-16 the maximum thickness Y_i is taken, and MadWT is sum(|Y_i - mean(Y)|) / 16. Unlike
    Hesse et al., no outliers are removed, to match the rest of the analysis. The apex (segment 17) is not included.

    Args:
        patient_dir: The patient's output folder from the LVM thickness pipeline.

    Returns:
        MadWT in mm, NaN if any of segments 1-16 has no measurements.
    """
    thickness = load_thickness(patient_dir)
    if any(len(thickness.get(segment, [])) == 0 for segment in WALL_SEGMENTS):
        return np.nan
    maxima = np.array([thickness[segment].max() for segment in WALL_SEGMENTS])
    return float(np.mean(np.abs(maxima - maxima.mean())))


def septal_lateral_ratio(patient_dir: str) -> float:
    """Septal-to-lateral ratio (SLR) of the maximum wall thickness of one patient, after Shiwani et al.

    SLR is the maximum thickness in the basal and mid septal segments (AHA 2, 3, 8, 9) divided by the maximum
    thickness in the basal and mid lateral segments (AHA 5, 6, 11, 12).

    Args:
        patient_dir: The patient's output folder from the LVM thickness pipeline.

    Returns:
        SLR, NaN if the septal or lateral segments have no measurements.
    """
    thickness = load_thickness(patient_dir)
    septal = np.concatenate([thickness.get(segment, np.array([])) for segment in SEPTAL_SEGMENTS])
    lateral = np.concatenate([thickness.get(segment, np.array([])) for segment in LATERAL_SEGMENTS])
    if len(septal) == 0 or len(lateral) == 0:
        return np.nan
    return float(septal.max() / lateral.max())

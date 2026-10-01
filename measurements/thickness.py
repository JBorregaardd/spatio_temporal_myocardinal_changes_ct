"""Left-ventricular wall thickness per AHA segment from the outputs of the LVM thickness pipeline."""
import json
import os

import numpy as np

SEGMENTS = range(1, 18)


def thickness_per_segment(patient_dir: str) -> dict[str, float]:
    """Median, mean and standard deviation of the wall thickness in each AHA segment of one patient.


    Args:
        patient_dir: The patient's output folder from the LVM thickness pipeline.

    Returns:
        ``segment_<n>_median_mm``, ``segment_<n>_mean_mm`` and ``segment_<n>_std_mm`` for segments 1-17.
    """
    with open(os.path.join(patient_dir, "thickness.json")) as f:
        thickness = {int(float(segment)): np.array(values) for segment, values in json.load(f).items()}

    result = {}
    for segment in SEGMENTS:
        values = thickness.get(segment, np.array([]))
        has_values = len(values) > 0
        result[f"segment_{segment}_median_mm"] = float(np.median(values)) if has_values else np.nan
        result[f"segment_{segment}_mean_mm"] = float(np.mean(values)) if has_values else np.nan
        result[f"segment_{segment}_std_mm"] = float(np.std(values)) if has_values else np.nan
    return result

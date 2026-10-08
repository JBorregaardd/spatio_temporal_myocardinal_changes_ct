#################################################################################################################################################
# This script measures the LV myocardium per AHA segment for all patients in the dataset and makes one table per segment. Each table has
# a row for n and for the median thickness, mean thickness, volume (mL) and mass (g) of the patients, given as median ± std over patients,
# and a column per group of patients. For now there is one group, "All"; the hypertension groups and a BSA metric will be added later.
# It reads the pipeline run folder <output>/<DATASET>_<AORTA_EXCLUSION_MM>mm and saves the tables in <df>/segments/segment_<n>.csv.
# e.g. uv run measurements/main_segmentwise_df.py
#################################################################################################################################################

import os
import sys
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "measurements"))

from thickness import SEGMENTS, thickness_per_segment  
from volume import segment_volume  

MYOCARDIAL_DENSITY_G_PER_ML = 1.055  # Fuchs et al. 2016
DATASET = "ImageCAS_1-200"
AORTA_EXCLUSION_MM = 2.5
#WORKERS = max(1, (os.cpu_count() or 2) - 1)
WORKERS = 8
ROW_LABELS = {
    "median_wall_thickness": "Median thickness (mm)",
    "mean_wall_thickness": "Mean thickness (mm)",
    "volume_ml": "Volume (mL)",
    "mass_g": "Mass (g)",
}


def measure_patient(patient_dir: str) -> list[dict] | None:
    """Wall thickness, volume and mass of each AHA segment of one patient.

    Args:
        patient_dir: The patient's output folder from the LVM thickness pipeline.

    Returns:
        One row per segment with ``patient_id``, ``segment`` and the columns in ``ROW_LABELS``,
        or ``None`` if the patient could not be measured.
    """
    patient_id = os.path.basename(patient_dir)
    try:
        thickness = thickness_per_segment(patient_dir)
        volumes = segment_volume(os.path.join(patient_dir, "segmentations", "lv17", "lv17.nii.gz"))
        rows = []
        for segment in SEGMENTS:
            volume_ml = volumes[f"segment_{segment}_volume_ml"]
            rows.append({
                "patient_id": patient_id,
                "segment": segment,
                "median_wall_thickness": thickness[f"segment_{segment}_median_mm"],
                "mean_wall_thickness": thickness[f"segment_{segment}_mean_mm"],
                "volume_ml": volume_ml,
                "mass_g": MYOCARDIAL_DENSITY_G_PER_ML * volume_ml,
            })
        print(f"Patient {patient_id} done")
        return rows
    except Exception as error:
        print(f"Patient {patient_id} failed: {error}")
        return None


def segment_table(groups: dict[str, pd.DataFrame], segment: int) -> pd.DataFrame:
    """Make the table of one segment: n and each measurement as median ± std over the patients of each group.

    Args:
        groups: Group name to that group's rows, one row per patient and segment with the columns in ``ROW_LABELS``.
        segment: AHA segment (1-17).

    Returns:
        A table with one row for n and per measurement, and one column per group.
    """
    columns = {}
    for name, data in groups.items():
        values = data[data["segment"] == segment]
        column = {"n": str(values["patient_id"].nunique())}
        for key, label in ROW_LABELS.items():
            column[label] = f"{values[key].median():.2f} ± {values[key].std():.2f}"
        columns[name] = column
    return pd.DataFrame(columns).rename_axis(f"Segment {segment}")


if __name__ == "__main__":
    load_dotenv(os.path.join(REPO_ROOT, ".env"))
    root = os.environ["PROJECT_ROOT"]
    run_folder = os.path.join(root, "output", f"{DATASET}_{AORTA_EXCLUSION_MM:g}mm")

    patient_ids = sorted(
        (name for name in os.listdir(run_folder) if os.path.isfile(os.path.join(run_folder, name, "done.json"))),
        key=lambda pid: int(pid) if pid.isdigit() else float("inf"),
    )

    patient_dirs = [os.path.join(run_folder, pid) for pid in patient_ids]
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        rows = [row for rows in executor.map(measure_patient, patient_dirs) if rows is not None for row in rows]

    data = pd.DataFrame(rows)
    groups = {"All": data}
    output_folder = os.path.join(root, "df", "segments")
    os.makedirs(output_folder, exist_ok=True)

    for segment in SEGMENTS:
        table = segment_table(groups, segment)
        print(f"\n{table}")
        table.to_csv(os.path.join(output_folder, f"segment_{segment}.csv"), encoding="utf-8-sig")
    print(f"\nSegment tables saved to: {output_folder}")

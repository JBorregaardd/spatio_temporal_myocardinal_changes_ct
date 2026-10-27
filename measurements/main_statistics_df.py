#################################################################################################################################################
# This script creates two csv files with statistics per segment and per patient, from the outputs of the LVM thickness pipeline.
# It reads the run folder <output>/<dataset>_<aorta-exclusion-mm> written by main_pipeline.py and saves the results to CSV files in
# <run folder>/statistics. Thickness and segment volumes come from the run folder, and the LV myocardial volume and mass and the sphericity
# index from the TotalSegmentator heart segmentation. Height, weight and BSA come from the metadata file via BSA.py.
# e.g. uv run measurements/main_statistics_df.py --dataset CGPS10 --aorta-exclusion-mm 0
#################################################################################################################################################

# Demographics data is only available for the CGPS dataset, so when --dataset ImageCAS_1-200 is used, the demographics columns will be empty.

import os
import sys
import json
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "measurements"))
sys.path.insert(0, os.path.join(REPO_ROOT, "LVM_thickness"))

from BSA import body_surface_area, gender_and_age, load_metadata_file, load_translationtable_file  # noqa: E402
from pipeline import AORTA_EXCLUSION_MM, load_environment, run_output_dir  # noqa: E402
from sphericity_index import sphericity_index  # noqa: E402
from thickness import SEGMENTS, madwt, thickness_per_segment  # noqa: E402
from volume import myocardium_volume, segment_volume  # noqa: E402

DATASETS_WITHOUT_DEMOGRAPHICS = {"ImageCAS_1-200"}
FINISHED_MARKER = "done.json"
STATISTICS_FOLDER = "statistics"
WALL_THICKNESS_SEGMENTS = range(1, 17)
WORKERS = max(1, (os.cpu_count() or 2) - 1)

SEGMENT_FIELDS = [
    "patient_id",
    "segment",
    "median_thickness",
    "mean_thickness",
    "std_thickness",
    "volume_ml",
    "mass_g",
    "BSA",
    "SLVDVI",
]

PATIENT_FIELDS = [
    "patient_id",
    "sex",
    "age",
    "height_cm",
    "weight_kg",
    "bsa_m2",
    "lvm_max_thickness_mm",
    "lvm_average_thickness_mm",
    "MadWT_mm",
    "lvm_volume_ml",
    "lvm_g",
    "Sphericity_index",
    "LVDVI",
]


def load_demographics(dataset: str) -> pd.DataFrame:
    """Sex, age, height, weight and BSA of each patient, via ``BSA.py``.

    Height, weight and BSA come from the metadata file, and sex (0 = man, 1 = woman) and age from the translation table
    at the examination of the dataset (``CGPS10``: 10-year follow-up, ``CGPS1``: baseline).

    Args:
        dataset: Dataset name, ``CGPS10`` or ``CGPS1``.

    Returns:
        One row per patient with ``patient_id``, ``sex``, ``age``, ``height_cm``, ``weight_kg`` and ``bsa_m2``.
    """
    patients = body_surface_area(pd.read_csv(load_metadata_file()))
    sex_and_age = gender_and_age(pd.read_csv(load_translationtable_file(), dtype=str), dataset)
    patients = patients.merge(sex_and_age, on="patient_id", how="left")
    return pd.DataFrame(
        {
            "patient_id": patients["patient_id"],
            "sex": patients["sex"],
            "age": patients["age"],
            "height_cm": patients["height_cm"],
            "weight_kg": patients["PatientWeight"],
            "bsa_m2": patients["BSA"],
        }
    )


def average_thickness(patient_dir: str) -> float:
    """Mean of all wall thickness measurements in AHA segments 1-16 of one patient.

    The apex (segment 17) is left out, as its thickness differs markedly from the other segments.

    Args:
        patient_dir: The patient's output folder from the LVM thickness pipeline.

    Returns:
        Average thickness in mm, NaN if there are no measurements.
    """
    with open(os.path.join(patient_dir, "thickness.json")) as f:
        thickness = {int(float(segment)): values for segment, values in json.load(f).items()}
    values = [value for segment in WALL_THICKNESS_SEGMENTS for value in thickness.get(segment, [])]
    return float(np.mean(values)) if values else np.nan


def measure_patient(patient_id: str, patient_dir: str, segmentation_path: str) -> tuple[list[dict], dict] | None:
    """Per-segment and patient-level measurements of one patient, without the demographics.

    Args:
        patient_id: Dataset id.
        patient_dir: The patient's output folder from the LVM thickness pipeline.
        segmentation_path: TotalSegmentator heart-chambers segmentation of the patient.

    Returns:
        Tuple of (one row per segment, one patient row), or ``None`` if the patient could not be measured.
    """
    try:
        thickness = thickness_per_segment(patient_dir)
        volumes = segment_volume(os.path.join(patient_dir, "segmentations", "lv17", "lv17.nii.gz"))
        segment_rows = [
            {
                "patient_id": patient_id,
                "segment": segment,
                "median_thickness": thickness[f"segment_{segment}_median_mm"],
                "mean_thickness": thickness[f"segment_{segment}_mean_mm"],
                "std_thickness": thickness[f"segment_{segment}_std_mm"],
                "volume_ml": volumes[f"segment_{segment}_volume_ml"],
                "mass_g": volumes[f"segment_{segment}_mass_g"],
            }
            for segment in SEGMENTS
        ]

        sphericity = sphericity_index(segmentation_path)
        myocardium = myocardium_volume(segmentation_path)
        patient_row = {
            "patient_id": patient_id,
            "lvm_max_thickness_mm": np.nanmax(
                [thickness[f"segment_{segment}_median_mm"] for segment in WALL_THICKNESS_SEGMENTS]
            ),
            "lvm_average_thickness_mm": average_thickness(patient_dir),
            "MadWT_mm": madwt(patient_dir),
            "lvm_volume_ml": myocardium["lvm_volume_ml"],
            "lvm_g": myocardium["lvm_mass_g"],
            "Sphericity_index": sphericity["si_vol"],
        }
        print(f"Patient {patient_id} done")
        return segment_rows, patient_row
    except Exception as error:
        print(f"Patient {patient_id} failed: {error}")
        return None


if __name__ == "__main__":
    root, data = load_environment()

    parser = argparse.ArgumentParser(description="Create the per-segment and per-patient statistics tables.")
    parser.add_argument(
        "--dataset", required=True, help="Dataset name of the pipeline run to analyse, as given to main_pipeline.py."
    )
    parser.add_argument(
        "--aorta-exclusion-mm",
        type=float,
        default=AORTA_EXCLUSION_MM,
        help=f"Aortic exclusion distance of the run to analyse (default: {AORTA_EXCLUSION_MM}).",
    )
    parser.add_argument(
        "--output-dir", default=None, help="Parent of the run folders (default: <PROJECT_ROOT>/output)."
    )
    parser.add_argument(
        "--workers", type=int, default=WORKERS, help=f"Patients analysed in parallel (default: {WORKERS})."
    )
    args = parser.parse_args()

    run_folder = run_output_dir(root, args.dataset, args.aorta_exclusion_mm, args.output_dir)
    if not os.path.isdir(run_folder):
        raise SystemExit(f"Run folder not found: {run_folder}")

    patient_ids = sorted(
        (name for name in os.listdir(run_folder) if os.path.isfile(os.path.join(run_folder, name, FINISHED_MARKER))),
        key=lambda pid: int(pid) if pid.isdigit() else float("inf"),
    )
    if not patient_ids:
        raise SystemExit(f"No finished patients ({FINISHED_MARKER}) found in {run_folder}.")

    patient_dirs = [os.path.join(run_folder, pid) for pid in patient_ids]
    segmentation_paths = [data.segmentation_path(pid) for pid in patient_ids]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = [
            result
            for result in executor.map(measure_patient, patient_ids, patient_dirs, segmentation_paths)
            if result is not None
        ]
    if not results:
        raise SystemExit("No patients could be measured.")

    segments = pd.DataFrame([row for segment_rows, _ in results for row in segment_rows])
    patients = pd.DataFrame([patient_row for _, patient_row in results])

    if args.dataset in DATASETS_WITHOUT_DEMOGRAPHICS:
        demographics = pd.DataFrame(columns=["patient_id", "sex", "age", "height_cm", "weight_kg", "bsa_m2"])
    else:
        demographics = load_demographics(args.dataset)
    patients = patients.merge(demographics, on="patient_id", how="left")
    patients["LVDVI"] = patients["lvm_volume_ml"] / patients["bsa_m2"].astype(float)

    segments = segments.merge(patients[["patient_id", "bsa_m2"]].rename(columns={"bsa_m2": "BSA"}), on="patient_id")
    segments["SLVDVI"] = segments["volume_ml"] / segments["BSA"].astype(float)

    statistics_folder = os.path.join(run_folder, STATISTICS_FOLDER)
    os.makedirs(statistics_folder, exist_ok=True)
    segment_csv = os.path.join(statistics_folder, "statistics_thickness_per_segment.csv")
    patient_csv = os.path.join(statistics_folder, "statistics_patient_summary.csv")
    segments[SEGMENT_FIELDS].to_csv(segment_csv, index=False)
    patients[PATIENT_FIELDS].to_csv(patient_csv, index=False)
    print(f"Results for {len(patients)} of {len(patient_ids)} patient(s) saved to:")
    print(f"  {segment_csv}")
    print(f"  {patient_csv}")

#################################################################################################################################################
# This script classifies each CGPS1 (baseline) patient as hypertrophic (HT) or not from the average myocardial wall thickness. A patient
# has HT when lvm_average_thickness_mm (the mean of all thickness measurements in AHA segments 1-16) is above the threshold, by default
# 15 mm. Only CGPS1 is classified; the CGPS10 subjects get the class of their CGPS1 scan via the links in spaghetti_plot.py.
# It reads <run folder>/statistics/statistics_patient_summary.csv, e.g. output/CGPS1_0mm/statistics/..., written by
# main_statistics_df.py, and saves one row per patient in <PROJECT_ROOT>/df/repeated_measures/CGPS1_HT.csv.
# e.g. uv run repeated_measures/ht_classifier.py --threshold-mm 15
#################################################################################################################################################

import os
import sys
import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "LVM_thickness"))

from pipeline import AORTA_EXCLUSION_MM, load_environment, run_output_dir  # noqa: E402

DATASET = "CGPS1"
THRESHOLD_MM = 15.0
THICKNESS_COLUMN = "lvm_average_thickness_mm"
PATIENT_SUMMARY = os.path.join("statistics", "statistics_patient_summary.csv")
HT_FOLDER = os.path.join("df", "repeated_measures")
HT_FILE = "CGPS1_HT.csv"


def classify_ht(patients: pd.DataFrame, threshold_mm: float = THRESHOLD_MM) -> pd.DataFrame:
    """Classify each patient as HT when the average myocardial wall thickness is above the threshold.

    Args:
        patients: One row per patient with ``patient_id`` and ``lvm_average_thickness_mm``.
        threshold_mm: Thickness in mm the average must be above for HT.

    Returns:
        One row per patient with ``patient_id``, ``lvm_average_thickness_mm`` and ``HT`` (``True``/``False``,
        missing if the patient has no average thickness).
    """
    table = patients[["patient_id", THICKNESS_COLUMN]].drop_duplicates("patient_id").reset_index(drop=True)
    table["HT"] = (table[THICKNESS_COLUMN] > threshold_mm).astype("boolean")
    table.loc[table[THICKNESS_COLUMN].isna(), "HT"] = pd.NA
    return table


if __name__ == "__main__":
    root, _ = load_environment()

    parser = argparse.ArgumentParser(description="Classify the CGPS1 patients as HT from the average wall thickness.")
    parser.add_argument("--threshold-mm", type=float, default=THRESHOLD_MM,
                        help=f"HT when the average thickness is above this (default: {THRESHOLD_MM}).")
    parser.add_argument("--aorta-exclusion-mm", type=float, default=AORTA_EXCLUSION_MM,
                        help=f"Aortic exclusion distance of the run (default: {AORTA_EXCLUSION_MM}).")
    parser.add_argument("--output-dir", default=None,
                        help="Parent of the run folders (default: <PROJECT_ROOT>/output).")
    args = parser.parse_args()

    summary_csv = Path(run_output_dir(root, DATASET, args.aorta_exclusion_mm, args.output_dir)) / PATIENT_SUMMARY
    if not summary_csv.is_file():
        raise SystemExit(f"Patient summary not found: {summary_csv}\nRun main_statistics_df.py for {DATASET} first.")
    print(f"{DATASET}: {summary_csv}")

    table = classify_ht(pd.read_csv(summary_csv, dtype={"patient_id": str}), args.threshold_mm)
    counts = table["HT"].value_counts()
    print(f"HT: {counts.get(True, 0)}, no HT: {counts.get(False, 0)}, unknown: {table['HT'].isna().sum()}")

    output_folder = Path(root) / HT_FOLDER
    output_folder.mkdir(parents=True, exist_ok=True)
    table_path = output_folder / HT_FILE
    table.to_csv(table_path, index=False)
    print(f"Table saved to: {table_path}")

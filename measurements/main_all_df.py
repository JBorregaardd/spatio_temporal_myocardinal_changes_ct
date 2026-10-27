#################################################################################################################################################
# This script makes a table of the whole LV per group of patients from the patient summary written by main_statistics_df.py. The table has
# a row for n and for each measurement given as mean ± sd over the patients, and a column per group of patients. For now there is one
# group, "All"; the hypertension groups will be added later.
# It reads <run folder>/statistics/statistics_patient_summary.csv and saves the table in
# <df>/<dataset>_<aorta-exclusion-mm>mm_n<number of patients>/df_myocardium.csv. With --sex male or --sex female only those patients are
# included and the table is saved as df_myocardium_<sex>.csv in the same folder.
# Run main_statistics_df.py for the same --dataset and --aorta-exclusion-mm first.
# e.g. uv run measurements/main_all_df.py --dataset CGPS10 --aorta-exclusion-mm 0 --sex female
#################################################################################################################################################

import os
import sys
import argparse
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "LVM_thickness"))

from pipeline import AORTA_EXCLUSION_MM, load_environment, run_output_dir, table_output_dir  # noqa: E402

PATIENT_SUMMARY = os.path.join("statistics", "statistics_patient_summary.csv")
FEMALE_LABEL = "Female (%)"
SEX_CODES = {"male": 0, "female": 1}
SEX_CHOICES = ["all", *SEX_CODES]
ROW_LABELS = {
    "age": "Age (years)",
    "lvm_volume_ml": "Volume (mL)",
    "lvm_g": "Mass (g)",
    "LVDVI": "LVDVI (mL/m^2)",
    "Sphericity_index": "Volumetric sphericity index",
    "lvm_average_thickness_mm": "Average thickness (mm)",
    "lvm_max_thickness_mm": "Max thickness (mm)",
    "MadWT_mm": "MadWT (mm)",
    "height_cm": "Height (cm)",
    "weight_kg": "Weight (kg)",
    "bsa_m2": "BSA (m^2)",
}


def select_sex(data: pd.DataFrame, sex: str) -> pd.DataFrame:
    """Keep the rows of the patients of one sex.

    Args:
        data: Rows with a ``sex`` column (0 = man, 1 = woman).
        sex: ``"all"``, ``"male"`` or ``"female"``.

    Returns:
        The rows of that sex, or all rows for ``"all"``.
    """
    if sex == "all":
        return data
    return data[data["sex"] == SEX_CODES[sex]]


def sex_suffix(sex: str) -> str:
    """File name suffix of the tables of one sex: empty for ``"all"``, else e.g. ``"_female"``."""
    return "" if sex == "all" else f"_{sex}"


def all_table(groups: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Make the table of the whole LV: n, the share of women and each measurement as mean ± sd per group of patients.

    The share of women is the percentage of patients with ``sex`` 1 among the patients with a known sex. Rows without
    any values in a group, e.g. BSA for a dataset without demographics, are left empty.

    Args:
        groups: Group name to that group's rows, one row per patient with ``patient_id``, ``sex`` (0 = man,
            1 = woman) and the columns in ``ROW_LABELS``.

    Returns:
        A table with one row for n and per measurement, and one column per group.
    """
    columns = {}
    for name, data in groups.items():
        column = {"n": str(data["patient_id"].nunique())}
        sex = data["sex"].dropna()
        column[FEMALE_LABEL] = f"{(sex == 1).mean() * 100:.2f}" if len(sex) else ""
        for key, label in ROW_LABELS.items():
            values = data[key]
            column[label] = f"{values.mean():.2f} ± {values.std():.2f}" if values.notna().any() else ""
        columns[name] = column
    return pd.DataFrame(columns).rename_axis("Whole LV")


if __name__ == "__main__":
    root, _ = load_environment()

    parser = argparse.ArgumentParser(description="Create the whole-LV table from the patient summary statistics.")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name of the pipeline run, as given to main_statistics_df.py.")
    parser.add_argument("--aorta-exclusion-mm", type=float, default=AORTA_EXCLUSION_MM,
                        help=f"Aortic exclusion distance of the run (default: {AORTA_EXCLUSION_MM}).")
    parser.add_argument("--output-dir", default=None,
                        help="Parent of the run folders (default: <PROJECT_ROOT>/output).")
    parser.add_argument("--sex", choices=SEX_CHOICES, default="all",
                        help="Only include the patients of this sex (default: all).")
    args = parser.parse_args()

    summary_csv = os.path.join(
        run_output_dir(root, args.dataset, args.aorta_exclusion_mm, args.output_dir), PATIENT_SUMMARY
    )
    if not os.path.isfile(summary_csv):
        raise SystemExit(f"Patient summary not found: {summary_csv}\nRun main_statistics_df.py first.")

    data = pd.read_csv(summary_csv)
    output_folder = table_output_dir(root, args.dataset, args.aorta_exclusion_mm, data["patient_id"].nunique())
    data = select_sex(data, args.sex)
    if data.empty:
        raise SystemExit(f"No patients with sex '{args.sex}' in {summary_csv}.")
    groups = {args.sex.capitalize(): data}
    os.makedirs(output_folder, exist_ok=True)

    table = all_table(groups)
    print(table)
    table_path = os.path.join(output_folder, f"df_myocardium{sex_suffix(args.sex)}.csv")
    table.to_csv(table_path, encoding="utf-8-sig")
    print(f"Table saved to: {table_path}")

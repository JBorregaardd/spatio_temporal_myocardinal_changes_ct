#################################################################################################################################################
# This script computes the change (delta = CGPS10 - CGPS1) of every feature of the patient summary for the subjects scanned in both CGPS1
# (baseline) and CGPS10 (10-year follow-up). The subjects are linked by <df>/CGPS1_CGPS10_links.csv, made by comparative_analysis/compare.py,
# and the values are read from output/<dataset>_<mm>mm/statistics/statistics_patient_summary.csv of each dataset.
# It saves one row per subject in <PROJECT_ROOT>/df/repeated_measures/deltas.csv with the ids, sex (from CGPS1, else CGPS10) and the
# delta of each feature, and the n and mean ± sd of each delta in delta_summary.csv in the same folder.
# A subject is kept when it is in both patient summaries; a delta is missing when the feature is missing at either examination.
# e.g. uv run repeated_measures/deltas.py
#################################################################################################################################################

import os
import sys
import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "LVM_thickness"))
sys.path.insert(0, os.path.join(REPO_ROOT, "repeated_measures"))

from ht_classifier import HT_FOLDER  # noqa: E402
from pipeline import AORTA_EXCLUSION_MM, load_environment  # noqa: E402
from spaghetti_plot import DATASETS, LINKS_FILE, patient_summary_path, read_ids  # noqa: E402

FEATURES = [
    "age",
    "height_cm",
    "weight_kg",
    "bsa_m2",
    "lvm_max_thickness_mm",
    "lvm_average_thickness_mm",
    "MadWT_mm",
    "SLR",
    "lvm_volume_ml",
    "lvm_g",
    "lv_blood_pool_volume_ml",
    "Sphericity_index",
    "LVDVI",
]
DELTAS_FILE = "deltas.csv"
SUMMARY_FILE = "delta_summary.csv"


def delta_table(summaries: dict[str, pd.DataFrame], links: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """The change of each feature, one row per linked subject.

    Args:
        summaries: Dataset name (``CGPS1``, ``CGPS10``) to its patient summary, with ``patient_id``, ``sex`` and the
            feature columns.
        links: One row per subject with the patient ids in the columns ``CGPS1`` and ``CGPS10``.
        features: Feature columns to compute the change of.

    Returns:
        One row per subject in both patient summaries with ``CGPS1``, ``CGPS10`` (the ids), ``sex`` and
        ``delta_<feature>`` (CGPS10 - CGPS1) per feature.
    """
    table = links[list(DATASETS)].drop_duplicates()
    for dataset, summary in summaries.items():
        columns = ["patient_id", "sex", *features]
        values = summary[columns].drop_duplicates("patient_id")
        values = values.rename(columns={column: f"{column}_{dataset}" for column in columns if column != "patient_id"})
        table = table.merge(values, left_on=dataset, right_on="patient_id", how="inner").drop(columns="patient_id")
    table["sex"] = table["sex_CGPS1"].astype("Float64").fillna(table["sex_CGPS10"].astype("Float64"))
    columns = ["CGPS1", "CGPS10", "sex"]
    for feature in features:
        table[f"delta_{feature}"] = table[f"{feature}_CGPS10"] - table[f"{feature}_CGPS1"]
        columns.append(f"delta_{feature}")
    return table[columns].reset_index(drop=True)


def delta_summary(table: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Number of subjects and mean ± sd of the change of each feature.

    Args:
        table: Table from ``delta_table``.
        features: Feature columns of the table.

    Returns:
        One row per feature with ``n``, ``mean_delta``, ``sd_delta`` and ``mean ± sd``.
    """
    rows = []
    for feature in features:
        delta = table[f"delta_{feature}"].dropna()
        rows.append(
            {
                "feature": feature,
                "n": len(delta),
                "mean_delta": delta.mean(),
                "sd_delta": delta.std(),
                "mean ± sd": f"{delta.mean():.2f} ± {delta.std():.2f}" if len(delta) else "",
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    root, _ = load_environment()

    parser = argparse.ArgumentParser(description="Compute the change of each feature from CGPS1 to CGPS10 per subject.")
    parser.add_argument("--links", default=None, help=f"Links table (default: <PROJECT_ROOT>/df/{LINKS_FILE}).")
    parser.add_argument("--aorta-exclusion-mm", type=float, default=AORTA_EXCLUSION_MM,
                        help=f"Aortic exclusion distance of the runs (default: {AORTA_EXCLUSION_MM}).")
    parser.add_argument("--output-dir", default=None,
                        help="Parent of the run folders (default: <PROJECT_ROOT>/output).")
    args = parser.parse_args()

    links_path = Path(args.links) if args.links else Path(root) / "df" / LINKS_FILE
    if not links_path.is_file():
        raise SystemExit(f"Links table not found: {links_path}\nRun comparative_analysis/compare.py first.")
    links = read_ids(links_path, list(DATASETS))

    summaries = {}
    for dataset in DATASETS:
        summary_path = patient_summary_path(root, dataset, args.aorta_exclusion_mm, args.output_dir)
        if not summary_path.is_file():
            raise SystemExit(
                f"Patient summary not found: {summary_path}\nRun main_statistics_df.py for {dataset} first."
            )
        print(f"{dataset}: {summary_path}")
        summaries[dataset] = read_ids(summary_path, ["patient_id"])
        missing = [column for column in ["sex", *FEATURES] if column not in summaries[dataset].columns]
        if missing:
            raise SystemExit(f"Columns {missing} not in {summary_path}.")

    table = delta_table(summaries, links, FEATURES)
    print(f"{len(table)} of {len(links)} linked subjects are in both patient summaries.")
    summary = delta_summary(table, FEATURES)
    print(summary[["feature", "n", "mean ± sd"]].to_string(index=False))

    output_folder = Path(root) / HT_FOLDER
    output_folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_folder / DELTAS_FILE, index=False)
    summary.to_csv(output_folder / SUMMARY_FILE, index=False, encoding="utf-8-sig")
    print(f"Tables saved to: {output_folder / DELTAS_FILE} and {output_folder / SUMMARY_FILE}")

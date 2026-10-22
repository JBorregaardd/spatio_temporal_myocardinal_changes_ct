#################################################################################################################################################
# This script makes a table linking the patient ids of the CGPS1 (baseline) and CGPS10 (10-year follow-up) datasets, so that the results
# of the two datasets can be compared per subject or overall.
# It reads the file in COMPARISON_DATA_FILE in the .env, where each row links a baseline and a follow-up scan in the columns
# "Baseline NIFTI file" and "Follow-up NIFTI file", e.g.
#   Baseline NIFTI file  = CGPS-1-august-2026_00001_SERIES0005
#   Follow-up NIFTI file = CGPS-10y-august-2026_0730_SERIES0004_volume_1
# and saves one row per subject with the columns CGPS1 and CGPS10, e.g. 00001 and 0730, in <PROJECT_ROOT>/df/CGPS1_CGPS10_links.csv.
# e.g. uv run comparative_analysis/compare.py
#################################################################################################################################################

import os
import re
import sys
import argparse
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "LVM_thickness"))

from pipeline import load_environment  # noqa: E402

DATASET_PREFIXES = {"CGPS1": "CGPS-1-august-2026", "CGPS10": "CGPS-10y-august-2026"}
LINK_COLUMNS = {"CGPS1": "Baseline NIFTI file", "CGPS10": "Follow-up NIFTI file"}
LINKS_FILE = "CGPS1_CGPS10_links.csv"


def load_comparison_file() -> str:
    """Path of the baseline/follow-up links file in ``COMPARISON_DATA_FILE`` from the repo-root .env or the environment."""
    from dotenv import load_dotenv

    load_dotenv(os.path.join(REPO_ROOT, ".env"))
    comparison_file = os.environ.get("COMPARISON_DATA_FILE")
    if not comparison_file:
        raise SystemExit(
            "COMPARISON_DATA_FILE is not set. Add it to the .env file in the repo root (see .env.example)."
        )
    return comparison_file


def extract_ids(values: pd.Series, prefix: str) -> pd.Series:
    """Patient ids in scan names of one dataset: the number after ``<prefix>_``, e.g. ``0730`` in
    ``CGPS-10y-august-2026_0730_SERIES0004_volume_1``.

    Args:
        values: Scan names, e.g. NIFTI file names or paths.
        prefix: Dataset prefix of the scan names, e.g. ``CGPS-10y-august-2026``.

    Returns:
        The patient ids as strings with the leading zeros kept, missing where a value has no id of the dataset.
    """
    return values.astype("string").str.extract(rf"{re.escape(prefix)}_(\d+)", expand=False)


def link_table(links: pd.DataFrame) -> pd.DataFrame:
    """Table linking the CGPS1 and CGPS10 patient ids of each subject.

    The CGPS1 ids come from the column ``Baseline NIFTI file`` and the CGPS10 ids from ``Follow-up NIFTI file``. Rows
    without an id of both datasets and duplicate pairs are dropped.

    Args:
        links: One row per subject with the columns in ``LINK_COLUMNS``.

    Returns:
        One row per subject with the patient ids in the columns ``CGPS1`` and ``CGPS10``.

    Raises:
        ValueError: If a column in ``LINK_COLUMNS`` is missing.
    """
    missing = [column for column in LINK_COLUMNS.values() if column not in links.columns]
    if missing:
        raise ValueError(f"Links file is missing the columns {missing}.")
    table = {
        dataset: extract_ids(links[column], DATASET_PREFIXES[dataset]) for dataset, column in LINK_COLUMNS.items()
    }
    return pd.DataFrame(table).dropna().drop_duplicates().reset_index(drop=True)


if __name__ == "__main__":
    root, _ = load_environment()

    parser = argparse.ArgumentParser(description="Create the table linking the CGPS1 and CGPS10 patient ids.")
    parser.add_argument("--output", default=None,
                        help=f"Path of the table (default: <PROJECT_ROOT>/df/{LINKS_FILE}).")
    args = parser.parse_args()

    comparison_csv = load_comparison_file()
    if not os.path.isfile(comparison_csv):
        raise SystemExit(f"Links file not found: {comparison_csv}")

    links = pd.read_csv(comparison_csv, dtype=str)
    table = link_table(links)
    print(table)
    print(f"{len(table)} of {len(links)} rows linked.")
    for dataset in DATASET_PREFIXES:
        repeated = table[dataset].duplicated(keep=False).sum()
        if repeated:
            print(f"[WARN] {repeated} rows share a {dataset} id with another row.")

    table_path = args.output or os.path.join(root, "df", LINKS_FILE)
    os.makedirs(os.path.dirname(os.path.abspath(table_path)), exist_ok=True)
    table.to_csv(table_path, index=False)
    print(f"Table saved to: {table_path}")

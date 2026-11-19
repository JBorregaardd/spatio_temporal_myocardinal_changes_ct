#################################################################################################################################################
# This script makes a table linking the patient ids of the CGPS1 (baseline) and CGPS10 (10-year follow-up) datasets, so that the results
# of the two datasets can be compared per subject or overall.
# It reads the file in COMPARISON_DATA_FILE in the .env, where each row with "Link_found (yes/no)" = yes and "Baseline scan type" = ED
# links a baseline and a follow-up scan in the columns "Baseline NIFTI file" and "Follow-up NIFTI file", e.g.
#   Baseline NIFTI file  = CGPS-1-august-2026_00001_SERIES0005
#   Follow-up NIFTI file = CGPS-10y-august-2026_0730_SERIES0004_volume_1
# and saves one row per subject with the columns CGPS1 and CGPS10, e.g. 00001 and 0730, in <PROJECT_ROOT>/df/CGPS1_CGPS10_links.csv.
# The point in the R-R interval of each scan is read from the "Baseline SeriesDescription" and "Follow-up SeriesDescription" columns
# of the same row, written either as a percentage (FULL 73% 1.00s Cardiac 0.5 CE) or as a delay and an R-R interval
# (HALF 648ms 0.90s PIQE 0.5 CE -> 0.648 / 0.90 = 72%), and saved in the columns CGPS1_rr_percent, CGPS10_rr_percent and
# rr_percent_difference (CGPS10 - CGPS1).
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
DESCRIPTION_COLUMNS = {
    "CGPS1": ("Baseline SeriesDescription", "Baseline SeriesDescribtion"),
    "CGPS10": ("Follow-up SeriesDescription", "Follow-up SeriesDescribtion"),
}
LINK_FOUND_COLUMN = "Link found (yes/no)"
SCAN_TYPE_COLUMN = "Baseline scan type"
FOLLOW_UP_SCAN_TYPE_COLUMN ="Follow-up scan type"
SCAN_TYPE = "ED"
LINKS_FILE = "CGPS1_CGPS10_links.csv"
PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
DELAY_MS = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*ms\b")
RR_INTERVAL_S = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*s\b")


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


def rr_percent(description: object) -> float:
    """Point in the R-R interval of a scan in percent, from its series description.

    Args:
        description: Series description, either with a percentage, e.g. ``FULL 73% 1.00s Cardiac 0.5 CE``, or with
            the delay after the R wave and the R-R interval, e.g. ``HALF 648ms 0.90s PIQE 0.5 CE`` (72%).

    Returns:
        The percentage, missing if the description has neither a percentage nor a delay and a non-zero R-R interval.
    """
    if not isinstance(description, str):
        return float("nan")
    percent = PERCENT.search(description)
    if percent:
        return float(percent.group(1))
    delay = DELAY_MS.search(description)
    interval = RR_INTERVAL_S.search(description)
    if delay is None or interval is None or float(interval.group(1)) == 0:
        return float("nan")
    return 100 * float(delay.group(1)) / (1000 * float(interval.group(1)))


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


def find_column(columns: pd.Index, names: tuple[str, ...]) -> str | None:
    """The first of ``names`` in ``columns``, or None if none of them is."""
    return next((name for name in names if name in columns), None)


def link_table(links: pd.DataFrame) -> pd.DataFrame:
    """Table linking the CGPS1 and CGPS10 patient ids of each subject, with the point in the R-R interval of each scan.

    Only rows with ``Link_found (yes/no)`` = yes and ``Baseline scan type`` = ED are used. The CGPS1 id and R-R percentage come from ``Baseline NIFTI
    file`` and ``Baseline SeriesDescription``, the CGPS10 ones from ``Follow-up NIFTI file`` and ``Follow-up
    SeriesDescription`` of the same row. Rows without an id of both datasets and duplicate id pairs are dropped.

    Args:
        links: Rows of the links file with the columns in ``LINK_COLUMNS``, ``DESCRIPTION_COLUMNS`` (either spelling)
            ``LINK_FOUND_COLUMN`` and ``SCAN_TYPE_COLUMN``.

    Returns:
        One row per subject with the patient ids in ``CGPS1`` and ``CGPS10``, ``CGPS1_rr_percent``,
        ``CGPS10_rr_percent`` and ``rr_percent_difference`` (CGPS10 - CGPS1).

    Raises:
        ValueError: If a needed column is missing.
    """
    links = links.rename(columns=lambda column: str(column).strip())
    descriptions = {dataset: find_column(links.columns, names) for dataset, names in DESCRIPTION_COLUMNS.items()}
    required = [*LINK_COLUMNS.values(), LINK_FOUND_COLUMN, SCAN_TYPE_COLUMN]
    missing = [column for column in required if column not in links.columns]
    missing += [DESCRIPTION_COLUMNS[dataset][0] for dataset, column in descriptions.items() if column is None]
    if missing:
        raise ValueError(f"Links file is missing the columns {missing}.")

    is_linked = links[LINK_FOUND_COLUMN].astype("string").str.strip().str.lower() == "yes"
    is_scan_type = links[SCAN_TYPE_COLUMN].astype("string").str.strip().str.upper() == SCAN_TYPE
    follow_scan_type = links[FOLLOW_UP_SCAN_TYPE_COLUMN].astype("string").str.strip().str.upper() == SCAN_TYPE
    linked = links[(is_linked & is_scan_type & follow_scan_type).fillna(False)]
    table = pd.DataFrame(
        {dataset: extract_ids(linked[column], DATASET_PREFIXES[dataset]) for dataset, column in LINK_COLUMNS.items()}
    )
    for dataset, column in descriptions.items():
        table[f"{dataset}_rr_percent"] = linked[column].map(rr_percent).astype(float)
    table["rr_percent_difference"] = table["CGPS10_rr_percent"] - table["CGPS1_rr_percent"]
    ids = list(LINK_COLUMNS)
    return table.dropna(subset=ids).drop_duplicates(subset=ids).reset_index(drop=True)


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
    try:
        table = link_table(links)
    except ValueError as error:
        raise SystemExit(f"{error}\nColumns in {comparison_csv}: {list(links.columns)}")
    print(table)
    print(f"{len(table)} of {len(links)} rows linked.")
    for dataset in LINK_COLUMNS:
        found = table[f"{dataset}_rr_percent"].notna().sum()
        print(f"{dataset}: R-R interval found for {found} of {len(table)} subjects.")
    print(f"R-R difference (CGPS10 - CGPS1):\n{table['rr_percent_difference'].describe().round(1)}")
    for dataset in DATASET_PREFIXES:
        repeated = table[dataset].duplicated(keep=False).sum()
        if repeated:
            print(f"[WARN] {repeated} rows share a {dataset} id with another row.")

    table_path = args.output or os.path.join(root, "df", LINKS_FILE)
    os.makedirs(os.path.dirname(os.path.abspath(table_path)), exist_ok=True)
    table.to_csv(table_path, index=False)
    print(f"Table saved to: {table_path}")

#################################################################################################################################################
# This script makes a table linking the patient ids of the CGPS1 (baseline) and CGPS10 (10-year follow-up) datasets, so that the results
# of the two datasets can be compared per subject or overall.
# It reads the file in COMPARISON_DATA_FILE in the .env, where each row links a baseline and a follow-up scan in the columns
# "Baseline NIFTI file" and "Follow-up NIFTI file", e.g.
#   Baseline NIFTI file  = CGPS-1-august-2026_00001_SERIES0005
#   Follow-up NIFTI file = CGPS-10y-august-2026_0730_SERIES0004_volume_1
# and saves one row per subject with the columns CGPS1 and CGPS10, e.g. 00001 and 0730, in <PROJECT_ROOT>/df/CGPS1_CGPS10_links.csv.
# When CGPS1_FILELIST_CSV and CGPS10_FILELIST_CSV are set in the .env, the point in the R-R interval of each scan is added from the
# SeriesDescription column of the file lists, written either as a percentage (FULL 73% 1.00s Cardiac 0.5 CE) or as a delay and an
# R-R interval (HALF 648ms 0.90s PIQE 0.5 CE -> 0.648 / 0.90 = 72%), in the columns <dataset>_rr_percent
# and rr_percent_difference (CGPS10 - CGPS1).
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
FILELIST_VARIABLES = {"CGPS1": "CGPS1_FILELIST_CSV", "CGPS10": "CGPS10_FILELIST_CSV"}
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


def load_filelists() -> dict[str, str]:
    """Paths of the CGPS1 and CGPS10 file lists in ``FILELIST_VARIABLES`` from the .env, only for the variables set."""
    from dotenv import load_dotenv

    load_dotenv(os.path.join(REPO_ROOT, ".env"))
    return {
        dataset: os.environ[variable] for dataset, variable in FILELIST_VARIABLES.items() if os.environ.get(variable)
    }


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


def extract_series(values: pd.Series, prefix: str) -> pd.Series:
    """Series of scan names of one dataset, e.g. ``SERIES0004`` in ``CGPS-10y-august-2026_0730_SERIES0004_volume_1``.

    Args:
        values: Scan names, e.g. NIFTI file names or paths.
        prefix: Dataset prefix of the scan names, e.g. ``CGPS-10y-august-2026``.

    Returns:
        The series, missing where a value has no series of the dataset.
    """
    return values.astype("string").str.extract(rf"{re.escape(prefix)}_\d+_(SERIES\d+)", expand=False)


def scan_phases(ids: pd.Series, series: dict[str, str], filelist: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Point in the R-R interval of the scan of each patient id in one dataset.

    The rows of a patient in the file list are found by ``PseudoID``. If a patient has several rows, only the rows
    mentioning the series of the linked scan are kept, when such rows exist. The percentage is missing if the kept
    rows give different percentages or any kept row has no percentage, e.g. a blank ``SeriesDescription``.

    Args:
        ids: Patient ids of the dataset, e.g. ``0730``.
        series: Patient id to the series of its linked scan, e.g. ``{"0730": "SERIES0004"}``.
        filelist: File list of the dataset with the columns ``PseudoID``, e.g. ``CGPS-10y-august-2026_0730``, and
            ``SeriesDescription``.
        prefix: Dataset prefix of the pseudo ids, e.g. ``CGPS-10y-august-2026``.

    Returns:
        The percentage of each id, with the index of ``ids``.

    Raises:
        ValueError: If ``PseudoID`` or ``SeriesDescription`` is missing.
    """
    filelist = filelist.rename(columns=lambda column: str(column).strip())
    missing = [column for column in ("PseudoID", "SeriesDescription") if column not in filelist.columns]
    if missing:
        raise ValueError(f"File list is missing the columns {missing}.")
    rows = pd.DataFrame(
        {
            "patient_id": extract_ids(filelist["PseudoID"], prefix),
            "series_description": filelist["SeriesDescription"].astype("string").str.strip(),
            "text": filelist.astype("string").fillna("").agg(" ".join, axis=1),
        }
    )
    rows["rr_percent"] = rows["series_description"].map(rr_percent).astype(float)
    groups = dict(tuple(rows.dropna(subset=["patient_id"]).groupby("patient_id")))

    phases = []
    for patient_id in ids:
        candidates = groups.get(patient_id, rows.iloc[0:0])
        if len(candidates) > 1 and patient_id in series:
            matching = candidates[candidates["text"].str.contains(series[patient_id], regex=False)]
            candidates = matching if len(matching) else candidates
        percents = candidates["rr_percent"].unique()
        phases.append(percents[0] if len(percents) == 1 else float("nan"))
    return pd.Series(phases, index=ids.index, dtype=float)


def add_phases(table: pd.DataFrame, links: pd.DataFrame, filelists: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The links table with the point in the R-R interval of each scan.

    Args:
        table: One row per subject with the patient ids in the columns ``CGPS1`` and ``CGPS10``.
        links: Rows of the links file with the columns in ``LINK_COLUMNS``, used for the series of the linked scans.
        filelists: Dataset name to its file list, see ``scan_phases``.

    Returns:
        A copy of ``table`` with ``<dataset>_rr_percent`` per dataset in ``filelists`` and, with both datasets, ``rr_percent_difference`` (CGPS10 - CGPS1).
    """
    table = table.copy()
    for dataset, filelist in filelists.items():
        prefix = DATASET_PREFIXES[dataset]
        column = links[LINK_COLUMNS[dataset]]
        scans = pd.DataFrame({"id": extract_ids(column, prefix), "series": extract_series(column, prefix)})
        scans = scans.dropna().drop_duplicates()
        unique = scans.drop_duplicates("id", keep=False)
        series = dict(zip(unique["id"], unique["series"]))
        table[f"{dataset}_rr_percent"] = scan_phases(table[dataset], series, filelist, prefix)
    if all(dataset in filelists for dataset in DATASET_PREFIXES):
        table["rr_percent_difference"] = table["CGPS10_rr_percent"] - table["CGPS1_rr_percent"]
    return table


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
    filelist_paths = load_filelists()
    for dataset, variable in FILELIST_VARIABLES.items():
        if dataset not in filelist_paths:
            print(f"[WARN] {variable} is not set, so the R-R interval of the {dataset} scans is not added.")
    filelists = {}
    for dataset, path in filelist_paths.items():
        if not os.path.isfile(path):
            raise SystemExit(f"File list not found: {path}")
        filelists[dataset] = pd.read_csv(path, dtype=str)
    table = add_phases(table, links, filelists)
    print(table)
    for dataset in filelists:
        found = table[f"{dataset}_rr_percent"].notna().sum()
        print(f"{dataset}: R-R interval found for {found} of {len(table)} subjects.")
    if "rr_percent_difference" in table:
        print(f"R-R difference (CGPS10 - CGPS1):\n{table['rr_percent_difference'].describe().round(1)}")
    print(f"{len(table)} of {len(links)} rows linked.")
    for dataset in DATASET_PREFIXES:
        repeated = table[dataset].duplicated(keep=False).sum()
        if repeated:
            print(f"[WARN] {repeated} rows share a {dataset} id with another row.")

    table_path = args.output or os.path.join(root, "df", LINKS_FILE)
    os.makedirs(os.path.dirname(os.path.abspath(table_path)), exist_ok=True)
    table.to_csv(table_path, index=False)
    print(f"Table saved to: {table_path}")

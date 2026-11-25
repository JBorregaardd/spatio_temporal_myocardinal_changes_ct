#################################################################################################################################################
# This script finds the point in the R-R interval and the heart rate of a set of scans from the DICOM series descriptions in a metadata
# file, e.g. to see when in the cycle the CGPS10 "ED" and "ES" best-phase scans were reconstructed.
# The series description of a best-phase scan holds the reconstruction delay after the R wave and the R-R interval
# (HALF 648ms 0.90s PIQE 0.5 CE -> 72%, 67 bpm); that of a multiphase scan the percentage (FULL 73% 1.00s Cardiac 0.5 CE).
# The scans are given per group as a file list (one NIfTI path per line) or a folder of NIfTI files, and matched to the metadata rows on the
# file name (column "filename"). Without groups the CGPS10 ED and ES file lists are used, with META_DATA_FILE from .env as metadata.
# It saves one row per scan (group, filename, patient_id, SeriesDescription, rr_percent, rr_interval_s, heart_rate_bpm) in
# <PROJECT_ROOT>/df/cfa/scan_phases.csv and prints the distribution of rr_percent and heart_rate_bpm per group.
# e.g. uv run cfa/scan_phases.py
#      uv run cfa/scan_phases.py --metadata /storage/DTU-CFA-2_v2/Metadata/<file>.csv --folder best=/storage/DTU-CFA-2_v2/NIFTI_highres
#################################################################################################################################################

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "LVM_thickness"))
sys.path.insert(0, os.path.join(REPO_ROOT, "comparative_analysis"))

from compare import RR_INTERVAL_S, rr_percent  # noqa: E402
from pipeline import load_environment  # noqa: E402

CGPS10_FILELISTS = {
    "ED": "/storage/Data/DTU-CGPS-10y-august-2026/Filelists/DTU-CGPS-10y-august-2026_all_cardiac_ED_CE_fullpath.txt",
    "ES": "/storage/Data/DTU-CGPS-10y-august-2026/Filelists/DTU-CGPS-10y-august-2026_all_cardiac_ES_CE_fullpath.txt",
}
OUTPUT_FILE = Path("df", "cfa", "scan_phases.csv")


def rr_interval_s(description: object) -> float:
    """R-R interval in seconds from a series description, e.g. 0.90 in ``HALF 648ms 0.90s PIQE 0.5 CE``.

    Args:
        description: Series description.

    Returns:
        The R-R interval, missing if the description has none or it is zero.
    """
    if not isinstance(description, str):
        return float("nan")
    match = RR_INTERVAL_S.search(description)
    return float(match.group(1)) if match and float(match.group(1)) > 0 else float("nan")


def group_files(filelists: dict[str, str], folders: dict[str, str]) -> pd.DataFrame:
    """The file names of the scans of each group.

    Args:
        filelists: Group name to a text file with one NIfTI path per line.
        folders: Group name to a folder of ``.nii.gz`` files.

    Returns:
        One row per scan with ``group`` and ``filename`` (the base name).
    """
    rows = []
    for group, path in filelists.items():
        with open(path) as file:
            rows += [{"group": group, "filename": os.path.basename(line.strip())} for line in file if line.strip()]
    for group, folder in folders.items():
        rows += [{"group": group, "filename": path.name} for path in sorted(Path(folder).glob("*.nii.gz"))]
    return pd.DataFrame(rows, columns=["group", "filename"])


def scan_phases(files: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Point in the R-R interval and heart rate of each scan.

    Args:
        files: Scans from ``group_files``.
        metadata: Metadata table with ``filename`` and ``SeriesDescription`` and optionally ``patient_id``.

    Returns:
        ``files`` with ``patient_id``, ``SeriesDescription``, ``rr_percent``, ``rr_interval_s`` and
        ``heart_rate_bpm`` (60 / R-R interval); missing for scans not in the metadata.
    """
    metadata = metadata.rename(columns=lambda column: column.strip())
    columns = [column for column in ("filename", "patient_id", "SeriesDescription") if column in metadata.columns]
    metadata = metadata[columns].assign(filename=metadata["filename"].astype(str).str.strip().map(os.path.basename))
    table = files.merge(metadata.drop_duplicates("filename"), on="filename", how="left")
    table["SeriesDescription"] = table["SeriesDescription"].astype("string").str.strip()
    table["rr_percent"] = table["SeriesDescription"].map(rr_percent).astype(float)
    table["rr_interval_s"] = table["SeriesDescription"].map(rr_interval_s).astype(float)
    table["heart_rate_bpm"] = 60 / table["rr_interval_s"]
    return table


def parse_groups(values: list[str]) -> dict[str, str]:
    """Parse ``NAME=path`` arguments into a dictionary."""
    groups = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise SystemExit(f"Expected NAME=path, got {value!r}.")
        groups[name] = path
    return groups


if __name__ == "__main__":
    root, _ = load_environment()

    parser = argparse.ArgumentParser(
        description="R-R percentage and heart rate of scans from their series descriptions."
    )
    parser.add_argument("--metadata", default=None, help="Metadata CSV (default: META_DATA_FILE in .env).")
    parser.add_argument("--filelist", action="append", default=[], help="NAME=file list; repeatable.")
    parser.add_argument("--folder", action="append", default=[], help="NAME=folder of .nii.gz files; repeatable.")
    parser.add_argument("--output", default=None, help=f"Output CSV (default: <PROJECT_ROOT>/{OUTPUT_FILE}).")
    args = parser.parse_args()

    metadata_path = args.metadata or os.environ.get("META_DATA_FILE")
    if not metadata_path or not os.path.isfile(metadata_path):
        raise SystemExit(f"Metadata file not found: {metadata_path}. Pass --metadata or set META_DATA_FILE.")
    filelists, folders = parse_groups(args.filelist), parse_groups(args.folder)
    if not filelists and not folders:
        filelists = CGPS10_FILELISTS

    files = group_files(filelists, folders)
    table = scan_phases(files, pd.read_csv(metadata_path, low_memory=False))
    output = Path(args.output) if args.output else Path(root) / OUTPUT_FILE
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)

    for group, scans in table.groupby("group", sort=False):
        print(
            f"\n{group}: {len(scans)} scans, {scans['SeriesDescription'].notna().sum()} in the metadata, "
            f"{scans['rr_percent'].notna().sum()} with an R-R percentage"
        )
        print(scans[["rr_percent", "heart_rate_bpm"]].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(1))
        print("Most common series descriptions:")
        print(scans["SeriesDescription"].str.replace(r"\d+ms", "<ms>", regex=True).value_counts().head(5).to_string())
    print(f"\nSaved {output}")

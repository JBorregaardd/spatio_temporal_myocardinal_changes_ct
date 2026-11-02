"""Collect the TotalSegmentator segmentations of the scans in the ED/ES dataset built by ``make_phase_dataset.py``.

The segmentation of a scan ``.../NIFTI/<scan>.nii.gz`` is read from
``<segmentation root>/Cardiac_<phase>_CE/<scan>/segmentations/heartchambers_highres.nii.gz`` and copied (or symlinked)
to ``<dataset>/segmentations/<phase>/<id>.heart.nii.gz``, the default segmentation naming of ``valve_planes.py``.
Scans without a segmentation are listed in ``<dataset>/segmentations/missing.csv``.

Example:
    uv run side_project/valve_planes/fetch_phase_segmentations.py
"""
from __future__ import annotations

import csv
import os

import typer

from make_phase_dataset import OUTPUT_FOLDER, place_scan

SEGMENTATION_ROOT = "/storage/Data/AortaExplorer/DTU-CGPS-10y-august-2026"
SEGMENTATION_FILE = os.path.join("segmentations", "heartchambers_highres.nii.gz")
MISSING_COLUMNS = ("patient_id", "phase", "expected_path")


def source_segmentation(segmentation_root: str, phase: str, scan_path: str) -> str:
    """Path of the TotalSegmentator heart-chambers segmentation of one scan.

    Args:
        segmentation_root: Folder holding the ``Cardiac_<phase>_CE`` folders.
        phase: ``"ED"`` or ``"ES"``.
        scan_path: Original path of the scan, e.g. ``.../CGPS-10y-august-2026_0000_SERIES0013_volume_1.nii.gz``.

    Returns:
        ``<segmentation_root>/Cardiac_<phase>_CE/<scan name>/segmentations/heartchambers_highres.nii.gz``.
    """
    scan_name = os.path.basename(scan_path).removesuffix(".nii.gz")
    return os.path.join(segmentation_root, f"Cardiac_{phase}_CE", scan_name, SEGMENTATION_FILE)


def fetch_segmentations(
    rows: list[dict[str, str]], segmentation_root: str, output_folder: str, copy: bool
) -> list[dict[str, str]]:
    """Place the segmentation of every dataset scan in ``<output_folder>/<phase>/<id>.heart.nii.gz``.

    Args:
        rows: Rows of ``dataset.csv`` with ``patient_id``, ``phase`` and ``source_path``.
        segmentation_root: Folder holding the ``Cardiac_<phase>_CE`` folders.
        output_folder: Folder the phase folders are created in.
        copy: Copy the segmentations instead of symlinking them.

    Returns:
        One row per scan without a segmentation, with the columns in ``MISSING_COLUMNS``.
    """
    missing = []
    for row in rows:
        source = source_segmentation(segmentation_root, row["phase"], row["source_path"])
        if not os.path.isfile(source):
            missing.append({"patient_id": row["patient_id"], "phase": row["phase"], "expected_path": source})
            continue
        os.makedirs(os.path.join(output_folder, row["phase"]), exist_ok=True)
        place_scan(source, os.path.join(output_folder, row["phase"], f"{row['patient_id']}.heart.nii.gz"), copy)
    return missing


def main(
    dataset_folder: str = typer.Option(OUTPUT_FOLDER, help="Dataset folder with dataset.csv from make_phase_dataset."),
    segmentation_root: str = typer.Option(SEGMENTATION_ROOT, help="Folder holding the Cardiac_<phase>_CE folders."),
    copy: bool = typer.Option(True, "--copy/--symlink", help="Copy the segmentations or symlink them."),
) -> None:
    """Collect the segmentations of the dataset and list the scans without one."""
    with open(os.path.join(dataset_folder, "dataset.csv"), newline="") as file:
        rows = list(csv.DictReader(file))
    output_folder = os.path.join(dataset_folder, "segmentations")
    missing = fetch_segmentations(rows, segmentation_root, output_folder, copy)

    missing_csv = os.path.join(output_folder, "missing.csv")
    os.makedirs(output_folder, exist_ok=True)
    with open(missing_csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MISSING_COLUMNS)
        writer.writeheader()
        writer.writerows(missing)
    print(f"{len(rows) - len(missing)} of {len(rows)} segmentations saved to {output_folder}")
    if missing:
        print(f"{len(missing)} scans have no segmentation, listed in {missing_csv}")


if __name__ == "__main__":
    typer.run(main)

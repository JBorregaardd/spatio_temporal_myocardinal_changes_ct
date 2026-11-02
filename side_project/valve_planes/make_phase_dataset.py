"""Build a balanced end-diastolic / end-systolic CT dataset from the CGPS 10-year file lists.

Draws the same random patients from the ED and the ES file list, so every patient contributes one scan of each phase,
and links (or copies) the scans to ``<output>/ED/<id>.nii.gz`` and ``<output>/ES/<id>.nii.gz``. A ``dataset.csv`` file
records the patient id, phase and source path of every scan. Each phase folder can be passed straight to
``valve_planes.py`` with ``--image-pattern "{id}.nii.gz"``.

Example:
    uv run side_project/valve_planes/make_phase_dataset.py --n-patients 100
"""
from __future__ import annotations

import csv
import os
import random
import re
import shutil

import typer

FILELIST_FOLDER = "/storage/Data/DTU-CGPS-10y-august-2026/Filelists"
ED_FILELIST = os.path.join(FILELIST_FOLDER, "DTU-CGPS-10y-august-2026_all_cardiac_ED_CE_fullpath.txt")
ES_FILELIST = os.path.join(FILELIST_FOLDER, "DTU-CGPS-10y-august-2026_all_cardiac_ES_CE_fullpath.txt")
OUTPUT_FOLDER = "/data/Data/JERI/data/CGPS10_ED_ES"
PATIENT_ID = re.compile(r"_(\d+)_SERIES\d+_volume_\d+\.nii\.gz$")
CSV_COLUMNS = ("patient_id", "phase", "path", "source_path")


def read_filelist(path: str) -> dict[str, str]:
    """Map patient ids to scan paths for one file list, keeping the first scan (sorted by path) of each patient.

    Args:
        path: Text file with one full NIfTI path per line, e.g.
            ``.../CGPS-10y-august-2026_0000_SERIES0013_volume_1.nii.gz``.

    Returns:
        ``{patient_id: scan_path}``.

    Raises:
        ValueError: If a line does not contain a patient id.
    """
    with open(path) as file:
        lines = sorted(line.strip() for line in file if line.strip())
    scans = {}
    for line in lines:
        match = PATIENT_ID.search(line)
        if match is None:
            raise ValueError(f"No patient id in line of {path}: {line}")
        scans.setdefault(match.group(1), line)
    return scans


def sample_patients(ed_scans: dict[str, str], es_scans: dict[str, str], n_patients: int, seed: int) -> list[str]:
    """Draw ``n_patients`` random patients that have both an ED and an ES scan, sorted numerically.

    Raises:
        ValueError: If fewer than ``n_patients`` patients have both phases.
    """
    common = sorted(ed_scans.keys() & es_scans.keys(), key=int)
    if len(common) < n_patients:
        raise ValueError(f"Only {len(common)} patients have both an ED and an ES scan, {n_patients} requested")
    return sorted(random.Random(seed).sample(common, n_patients), key=int)


def place_scan(source: str, destination: str, copy: bool) -> None:
    """Copy or symlink ``source`` to ``destination``, replacing an existing link or file."""
    if os.path.lexists(destination):
        os.remove(destination)
    if copy:
        shutil.copy2(source, destination)
    else:
        os.symlink(source, destination)


def build_dataset(
    ed_scans: dict[str, str], es_scans: dict[str, str], patient_ids: list[str], output_folder: str, copy: bool
) -> list[dict[str, str]]:
    """Place the ED and ES scans of the patients in ``<output_folder>/<phase>/<id>.nii.gz``.

    Returns:
        One row per scan with the columns in ``CSV_COLUMNS``.
    """
    rows = []
    for phase, scans in (("ED", ed_scans), ("ES", es_scans)):
        os.makedirs(os.path.join(output_folder, phase), exist_ok=True)
        for patient_id in patient_ids:
            destination = os.path.join(output_folder, phase, f"{patient_id}.nii.gz")
            place_scan(scans[patient_id], destination, copy)
            rows.append({"patient_id": patient_id, "phase": phase, "path": destination, "source_path": scans[patient_id]})
    return rows


def main(
    output_folder: str = typer.Option(OUTPUT_FOLDER, help="Folder the ED/ and ES/ folders and dataset.csv go in."),
    n_patients: int = typer.Option(100, help="Number of patients; each gives one ED and one ES scan."),
    seed: int = typer.Option(42, help="Seed of the random patient selection."),
    ed_filelist: str = typer.Option(ED_FILELIST, help="File list of the end-diastolic scans."),
    es_filelist: str = typer.Option(ES_FILELIST, help="File list of the end-systolic scans."),
    copy: bool = typer.Option(False, help="Copy the scans instead of symlinking them."),
) -> None:
    """Build the ED/ES dataset and write ``dataset.csv``."""
    ed_scans = read_filelist(ed_filelist)
    es_scans = read_filelist(es_filelist)
    patient_ids = sample_patients(ed_scans, es_scans, n_patients, seed)
    rows = build_dataset(ed_scans, es_scans, patient_ids, output_folder, copy)

    csv_path = os.path.join(output_folder, "dataset.csv")
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(patient_ids)} patients ({len(rows)} scans) of {len(ed_scans.keys() & es_scans.keys())} with both "
          f"phases saved to {output_folder}")


if __name__ == "__main__":
    typer.run(main)

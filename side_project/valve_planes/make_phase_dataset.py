"""Build a balanced end-diastolic / end-systolic CT dataset from the CGPS 10-year file lists.

Draws random patients from the ED and, independently, from the ES file list, so a patient can appear in one or both
phases, and links (or copies) the scans to ``<output>/ED/<id>.nii.gz`` and ``<output>/ES/<id>.nii.gz``. A
``dataset.csv`` file records the patient id, phase and source path of every scan. Each phase folder can be passed
straight to ``valve_planes.py`` with ``--image-pattern "{id}.nii.gz"``.

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
PATIENT_ID = re.compile(r"_(\d+)_SERIES\d+")
CSV_COLUMNS = ("patient_id", "phase", "path", "source_path")


def read_filelist(path: str) -> dict[str, str]:
    """Map patient ids to scan paths for one file list, keeping one scan per patient.

    Scans of a multi-phase series (``_volume_<n>`` suffix) are preferred over scans without the suffix; ties are broken
    by the first path in sorted order.

    Args:
        path: Text file with one full NIfTI path per line, e.g.
            ``.../CGPS-10y-august-2026_0000_SERIES0013_volume_1.nii.gz``; the ``_volume_<n>`` suffix is missing for
            some scans, e.g. ``.../CGPS-10y-august-2026_0013_SERIES0005.nii.gz``.

    Returns:
        ``{patient_id: scan_path}``.

    Raises:
        ValueError: If a line does not contain a patient id.
    """
    with open(path) as file:
        lines = sorted((line.strip() for line in file if line.strip()), key=lambda line: ("_volume_" not in line, line))
    scans = {}
    for line in lines:
        match = PATIENT_ID.search(line)
        if match is None:
            raise ValueError(f"No patient id in line of {path}: {line}")
        scans.setdefault(match.group(1), line)
    return scans


def sample_patients(scans: dict[str, str], n_patients: int, rng: random.Random) -> list[str]:
    """Draw ``n_patients`` random patients of one file list, sorted numerically.

    Raises:
        ValueError: If the file list has fewer than ``n_patients`` patients.
    """
    if len(scans) < n_patients:
        raise ValueError(f"Only {len(scans)} patients in the file list, {n_patients} requested")
    return sorted(rng.sample(sorted(scans, key=int), n_patients), key=int)


def place_scan(source: str, destination: str, copy: bool) -> None:
    """Copy or symlink ``source`` to ``destination``, replacing an existing link or file."""
    if os.path.lexists(destination):
        os.remove(destination)
    if copy:
        shutil.copy2(source, destination)
    else:
        os.symlink(source, destination)


def build_dataset(selection: dict[str, dict[str, str]], output_folder: str, copy: bool) -> list[dict[str, str]]:
    """Place the selected scans in ``<output_folder>/<phase>/<id>.nii.gz``.

    Args:
        selection: Phase (``"ED"``/``"ES"``) to ``{patient_id: scan_path}`` of the scans to include.
        output_folder: Folder the phase folders are created in.
        copy: Copy the scans instead of symlinking them.

    Returns:
        One row per scan with the columns in ``CSV_COLUMNS``.
    """
    rows = []
    for phase, scans in selection.items():
        os.makedirs(os.path.join(output_folder, phase), exist_ok=True)
        for patient_id, source in scans.items():
            destination = os.path.join(output_folder, phase, f"{patient_id}.nii.gz")
            place_scan(source, destination, copy)
            rows.append({"patient_id": patient_id, "phase": phase, "path": destination, "source_path": source})
    return rows


def main(
    output_folder: str = typer.Option(OUTPUT_FOLDER, help="Folder the ED/ and ES/ folders and dataset.csv go in."),
    n_patients: int = typer.Option(100, help="Number of scans drawn per phase."),
    seed: int = typer.Option(42, help="Seed of the random patient selection."),
    ed_filelist: str = typer.Option(ED_FILELIST, help="File list of the end-diastolic scans."),
    es_filelist: str = typer.Option(ES_FILELIST, help="File list of the end-systolic scans."),
    copy: bool = typer.Option(False, help="Copy the scans instead of symlinking them."),
) -> None:
    """Build the ED/ES dataset and write ``dataset.csv``."""
    rng = random.Random(seed)
    selection = {}
    for phase, filelist in (("ED", ed_filelist), ("ES", es_filelist)):
        scans = read_filelist(filelist)
        selection[phase] = {patient_id: scans[patient_id] for patient_id in sample_patients(scans, n_patients, rng)}
    rows = build_dataset(selection, output_folder, copy)

    csv_path = os.path.join(output_folder, "dataset.csv")
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    overlap = len(selection["ED"].keys() & selection["ES"].keys())
    print(f"{len(rows)} scans ({n_patients} per phase, {overlap} patients in both) saved to {output_folder}")


if __name__ == "__main__":
    typer.run(main)

"""Mix the ED and ES scans of the phase dataset under random scan ids, so the valves can be labelled blind to the phase.

Reads ``<dataset>/dataset.csv`` (from ``make_phase_dataset.py``) and the segmentations from
``fetch_phase_segmentations.py``, shuffles all scans and symlinks them to ``<dataset>/blinded/images/<scan_id>.nii.gz``
and ``<dataset>/blinded/segmentations/<scan_id>.heart.nii.gz``. The scan ids (``S001``, ``S002``, ...) say nothing
about the phase; the key linking them to patient id and phase is written to ``<dataset>/blinded_key.csv``, outside the
folder that is labelled. If ``blinded_key.csv`` exists, its scan ids are kept and only scans new in ``dataset.csv``
(added with ``make_phase_dataset.py --extend``) are shuffled and numbered after the highest existing id, so existing
labels stay valid.

Example:
    uv run side_project/valve_planes/blind_phase_dataset.py
"""
from __future__ import annotations

import csv
import os
import random

import typer

from make_phase_dataset import OUTPUT_FOLDER, place_scan

KEY_COLUMNS = ("scan_id", "patient_id", "phase")


def assign_scan_ids(
    rows: list[dict[str, str]], seed: int, key: list[dict[str, str]] | None = None
) -> list[dict[str, str]]:
    """Shuffle the dataset scans that are not in ``key`` yet and number them after the highest id in ``key``.

    Args:
        rows: Rows of ``dataset.csv`` with ``patient_id`` and ``phase``.
        seed: Seed of the shuffle.
        key: Existing key rows (columns in ``KEY_COLUMNS``), kept unchanged; none for a new key (``S001``, ...).

    Returns:
        The existing key rows followed by one row per new scan, sorted by scan id.
    """
    key = key or []
    known = {(row["patient_id"], row["phase"]) for row in key}
    shuffled = [row for row in rows if (row["patient_id"], row["phase"]) not in known]
    random.Random(seed).shuffle(shuffled)
    start = max((int(row["scan_id"][1:]) for row in key), default=0) + 1
    width = max(3, len(str(start + len(shuffled) - 1)))
    return key + [
        {"scan_id": f"S{number:0{width}d}", "patient_id": row["patient_id"], "phase": row["phase"]}
        for number, row in enumerate(shuffled, start=start)
    ]


def link_blinded(key: list[dict[str, str]], dataset_folder: str, blinded_folder: str) -> int:
    """Symlink each scan and its segmentation under its scan id.

    Args:
        key: Rows with the columns in ``KEY_COLUMNS``.
        dataset_folder: Folder with ``<phase>/<id>.nii.gz`` and ``segmentations/<phase>/<id>.heart.nii.gz``.
        blinded_folder: Folder the ``images`` and ``segmentations`` folders are created in.

    Returns:
        Number of scans without a segmentation (``valve_planes.py`` runs TotalSegmentator for those).
    """
    for subfolder in ("images", "segmentations"):
        os.makedirs(os.path.join(blinded_folder, subfolder), exist_ok=True)
    missing = 0
    for row in key:
        image = os.path.join(dataset_folder, row["phase"], f"{row['patient_id']}.nii.gz")
        place_scan(os.path.abspath(image), os.path.join(blinded_folder, "images", f"{row['scan_id']}.nii.gz"), False)
        segmentation = os.path.join(dataset_folder, "segmentations", row["phase"], f"{row['patient_id']}.heart.nii.gz")
        if os.path.isfile(segmentation):
            destination = os.path.join(blinded_folder, "segmentations", f"{row['scan_id']}.heart.nii.gz")
            place_scan(os.path.abspath(segmentation), destination, False)
        else:
            missing += 1
    return missing


def main(
    dataset_folder: str = typer.Option(OUTPUT_FOLDER, help="Dataset folder with dataset.csv from make_phase_dataset."),
    seed: int = typer.Option(42, help="Seed of the shuffle."),
) -> None:
    """Build the blinded folder and write ``blinded_key.csv``."""
    with open(os.path.join(dataset_folder, "dataset.csv"), newline="") as file:
        rows = list(csv.DictReader(file))
    key_path = os.path.join(dataset_folder, "blinded_key.csv")
    existing = []
    if os.path.isfile(key_path):
        with open(key_path, newline="") as file:
            existing = list(csv.DictReader(file))
    key = assign_scan_ids(rows, seed, existing)
    blinded_folder = os.path.join(dataset_folder, "blinded")
    missing = link_blinded(key, dataset_folder, blinded_folder)

    with open(key_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=KEY_COLUMNS)
        writer.writeheader()
        writer.writerows(key)
    print(
        f"{len(key)} scans ({len(key) - len(existing)} new) linked to {blinded_folder} ({missing} without a "
        f"segmentation); key in {key_path}"
    )


if __name__ == "__main__":
    typer.run(main)

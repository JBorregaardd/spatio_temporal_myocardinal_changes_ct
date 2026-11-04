"""Export valve-aligned 3D CT crops and open/closed labels as a GloViTa ``generic_3d_dataset``.

Reads the ``<id>_valves.json`` files written by ``valve_planes.py`` (valve centre and axes, CT path and per-scan HU
window) and the ``labels.csv`` written by ``label_valves.py``. For every scan and valve a cube is sampled from the CT
around the valve centre with its axes along ``in_plane_u``, ``in_plane_v`` and the flow normal, so every crop has the
same orientation (blood flows along the last axis). Intensities are normalised per scan so the median myocardium is 0
and the median blood pool 1. One dataset folder is written per valve:

    <output>/<valve>/images/<id>.b2nd   float32 crop of shape (1, u, v, normal)
    <output>/<valve>/dataset.json       {"num_classes": 2, "subtask": "multiclass"}
    <output>/<valve>/labels.json        {"<id>": 0 (closed) or 1 (open)}
    <output>/<valve>/splits.json        {"train": [...], "val": [...], "test": [...]}, split by patient

Example:
    uv run side_project/valve_planes/export_crops.py side_project/valve_planes/output side_project/valve_planes/output/crops
"""
from __future__ import annotations

import csv
import glob
import json
import logging
import os
import random

import blosc2
import numpy as np
import SimpleITK as sitk
import typer
from scipy.ndimage import map_coordinates

from valve_planes import AIR_HU, VALVES, physical_to_index

CROP_VOXELS = 96
CROP_RESOLUTION_MM = 0.5
LABEL_CODES = {"closed": 0, "open": 1}
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)
SPLIT_NAMES = ("train", "val", "test")

logger = logging.getLogger("export_crops")


def intensity_reference(hu_window: list[float]) -> tuple[float, float]:
    """Median myocardium HU and blood-myocardium contrast recovered from the adaptive window of ``valve_planes.py``.

    The adaptive window is ``[myocardium - c, blood + c / 2]`` with ``c = blood - myocardium``, so its width is
    ``2.5 * c``.

    Returns:
        Tuple of (myocardium HU, contrast in HU).
    """
    hu_min, hu_max = hu_window
    contrast = (hu_max - hu_min) / 2.5
    return hu_min + contrast, contrast


def scan_hu_window(metadata: dict) -> list[float] | None:
    """The per-scan HU window of a ``<id>_valves.json`` file, also from files written before ``hu_window`` existed.

    Older files store it as ``settings.hu_min``/``settings.hu_max`` with ``settings.window == "adaptive"``; a fixed
    window carries no per-scan intensity reference, so None is returned for it.
    """
    if "hu_window" in metadata:
        return metadata["hu_window"]
    settings = metadata.get("settings", {})
    if settings.get("window") == "adaptive":
        return [settings["hu_min"], settings["hu_max"]]
    return None


def sample_crop(
    ct: sitk.Image, plane: dict, crop_voxels: int = CROP_VOXELS, resolution_mm: float = CROP_RESOLUTION_MM
) -> np.ndarray:
    """Sample a cube of HU values centred on a valve, with axes (``in_plane_u``, ``in_plane_v``, normal).

    Args:
        ct: CT image in HU.
        plane: Valve entry of a ``<id>_valves.json`` file with ``center_mm``, ``in_plane_u``, ``in_plane_v`` and
            ``normal``.
        crop_voxels: Side length of the cube in voxels.
        resolution_mm: Voxel size in mm.

    Returns:
        HU array of shape (crop_voxels,) * 3 indexed (u, v, normal); points outside the scan are ``AIR_HU``.
    """
    axis = (np.arange(crop_voxels) - (crop_voxels - 1) / 2) * resolution_mm
    a, b, c = np.meshgrid(axis, axis, axis, indexing="ij")
    basis = np.array([plane["in_plane_u"], plane["in_plane_v"], plane["normal"]], dtype=float)
    points = np.asarray(plane["center_mm"]) + np.stack([a, b, c], axis=-1).reshape(-1, 3) @ basis
    index_zyx = physical_to_index(ct, points)[:, ::-1].T
    volume = sitk.GetArrayViewFromImage(ct)
    return map_coordinates(volume, index_zyx, order=1, cval=AIR_HU).reshape(crop_voxels, crop_voxels, crop_voxels)


def read_labels(path: str) -> dict[str, dict[str, int]]:
    """Read ``labels.csv`` into ``{valve: {scan_id: 0 or 1}}``, leaving out unclear and missing labels."""
    labels: dict[str, dict[str, int]] = {valve.name: {} for valve in VALVES}
    with open(path, newline="") as file:
        for row in csv.DictReader(file):
            for valve in labels:
                if row.get(valve, "") in LABEL_CODES:
                    labels[valve][row["patient_id"]] = LABEL_CODES[row[valve]]
    return labels


def read_groups(path: str | None) -> dict[str, str]:
    """Map scan ids to patient ids from a ``blinded_key.csv`` (``scan_id,patient_id,...``); empty if no key."""
    if not path:
        return {}
    with open(path, newline="") as file:
        return {row["scan_id"]: row["patient_id"] for row in csv.DictReader(file)}


def split_by_patient(
    labels: dict[str, int],
    groups: dict[str, str],
    fractions: tuple[float, float, float] = SPLIT_FRACTIONS,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Split labelled scans into train/val/test so all scans of a patient land in the same split.

    Patients are stratified by the set of labels of their scans, so the rare class is spread over all splits.

    Args:
        labels: Scan id to label.
        groups: Scan id to patient id; scans missing from it are their own patient.
        fractions: Train, validation and test fractions of the patients in each stratum.
        seed: Seed of the shuffle.

    Returns:
        ``{"train": [...], "val": [...], "test": [...]}`` with sorted scan ids.
    """
    patients: dict[str, list[str]] = {}
    for scan_id in labels:
        patients.setdefault(groups.get(scan_id, scan_id), []).append(scan_id)
    strata: dict[tuple[int, ...], list[str]] = {}
    for patient, scans in sorted(patients.items()):
        strata.setdefault(tuple(sorted({labels[s] for s in scans})), []).append(patient)

    rng = random.Random(seed)
    splits: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for members in strata.values():
        rng.shuffle(members)
        n_val = round(len(members) * fractions[1])
        n_test = round(len(members) * fractions[2])
        parts = {"val": members[:n_val], "test": members[n_val:n_val + n_test], "train": members[n_val + n_test:]}
        for name, part in parts.items():
            splits[name].extend(scan for patient in part for scan in patients[patient])
    return {name: sorted(scans) for name, scans in splits.items()}


def save_b2nd(array: np.ndarray, path: str) -> None:
    """Write an array as a ZSTD-compressed Blosc2 ``.b2nd`` file, creating its folder and overwriting an existing one."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    blosc2.asarray(
        np.ascontiguousarray(array), urlpath=path, cparams={"codec": blosc2.Codec.ZSTD, "clevel": 8}, mmap_mode="w+"
    )


def export_scan(
    metadata: dict, valve_names: list[str], output_folder: str, crop_voxels: int, resolution_mm: float, overwrite: bool
) -> None:
    """Write the normalised crops of one scan to ``<output_folder>/<valve>/images/<id>.b2nd``."""
    targets = {
        name: os.path.join(output_folder, name, "images", f"{metadata['patient_id']}.b2nd") for name in valve_names
    }
    targets = {name: path for name, path in targets.items() if overwrite or not os.path.isfile(path)}
    if not targets:
        return
    ct = sitk.Cast(sitk.ReadImage(metadata["image"]), sitk.sitkFloat32)
    myocardium, contrast = intensity_reference(scan_hu_window(metadata))
    for name, path in targets.items():
        crop = (sample_crop(ct, metadata["valves"][name], crop_voxels, resolution_mm) - myocardium) / contrast
        save_b2nd(crop[None].astype(np.float32), path)


def main(
    planes_folder: str = typer.Argument(..., help="Output folder of valve_planes.py with the <id>_valves.json files."),
    output_folder: str = typer.Argument(..., help="Folder the per-valve dataset folders are written to."),
    labels_path: str = typer.Option("", help="Labels CSV of label_valves.py; defaults to <planes_folder>/labels.csv."),
    key_path: str = typer.Option("", help="blinded_key.csv mapping scan ids to patients, for patient-level splits."),
    valves: list[str] = typer.Option(["mitral", "aortic"], help="Valves to export."),
    crop_voxels: int = typer.Option(CROP_VOXELS, help="Side length of the crop cube in voxels."),
    resolution_mm: float = typer.Option(CROP_RESOLUTION_MM, help="Voxel size of the crop in mm."),
    seed: int = typer.Option(42, help="Seed of the patient-level split."),
    overwrite: bool = typer.Option(False, help="Rewrite crops that already exist."),
) -> None:
    """Export valve crops, labels and patient-level splits for GloViTa."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    unknown = set(valves) - {valve.name for valve in VALVES}
    if unknown:
        raise typer.BadParameter(f"Unknown valves {sorted(unknown)}")
    labels = read_labels(labels_path or os.path.join(planes_folder, "labels.csv"))
    groups = read_groups(key_path)

    metadata_paths = sorted(glob.glob(os.path.join(planes_folder, "*_valves.json")))
    logger.info(f"Exporting {len(metadata_paths)} scans to {output_folder}")
    exported = set()
    for path in metadata_paths:
        with open(path) as file:
            metadata = json.load(file)
        if scan_hu_window(metadata) is None:
            logger.error(f"{path}: no per-scan HU window; rerun valve_planes.py on this scan")
            continue
        try:
            export_scan(metadata, valves, output_folder, crop_voxels, resolution_mm, overwrite)
            exported.add(metadata["patient_id"])
        except Exception as error:
            logger.error(f"{metadata['patient_id']}: failed - {type(error).__name__}: {error}")

    for name in valves:
        valve_labels = {scan: label for scan, label in labels[name].items() if scan in exported}
        folder = os.path.join(output_folder, name)
        os.makedirs(folder, exist_ok=True)
        files = {
            "dataset.json": {"num_classes": 2, "subtask": "multiclass"},
            "labels.json": valve_labels,
            "splits.json": split_by_patient(valve_labels, groups, seed=seed),
        }
        for filename, content in files.items():
            with open(os.path.join(folder, filename), "w") as file:
                json.dump(content, file, indent=2)
        counts = {code: sum(label == value for label in valve_labels.values()) for code, value in LABEL_CODES.items()}
        logger.info(f"{name}: {len(valve_labels)} labelled scans {counts}")


if __name__ == "__main__":
    typer.run(main)

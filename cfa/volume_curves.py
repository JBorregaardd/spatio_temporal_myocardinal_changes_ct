#################################################################################################################################################
# This script measures the volume of every heart chamber in every phase of the multiphase CFA-2 scans, the shared first step of the
# ejection fraction calibration and the valve timing analysis.
# The CFA folder (CFA_DIR in .env, default /storage/DTU-CFA-2_v2) holds the 20 phases of the R-R interval in NIFTI/CFA-2_<id>_<00..95>.nii.gz
# (1 mm) and the scanner's best phase in NIFTI_highres/CFA-2_<id>_<pct>%.nii.gz (0.5 mm), with their TotalSegmentator heartchambers_highres
# segmentations in TotalSegmentator/<scan>/segmentations/heartchambers_highres/heartchambers_highres.nii.gz and in TotalSegmentator_highres/.
# It saves in <PROJECT_ROOT>/df/cfa/:
#   volume_curves.csv   one row per scan (patient_id, series grid/best, phase_pct, the volume of each chamber in ml, slice spacing)
#   patient_curves.csv  one row per patient: EDV and ESV (maximum and minimum LV volume over the grid phases) and their phases, the
#                       ejection fraction from them, the best phase with its high-res LV volume and the grid LV volume interpolated at the
#                       same phase (to see the effect of the slice spacing)
# e.g. uv run cfa/volume_curves.py --workers 8
#################################################################################################################################################

import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "LVM_thickness"))

from pipeline import load_environment  # noqa: E402

DEFAULT_CFA_DIR = "/storage/DTU-CFA-2_v2"
CHAMBERS = {1: "lvm", 2: "la", 3: "lv", 4: "ra", 5: "rv"}
SERIES = {
    "grid": ("NIFTI", "TotalSegmentator", re.compile(r"(?P<patient>CFA-2_\d+)_(?P<phase>\d{2})\.nii\.gz")),
    "best": (
        "NIFTI_highres",
        "TotalSegmentator_highres",
        re.compile(r"(?P<patient>CFA-2_\d+)_(?P<phase>\d+(?:\.\d+)?)%\.nii\.gz"),
    ),
}
SEGMENTATION_FILE = Path("segmentations", "heartchambers_highres", "heartchambers_highres.nii.gz")
OUTPUT_FOLDER = Path("df", "cfa")
CURVES_FILE = "volume_curves.csv"
PATIENTS_FILE = "patient_curves.csv"


def find_scans(cfa_dir: Path) -> pd.DataFrame:
    """List the grid and best-phase scans of the CFA folder with the path of their segmentation.

    Args:
        cfa_dir: CFA folder with ``NIFTI``, ``NIFTI_highres``, ``TotalSegmentator`` and ``TotalSegmentator_highres``.

    Returns:
        One row per scan with ``patient_id``, ``series`` (``grid`` or ``best``), ``phase_pct``, ``scan_id`` (the file
        name without ``.nii.gz``) and ``segmentation`` (its path, which may not exist).
    """
    rows = []
    for series, (image_folder, segmentation_folder, pattern) in SERIES.items():
        folder = cfa_dir / image_folder
        for path in sorted(folder.glob("*.nii.gz")) if folder.is_dir() else []:
            match = pattern.fullmatch(path.name)
            if not match:
                continue
            scan_id = path.name.removesuffix(".nii.gz")
            rows.append(
                {
                    "patient_id": match["patient"],
                    "series": series,
                    "phase_pct": float(match["phase"]),
                    "scan_id": scan_id,
                    "segmentation": str(cfa_dir / segmentation_folder / scan_id / SEGMENTATION_FILE),
                }
            )
    return pd.DataFrame(rows, columns=["patient_id", "series", "phase_pct", "scan_id", "segmentation"])


def chamber_volumes(segmentation_path: str) -> dict[str, float]:
    """Volume of each heart chamber in one TotalSegmentator heartchambers_highres segmentation.

    Args:
        segmentation_path: Path of the segmentation.

    Returns:
        ``<chamber>_volume_ml`` for the LV myocardium (``lvm``), LA, LV, RA and RV, and ``slice_spacing_mm``; NaN
        volumes if the segmentation does not exist.
    """
    if not os.path.isfile(segmentation_path):
        return {**{f"{name}_volume_ml": np.nan for name in CHAMBERS.values()}, "slice_spacing_mm": np.nan}
    segmentation = sitk.ReadImage(segmentation_path)
    counts = np.bincount(sitk.GetArrayViewFromImage(segmentation).ravel(), minlength=max(CHAMBERS) + 1)
    voxel_ml = float(np.prod(segmentation.GetSpacing())) / 1000
    volumes = {f"{name}_volume_ml": counts[label] * voxel_ml for label, name in CHAMBERS.items()}
    return {**volumes, "slice_spacing_mm": segmentation.GetSpacing()[2]}


def volume_curves(scans: pd.DataFrame, workers: int) -> pd.DataFrame:
    """Measure the chamber volumes of every scan.

    Args:
        scans: Scans from ``find_scans``.
        workers: Parallel processes.

    Returns:
        ``scans`` with the columns of ``chamber_volumes`` added, sorted by patient, series and phase.
    """
    with ProcessPoolExecutor(max_workers=workers) as pool:
        volumes = list(pool.map(chamber_volumes, scans["segmentation"], chunksize=8))
    table = pd.concat([scans.reset_index(drop=True), pd.DataFrame(volumes)], axis=1)
    return table.sort_values(["patient_id", "series", "phase_pct"], ignore_index=True)


def patient_summary(curves: pd.DataFrame) -> pd.DataFrame:
    """Summarise the LV volume curve of each patient.

    The EDV and ESV are the largest and smallest LV volume over the grid phases. The grid volume at the best phase is
    interpolated linearly between the neighbouring grid phases, wrapping from 95% to the 0% of the next beat.

    Args:
        curves: Table from ``volume_curves``.

    Returns:
        One row per patient with ``n_grid_phases``, ``edv_ml``, ``ed_phase_pct``, ``esv_ml``, ``es_phase_pct``,
        ``ef_pct``, ``best_phase_pct``, ``best_lv_volume_ml`` (high-res) and ``grid_lv_volume_at_best_ml``. The
        best-phase columns are NaN without a best-phase scan; with several, the first is used.
    """
    rows = []
    for patient_id, scans in curves.groupby("patient_id", sort=True):
        grid = scans[(scans["series"] == "grid") & scans["lv_volume_ml"].notna()]
        best = scans[(scans["series"] == "best") & scans["lv_volume_ml"].notna()]
        row = {"patient_id": patient_id, "n_grid_phases": len(grid)}
        if len(grid):
            ed, es = grid.loc[grid["lv_volume_ml"].idxmax()], grid.loc[grid["lv_volume_ml"].idxmin()]
            row |= {
                "edv_ml": ed["lv_volume_ml"],
                "ed_phase_pct": ed["phase_pct"],
                "esv_ml": es["lv_volume_ml"],
                "es_phase_pct": es["phase_pct"],
                "ef_pct": (ed["lv_volume_ml"] - es["lv_volume_ml"]) / ed["lv_volume_ml"] * 100,
            }
        if len(best):
            best_phase = best.iloc[0]["phase_pct"]
            row |= {"best_phase_pct": best_phase, "best_lv_volume_ml": best.iloc[0]["lv_volume_ml"]}
            if len(grid) > 1:
                row["grid_lv_volume_at_best_ml"] = float(
                    np.interp(best_phase, grid["phase_pct"], grid["lv_volume_ml"], period=100)
                )
        rows.append(row)
    columns = [
        "patient_id",
        "n_grid_phases",
        "edv_ml",
        "ed_phase_pct",
        "esv_ml",
        "es_phase_pct",
        "ef_pct",
        "best_phase_pct",
        "best_lv_volume_ml",
        "grid_lv_volume_at_best_ml",
    ]
    return pd.DataFrame(rows).reindex(columns=columns)


if __name__ == "__main__":
    root, _ = load_environment()

    parser = argparse.ArgumentParser(description="Measure the heart chamber volumes in every phase of the CFA-2 scans.")
    parser.add_argument(
        "--cfa-dir", default=None, help=f"CFA folder (default: CFA_DIR in .env, else {DEFAULT_CFA_DIR})."
    )
    parser.add_argument("--output-dir", default=None, help=f"Output folder (default: <PROJECT_ROOT>/{OUTPUT_FOLDER}).")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="Parallel processes.")
    parser.add_argument("--limit", type=int, default=0, help="Only the first N patients (0 measures all).")
    args = parser.parse_args()

    cfa_dir = Path(args.cfa_dir or os.environ.get("CFA_DIR") or DEFAULT_CFA_DIR)
    scans = find_scans(cfa_dir)
    if scans.empty:
        raise SystemExit(f"No CFA-2 scans found in {cfa_dir / 'NIFTI'} or {cfa_dir / 'NIFTI_highres'}.")
    if args.limit > 0:
        scans = scans[scans["patient_id"].isin(sorted(scans["patient_id"].unique())[: args.limit])]
    for series in SERIES:
        part = scans[scans["series"] == series]
        found = part["segmentation"].map(os.path.isfile).sum()
        print(f"{series}: {len(part)} scans of {part['patient_id'].nunique()} patients, {found} with a segmentation")

    curves = volume_curves(scans, args.workers)
    patients = patient_summary(curves)
    output_dir = Path(args.output_dir) if args.output_dir else Path(root) / OUTPUT_FOLDER
    output_dir.mkdir(parents=True, exist_ok=True)
    curves.to_csv(output_dir / CURVES_FILE, index=False)
    patients.to_csv(output_dir / PATIENTS_FILE, index=False)

    complete = patients["n_grid_phases"].eq(len(range(0, 100, 5))).sum()
    print(f"{complete} of {len(patients)} patients have all 20 grid phases")
    print(
        patients[["ed_phase_pct", "es_phase_pct", "best_phase_pct", "edv_ml", "esv_ml", "ef_pct"]].describe().round(1)
    )
    ratio = patients["best_lv_volume_ml"] / patients["grid_lv_volume_at_best_ml"]
    print(
        f"High-res / grid LV volume at the best phase: median {ratio.median():.3f} (IQR {ratio.quantile(0.25):.3f}-"
        f"{ratio.quantile(0.75):.3f})"
    )
    print(f"Saved {output_dir / CURVES_FILE} and {output_dir / PATIENTS_FILE}")

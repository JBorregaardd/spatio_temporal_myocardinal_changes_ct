#################################################################################################################################################
# This script takes as input the output folder from the LVM thickness pipeline and calculates the median thickness for each of the 17 segments,
# from each patient's thickness.json (measurements near the aorta were already left out by the pipeline run).
# It also calculates myocardial volume and mass per segment, and patient-level measures (LV mass, LV diastolic volume, LV M/V ratio,
# LA volume, maximal wall thickness), indexed to body surface area when a demographics table is given.
# It reads the run folder <output>/<dataset>_<aorta-exclusion-mm> written by main_pipeline.py and saves the results to CSV files in that folder.
# e.g. uv run initial_data_investigation/statistics_analysis.py --dataset ImageCAS_1-200 --aorta-exclusion-mm 2.5 --all
#################################################################################################################################################

import os
import sys
import csv
import time
import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

import SimpleITK as sitk
from dotenv import load_dotenv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "LVM_thickness"))
sys.path.insert(0, os.path.join(REPO_ROOT, "measurements"))

from pipeline import AORTA_EXCLUSION_MM, run_output_dir  # noqa: E402
from thickness import thickness_per_segment  # noqa: E402


LVM_ID = 1
LA_ID = 2
LV_ID = 3
UNASSIGNED_ID = -1
FINISHED_MARKER = "done.json"
STATISTICS_FOLDER = "statistics"
MYOCARDIAL_DENSITY_G_PER_ML = 1.055  # Fuchs et al. 2016, EHJ-CVI 17:1009
SEGMENTS = range(1, 18)
WALL_THICKNESS_SEGMENTS = range(1, 17)

SEGMENT_FIELDS = ["patient_id", "segment", "median_thickness", "mean_thickness", "std_thickness", "volume_ml", "mass_g"]
PATIENT_FIELDS = [
    "patient_id",
    "lvm_g",
    "lvm_segments_g",
    "lvm_unassigned_g",
    "lvdv_ml",
    "lv_mv_ratio",
    "lav_ml",
    "lv_max_wall_thickness_mm",
    "lv_max_wall_thickness_segment",
    "sex",
    "age",
    "height_cm",
    "weight_kg",
    "bsa_m2",
    "lvmi",
    "lvdvi",
    "lavi",
]

def calculate_statistics_thickness_per_segment(folder: str) -> list[dict]:
    """Median, mean and standard deviation of the wall thickness per AHA segment, from the pipeline's thickness.json.

    Args:
        folder: The patient's output folder.

    Returns:
        One row per segment 1-17; NaN for segments without measurements.
    """
    stats = thickness_per_segment(folder)
    return [
        {
            "segment": segment,
            "median_thickness": stats[f"segment_{segment}_median_mm"],
            "mean_thickness": stats[f"segment_{segment}_mean_mm"],
            "std_thickness": stats[f"segment_{segment}_std_mm"],
        }
        for segment in SEGMENTS
    ]


def label_volume_ml(image: sitk.Image, mask: np.ndarray) -> float:
    """Volume of a voxel mask in mL.

    Args:
        image: Image the mask was taken from; its spacing (the slice increment, not the slice thickness) sets the
            voxel volume.
        mask: Boolean voxel mask.

    Returns:
        Volume in mL.
    """
    return float(np.count_nonzero(mask) * np.prod(image.GetSpacing()) / 1000.0)


def bsa_dubois(height_cm: float, weight_kg: float) -> float:
    """Body surface area by the DuBois & DuBois (1916) formula.

    Args:
        height_cm: Height in centimetres.
        weight_kg: Weight in kilograms.

    Returns:
        Body surface area in m^2.
    """
    return 0.007184 * height_cm**0.725 * weight_kg**0.425


def load_demographics(path: str) -> dict[str, dict[str, str]]:
    """Read a demographics table keyed by patient id.

    Args:
        path: CSV with a ``patient_id`` column and optionally ``height_cm``, ``weight_kg``, ``sex`` and ``age``.

    Returns:
        Mapping of patient id to that patient's row.
    """
    with open(path, newline="") as f:
        return {row["patient_id"].strip(): row for row in csv.DictReader(f)}


def _as_float(value: str | None) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def calculate_volume_per_segment(lv17: sitk.Image) -> dict[int, float]:
    """Myocardial volume of each AHA segment, counted in the native (untransformed) 17-segment label image.

    Args:
        lv17: The pipeline's ``segmentations/lv17/lv17.nii.gz``.

    Returns:
        Mapping of segment (1-17) to volume in mL.
    """
    labels = sitk.GetArrayViewFromImage(lv17)
    return {segment: label_volume_ml(lv17, labels == segment) for segment in SEGMENTS}


def calculate_patient_summary(
    heart: sitk.Image,
    lv17: sitk.Image,
    thickness_table: list[dict],
    demographics: dict[str, str] | None = None,
) -> dict[str, float | str]:
    """Patient-level LV mass, volumes and wall thickness, indexed to BSA when height and weight are known.

    Two LV masses are reported: ``lvm_g`` from the whole myocardium label, and ``lvm_segments_g`` from the 17 AHA
    segments only. The difference, ``lvm_unassigned_g``, is basal myocardium beyond the 17-segment model.

    Args:
        heart: TotalSegmentator heart-chambers segmentation.
        lv17: The pipeline's 17-segment label image.
        thickness_table: Output of ``calculate_statistics_thickness_per_segment``.
        demographics: This patient's demographics row, if available.

    Returns:
        One row of the patient summary table; indexed measures are NaN without height and weight.
    """
    heart_labels = sitk.GetArrayViewFromImage(heart)
    lv17_labels = sitk.GetArrayViewFromImage(lv17)

    lvm_g = label_volume_ml(heart, heart_labels == LVM_ID) * MYOCARDIAL_DENSITY_G_PER_ML
    lvm_segments_g = label_volume_ml(lv17, (lv17_labels >= 1) & (lv17_labels <= 17)) * MYOCARDIAL_DENSITY_G_PER_ML
    lvm_unassigned_g = label_volume_ml(lv17, lv17_labels == UNASSIGNED_ID) * MYOCARDIAL_DENSITY_G_PER_ML
    lvdv_ml = label_volume_ml(heart, heart_labels == LV_ID)
    lav_ml = label_volume_ml(heart, heart_labels == LA_ID)

    medians = {row["segment"]: row["median_thickness"] for row in thickness_table}
    wall = {segment: medians.get(segment, np.nan) for segment in WALL_THICKNESS_SEGMENTS}
    measured = {segment: value for segment, value in wall.items() if np.isfinite(value)}
    max_segment = max(measured, key=measured.get) if measured else None

    demographics = demographics or {}
    height_cm = _as_float(demographics.get("height_cm"))
    weight_kg = _as_float(demographics.get("weight_kg"))
    bsa_m2 = bsa_dubois(height_cm, weight_kg) if np.isfinite(height_cm) and np.isfinite(weight_kg) else np.nan

    return {
        "lvm_g": lvm_g,
        "lvm_segments_g": lvm_segments_g,
        "lvm_unassigned_g": lvm_unassigned_g,
        "lvdv_ml": lvdv_ml,
        "lv_mv_ratio": lvm_g / lvdv_ml if lvdv_ml > 0 else np.nan,
        "lav_ml": lav_ml,
        "lv_max_wall_thickness_mm": measured[max_segment] if max_segment else np.nan,
        "lv_max_wall_thickness_segment": max_segment if max_segment else "",
        "sex": demographics.get("sex", ""),
        "age": demographics.get("age", ""),
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "bsa_m2": bsa_m2,
        "lvmi": lvm_g / bsa_m2,
        "lvdvi": lvdv_ml / bsa_m2,
        "lavi": lav_ml / bsa_m2,
    }


def find_finished_patients(output_dir: str) -> list[str]:
    """Patients whose pipeline run completed, i.e. whose output folder has ``done.json``.

    Args:
        output_dir: The pipeline's output folder.

    Returns:
        Patient ids, numeric ids in numeric order.
    """
    ids = [name for name in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, name, FINISHED_MARKER))]
    return sorted(ids, key=lambda pid: (0, int(pid), "") if pid.isdigit() else (1, 0, pid))


def write_csv(path: str, fieldnames: list[str], rows: list[dict]) -> None:
    """Write rows to a CSV file with the given column order."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyse_patient(
    patient_id: str,
    output_dir: str,
    segmentation_folder: str,
    demographics: dict[str, str] | None = None,
) -> tuple[list[dict], dict]:
    """Run the full analysis for one patient and write that patient's CSVs into its output folder.

    Args:
        patient_id: Dataset id.
        output_dir: The pipeline's run folder, containing ``<patient_id>/``.
        segmentation_folder: Folder containing ``<patient_id>.heart.nii.gz``.
        demographics: This patient's demographics row, if available.

    Returns:
        Tuple of (per-segment rows, patient summary row).
    """
    folder = os.path.join(output_dir, patient_id)
    total_path = os.path.join(segmentation_folder, f"{patient_id}.heart.nii.gz")
    lv17_path = os.path.join(folder, "segmentations", "lv17", "lv17.nii.gz")

    table = calculate_statistics_thickness_per_segment(folder)

    heart = sitk.ReadImage(total_path)
    lv17 = sitk.ReadImage(lv17_path)
    segment_volumes = calculate_volume_per_segment(lv17)

    for row in table:
        row["patient_id"] = patient_id
        row["volume_ml"] = segment_volumes[row["segment"]]
        row["mass_g"] = segment_volumes[row["segment"]] * MYOCARDIAL_DENSITY_G_PER_ML

    summary = {"patient_id": patient_id, **calculate_patient_summary(heart, lv17, table, demographics)}

    write_csv(os.path.join(folder, "statistics_thickness_per_segment.csv"), SEGMENT_FIELDS, table)
    write_csv(os.path.join(folder, "statistics_patient_summary.csv"), PATIENT_FIELDS, [summary])
    return table, summary


def _analyse_patient_safely(patient_id: str, *args) -> tuple[str, list[dict] | None, dict | None, str, float]:
    start = time.perf_counter()
    try:
        table, summary = analyse_patient(patient_id, *args)
        return patient_id, table, summary, "", time.perf_counter() - start
    except Exception:
        return patient_id, None, None, traceback.format_exc(), time.perf_counter() - start


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(project_root, ".env"))

    root = os.environ["PROJECT_ROOT"]
    segmentation_folder = os.environ.get("SEGMENTATION_FOLDER", os.path.join(root, "data", "TotalSegmentator"))

    parser = argparse.ArgumentParser(description="Run the LVM thickness analysis for one, several or all patients.")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name of the pipeline run to analyse, as given to main_pipeline.py.")
    parser.add_argument("--aorta-exclusion-mm", type=float, default=AORTA_EXCLUSION_MM,
                        help=f"Aortic exclusion distance of the run to analyse (default: {AORTA_EXCLUSION_MM}). "
                             f"Together with --dataset it selects <output-dir>/<dataset>_<aorta-exclusion-mm>.")
    parser.add_argument("--output-dir", default=None,
                        help="Parent of the run folders (default: <PROJECT_ROOT>/output).")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--patient-ids",
        "--patient-id",
        dest="patient_ids",
        nargs="+",
        default=None,
        help="Patient ids to process, e.g. --patient-ids 1 2 3 4 (default: $PATIENT_IDS or $PATIENT_ID from .env). "
             "Useful for cluster array jobs, e.g. --patient-id $SLURM_ARRAY_TASK_ID.",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help=f"Process every patient in the output folder whose pipeline run finished ({FINISHED_MARKER} present).",
    )
    parser.add_argument(
        "--demographics",
        default=os.environ.get("DEMOGRAPHICS_CSV"),
        help="Optional CSV with patient_id, height_cm, weight_kg, sex, age (default: $DEMOGRAPHICS_CSV). "
             "Without it the BSA-indexed measures are left empty.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Patients analysed in parallel (default: 1). Each needs ~2-3 GB of RAM.",
    )
    args = parser.parse_args()

    output_dir = run_output_dir(root, args.dataset, args.aorta_exclusion_mm, args.output_dir)
    if not os.path.isdir(output_dir):
        raise SystemExit(f"Run folder not found: {output_dir}")
    print(f"Run folder: {output_dir}")

    if args.all:
        patient_ids = find_finished_patients(output_dir)
        if not patient_ids:
            raise SystemExit(f"No finished patients ({FINISHED_MARKER}) found in {output_dir}.")
    elif args.patient_ids:
        patient_ids = args.patient_ids
    else:
        patient_ids = (os.environ.get("PATIENT_IDS") or os.environ.get("PATIENT_ID", "")).split()
        if not patient_ids:
            raise SystemExit("No patient IDs provided. Use --patient-ids, --all, or set PATIENT_IDS in .env.")

    demographics = load_demographics(args.demographics) if args.demographics else {}
    if args.demographics:
        missing = [pid for pid in patient_ids if pid not in demographics]
        if missing:
            shown = " ".join(missing[:20]) + (" ..." if len(missing) > 20 else "")
            print(f"[WARN] {len(missing)} of {len(patient_ids)} patients not in {args.demographics}; "
                  f"their BSA-indexed measures are left empty: {shown}")

    print(f"Analysing {len(patient_ids)} patient(s) with {args.workers} worker(s)...")
    results = {}
    failed = {}
    jobs = [
        (pid, output_dir, segmentation_folder, demographics.get(pid))
        for pid in patient_ids
    ]

    def record(done: int, outcome: tuple) -> None:
        pid, table, summary, error, seconds = outcome
        if error:
            failed[pid] = error
        else:
            results[pid] = (table, summary)
        print(f"  [{done}/{len(jobs)}] patient {pid} {'FAILED' if error else 'done'} ({seconds:.0f}s)", flush=True)

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_analyse_patient_safely, *job) for job in jobs]
            for done, future in enumerate(as_completed(futures), 1):
                record(done, future.result())
    else:
        for done, job in enumerate(jobs, 1):
            record(done, _analyse_patient_safely(*job))

    ordered = [pid for pid in patient_ids if pid in results]
    if ordered:
        statistics_folder = os.path.join(output_dir, STATISTICS_FOLDER)
        os.makedirs(statistics_folder, exist_ok=True)
        segment_csv = os.path.join(statistics_folder, "statistics_thickness_per_segment.csv")
        summary_csv = os.path.join(statistics_folder, "statistics_patient_summary.csv")
        write_csv(segment_csv, SEGMENT_FIELDS, [row for pid in ordered for row in results[pid][0]])
        write_csv(summary_csv, PATIENT_FIELDS, [results[pid][1] for pid in ordered])
        print(f"Results for {len(ordered)} patient(s) saved to:")
        print(f"  {segment_csv}")
        print(f"  {summary_csv}")

    if failed:
        print(f"{len(failed)} patient(s) failed: {' '.join(failed)}")
        for pid, error in failed.items():
            print(f"--- patient {pid}: {error.strip().splitlines()[-1]}")
        sys.exit(1)

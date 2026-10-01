#################################################################################################################################################
# Sensitivity analysis for the aortic exclusion distance used with utils.aorta_distance_at_points.
# For every finished patient it sweeps the exclusion distance, drops mesh vertices closer than that to the aorta, and recomputes the
# per-segment thickness exactly like statistics_analysis.py (ring voxels -> lv17 transform -> segment). It writes CSVs and a figure that
# show how the segment bordering the aorta recovers relative to its neighbours, how many of its voxels are lost, and how much the other
# segments move, so a sweet spot for the distance can be chosen.
#################################################################################################################################################

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from dotenv import load_dotenv
from skimage import morphology
from vtk.util.numpy_support import vtk_to_numpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import utils  # noqa: E402

LVM_ID = 1
LV_ID = 3
FINISHED_MARKER = "done.json"
SEGMENTS = range(1, 18)
DEFAULT_THRESHOLDS = [float(t) for t in np.arange(0.0, 8.01, 0.5)]
PROFILE_BIN_MM = 0.5
OUTPUT_SUBFOLDER = os.path.join("statistics", "aorta_sensitivity")

TARGET_COLOR = "#2a78d6"
OTHER_COLOR = "#eb6834"
MUTED_INK = "#6b6a63"


def ring_voxel_segments(
    heart: sitk.Image, ring_nnz: tuple[np.ndarray, ...], folder: str
) -> tuple[np.ndarray, np.ndarray]:
    """Pair each ring voxel with the lv17 segment it lands in, exactly as statistics_analysis.py resamples them.

    statistics_analysis.py resamples the thickness ring image through ``lv17_transform.txt`` with nearest-neighbour
    interpolation and reads the lv17 labels at the same target voxels. Resampling an image of ring-voxel ids instead
    gives the same source-to-target mapping once, so every threshold can reuse it without resampling again.

    Args:
        heart: TotalSegmentator heart-chambers segmentation.
        ring_nnz: Ring voxel indices in (x, y, z) array order.
        folder: The patient's pipeline output folder.

    Returns:
        Tuple of (ring voxel id, segment label) arrays, one entry per target voxel; a ring voxel can appear more than
        once or not at all, matching the resampled image.
    """
    ids = np.zeros(sitk.GetArrayViewFromImage(heart).shape[::-1], dtype=np.int32)
    ids[ring_nnz] = np.arange(1, len(ring_nnz[0]) + 1, dtype=np.int32)
    id_image = sitk.GetImageFromArray(ids.transpose(2, 1, 0))
    id_image.CopyInformation(heart)

    transform = sitk.ReadTransform(os.path.join(folder, "lv17_transform.txt"))
    lv17 = sitk.ReadImage(os.path.join(folder, "segmentations", "lv17", "lv17.nii.gz"))
    lv17_transformed = sitk.Resample(lv17, transform, sitk.sitkNearestNeighbor, 0, lv17.GetPixelID())
    id_transformed = sitk.Resample(id_image, transform, sitk.sitkNearestNeighbor, 0, id_image.GetPixelID())

    segments = sitk.GetArrayViewFromImage(lv17_transformed).ravel()
    mapped = sitk.GetArrayViewFromImage(id_transformed).ravel()
    valid = (mapped > 0) & np.isin(segments, list(SEGMENTS))
    return mapped[valid].astype(np.int64) - 1, segments[valid].astype(np.int64)


def analyse_patient(
    patient_id: str,
    output_dir: str,
    segmentation_folder: str,
    thresholds: list[float],
) -> tuple[list[dict], list[dict]]:
    """Per-segment thickness for every exclusion distance, plus a thickness-vs-aorta-distance profile.

    Ring voxel values follow ``utils.MeshNeighbourhood.values``, the rule statistics_analysis.py uses, so a threshold
    equal to ``utils.AORTA_EXCLUSION_MM`` reproduces its per-segment statistics and 0 reproduces them without the
    exclusion.

    Args:
        patient_id: Dataset id.
        output_dir: The pipeline's output folder, containing ``<patient_id>/``.
        segmentation_folder: Folder containing ``<patient_id>.heart.nii.gz``.
        thresholds: Exclusion distances in mm.

    Returns:
        Tuple of (per-threshold per-segment rows, per-segment per-distance-bin profile rows).
    """
    folder = os.path.join(output_dir, patient_id)
    heart = sitk.ReadImage(os.path.join(segmentation_folder, f"{patient_id}.heart.nii.gz"))
    mesh = utils.read_vtk_mesh(os.path.join(folder, "surfaces", "dist_source.vtk"))

    labels = sitk.GetArrayViewFromImage(heart).transpose(2, 1, 0)
    im_bin = np.isin(labels, (LV_ID, LVM_ID))
    ring_nnz = (im_bin & ~morphology.erosion(im_bin)).nonzero()
    ring_points = utils.continuous_index_to_physical(heart, np.stack(ring_nnz, axis=1))

    vertices = vtk_to_numpy(mesh.GetPoints().GetData()).astype(np.float64)
    scalars = vtk_to_numpy(mesh.GetPointData().GetScalars()).astype(np.float64)
    vertex_distance = utils.aorta_distance_at_points(heart, vertices, max(thresholds))

    neighbourhood = utils.MeshNeighbourhood.build(ring_points, vertices)

    ring_id, segment = ring_voxel_segments(heart, ring_nnz, folder)

    rows = []
    for threshold in thresholds:
        keep = vertex_distance > threshold if threshold > 0 else np.ones(len(vertices), dtype=bool)
        values = neighbourhood.values(scalars, keep)

        voxel_values = values[ring_id] + 1e-10
        for seg in SEGMENTS:
            seg_values = voxel_values[segment == seg]
            seg_values = seg_values[seg_values > 0]
            rows.append(
                {
                    "patient_id": patient_id,
                    "threshold_mm": threshold,
                    "segment": seg,
                    "n_voxels": len(seg_values),
                    "median_thickness": np.median(seg_values) if len(seg_values) else np.nan,
                    "mean_thickness": np.mean(seg_values) if len(seg_values) else np.nan,
                    "vertices_removed_fraction": 1 - keep.mean(),
                }
            )

    voxel_distance = vertex_distance[neighbourhood.nearest][ring_id]
    voxel_thickness = neighbourhood.values(scalars)[ring_id]
    bin_index = np.floor(np.minimum(voxel_distance, max(thresholds)) / PROFILE_BIN_MM).astype(int)
    profile = []
    for seg in SEGMENTS:
        for b in np.unique(bin_index[segment == seg]):
            in_bin = (segment == seg) & (bin_index == b)
            profile.append(
                {
                    "patient_id": patient_id,
                    "segment": seg,
                    "distance_bin_mm": b * PROFILE_BIN_MM,
                    "n_voxels": int(in_bin.sum()),
                    "median_thickness": float(np.median(voxel_thickness[in_bin])),
                }
            )
    return rows, profile


def _analyse_patient_safely(patient_id: str, *args) -> tuple[str, list[dict] | None, list[dict] | None, str, float]:
    start = time.perf_counter()
    try:
        rows, profile = analyse_patient(patient_id, *args)
        return patient_id, rows, profile, "", time.perf_counter() - start
    except Exception:
        return patient_id, None, None, traceback.format_exc(), time.perf_counter() - start


def summarise(data: pd.DataFrame, target: int, neighbours: list[int]) -> pd.DataFrame:
    """Per-patient sensitivity metrics for every threshold.

    Args:
        data: Per-threshold per-segment rows from ``analyse_patient``.
        target: Segment bordering the aorta.
        neighbours: Segments the target is compared against.

    Returns:
        One row per patient and threshold with the target-vs-neighbour gap (mm), the fraction of the target segment's
        voxels lost relative to no exclusion, and the largest absolute change in any other segment's median (mm).
    """
    medians = data.pivot_table(index=["patient_id", "threshold_mm"], columns="segment", values="median_thickness")
    voxels = data.pivot_table(index=["patient_id", "threshold_mm"], columns="segment", values="n_voxels")
    baseline_medians = medians.xs(0.0, level="threshold_mm")
    baseline_voxels = voxels.xs(0.0, level="threshold_mm")

    others = [seg for seg in medians.columns if seg != target and seg != 17]
    patient_index = medians.index.get_level_values("patient_id")
    shift = (medians[others] - baseline_medians.loc[patient_index, others].to_numpy()).abs()

    return pd.DataFrame(
        {
            "target_gap_mm": medians[target] - medians[neighbours].mean(axis=1),
            "target_voxels_lost_fraction": 1 - voxels[target] / baseline_voxels.loc[patient_index, target].to_numpy(),
            "max_other_shift_mm": shift.max(axis=1),
            "target_empty": voxels[target] == 0,
        }
    ).reset_index()


def suggest_threshold(summary: pd.DataFrame, tolerance_mm: float) -> float:
    """Smallest threshold whose median target gap is within ``tolerance_mm`` of the gap at the largest threshold.

    Args:
        summary: Output of ``summarise``.
        tolerance_mm: How close to the plateau the gap must be.

    Returns:
        The suggested exclusion distance in mm.
    """
    gap = summary.groupby("threshold_mm")["target_gap_mm"].median()
    return float(gap.index[(gap - gap.iloc[-1]).abs() <= tolerance_mm].min())


def _band(ax: plt.Axes, x: np.ndarray, grouped: pd.core.groupby.SeriesGroupBy, color: str, label: str) -> None:
    ax.fill_between(x, grouped.quantile(0.25), grouped.quantile(0.75), color=color, alpha=0.2, linewidth=0)
    ax.plot(x, grouped.median(), color=color, linewidth=2, marker="o", markersize=5, label=label)


def plot_sensitivity(
    summary: pd.DataFrame,
    profile: pd.DataFrame,
    target: int,
    neighbours: list[int],
    suggested: float,
    output_path: str,
) -> None:
    """Four-panel figure: target gap, target voxels lost, shift in other segments, and thickness vs aorta distance.

    Args:
        summary: Output of ``summarise``.
        profile: Profile rows from ``analyse_patient``.
        target: Segment bordering the aorta.
        neighbours: Segments the target is compared against.
        suggested: Threshold to mark on the sweep panels.
        output_path: PNG path.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    by_threshold = summary.groupby("threshold_mm")
    x = np.array(sorted(summary["threshold_mm"].unique()))
    neighbour_text = "/".join(str(n) for n in neighbours)

    ax = axes[0, 0]
    _band(ax, x, by_threshold["target_gap_mm"], TARGET_COLOR, "Median across patients (IQR shaded)")
    ax.axhline(0, color=MUTED_INK, linewidth=1)
    ax.set_ylabel(f"Segment {target} minus mean of {neighbour_text} (mm)", fontsize=16)
    ax.set_title(f"Segment {target} vs. its neighbours", fontsize=18)

    ax = axes[0, 1]
    lost = summary.assign(target_voxels_lost_fraction=summary["target_voxels_lost_fraction"] * 100)
    _band(
        ax,
        x,
        lost.groupby("threshold_mm")["target_voxels_lost_fraction"],
        TARGET_COLOR,
        "Median across patients (IQR shaded)",
    )
    ax.set_ylabel(f"Segment {target} voxels removed (%)", fontsize=16)
    ax.set_title(f"Cost: segment {target} coverage", fontsize=18)

    ax = axes[1, 0]
    shift = by_threshold["max_other_shift_mm"]
    ax.plot(x, shift.median(), color=TARGET_COLOR, linewidth=2, marker="o", markersize=5, label="Median")
    ax.plot(x, shift.quantile(0.95), color=OTHER_COLOR, linewidth=2, marker="o", markersize=5, label="95th percentile")
    ax.set_ylabel("Largest change in another segment (mm)", fontsize=16)
    ax.set_title("Side effect on the other segments", fontsize=18)

    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.axvline(suggested, color=MUTED_INK, linewidth=1, linestyle="--")
        ax.text(
            suggested,
            0.98,
            f" suggested {suggested:g} mm",
            transform=ax.get_xaxis_transform(),
            va="top",
            fontsize=13,
            color=MUTED_INK,
        )
        ax.set_xlabel("Exclusion distance to aorta (mm)", fontsize=16)
        ax.legend(fontsize=13, loc="best")

    ax = axes[1, 1]
    per_patient = profile.assign(group=np.where(profile["segment"] == target, "target", "other"))
    per_patient = per_patient[per_patient["segment"] != 17]
    per_patient = per_patient.groupby(["group", "patient_id", "distance_bin_mm"])["median_thickness"].median()
    top_bin = profile["distance_bin_mm"].max()
    for group, color, label in (
        ("target", TARGET_COLOR, f"Segment {target}"),
        ("other", OTHER_COLOR, "Segments 1-16 except " + str(target)),
    ):
        if group not in per_patient.index.get_level_values("group"):
            continue
        curve = per_patient.loc[group].groupby("distance_bin_mm")
        bins = np.array(sorted(curve.groups)) + PROFILE_BIN_MM / 2
        bins[-1] = top_bin + PROFILE_BIN_MM
        _band(ax, bins, curve, color, label)
    ax.set_xlabel(f"Distance to aorta (mm; last point = beyond {top_bin:g})", fontsize=16)
    ax.set_ylabel("Thickness (mm)", fontsize=16)
    ax.set_title("Thickness vs. distance to the aorta, no exclusion", fontsize=18)
    ax.legend(fontsize=13, loc="best")

    for ax in axes.ravel():
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(color="#e1e0d9", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def find_finished_patients(output_dir: str) -> list[str]:
    """Patients whose pipeline run completed, i.e. whose output folder has ``done.json``."""
    ids = [name for name in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, name, FINISHED_MARKER))]
    return sorted(ids, key=lambda pid: (0, int(pid), "") if pid.isdigit() else (1, 0, pid))


def main() -> None:
    """Parse arguments, run the sweep for the selected patients, and write the CSVs and figure."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    load_dotenv(os.path.join(project_root, ".env"))
    root = os.environ["PROJECT_ROOT"]
    output_dir = os.path.join(root, "output")
    segmentation_folder = os.environ.get("SEGMENTATION_FOLDER", os.path.join(root, "data", "TotalSegmentator"))

    parser = argparse.ArgumentParser(description="Sensitivity analysis of the aortic exclusion distance.")
    parser.add_argument(
        "--patient-ids", nargs="+", default=None, help="Patients to include (default: every patient with done.json)."
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
        help="Exclusion distances in mm; 0 (no exclusion) is always added as the baseline.",
    )
    parser.add_argument(
        "--target-segment", type=int, default=2, help="Segment bordering the aorta (default: 2, AHA anteroseptal)."
    )
    parser.add_argument(
        "--neighbour-segments",
        nargs="+",
        type=int,
        default=[1, 3],
        help="Segments the target is compared against (default: 1 3).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.25,
        help="Suggest the smallest distance whose median gap is within this many mm of the plateau.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Patients analysed in parallel (default: 1).")
    parser.add_argument(
        "--plot-only", action="store_true", help="Skip the sweep and replot from the CSVs written by a previous run."
    )
    args = parser.parse_args()

    save_dir = os.path.join(output_dir, OUTPUT_SUBFOLDER)
    os.makedirs(save_dir, exist_ok=True)
    segments_csv = os.path.join(save_dir, "thickness_per_threshold.csv")
    profile_csv = os.path.join(save_dir, "thickness_vs_aorta_distance.csv")

    if args.plot_only:
        data, profile = pd.read_csv(segments_csv), pd.read_csv(profile_csv)
        data["patient_id"] = data["patient_id"].astype(str)
        profile["patient_id"] = profile["patient_id"].astype(str)
    else:
        thresholds = sorted({0.0, *args.thresholds})
        patient_ids = args.patient_ids or find_finished_patients(output_dir)
        print(
            f"Sweeping {len(thresholds)} distances for {len(patient_ids)} patient(s) with {args.workers} worker(s)..."
        )
        jobs = [(pid, output_dir, segmentation_folder, thresholds) for pid in patient_ids]
        rows, profiles, failed = [], [], {}

        def record(done: int, outcome: tuple) -> None:
            pid, patient_rows, patient_profile, error, seconds = outcome
            if error:
                failed[pid] = error
            else:
                rows.extend(patient_rows)
                profiles.extend(patient_profile)
            print(f"  [{done}/{len(jobs)}] patient {pid} {'FAILED' if error else 'done'} ({seconds:.0f}s)", flush=True)

        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(_analyse_patient_safely, *job) for job in jobs]
                for done, future in enumerate(as_completed(futures), 1):
                    record(done, future.result())
        else:
            for done, job in enumerate(jobs, 1):
                record(done, _analyse_patient_safely(*job))

        for pid, error in failed.items():
            print(f"--- patient {pid} failed: {error.strip().splitlines()[-1]}")
        if not rows:
            raise SystemExit("No patients were analysed.")
        data, profile = pd.DataFrame(rows), pd.DataFrame(profiles)
        data.to_csv(segments_csv, index=False)
        profile.to_csv(profile_csv, index=False)

    summary = summarise(data, args.target_segment, args.neighbour_segments)
    summary.to_csv(os.path.join(save_dir, "summary_per_patient.csv"), index=False)
    suggested = suggest_threshold(summary, args.tolerance)

    table = (
        summary.groupby("threshold_mm")
        .agg(
            gap_median=("target_gap_mm", "median"),
            gap_q25=("target_gap_mm", lambda s: s.quantile(0.25)),
            gap_q75=("target_gap_mm", lambda s: s.quantile(0.75)),
            voxels_lost_median_pct=("target_voxels_lost_fraction", lambda s: 100 * s.median()),
            voxels_lost_max_pct=("target_voxels_lost_fraction", lambda s: 100 * s.max()),
            other_shift_median=("max_other_shift_mm", "median"),
            other_shift_p95=("max_other_shift_mm", lambda s: s.quantile(0.95)),
            patients_target_empty=("target_empty", "sum"),
        )
        .round(3)
    )
    table.to_csv(os.path.join(save_dir, "summary_per_threshold.csv"))

    plot_path = os.path.join(save_dir, "aorta_exclusion_sensitivity.png")
    plot_sensitivity(summary, profile, args.target_segment, args.neighbour_segments, suggested, plot_path)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table)
    print(f"Suggested exclusion distance: {suggested:g} mm (median gap within {args.tolerance:g} mm of the plateau)")
    print(f"Results for {summary['patient_id'].nunique()} patient(s) saved to: {save_dir}")


if __name__ == "__main__":
    main()

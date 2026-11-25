#################################################################################################################################################
# This script looks for the cause of the bias between the mesh method and the segment method of the CGPS10 - CGPS1 change in wall
# thickness per AHA segment (compare_segment_methods.py; mesh - segment was about -0.8 mm in the basal septum), over all subjects and
# without grouping. Per subject and segment it brings together, in mm:
#   segment_delta_mm:  change of the segment medians of each scan's own 17-segment model (thickness.json, no registration),
#   mesh_delta_mm:     the same with the CGPS10 thickness taken where the registration put it on the CGPS1 vertices,
#   quarter_delta_mm:  a third method: change of the segment medians of each scan's own bullseye points (the points of
#                      bullseye_quarter_distribution.py, its four quarters pooled; no registration, the drawn wedges as segments),
# and the possible causes of the difference (mesh - segment):
#   1. the mesh method misses thickening: Spearman correlation of the difference with the size of the change, measured with the
#      independent quarter method (the segment method itself would share its errors with the difference),
#   2. unmatched vertices are left out: correlation with the share of unmatched vertices of the segment (match_distance.py, if run),
#   3. the scans' segment models cover different parts of the wall: correlation with log2 of the CGPS10 / CGPS1 share of the wall
#      (segments 1-16) that the segment holds in thickness.json,
# and how much single subjects drive it (mean vs median difference, share of subjects more than 2 mm apart).
# It needs segment_method_comparison.csv (compare_segment_methods.py) in <PROJECT_ROOT>/output/CGPS1_CGPS10_mesh_delta_0mm/, the points
# of bullseye_quarter_distribution.py in <PROJECT_ROOT>/output/CGPS1_CGPS10_quarters_0mm/, the pipeline folders of both datasets and,
# optionally, match_distance_per_segment.csv (match_distance.py). The per-subject values (computed once, --overwrite to redo) and the
# per-segment table are saved in the mesh delta run folder and <PROJECT_ROOT>/reports/figures/repeated_measures/bullseye, with a figure.
# e.g. uv run repeated_measures/bullseye/diagnose_segment_bias.py --workers 8
#################################################################################################################################################

import argparse
import os
import sys
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "measurements"))

from bullseye_mesh_delta import FIGURE_FOLDER, load_polar  # noqa: E402
from bullseye_quarter_distribution import POINTS_FILE, quarter_labels  # noqa: E402
from bullseye_quarter_distribution import RUN_NAME as QUARTER_RUN_NAME  # noqa: E402
from compare_segment_methods import SUBJECT_TABLE as COMPARISON_TABLE  # noqa: E402
from match_distance import SUBJECT_TABLE as MATCH_TABLE  # noqa: E402
from mesh_delta import AORTA_EXCLUSION_MM, RUN_NAME, pair_list, run_subjects  # noqa: E402
from pipeline import load_environment, run_output_dir  # noqa: E402
from spaghetti_plot import DATASETS, read_ids  # noqa: E402
from thickness import load_thickness  # noqa: E402

WALL_SEGMENTS = list(range(1, 17))
BASAL_SEGMENTS = list(range(1, 7))
SUBJECT_FILE = "segment_bias_inputs.csv"
SUMMARY_NAME = "segment_bias_diagnosis"
FAR_MM = 2.0
METHODS = {"segment_delta_mm": "Segment method", "mesh_delta_mm": "Mesh method", "quarter_delta_mm": "Quarter method"}


def wall_shares(scan_dir: Path) -> dict[int, float]:
    """Share of the wall measurements (segments 1-16) of a scan's thickness.json in each segment."""
    counts = {segment: len(values) for segment, values in load_thickness(str(scan_dir)).items()}
    total = sum(counts.get(segment, 0) for segment in WALL_SEGMENTS)
    return {segment: counts.get(segment, 0) / total for segment in WALL_SEGMENTS}


def bullseye_medians(points: dict[str, np.ndarray]) -> dict[int, float]:
    """Median of the bullseye points in each drawn wedge (segments 1-16)."""
    segments, _ = quarter_labels(points["theta"], points["radius"])
    return {
        segment: float(np.median(points["values"][segments == segment])) if np.any(segments == segment) else np.nan
        for segment in WALL_SEGMENTS
    }


def subject_inputs(cgps1: str, cgps10: str, folders: dict[str, Path], quarter_folder: Path) -> dict:
    """Quarter method change and wall share ratio per segment of one subject.

    Args:
        cgps1: CGPS1 patient id.
        cgps10: CGPS10 patient id.
        folders: Dataset name to its pipeline run folder.
        quarter_folder: Run folder of bullseye_quarter_distribution.py.

    Returns:
        ``CGPS1``, ``CGPS10`` and ``rows``: one row per segment 1-16 with ``quarter_delta_mm`` and ``share_log2_ratio``.
    """
    ids = {"CGPS1": cgps1, "CGPS10": cgps10}
    medians, shares = {}, {}
    for dataset in DATASETS:
        points_path = quarter_folder / f"{cgps1}_{cgps10}" / POINTS_FILE.format(dataset=dataset)
        medians[dataset] = bullseye_medians(load_polar(points_path))
        shares[dataset] = wall_shares(folders[dataset] / ids[dataset])
    rows = []
    for segment in WALL_SEGMENTS:
        ratio = shares["CGPS10"][segment] / shares["CGPS1"][segment] if shares["CGPS1"][segment] > 0 else np.nan
        rows.append(
            {
                "CGPS1": cgps1,
                "CGPS10": cgps10,
                "segment": segment,
                "quarter_delta_mm": medians["CGPS10"][segment] - medians["CGPS1"][segment],
                "share_log2_ratio": np.log2(ratio) if ratio > 0 else np.nan,
            }
        )
    return {"CGPS1": cgps1, "CGPS10": cgps10, "rows": rows}


def spearman(x: pd.Series, y: pd.Series) -> float:
    """Spearman correlation over the rows where both are finite, NaN with fewer than 10 rows."""
    finite = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[finite], y[finite]).statistic) if finite.sum() >= 10 else np.nan


def segment_diagnosis(table: pd.DataFrame) -> pd.DataFrame:
    """Bias of the mesh method and its possible causes per segment.

    Args:
        table: One row per subject and segment with the three methods' changes, ``difference_mm`` (mesh - segment),
            ``share_log2_ratio`` and, if available, ``unmatched_pct``.

    Returns:
        One row per segment with the mean change of each method, the mean and median difference, the share of subjects
        more than ``FAR_MM`` apart, the mean quarter - segment and mesh - quarter differences, and the Spearman
        correlations of the difference with the quarter method change, the unmatched share and the wall share ratio.
    """
    rows = []
    for segment, group in table.groupby("segment"):
        difference = group["difference_mm"]
        row = {"segment": segment, "n": int(difference.notna().sum())}
        for column in METHODS:
            row[f"mean_{column}"] = group[column].mean()
        row["mean_difference_mm"] = difference.mean()
        row["median_difference_mm"] = difference.median()
        row[f"pct_beyond_{FAR_MM:g}mm"] = 100 * (difference.abs() > FAR_MM).mean()
        row["mean_quarter_minus_segment_mm"] = (group["quarter_delta_mm"] - group["segment_delta_mm"]).mean()
        row["mean_mesh_minus_quarter_mm"] = (group["mesh_delta_mm"] - group["quarter_delta_mm"]).mean()
        row["r_difference_vs_change"] = spearman(difference, group["quarter_delta_mm"])
        row["r_difference_vs_share_ratio"] = spearman(difference, group["share_log2_ratio"])
        if "unmatched_pct" in group:
            row["r_difference_vs_unmatched"] = spearman(difference, group["unmatched_pct"])
            row["mean_difference_all_matched_mm"] = difference[group["unmatched_pct"] == 0].mean()
            row["mean_difference_some_unmatched_mm"] = difference[group["unmatched_pct"] > 0].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def diagnosis_figure(table: pd.DataFrame, summary: pd.DataFrame) -> plt.Figure:
    """Mean change per segment by method, and the basal differences against the possible causes.

    Args:
        table: Per subject and segment table (see ``segment_diagnosis``).
        summary: Table from ``segment_diagnosis``.

    Returns:
        The figure.
    """
    causes = [
        ("quarter_delta_mm", "Change, quarter method (mm)", "1. Size of the change"),
        ("share_log2_ratio", "log2 CGPS10 / CGPS1 share of the wall", "3. Segment covers more or less wall"),
    ]
    if "unmatched_pct" in table:
        causes.insert(1, ("unmatched_pct", "Unmatched vertices of the segment (%)", "2. Unmatched vertices"))
    fig, axes = plt.subplots(1, 1 + len(causes), figsize=(5.2 * (1 + len(causes)), 4.6), layout="constrained")

    positions = np.arange(len(WALL_SEGMENTS))
    width = 0.27
    for offset, (column, label) in zip((-width, 0, width), METHODS.items()):
        means = summary.set_index("segment")[f"mean_{column}"].reindex(WALL_SEGMENTS)
        axes[0].bar(positions + offset, means, width, label=label)
    axes[0].axhline(0, color="k", linewidth=0.8)
    axes[0].set(
        xticks=positions,
        xticklabels=WALL_SEGMENTS,
        xlabel="AHA segment",
        ylabel="Mean change (mm)",
        title="Mean CGPS10 − CGPS1 change per method",
    )
    axes[0].legend(fontsize=8)

    basal = table[table["segment"].isin(BASAL_SEGMENTS)]
    y_limits = np.nanpercentile(basal["difference_mm"], [1, 99])
    colours = plt.get_cmap("tab10")
    for ax, (column, label, title) in zip(axes[1:], causes):
        x_limits = np.nanpercentile(basal[column], [1, 99])
        for index, segment in enumerate(BASAL_SEGMENTS):
            rows = basal[basal["segment"] == segment]
            r = summary.set_index("segment").loc[segment]
            key = {
                "quarter_delta_mm": "r_difference_vs_change",
                "unmatched_pct": "r_difference_vs_unmatched",
                "share_log2_ratio": "r_difference_vs_share_ratio",
            }[column]
            ax.scatter(
                rows[column],
                rows["difference_mm"],
                s=4,
                alpha=0.3,
                color=colours(index),
                label=f"{segment} (r = {r[key]:+.2f})",
            )
        ax.axhline(0, color="k", linewidth=0.8)
        ax.set(xlim=x_limits, ylim=y_limits, xlabel=label, ylabel="Mesh − segment method (mm)", title=title)
        ax.legend(title="Segment (Spearman r)", fontsize=7, markerscale=3)
    fig.suptitle(
        f"Where does the mesh − segment bias come from? ({table[list(DATASETS)].drop_duplicates().shape[0]} "
        "subjects; scatter: basal segments, 1st-99th percentile shown)",
        fontsize=12,
    )
    return fig


def parse_args() -> argparse.Namespace:
    """Command line arguments."""
    parser = argparse.ArgumentParser(description="Possible causes of the mesh vs segment method bias per segment.")
    parser.add_argument("--workers", type=int, default=1, help="Number of subjects processed in parallel.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute the per-subject inputs even if saved.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root, _ = load_environment()
    mesh_folder = Path(run_output_dir(root, RUN_NAME, AORTA_EXCLUSION_MM))
    quarter_folder = Path(run_output_dir(root, QUARTER_RUN_NAME, AORTA_EXCLUSION_MM))
    folders = {dataset: Path(run_output_dir(root, dataset, AORTA_EXCLUSION_MM)) for dataset in DATASETS}

    comparison = read_ids(mesh_folder / COMPARISON_TABLE, list(DATASETS))
    inputs_path = mesh_folder / SUBJECT_FILE
    if args.overwrite or not inputs_path.is_file():
        task = partial(subject_inputs, folders=folders, quarter_folder=quarter_folder)
        results = run_subjects(
            task, pair_list(comparison.drop_duplicates(list(DATASETS))), args.workers, lambda result: "done"
        )
        pd.DataFrame([row for result in results for row in result["rows"]]).to_csv(inputs_path, index=False)
    inputs = read_ids(inputs_path, list(DATASETS))

    keys = [*DATASETS, "segment"]
    table = comparison.merge(inputs, on=keys, how="inner")
    match_path = mesh_folder / MATCH_TABLE
    if match_path.is_file():
        match = read_ids(match_path, list(DATASETS))[[*keys, "unmatched_pct"]]
        table = table.merge(match, on=keys, how="left")
    else:
        print(f"[WARN] {match_path} not found (run match_distance.py); cause 2 (unmatched vertices) is left out.")
    print(f"{table[list(DATASETS)].drop_duplicates().shape[0]} subjects with all three methods.")

    summary = segment_diagnosis(table)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(summary.round(2).to_string(index=False))

    figure_folder = Path(root) / FIGURE_FOLDER
    figure_folder.mkdir(parents=True, exist_ok=True)
    summary.to_csv(figure_folder / f"{SUMMARY_NAME}.csv", index=False)
    table.to_csv(mesh_folder / f"{SUMMARY_NAME}_per_subject.csv", index=False)
    figure = diagnosis_figure(table, summary)
    figure.savefig(figure_folder / f"{SUMMARY_NAME}.png", dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Per subject table saved to: {mesh_folder / f'{SUMMARY_NAME}_per_subject.csv'}")
    print(f"Summary and figure saved to: {figure_folder / SUMMARY_NAME}.csv/.png")

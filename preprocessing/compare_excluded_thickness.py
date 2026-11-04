"""Compare wall thickness and LV metrics between all, kept and excluded scans.

Combines the statistics tables written by measurements/main_statistics_df.py with the exclusion table written by
analyse_ct_quality.py, and checks whether the scans excluded for limited RV contrast or low CNR have different
measurements than the kept ones. The septal segments (2, 3, 8, 9 and 14) border the RV, so they are the ones most
likely to be affected by limited RV contrast.

For every AHA segment and every patient-level metric the mean and standard deviation per group are reported, with
the difference between kept and excluded scans and the p-value of a Mann-Whitney U test (not corrected for multiple
testing).

Inputs and outputs:
    <run folder>/statistics/statistics_thickness_per_segment.csv            (from main_statistics_df.py)
    <run folder>/statistics/statistics_patient_summary.csv                  (from main_statistics_df.py)
    <PROJECT_ROOT>/df/preprocessing/<dataset>_ct_quality_exclusion.csv      (from analyse_ct_quality.py)
    <PROJECT_ROOT>/df/preprocessing/<dataset>_<mm>mm_thickness_per_segment_by_quality.csv
    <PROJECT_ROOT>/df/preprocessing/<dataset>_<mm>mm_patient_summary_by_quality.csv
    <PROJECT_ROOT>/reports/figures/preprocessing/<dataset>_<mm>mm_thickness_by_quality.png

Examples:
    uv run preprocessing/compare_excluded_thickness.py --dataset ImageCAS_1-200
    uv run preprocessing/compare_excluded_thickness.py --dataset ImageCAS_1-200 --aorta-exclusion-mm 2.5
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "LVM_thickness"))

from pipeline import AORTA_EXCLUSION_MM, load_environment, run_output_dir  # noqa: E402

SEPTAL_SEGMENTS = [2, 3, 8, 9, 14]
PATIENT_METRICS = [
    "lvm_max_thickness_mm",
    "lvm_average_thickness_mm",
    "MadWT_mm",
    "SLR",
    "lvm_volume_ml",
    "lvm_g",
    "lv_blood_pool_volume_ml",
    "Sphericity_index",
]
GROUPS = {"All": None, "Kept": False, "Excluded": True}
GROUP_COLORS = {"All": "#7a7974", "Kept": "#2a78d6", "Excluded": "#eb6834"}
INK_COLOR = "#3d3d3a"
GRID_COLOR = "#e4e3dd"
SEPTUM_COLOR = "#f1f0ec"


def compare_groups(table: pd.DataFrame, value: str) -> dict[str, float]:
    """Mean and standard deviation of ``value`` per group, and the kept-vs-excluded difference and p-value.

    Args:
        table: Measurements with a boolean ``excluded`` column.
        value: Column to compare.

    Returns:
        One result row.
    """
    row: dict[str, float] = {}
    for group, excluded in GROUPS.items():
        values = table[value] if excluded is None else table.loc[table["excluded"] == excluded, value]
        values = values.dropna()
        row[f"n_{group.lower()}"] = len(values)
        row[f"mean_{group.lower()}"] = values.mean()
        row[f"std_{group.lower()}"] = values.std()
    kept = table.loc[~table["excluded"], value].dropna()
    excluded = table.loc[table["excluded"], value].dropna()
    row["difference_kept_minus_excluded"] = row["mean_kept"] - row["mean_excluded"]
    row["p_value"] = mannwhitneyu(kept, excluded).pvalue if len(kept) and len(excluded) else np.nan
    return row


def segment_comparison(segments: pd.DataFrame) -> pd.DataFrame:
    """Compare the median thickness per AHA segment between the groups.

    Args:
        segments: Per-segment statistics with a boolean ``excluded`` column.

    Returns:
        One row per segment.
    """
    rows = [
        {"segment": segment, "septal": segment in SEPTAL_SEGMENTS, **compare_groups(group, "median_thickness")}
        for segment, group in segments.groupby("segment")
    ]
    return pd.DataFrame(rows)


def patient_comparison(patients: pd.DataFrame) -> pd.DataFrame:
    """Compare the patient-level metrics between the groups.

    Args:
        patients: Patient summary with a boolean ``excluded`` column.

    Returns:
        One row per metric.
    """
    return pd.DataFrame([{"metric": metric, **compare_groups(patients, metric)} for metric in PATIENT_METRICS])


def plot_segments(comparison: pd.DataFrame, title: str, out_path: str) -> None:
    """Mean median thickness ± one standard deviation per segment for all, kept and excluded scans.

    Args:
        comparison: Table from ``segment_comparison``.
        title: Figure title.
        out_path: Path of the saved figure.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    x = comparison["segment"].to_numpy()
    for segment in SEPTAL_SEGMENTS:
        ax.axvspan(segment - 0.45, segment + 0.45, color=SEPTUM_COLOR, zorder=0)

    offsets = {"All": -0.22, "Kept": 0.0, "Excluded": 0.22}
    for group, color in GROUP_COLORS.items():
        key = group.lower()
        n = int(comparison[f"n_{key}"].max())
        ax.errorbar(x + offsets[group], comparison[f"mean_{key}"], yerr=comparison[f"std_{key}"], fmt="o",
                    color=color, markersize=7, markeredgecolor="white", elinewidth=2, capsize=0,
                    label=f"{group} (n = {n})")

    for _, row in comparison[comparison["p_value"] < 0.05].iterrows():
        top = max(row["mean_kept"] + row["std_kept"], row["mean_excluded"] + row["std_excluded"])
        ax.text(row["segment"], top + 0.3, "*", ha="center", color=INK_COLOR, fontsize=14)

    ax.set_xticks(x, [f"{s}\nseptum" if s in SEPTAL_SEGMENTS else str(s) for s in x])
    ax.set(xlabel="AHA segment", ylabel="Median wall thickness (mm), mean ± SD", title=title)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=INK_COLOR)
    ax.legend(frameon=False, loc="upper right", ncols=3)
    ax.text(0.0, -0.2, "* p < 0.05, kept vs excluded (Mann-Whitney U, uncorrected)", transform=ax.transAxes,
            color=INK_COLOR, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compare thickness and LV metrics of all, kept and excluded scans.")
    parser.add_argument("--dataset", required=True, help="Dataset name, as given to main_pipeline.py.")
    parser.add_argument("--aorta-exclusion-mm", type=float, default=AORTA_EXCLUSION_MM,
                        help=f"Aortic exclusion distance of the run to analyse (default: {AORTA_EXCLUSION_MM}).")
    parser.add_argument("--output-dir", default=None, help="Parent of the run folders (default: <PROJECT_ROOT>/output).")
    return parser.parse_args()


def main() -> None:
    """Print the comparison and save the tables and figure."""
    args = parse_args()
    root, _ = load_environment()
    run_folder = run_output_dir(root, args.dataset, args.aorta_exclusion_mm, args.output_dir)
    statistics_folder = os.path.join(run_folder, "statistics")
    table_dir = os.path.join(root, "df", "preprocessing")
    figure_dir = os.path.join(root, "reports", "figures", "preprocessing")
    os.makedirs(figure_dir, exist_ok=True)

    exclusion_path = os.path.join(table_dir, f"{args.dataset}_ct_quality_exclusion.csv")
    if not os.path.isfile(exclusion_path):
        raise SystemExit(f"[ERROR] {exclusion_path} not found; run preprocessing/analyse_ct_quality.py first.")
    if not os.path.isdir(statistics_folder):
        raise SystemExit(f"[ERROR] {statistics_folder} not found; run measurements/main_statistics_df.py first.")

    exclusion = pd.read_csv(exclusion_path, dtype={"patient_id": str})[["patient_id", "excluded"]]
    segments = pd.read_csv(os.path.join(statistics_folder, "statistics_thickness_per_segment.csv"),
                           dtype={"patient_id": str})
    patients = pd.read_csv(os.path.join(statistics_folder, "statistics_patient_summary.csv"), dtype={"patient_id": str})

    missing = sorted(set(patients["patient_id"]) - set(exclusion["patient_id"]), key=lambda p: (len(p), p))
    if missing:
        print(f"[WARN] {len(missing)} patients have no quality check and are left out: {', '.join(missing)}")
    segments = segments.merge(exclusion, on="patient_id")
    patients = patients.merge(exclusion, on="patient_id")

    n_excluded = int(patients["excluded"].sum())
    print(f"{args.dataset}, aorta exclusion {args.aorta_exclusion_mm:g} mm: {len(patients)} patients, "
          f"{len(patients) - n_excluded} kept, {n_excluded} excluded")

    columns = ["mean_all", "mean_kept", "mean_excluded", "difference_kept_minus_excluded", "p_value"]
    segment_table = segment_comparison(segments)
    print("\nMedian thickness per segment (mm):")
    print(segment_table.set_index("segment")[["septal", *columns]].round(3).to_string())

    patient_table = patient_comparison(patients)
    print("\nPatient-level metrics:")
    print(patient_table.set_index("metric")[columns].round(3).to_string())

    name = f"{args.dataset}_{args.aorta_exclusion_mm:g}mm"
    segment_path = os.path.join(table_dir, f"{name}_thickness_per_segment_by_quality.csv")
    patient_path = os.path.join(table_dir, f"{name}_patient_summary_by_quality.csv")
    figure_path = os.path.join(figure_dir, f"{name}_thickness_by_quality.png")
    segment_table.to_csv(segment_path, index=False)
    patient_table.to_csv(patient_path, index=False)
    plot_segments(segment_table, f"Wall thickness per segment, {name}: all vs kept vs excluded scans", figure_path)
    print(f"\nTables:\n  {segment_path}\n  {patient_path}\nFigure: {figure_path}")


if __name__ == "__main__":
    main()

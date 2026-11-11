"""
Analyse the output of ct_quality_checks.py and decide which scans to exclude.

A scan is excluded only if it has both limited RV contrast (k >= K_THRESHOLD) and low image quality
(CNR < CNR_THRESHOLD).
The script prints how many scans each criterion excludes and saves histograms of k and CNR and a k-vs-CNR scatter
plot with the thresholds, and a table with the exclusion flags.

Inputs and outputs:
    <PROJECT_ROOT>/df/preprocessing/<dataset>_ct_quality_exclusion.csv
    <PROJECT_ROOT>/reports/figures/preprocessing/<dataset>_ct_quality.png

Examples:
    uv run preprocessing/analyse_ct_quality.py --dataset ImageCAS_1-200 --k-threshold 5 --cnr-threshold 6
    uv run preprocessing/analyse_ct_quality.py --dataset ImageCAS_1-200 --k-threshold 1 --k-only
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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "LVM_thickness"))

from pipeline import load_environment  # noqa: E402

K_THRESHOLD = 1
CNR_THRESHOLD = 5.0

KEPT_COLOR = "#2a78d6"
EXCLUDED_COLOR = "#eb6834"
INK_COLOR = "#3d3d3a"
GRID_COLOR = "#e4e3dd"


def add_exclusion_flags(
    results: pd.DataFrame, k_threshold: int, cnr_threshold: float, k_only: bool = False
) -> pd.DataFrame:
    """Add a boolean column per exclusion criterion and one for the combined decision.

    Args:
        results: Output table of ct_quality_checks.py.
        k_threshold: Scans with k >= this value have limited RV contrast.
        cnr_threshold: Scans with CNR below this value have low image quality.
        k_only: Exclude on limited RV contrast alone and ignore CNR.

    Returns:
        Table with ``patient_id``, ``k``, ``cnr`` and the columns ``low_rv_contrast``, ``low_cnr`` and ``excluded``.
    """
    flagged = results[["patient_id", "k", "cnr"]].copy()
    flagged["low_rv_contrast"] = flagged["k"] >= k_threshold
    flagged["low_cnr"] = flagged["cnr"] < cnr_threshold
    if k_only:
        flagged["excluded"] = flagged["low_rv_contrast"]
    else:
        flagged["excluded"] = flagged["low_rv_contrast"] & flagged["low_cnr"]
    return flagged


def exclusion_summary(flagged: pd.DataFrame, k_only: bool = False) -> pd.DataFrame:
    """Number and percentage of scans per exclusion group.

    Args:
        flagged: Table from ``add_exclusion_flags``.
        k_only: Whether scans were excluded on limited RV contrast alone.

    Returns:
        One row per group with the columns ``n`` and ``percent``.
    """
    if k_only:
        groups = {
            "Low RV contrast (excluded)": flagged["excluded"],
            "Kept in total": ~flagged["excluded"],
        }
    else:
        groups = {
            "Low RV contrast only (kept)": flagged["low_rv_contrast"] & ~flagged["low_cnr"],
            "Low CNR only (kept)": flagged["low_cnr"] & ~flagged["low_rv_contrast"],
            "Both (excluded)": flagged["excluded"],
            "Kept in total": ~flagged["excluded"],
        }
    counts = pd.Series({name: int(mask.sum()) for name, mask in groups.items()}, name="n")
    return pd.DataFrame({"n": counts, "percent": (100 * counts / len(flagged)).round(1)})


def style_axis(ax: plt.Axes) -> None:
    """Recessive grid and spines, matching the ink colour of the labels."""
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=INK_COLOR)


def plot_quality(flagged: pd.DataFrame, k_threshold: int, cnr_threshold: float, out_path: str) -> None:
    """Histograms of k and CNR and a k-vs-CNR scatter plot, with the thresholds marked.

    Args:
        flagged: Table from ``add_exclusion_flags``.
        k_threshold: k threshold used for exclusion.
        cnr_threshold: CNR threshold used for exclusion.
        out_path: Path of the saved figure.
    """
    fig, (ax_k, ax_cnr, ax_scatter) = plt.subplots(1, 3, figsize=(16, 4.8))
    edgecolor = "white"

    k_bins = np.concatenate([[0, 1], np.arange(6, 102, 5)])
    ax_k.hist(flagged["k"], bins=k_bins, color=KEPT_COLOR, edgecolor=edgecolor, linewidth=1)
    ax_k.axvline(k_threshold, color=INK_COLOR, linestyle="--", linewidth=1.5)
    n_zero = int((flagged["k"] == 0).sum())
    ax_k.text(k_threshold + 2, n_zero, f"k = 0: {n_zero} scans\nthreshold k ≥ {k_threshold}", color=INK_COLOR,
              va="top")
    ax_k.set(xlabel="k (t-tests with no RV–myocardium difference, of 100)", ylabel="Number of scans",
             title="RV contrast")

    cnr_bins = np.arange(np.floor(flagged["cnr"].min()) - 1, np.ceil(flagged["cnr"].max()) + 0.5, 0.5)
    ax_cnr.hist(flagged["cnr"], bins=cnr_bins, color=KEPT_COLOR, edgecolor=edgecolor, linewidth=2)
    ax_cnr.axvline(cnr_threshold, color=INK_COLOR, linestyle="--", linewidth=1.5)
    ax_cnr.text(cnr_threshold + 0.1, ax_cnr.get_ylim()[1] * 0.95, f"CNR < {cnr_threshold:g}", color=INK_COLOR,
                va="top")
    ax_cnr.set(xlabel="Contrast-to-noise ratio (aorta core vs myocardium)", ylabel="Number of scans",
               title="Image quality")

    for excluded, color, label in ((False, KEPT_COLOR, "Kept"), (True, EXCLUDED_COLOR, "Excluded")):
        group = flagged[flagged["excluded"] == excluded]
        ax_scatter.scatter(group["k"], group["cnr"], s=40, color=color, edgecolor="white", linewidth=1,
                           label=f"{label} (n = {len(group)})")
    ax_scatter.axvline(k_threshold, color=INK_COLOR, linestyle="--", linewidth=1.5)
    ax_scatter.axhline(cnr_threshold, color=INK_COLOR, linestyle="--", linewidth=1.5)
    ax_scatter.set(xlabel="k", ylabel="CNR", title="Exclusion per scan")
    ax_scatter.legend(frameon=False, loc="upper right")

    for ax in (ax_k, ax_cnr, ax_scatter):
        style_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyse RV contrast and CNR and choose which scans to exclude.")
    parser.add_argument("--dataset", required=True, help="Dataset name used when running ct_quality_checks.py.")
    parser.add_argument("--k-threshold", type=int, default=K_THRESHOLD,
                        help=f"Exclude scans with k >= this value (default: {K_THRESHOLD}).")
    parser.add_argument("--cnr-threshold", type=float, default=CNR_THRESHOLD,
                        help=f"Exclude scans with CNR below this value (default: {CNR_THRESHOLD:g}).")
    parser.add_argument("--k-only", action="store_true",
                        help="Exclude on k >= --k-threshold alone and ignore CNR. Outputs get a '_k_only' suffix.")
    return parser.parse_args()


def main() -> None:
    """Print the analysis and save the figure and exclusion table."""
    args = parse_args()
    root, _ = load_environment()
    table_dir = os.path.join(root, "df", "preprocessing")
    figure_dir = os.path.join(root, "reports", "figures", "preprocessing")
    os.makedirs(figure_dir, exist_ok=True)

    results = pd.read_csv(os.path.join(table_dir, f"{args.dataset}_ct_quality.csv"), dtype={"patient_id": str})
    failed = results["error"].notna()
    if failed.any():
        print(f"Skipping {failed.sum()} failed patients: {', '.join(results.loc[failed, 'patient_id'])}")
    results = results[~failed].reset_index(drop=True)

    flagged = add_exclusion_flags(results, args.k_threshold, args.cnr_threshold, args.k_only)
    if args.k_only:
        rule = f"k >= {args.k_threshold}"
    else:
        rule = f"k >= {args.k_threshold} and CNR < {args.cnr_threshold:g}"
    suffix = "_k_only" if args.k_only else ""

    print(f"\n{len(results)} scans in {args.dataset}\n")
    print(results[["k", "cnr"]].describe().round(2).to_string())
    print(f"\nScans with k = 0: {(results['k'] == 0).sum()}")
    print(f"Scans with k >= {args.k_threshold}: {flagged['low_rv_contrast'].sum()}")
    print(f"Scans with CNR < {args.cnr_threshold:g}: {flagged['low_cnr'].sum()}")
    print(f"Scans to exclude ({rule}): {flagged['excluded'].sum()} of {len(flagged)}")

    print(f"\nExclusion with {rule}:")
    print(exclusion_summary(flagged, args.k_only).to_string())
    if args.k_only:
        print(f"\nExcluded patient IDs: {', '.join(flagged.loc[flagged['excluded'], 'patient_id'])}")

    table_path = os.path.join(table_dir, f"{args.dataset}_ct_quality_exclusion{suffix}.csv")
    flagged.to_csv(table_path, index=False)
    figure_path = os.path.join(figure_dir, f"{args.dataset}_ct_quality{suffix}.png")
    plot_quality(flagged, args.k_threshold, args.cnr_threshold, figure_path)
    print(f"\nExclusion table: {table_path}\nFigure: {figure_path}")


if __name__ == "__main__":
    main()

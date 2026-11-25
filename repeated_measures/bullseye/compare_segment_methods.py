#################################################################################################################################################
# This script checks the per-subject change in wall thickness computed on the mesh (mesh_delta.py) against the per-segment
# change of bullseye_segment_delta.py, which needs no registration. Per subject and AHA segment it compares, in mm:
#   segment_delta_mm:  median CGPS10 thickness in the segment of the CGPS10 17-segment model minus median CGPS1 thickness in the segment
#                      of the CGPS1 model (thickness.json of each scan, as in statistics_thickness_per_segment.csv),
#   mesh_delta_mm:     median of the CGPS10 thickness moved onto the CGPS1 vertices of the segment minus the median CGPS1 thickness of
#                      those vertices (matched vertices only), so the same statistic but with the CGPS10 values taken where the
#                      registration put them,
#   vertex_delta_mm:   median of the per-vertex differences of the segment, as drawn in the mesh bullseye.
# With a correct registration mesh_delta_mm and segment_delta_mm agree; their difference (mesh - segment) shows where they do not.
# The CGPS1 vertices get their segment as in the pipeline: the nearest point of the CGPS1 surfaces/myocardium_17.vtk.
# It uses every subject in <PROJECT_ROOT>/output/CGPS1_CGPS10_mesh_delta_0mm/registration_qc.csv, so run mesh_delta.py first.
# The per-subject table is saved in that run folder; the agreement per segment and the figure (scatter, Bland-Altman and the mean
# difference per segment) in <PROJECT_ROOT>/reports/figures/repeated_measures/bullseye.
# e.g. uv run repeated_measures/bullseye/compare_segment_methods.py
#################################################################################################################################################

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from vtk.util.numpy_support import vtk_to_numpy  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "measurements"))

from bullseye_mesh_delta import FIGURE_FOLDER  # noqa: E402
from mesh_delta import AORTA_EXCLUSION_MM, DELTA_MESH, QC_FILE, RUN_NAME, mesh_points  # noqa: E402
from bullseye_segment_delta import WALL_SEGMENTS, draw_bullseye  # noqa: E402
from match_distance import SEGMENT_MESH, vertex_segments  # noqa: E402
from pipeline import load_environment, run_output_dir  # noqa: E402
from spaghetti_plot import DATASETS, read_ids  # noqa: E402
from thickness import load_thickness  # noqa: E402
from utils import utils  # noqa: E402

SUBJECT_TABLE = "segment_method_comparison.csv"
FIGURE_NAME = "segment_method_comparison"
CMAP = "RdBu_r"
N_WORST = 5


def mesh_segment_deltas(
    segments: np.ndarray, thickness_cgps1: np.ndarray, thickness_cgps10: np.ndarray
) -> pd.DataFrame:
    """Change per segment from the difference mesh, as a difference of medians and as a median of differences.

    Args:
        segments: (N,) AHA segment of each CGPS1 vertex.
        thickness_cgps1: (N,) CGPS1 thickness of each vertex.
        thickness_cgps10: (N,) CGPS10 thickness moved onto each vertex, NaN where unmatched.

    Returns:
        One row per segment with ``segment``, ``n_matched``, ``mesh_delta_mm`` and ``vertex_delta_mm``.
    """
    table = pd.DataFrame({"segment": segments, "cgps1": thickness_cgps1, "cgps10": thickness_cgps10}).dropna()
    table["delta"] = table["cgps10"] - table["cgps1"]
    grouped = table.groupby("segment")
    medians = grouped[["cgps1", "cgps10", "delta"]].median()
    return pd.DataFrame(
        {
            "n_matched": grouped.size(),
            "mesh_delta_mm": medians["cgps10"] - medians["cgps1"],
            "vertex_delta_mm": medians["delta"],
        }
    ).reset_index()


def segment_method_deltas(cgps1_dir: Path, cgps10_dir: Path) -> pd.DataFrame:
    """Change per segment of the segment method: difference of each scan's own segment medians.

    Args:
        cgps1_dir: Pipeline output folder of the CGPS1 scan.
        cgps10_dir: Pipeline output folder of the CGPS10 scan.

    Returns:
        One row per segment present in both scans with ``segment`` and ``segment_delta_mm``.
    """
    medians = [
        {segment: float(np.median(values)) for segment, values in load_thickness(str(folder)).items() if len(values)}
        for folder in (cgps1_dir, cgps10_dir)
    ]
    shared = sorted(set(medians[0]) & set(medians[1]))
    return pd.DataFrame(
        {"segment": shared, "segment_delta_mm": [medians[1][segment] - medians[0][segment] for segment in shared]}
    )


def subject_comparison(run_folder: Path, folders: dict[str, Path], subjects: pd.DataFrame) -> pd.DataFrame:
    """Both methods per subject and wall segment for every subject with a difference mesh.

    Args:
        run_folder: Run folder of mesh_delta.py.
        folders: Dataset name to its pipeline run folder.
        subjects: Table with the ids in ``CGPS1`` and ``CGPS10``.

    Returns:
        One row per subject and segment 1-16 with the ids, ``n_matched``, ``segment_delta_mm``, ``mesh_delta_mm``,
        ``vertex_delta_mm`` and ``difference_mm`` (mesh - segment).
    """
    rows = []
    for cgps1, cgps10 in subjects[list(DATASETS)].itertuples(index=False):
        mesh_path = run_folder / f"{cgps1}_{cgps10}" / DELTA_MESH
        if not mesh_path.is_file():
            print(f"[WARN] {cgps1} / {cgps10}: no {DELTA_MESH}, left out.")
            continue
        mesh = utils.read_vtk_mesh(str(mesh_path))
        data = mesh.GetPointData()
        segments = vertex_segments(mesh_points(mesh), folders["CGPS1"] / cgps1 / SEGMENT_MESH)
        mesh_table = mesh_segment_deltas(
            segments,
            vtk_to_numpy(data.GetArray("thickness_CGPS1")).astype(np.float64),
            vtk_to_numpy(data.GetArray("thickness_CGPS10")).astype(np.float64),
        )
        segment_table = segment_method_deltas(folders["CGPS1"] / cgps1, folders["CGPS10"] / cgps10)
        table = mesh_table.merge(segment_table, on="segment", how="inner")
        table = table[table["segment"].isin(WALL_SEGMENTS)]
        table.insert(0, "CGPS10", cgps10)
        table.insert(0, "CGPS1", cgps1)
        rows.append(table)
    comparison = pd.concat(rows, ignore_index=True)
    comparison["difference_mm"] = comparison["mesh_delta_mm"] - comparison["segment_delta_mm"]
    return comparison


def agreement(differences: pd.Series, mesh: pd.Series, segment: pd.Series) -> dict[str, float]:
    """Bland-Altman agreement and correlation of the two methods.

    Args:
        differences: Mesh minus segment method.
        mesh: Mesh method values.
        segment: Segment method values.

    Returns:
        ``n``, ``bias_mm``, ``sd_mm``, ``loa_low_mm``, ``loa_high_mm`` (bias ± 1.96 sd), ``mean_abs_mm`` and ``r``.
    """
    bias, sd = float(differences.mean()), float(differences.std())
    return {
        "n": len(differences),
        "bias_mm": bias,
        "sd_mm": sd,
        "loa_low_mm": bias - 1.96 * sd,
        "loa_high_mm": bias + 1.96 * sd,
        "mean_abs_mm": float(differences.abs().mean()),
        "r": float(np.corrcoef(mesh, segment)[0, 1]) if len(differences) > 2 else np.nan,
    }


def agreement_table(comparison: pd.DataFrame) -> pd.DataFrame:
    """Agreement of the two methods over all wall segments and per segment.

    Args:
        comparison: Table from ``subject_comparison``.

    Returns:
        One row for all segments (``segment`` "all") and one per segment, with the columns of ``agreement``.
    """
    rows = [
        {
            "segment": "all",
            **agreement(comparison["difference_mm"], comparison["mesh_delta_mm"], comparison["segment_delta_mm"]),
        }
    ]
    for segment, group in comparison.groupby("segment"):
        rows.append(
            {"segment": segment, **agreement(group["difference_mm"], group["mesh_delta_mm"], group["segment_delta_mm"])}
        )
    return pd.DataFrame(rows)


def comparison_figure(comparison: pd.DataFrame, summary: pd.DataFrame) -> plt.Figure:
    """Scatter of the two methods, Bland-Altman plot and bullseye of the mean difference per segment.

    Args:
        comparison: Table from ``subject_comparison``.
        summary: Table from ``agreement_table``.

    Returns:
        The figure.
    """
    overall = summary.iloc[0]
    fig = plt.figure(figsize=(16, 5.4), layout="constrained")
    scatter_ax = fig.add_subplot(1, 3, 1)
    bland_ax = fig.add_subplot(1, 3, 2)
    bullseye_ax = fig.add_subplot(1, 3, 3, polar=True)

    ring = np.select([comparison["segment"] <= 6, comparison["segment"] <= 12], ["basal", "mid"], "apical")
    for name, colour in [("basal", "#1b9e77"), ("mid", "#d95f02"), ("apical", "#7570b3")]:
        rows = comparison[ring == name]
        scatter_ax.scatter(rows["segment_delta_mm"], rows["mesh_delta_mm"], s=10, alpha=0.6, color=colour, label=name)
        mean = (rows["segment_delta_mm"] + rows["mesh_delta_mm"]) / 2
        bland_ax.scatter(mean, rows["difference_mm"], s=10, alpha=0.6, color=colour, label=name)
    limits = np.nanpercentile(np.abs(comparison[["segment_delta_mm", "mesh_delta_mm"]].to_numpy()), 99.5)
    scatter_ax.plot([-limits, limits], [-limits, limits], "k--", linewidth=1)
    scatter_ax.set(
        xlim=(-limits, limits),
        ylim=(-limits, limits),
        aspect="equal",
        xlabel="Segment method: change of segment medians (mm)",
        ylabel="Mesh method: change of segment medians (mm)",
        title=f"Per subject and segment (r = {overall['r']:.2f})",
    )
    scatter_ax.legend(title="Ring", fontsize=8)

    for value, style in [(overall["bias_mm"], "-"), (overall["loa_low_mm"], "--"), (overall["loa_high_mm"], "--")]:
        bland_ax.axhline(value, color="k", linestyle=style, linewidth=1)
    bland_ax.set(
        xlabel="Mean of the two methods (mm)",
        ylabel="Mesh − segment method (mm)",
        title=f"Bland-Altman: bias {overall['bias_mm']:+.2f} mm, "
        f"LoA {overall['loa_low_mm']:+.2f} to {overall['loa_high_mm']:+.2f}",
    )

    per_segment = summary[summary["segment"] != "all"].astype({"segment": int}).set_index("segment")
    values = per_segment["bias_mm"].reindex(WALL_SEGMENTS).to_numpy(dtype=float)
    vmax = max(float(np.nanmax(np.abs(values))), 0.1)
    draw_bullseye(
        bullseye_ax, values, Normalize(-vmax, vmax), CMAP, "Mean difference per segment (mesh − segment)", "{:+.2f}"
    )
    fig.suptitle(
        f"Mesh vs segment method for the CGPS10 − CGPS1 wall thickness change "
        f"(n = {comparison[list(DATASETS)].drop_duplicates().shape[0]} subjects)",
        fontsize=13,
    )
    return fig


if __name__ == "__main__":
    root, _ = load_environment()
    run_folder = Path(run_output_dir(root, RUN_NAME, AORTA_EXCLUSION_MM))
    folders = {dataset: Path(run_output_dir(root, dataset, AORTA_EXCLUSION_MM)) for dataset in DATASETS}
    subjects = read_ids(run_folder / QC_FILE, list(DATASETS))

    comparison = subject_comparison(run_folder, folders, subjects)
    comparison.to_csv(run_folder / SUBJECT_TABLE, index=False)
    summary = agreement_table(comparison)
    print("Agreement, mesh - segment method (mm):")
    print(summary.round(2).to_string(index=False))

    worst = comparison.assign(abs_difference=comparison["difference_mm"].abs())
    worst = worst.groupby(list(DATASETS))["abs_difference"].mean().sort_values(ascending=False).head(N_WORST)
    print(f"\nSubjects with the largest mean |mesh - segment| over the wall segments (mm):\n{worst.round(2)}")

    figure_folder = Path(root) / FIGURE_FOLDER
    figure_folder.mkdir(parents=True, exist_ok=True)
    summary.to_csv(figure_folder / f"{FIGURE_NAME}.csv", index=False)
    figure = comparison_figure(comparison, summary)
    figure.savefig(figure_folder / f"{FIGURE_NAME}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Per subject table saved to: {run_folder / SUBJECT_TABLE}")
    print(f"Agreement and figure saved to: {figure_folder / FIGURE_NAME}.csv/.png")

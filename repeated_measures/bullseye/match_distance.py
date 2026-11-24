#################################################################################################################################################
# This script measures, per AHA segment, how far the CGPS1 endocardial vertices had to reach for their CGPS10 partner when the wall
# thickness was moved between the examinations by bullseye_mesh_delta.py. A large match distance means the endocardium moved (e.g. the
# wall grew into the cavity) or the registration is off there, so the thickness change of that area is unreliable or missing.
# Every vertex of each subject's delta_mesh.vtk gets the AHA segment of its nearest point on the CGPS1 surfaces/myocardium_17.vtk (the rule
# of the pipeline's thickness per segment; -1 is outside the 17-segment model). Per subject and segment it counts the vertices, the median
# and 95th percentile match distance, the share above --threshold-mm and the share without a match (no CGPS10 vertex within the maximum
# distance used by bullseye_mesh_delta.py, so no change).
# It uses every subject listed in <PROJECT_ROOT>/output/CGPS1_CGPS10_mesh_delta_0mm/registration_qc.csv, so run bullseye_mesh_delta.py
# first. The per-subject table is saved in that run folder, and the per-segment summary over subjects and its bullseye figure in
# <PROJECT_ROOT>/reports/figures/repeated_measures/bullseye.
# e.g. uv run repeated_measures/bullseye/match_distance.py --threshold-mm 3
#################################################################################################################################################

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from vtk.util.numpy_support import vtk_to_numpy  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bullseye_mesh_delta import (  # noqa: E402
    AORTA_EXCLUSION_MM,
    DELTA_MESH,
    FIGURE_FOLDER,
    QC_FILE,
    RUN_NAME,
    mesh_points,
)
from bullseye_segment_delta import WALL_SEGMENTS, draw_bullseye  # noqa: E402
from pipeline import load_environment, run_output_dir  # noqa: E402
from spaghetti_plot import DATASETS, read_ids  # noqa: E402
from utils import utils  # noqa: E402

SEGMENT_MESH = os.path.join("surfaces", "myocardium_17.vtk")
THRESHOLD_MM = 3.0
SUBJECT_TABLE = "match_distance_per_segment.csv"
SUMMARY_NAME = "match_distance_summary"
OUTSIDE_SEGMENT = -1
CMAP = "Reds"


def vertex_segments(vertices: np.ndarray, segment_mesh_path: Path) -> np.ndarray:
    """AHA segment of every vertex, from the nearest point of the 17-segment mesh.

    Args:
        vertices: (N, 3) vertices in the physical space of the 17-segment mesh.
        segment_mesh_path: Mesh whose point scalars are segment labels (-1 outside the model).

    Returns:
        (N,) integer segment labels.
    """
    mesh = utils.read_vtk_mesh(str(segment_mesh_path))
    labels = vtk_to_numpy(mesh.GetPointData().GetScalars()).astype(int)
    _, nearest = cKDTree(mesh_points(mesh)).query(vertices)
    return labels[nearest]


def segment_match_distance(
    distance: np.ndarray, matched: np.ndarray, segments: np.ndarray, threshold: float
) -> pd.DataFrame:
    """Match distance statistics per segment of one subject.

    Args:
        distance: (N,) distance in mm of each CGPS1 vertex to its nearest registered CGPS10 vertex.
        matched: (N,) whether the vertex got a CGPS10 thickness.
        segments: (N,) AHA segment of each vertex.
        threshold: Distance in mm above which a match counts as far.

    Returns:
        One row per segment with ``segment``, ``n_vertices``, ``median_mm``, ``p95_mm``, ``far_pct`` (% above
        ``threshold``) and ``unmatched_pct``.
    """
    table = pd.DataFrame(
        {"segment": segments, "distance": distance, "far": distance > threshold, "unmatched": ~matched}
    )
    grouped = table.groupby("segment")
    return pd.DataFrame(
        {
            "n_vertices": grouped.size(),
            "median_mm": grouped["distance"].median(),
            "p95_mm": grouped["distance"].quantile(0.95),
            "far_pct": 100 * grouped["far"].mean(),
            "unmatched_pct": 100 * grouped["unmatched"].mean(),
        }
    ).reset_index()


def subject_table(run_folder: Path, cgps1_folder: Path, subjects: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Match distance statistics per subject and segment of every subject with a difference mesh.

    Args:
        run_folder: Run folder of bullseye_mesh_delta.py.
        cgps1_folder: Pipeline run folder of CGPS1.
        subjects: Table with the ids in ``CGPS1`` and ``CGPS10``.
        threshold: Distance in mm above which a match counts as far.

    Returns:
        The rows of ``segment_match_distance`` of every subject, with ``CGPS1`` and ``CGPS10`` in front.
    """
    rows = []
    for cgps1, cgps10 in subjects[list(DATASETS)].itertuples(index=False):
        mesh_path = run_folder / f"{cgps1}_{cgps10}" / DELTA_MESH
        if not mesh_path.is_file():
            print(f"[WARN] {cgps1} / {cgps10}: no {DELTA_MESH}, left out.")
            continue
        mesh = utils.read_vtk_mesh(str(mesh_path))
        data = mesh.GetPointData()
        distance = vtk_to_numpy(data.GetArray("match_distance_mm")).astype(np.float64)
        matched = np.isfinite(vtk_to_numpy(data.GetArray("delta_mm")))
        segments = vertex_segments(mesh_points(mesh), cgps1_folder / cgps1 / SEGMENT_MESH)
        table = segment_match_distance(distance, matched, segments, threshold)
        table.insert(0, "CGPS10", cgps10)
        table.insert(0, "CGPS1", cgps1)
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def segment_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Per segment summary over subjects of the table from ``subject_table``.

    Args:
        table: One row per subject and segment.

    Returns:
        One row per segment with ``n_subjects``, the mean over subjects of ``far_pct`` and ``unmatched_pct``, the median
        over subjects of ``median_mm`` and ``p95_mm``, and ``max_far_pct`` (the worst subject).
    """
    grouped = table.groupby("segment")
    return pd.DataFrame(
        {
            "n_subjects": grouped.size(),
            "mean_far_pct": grouped["far_pct"].mean(),
            "max_far_pct": grouped["far_pct"].max(),
            "mean_unmatched_pct": grouped["unmatched_pct"].mean(),
            "median_median_mm": grouped["median_mm"].median(),
            "median_p95_mm": grouped["p95_mm"].median(),
        }
    ).reset_index()


def summary_figure(summary: pd.DataFrame, threshold: float, n_subjects: int) -> plt.Figure:
    """Bullseyes of the mean share of far and of unmatched vertices per segment.

    Args:
        summary: Table from ``segment_summary``.
        threshold: Distance in mm above which a match counts as far.
        n_subjects: Number of subjects in the summary.

    Returns:
        The figure.
    """
    panels = [
        ("mean_far_pct", f"Vertices matched > {threshold:g} mm away"),
        ("mean_unmatched_pct", "Vertices without a match (no change)"),
    ]
    values = summary.set_index("segment")
    vmax = max(float(np.nanmax(values.reindex(WALL_SEGMENTS)[column])) for column, _ in panels)
    norm = Normalize(vmin=0, vmax=max(vmax, 1))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 5.6), subplot_kw=dict(polar=True), layout="constrained")
    for ax, (column, title) in zip(axes, panels):
        draw_bullseye(ax, values[column].reindex(WALL_SEGMENTS).to_numpy(dtype=float), norm, CMAP, title, "{:.1f}%")
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=CMAP), ax=axes, orientation="horizontal", shrink=0.5, aspect=40, pad=0.04
    )
    colorbar.set_label("Mean share of the segment's CGPS1 endocardial vertices over subjects (%)")
    fig.suptitle(f"How far the CGPS1 vertices reach for their CGPS10 partner (n = {n_subjects})", fontsize=13, y=1.06)
    return fig


def parse_args() -> argparse.Namespace:
    """Command line arguments."""
    parser = argparse.ArgumentParser(description="Match distance between CGPS1 and CGPS10 vertices per AHA segment.")
    parser.add_argument("--threshold-mm", type=float, default=THRESHOLD_MM, help="Distance that counts as far.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root, _ = load_environment()
    run_folder = Path(run_output_dir(root, RUN_NAME, AORTA_EXCLUSION_MM))
    subjects = read_ids(run_folder / QC_FILE, list(DATASETS))
    cgps1_folder = Path(run_output_dir(root, "CGPS1", AORTA_EXCLUSION_MM))

    table = subject_table(run_folder, cgps1_folder, subjects, args.threshold_mm)
    n_subjects = table[list(DATASETS)].drop_duplicates().shape[0]
    table.to_csv(run_folder / SUBJECT_TABLE, index=False)
    summary = segment_summary(table)
    print(f"{n_subjects} subjects, threshold {args.threshold_mm:g} mm (segment {OUTSIDE_SEGMENT} = outside the model):")
    print(summary.round(2).to_string(index=False))

    figure_folder = Path(root) / FIGURE_FOLDER
    figure_folder.mkdir(parents=True, exist_ok=True)
    name = f"{SUMMARY_NAME}_{args.threshold_mm:g}mm"
    summary.to_csv(figure_folder / f"{name}.csv", index=False)
    figure = summary_figure(summary, args.threshold_mm, n_subjects)
    figure.savefig(figure_folder / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Per subject table saved to: {run_folder / SUBJECT_TABLE}")
    print(f"Summary and figure saved to: {figure_folder / name}.csv/.png")

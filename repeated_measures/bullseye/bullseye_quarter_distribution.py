#################################################################################################################################################
# This script compares the distribution of the LV wall thickness of CGPS1 and CGPS10 in four parts of every AHA segment, per subject and
# without registration, and shows the changes as bullseye plots with 64 cells (no grouping, no means over subjects).
# Each scan's thickness (surfaces/dist_source.vtk) is projected into its own bullseye, i.e. with its own long axis and RV insertion
# (the lv17 frame of that scan), using the projected ring voxels themselves (one measurement each, no hole filling). Every point gets the
# AHA wedge it falls in as drawn in the bullseye (segments 1-16; the apex is left out) and one of four quarters of that wedge: split in
# two by radius (the half towards the base, i.e. outwards, or towards the apex) and in two by angle (the half towards the previous or
# the next segment number). Note that the drawn wedges follow the pipeline's segment borders only approximately (equal radial thirds).
# For each subject, segment and quarter the percentiles P10, P25, P50, P75 and P90 of the thickness are taken at both examinations, the
# change of each percentile (CGPS10 - CGPS1, in mm), and the Wasserstein (earth mover's) distance between the two distributions (mm, no
# sign). A quarter with fewer than --min-points points at either examination has no value.
# Each scan's bullseye points are saved in <PROJECT_ROOT>/output/CGPS1_CGPS10_quarters_0mm/<CGPS1 id>_<CGPS10 id>/ and the table of all
# processed subjects in quarter_distribution.csv in that run folder. One figure per measure of the --n random subjects (the same as in
# the mesh delta scripts for the same --n and --seed), and a figure of all measures side by side for the first --compare of them, are
# saved in <PROJECT_ROOT>/reports/figures/repeated_measures/bullseye. With --all every subject is processed (--workers in parallel).
# It needs the links table and the finished pipeline runs with 0 mm aortic exclusion of both datasets; no registration is used.
# e.g. uv run repeated_measures/bullseye/bullseye_quarter_distribution.py --n 20 --seed 0 --workers 4
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
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bullseye_mesh_delta import (  # noqa: E402
    CMAP,
    FIGURE_FOLDER,
    MISSING_COLOUR,
    RADII,
    RINGS,
    draw_outline,
    figure_name,
    load_polar,
    project_mesh,
)
from mesh_delta import (  # noqa: E402
    AORTA_EXCLUSION_MM,
    THICKNESS_MESH,
    add_subject_arguments,
    pair_list,
    run_subjects,
    setup_subjects,
)
from pipeline import run_output_dir  # noqa: E402
from spaghetti_plot import DATASETS, read_ids  # noqa: E402
from utils import utils  # noqa: E402

RUN_NAME = "CGPS1_CGPS10_quarters"
POINTS_FILE = "points_{dataset}.npz"
TABLE_FILE = "quarter_distribution.csv"
PERCENTILES = [10, 25, 50, 75, 90]
QUARTERS = 4
MIN_POINTS = 30
N_COMPARE = 5
WASSERSTEIN = "wasserstein_mm"
WASSERSTEIN_CMAP = "Reds"


def quarter_labels(theta: np.ndarray, radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """AHA segment (1-16) and quarter (0-3) of bullseye points, from the wedges as drawn in the bullseye.

    The quarter is ``2 * outer + next``: ``outer`` is the half of the wedge's ring towards the base (larger radius),
    ``next`` the half of its angle towards the next segment number. Points in the apex or outside the bullseye get -1.

    Args:
        theta: Angles in radians.
        radius: Radii normalised so the AHA rings lie at 0.2, 0.47, 0.73 and 1.

    Returns:
        (N,) segments and (N,) quarters.
    """
    segments = np.full(len(theta), -1)
    quarters = np.full(len(theta), -1)
    for (start, stop), r_in, r_out in zip(RINGS, RADII[2::-1], RADII[:0:-1]):
        n = stop - start
        width = 2 * np.pi / n
        inside = (radius >= r_in) & (radius < r_out)
        relative = np.mod(theta - np.pi / 2 + width / 2, 2 * np.pi)
        wedge = np.floor(relative / width).astype(int) % n
        outer = radius >= (r_in + r_out) / 2
        upper = relative - wedge * width >= width / 2
        segments[inside] = start + wedge[inside] + 1
        quarters[inside] = (2 * outer + upper)[inside]
    return segments, quarters


def scan_points(scan_dir: Path, points_path: Path, overwrite: bool) -> dict[str, np.ndarray]:
    """A scan's thickness as bullseye points in its own frame, projected once and then read from ``points_path``."""
    if overwrite or not points_path.is_file():
        mesh = utils.read_vtk_mesh(str(scan_dir / THICKNESS_MESH))
        np.savez_compressed(points_path, **project_mesh(scan_dir, mesh, fill=False))
    return load_polar(points_path)


def distribution_changes(points: dict[str, dict[str, np.ndarray]], min_points: int = MIN_POINTS) -> pd.DataFrame:
    """Percentiles at both examinations, their change and the Wasserstein distance per segment and quarter.

    Args:
        points: Dataset name (``CGPS1``, ``CGPS10``) to that scan's bullseye points.
        min_points: Fewest points a quarter needs at both examinations to get values.

    Returns:
        One row per segment 1-16 and quarter 0-3 with ``n_CGPS1``, ``n_CGPS10``, ``P<p>_CGPS1``, ``P<p>_CGPS10``,
        ``P<p>_change_mm`` for every percentile and ``wasserstein_mm``; NaN where there are too few points.
    """
    grouped = {}
    for dataset, scan in points.items():
        segments, quarters = quarter_labels(scan["theta"], scan["radius"])
        grouped[dataset] = (segments, quarters, scan["values"])
    rows = []
    for segment in range(1, 17):
        for quarter in range(QUARTERS):
            values = {
                dataset: scan_values[(segments == segment) & (quarters == quarter)]
                for dataset, (segments, quarters, scan_values) in grouped.items()
            }
            row = {"segment": segment, "quarter": quarter}
            enough = all(len(scan_values) >= min_points for scan_values in values.values())
            for dataset, scan_values in values.items():
                row[f"n_{dataset}"] = len(scan_values)
                percentiles = np.percentile(scan_values, PERCENTILES) if enough else np.full(len(PERCENTILES), np.nan)
                for percentile, value in zip(PERCENTILES, percentiles):
                    row[f"P{percentile}_{dataset}"] = value
            for percentile in PERCENTILES:
                row[f"P{percentile}_change_mm"] = row[f"P{percentile}_CGPS10"] - row[f"P{percentile}_CGPS1"]
            row[WASSERSTEIN] = wasserstein_distance(values["CGPS1"], values["CGPS10"]) if enough else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def quarter_task(
    cgps1: str, cgps10: str, folders: dict[str, Path], run_folder: Path, min_points: int, overwrite: bool
) -> dict:
    """Project both scans of one subject and compare their distributions per segment quarter."""
    work_dir = run_folder / f"{cgps1}_{cgps10}"
    work_dir.mkdir(parents=True, exist_ok=True)
    ids = {"CGPS1": cgps1, "CGPS10": cgps10}
    points = {
        dataset: scan_points(folders[dataset] / ids[dataset], work_dir / POINTS_FILE.format(dataset=dataset), overwrite)
        for dataset in DATASETS
    }
    table = distribution_changes(points, min_points)
    table.insert(0, "CGPS10", cgps10)
    table.insert(0, "CGPS1", cgps1)
    return {"CGPS1": cgps1, "CGPS10": cgps10, "table": table}


def measure_columns() -> list[str]:
    """The measures shown: the change of every percentile and the Wasserstein distance."""
    return [f"P{percentile}_change_mm" for percentile in PERCENTILES] + [WASSERSTEIN]


def measure_title(measure: str) -> str:
    """Short description of a measure."""
    return "Wasserstein distance" if measure == WASSERSTEIN else f"Change of {measure.split('_')[0]}"


def draw_quarters(ax: plt.Axes, values: np.ndarray, norm: Normalize, cmap: str, title: str) -> None:
    """Draw a bullseye of segments 1-16 with every wedge split into its four quarters.

    Args:
        ax: Polar axis to draw on.
        values: (16, 4) values per segment and quarter (see ``quarter_labels``); NaN is drawn grey.
        norm: Colour normalisation.
        cmap: Name of the colormap.
        title: Title of the axis.
    """
    colormap = plt.get_cmap(cmap)
    for (start, stop), r_in, r_out in zip(RINGS, RADII[2::-1], RADII[:0:-1]):
        n = stop - start
        width = 2 * np.pi / n
        r_mid = (r_in + r_out) / 2
        for wedge in range(n):
            centre = np.pi / 2 + wedge * width
            for quarter in range(QUARTERS):
                outer, upper = divmod(quarter, 2)
                value = values[start + wedge, quarter]
                colour = colormap(norm(value)) if np.isfinite(value) else MISSING_COLOUR
                ax.bar(
                    centre + (width / 4 if upper else -width / 4),
                    r_out - r_mid if outer else r_mid - r_in,
                    width / 2,
                    r_mid if outer else r_in,
                    color=colour,
                    edgecolor="white",
                    linewidth=0.3,
                )
    draw_outline(ax)
    ax.set(ylim=[0, 1])
    ax.axis("off")
    ax.set_title(title, fontsize=9)


def quarter_values(table: pd.DataFrame, measure: str) -> np.ndarray:
    """(16, 4) values of one measure per segment and quarter of one subject's rows."""
    grid = table.pivot(index="segment", columns="quarter", values=measure).reindex(index=range(1, 17))
    return grid.reindex(columns=range(QUARTERS)).to_numpy(dtype=float)


def measure_norms(tables: dict[str, pd.DataFrame], vmax: float | None) -> dict[str, Normalize]:
    """Colour scales: one symmetric scale shared by all percentile changes and one from 0 for the Wasserstein distance.

    Args:
        tables: Subject label to its rows.
        vmax: Limit of the percentile change scale in mm; None takes the 98th percentile of the absolute changes.

    Returns:
        Measure to its colour normalisation.
    """
    rows = pd.concat(tables.values())
    changes = np.abs(rows[measure_columns()[:-1]].to_numpy(dtype=float))
    change_limit = vmax or float(np.nanpercentile(changes, 98))
    distance_limit = float(np.nanpercentile(rows[WASSERSTEIN], 98))
    norms = {measure: Normalize(-change_limit, change_limit, clip=True) for measure in measure_columns()[:-1]}
    norms[WASSERSTEIN] = Normalize(0, distance_limit, clip=True)
    return norms


def measure_cmap(measure: str) -> str:
    """Diverging colours for signed changes, sequential for the Wasserstein distance."""
    return WASSERSTEIN_CMAP if measure == WASSERSTEIN else CMAP


def colour_label(measure: str) -> str:
    """Colour bar label of a measure."""
    if measure == WASSERSTEIN:
        return "Wasserstein distance between the CGPS1 and CGPS10 thickness distributions (mm, no sign)"
    return f"{measure_title(measure)} of the wall thickness (mm), CGPS10 − CGPS1 (grey = too few points)"


def measure_grid(tables: dict[str, pd.DataFrame], measure: str, norm: Normalize, n_columns: int = 5) -> plt.Figure:
    """Grid of single-subject quarter bullseyes of one measure.

    Args:
        tables: Subject label to its rows.
        measure: Column to show.
        norm: Colour normalisation.
        n_columns: Bullseyes per row.

    Returns:
        The figure.
    """
    n_rows = int(np.ceil(len(tables) / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(3.2 * n_columns, 3.4 * n_rows),
        subplot_kw=dict(polar=True),
        layout="constrained",
        squeeze=False,
    )
    for ax, (label, table) in zip(axes.flat, tables.items()):
        draw_quarters(ax, quarter_values(table, measure), norm, measure_cmap(measure), label)
    for ax in axes.flat[len(tables) :]:
        ax.axis("off")
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=measure_cmap(measure)), ax=axes, orientation="horizontal", shrink=0.4, aspect=40
    )
    colorbar.set_label(colour_label(measure))
    return fig


def comparison_figure(tables: dict[str, pd.DataFrame], norms: dict[str, Normalize]) -> plt.Figure:
    """Quarter bullseyes of some subjects (rows) for every measure (columns).

    Args:
        tables: Subject label to its rows, one row of bullseyes each.
        norms: Measure to its colour normalisation.

    Returns:
        The figure.
    """
    measures = measure_columns()
    fig, axes = plt.subplots(
        len(tables),
        len(measures),
        figsize=(3.1 * len(measures), 3.3 * len(tables)),
        subplot_kw=dict(polar=True),
        layout="constrained",
        squeeze=False,
    )
    for row, (label, table) in enumerate(tables.items()):
        for column, measure in enumerate(measures):
            title = f"{label}\n{measure_title(measure)}" if column == 0 else measure_title(measure)
            draw_quarters(
                axes[row, column], quarter_values(table, measure), norms[measure], measure_cmap(measure), title
            )
    for measure, axes_part in [(measures[0], axes[:, :-1]), (WASSERSTEIN, axes[:, -1])]:
        colorbar = fig.colorbar(
            ScalarMappable(norm=norms[measure], cmap=measure_cmap(measure)),
            ax=axes_part,
            orientation="horizontal",
            shrink=0.6,
            aspect=30,
        )
        colorbar.set_label("Change of the percentile (mm)" if measure != WASSERSTEIN else "Wasserstein distance (mm)")
    return fig


def parse_args() -> argparse.Namespace:
    """Command line arguments."""
    parser = argparse.ArgumentParser(description="Per-subject thickness distribution change per AHA segment quarter.")
    add_subject_arguments(parser)
    parser.add_argument("--min-points", type=int, default=MIN_POINTS, help="Fewest points per quarter and scan.")
    parser.add_argument("--compare", type=int, default=N_COMPARE, help="Subjects in the side-by-side figure.")
    parser.add_argument("--vmax", type=float, help="Limit of the percentile change colour scale in mm.")
    parser.add_argument("--overwrite", action="store_true", help="Project the scans again even if saved.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root, folders, _, to_process, chosen = setup_subjects(args)
    run_folder = Path(run_output_dir(root, RUN_NAME, AORTA_EXCLUSION_MM))
    task = partial(
        quarter_task, folders=folders, run_folder=run_folder, min_points=args.min_points, overwrite=args.overwrite
    )
    results = run_subjects(task, pair_list(to_process), args.workers, lambda result: "compared")
    if not results:
        raise SystemExit("No subject could be processed.")

    table = pd.concat([result["table"] for result in results], ignore_index=True)
    table_path = run_folder / TABLE_FILE
    if table_path.is_file():
        old = read_ids(table_path, list(DATASETS))
        processed = set(zip(table["CGPS1"], table["CGPS10"]))
        old = old[[pair not in processed for pair in zip(old["CGPS1"], old["CGPS10"])]]
        table = pd.concat([old, table], ignore_index=True)
    table.to_csv(table_path, index=False)

    by_subject = {(result["CGPS1"], result["CGPS10"]): result["table"] for result in results}
    shown = {
        f"{cgps1} / {cgps10}": by_subject[(cgps1, cgps10)]
        for cgps1, cgps10 in pair_list(chosen)
        if (cgps1, cgps10) in by_subject
    }
    if not shown:
        raise SystemExit("None of the chosen subjects could be processed.")
    norms = measure_norms(shown, args.vmax)
    print(
        f"Colour scales: percentile change ±{norms[measure_columns()[0]].vmax:.1f} mm, "
        f"Wasserstein 0 to {norms[WASSERSTEIN].vmax:.1f} mm"
    )

    figure_folder = Path(root) / FIGURE_FOLDER
    figure_folder.mkdir(parents=True, exist_ok=True)
    for measure in measure_columns():
        path = figure_folder / f"bullseye_quarter_{measure.removesuffix('_mm')}_{figure_name(args)}.png"
        figure = measure_grid(shown, measure, norms[measure])
        figure.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        print(f"Figure saved to: {path}")
    path = figure_folder / f"bullseye_quarter_comparison_{figure_name(args)}.png"
    figure = comparison_figure(dict(list(shown.items())[: args.compare]), norms)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Table saved to: {table_path}")
    print(f"Figure saved to: {path}")

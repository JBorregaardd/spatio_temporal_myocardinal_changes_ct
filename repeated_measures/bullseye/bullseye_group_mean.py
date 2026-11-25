#################################################################################################################################################
# This script makes the group maps of the change in LV wall thickness (CGPS10 - CGPS1, in mm) from baseline to 10-year follow-up: the
# mean and the median over the subjects of each group, for women and men with and without HT (the HT class of the CGPS1 scan), as
# bullseye plots. No smoothing is added; each subject's map is already smoothed on the mesh.
# Two figures with the same layout (top row mean, bottom row median, one column per group) and one shared colour scale:
#   1. the mesh method: each subject's bullseye after Gaussian smoothing on the mesh (bullseye_smooth_delta.py, --fwhm-mm, default
#      10 mm) is put on one common polar grid (the normalised bullseye coordinates, so a cell is the same position in every subject)
#      and the mean and median are taken per cell over the subjects with data there,
#   2. the quarter method: the change of a percentile (--percentile, default P50) per quarter of every AHA segment
#      (bullseye_quarter_distribution.py, no registration), mean and median per quarter over the subjects.
# A cell or quarter with data in less than --coverage (default 50%) of the group's subjects is grey. The mesh method underestimates
# thickening in the basal septum (diagnose_segment_bias.py), so trust the quarter method where the two differ there.
# It needs bullseye_smooth_delta.py --all --fwhm-mm <F> and bullseye_quarter_distribution.py --all to have run, the HT classes in
# <PROJECT_ROOT>/df/repeated_measures/CGPS1_HT.csv (ht_classifier.py) and each dataset's statistics_patient_summary.csv (sex).
# The figures, the quarter group table, the group maps (npz) and the number of subjects per group are saved in
# <PROJECT_ROOT>/reports/figures/repeated_measures/bullseye.
# e.g. uv run repeated_measures/bullseye/bullseye_group_mean.py --fwhm-mm 10 --coverage 0.5 --percentile 50 --workers 8
#################################################################################################################################################

import argparse
import os
import sys
import warnings
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "measurements"))

from bullseye_mesh_delta import CMAP, FIGURE_FOLDER, draw_image, load_polar, polar_image  # noqa: E402
from bullseye_quarter_distribution import PERCENTILES, QUARTERS, draw_quarters  # noqa: E402
from bullseye_quarter_distribution import RUN_NAME as QUARTER_RUN_NAME  # noqa: E402
from bullseye_quarter_distribution import TABLE_FILE as QUARTER_TABLE  # noqa: E402
from bullseye_segment_delta import HT_TITLES, PANELS  # noqa: E402
from bullseye_smooth_delta import polar_name  # noqa: E402
from ht_classifier import HT_FILE, HT_FOLDER  # noqa: E402
from main_all_df import SEX_CODES  # noqa: E402
from mesh_delta import AORTA_EXCLUSION_MM, RUN_NAME, pair_list, run_subjects  # noqa: E402
from pipeline import load_environment, run_output_dir  # noqa: E402
from spaghetti_plot import DATASETS, LINKS_FILE, PATIENT_SUMMARY, SEX_TITLES, read_ids  # noqa: E402

FWHM_MM = 10.0
COVERAGE = 0.5
PERCENTILE = 50
GROUP_ANGLES = 180
GROUP_RADII = 150
STATISTICS = {"mean": "Mean", "median": "Median"}
FIGURE_NAME = "bullseye_group"


def subject_classes(pairs: pd.DataFrame, summaries: dict[str, pd.DataFrame], ht: pd.DataFrame) -> pd.DataFrame:
    """Sex and HT class of every linked subject.

    Args:
        pairs: One row per subject with the ids in ``CGPS1`` and ``CGPS10``.
        summaries: Dataset name to its patient summary, with ``patient_id`` and ``sex``.
        ht: One row per CGPS1 patient with ``patient_id`` and ``HT``, from ht_classifier.py.

    Returns:
        ``pairs`` with ``sex`` (0 = man, 1 = woman; from CGPS1, else CGPS10) and ``HT`` (of the CGPS1 scan).
    """
    table = pairs[list(DATASETS)].drop_duplicates().reset_index(drop=True)
    sex = pd.Series(pd.NA, index=table.index, dtype="Float64")
    for dataset, summary in summaries.items():
        codes = summary.drop_duplicates("patient_id").set_index("patient_id")["sex"].astype("Float64")
        sex = sex.fillna(table[dataset].map(codes).astype("Float64"))
    table["sex"] = sex
    classes = ht.drop_duplicates("patient_id").set_index("patient_id")["HT"].astype("boolean")
    table["HT"] = table["CGPS1"].map(classes).astype("boolean")
    return table


def group_title(ht: bool, sex: str) -> str:
    """Title of a group, e.g. ``Women with HT``."""
    return f"{SEX_TITLES[sex]} {HT_TITLES[ht]}"


def in_group(classes: pd.DataFrame, ht: bool, sex: str) -> pd.DataFrame:
    """The subjects of one HT class and sex."""
    return classes[(classes["HT"] == ht) & (classes["sex"] == SEX_CODES[sex])]


def subject_image(cgps1: str, cgps10: str, run_folder: Path, name: str) -> dict:
    """One subject's saved bullseye points on the common group grid."""
    points = load_polar(run_folder / f"{cgps1}_{cgps10}" / f"{name}.npz")
    _, _, image = polar_image(points, GROUP_ANGLES, GROUP_RADII)
    return {"CGPS1": cgps1, "CGPS10": cgps10, "image": image.astype(np.float32)}


def summarise_stack(stack: np.ndarray, coverage: float) -> dict[str, np.ndarray]:
    """Mean, median and number of subjects per cell over the subjects with data, NaN where coverage is too low.

    Args:
        stack: (subjects, ...) values, NaN where a subject has no data.
        coverage: Smallest share of the subjects that must have data in a cell.

    Returns:
        ``mean``, ``median`` and ``n`` with the shape of one subject's values.
    """
    n = np.isfinite(stack).sum(axis=0)
    enough = n >= coverage * len(stack)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.where(enough, np.nanmean(stack, axis=0), np.nan)
        median = np.where(enough, np.nanmedian(stack, axis=0), np.nan)
    return {"mean": mean, "median": median, "n": n}


def quarter_stack(table: pd.DataFrame, subjects: pd.DataFrame, column: str) -> np.ndarray:
    """(subjects, 16, 4) values of one quarter table column for the given subjects, NaN where missing."""
    rows = table.merge(subjects[list(DATASETS)], on=list(DATASETS), how="inner")
    stack = []
    for _, subject in rows.groupby(list(DATASETS)):
        grid = subject.pivot(index="segment", columns="quarter", values=column)
        stack.append(grid.reindex(index=range(1, 17), columns=range(QUARTERS)).to_numpy(dtype=float))
    return np.array(stack) if stack else np.full((0, 16, QUARTERS), np.nan)


def group_figure(maps: dict[str, dict[str, np.ndarray]], counts: dict[str, int], draw, vmax: float, label: str):
    """Group bullseyes: one column per group, the mean in the top row and the median in the bottom row.

    Args:
        maps: Group title to its ``summarise_stack`` result.
        counts: Group title to its number of subjects.
        draw: Draws one map: ``draw(ax, values, norm, title)``.
        vmax: Largest absolute change in mm of the shared colour scale.
        label: Colour bar label.

    Returns:
        The figure.
    """
    fig, axes = plt.subplots(
        len(STATISTICS),
        len(maps),
        figsize=(4.2 * len(maps), 4.4 * len(STATISTICS)),
        subplot_kw=dict(polar=True),
        layout="constrained",
        squeeze=False,
    )
    norm = Normalize(-vmax, vmax, clip=True)
    for column, (title, summary) in enumerate(maps.items()):
        for row, (statistic, statistic_title) in enumerate(STATISTICS.items()):
            draw(axes[row, column], summary[statistic], norm, f"{title} (n = {counts[title]})\n{statistic_title}")
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=CMAP), ax=axes, orientation="horizontal", shrink=0.45, aspect=40
    )
    colorbar.set_label(label)
    return fig


def parse_args() -> argparse.Namespace:
    """Command line arguments."""
    parser = argparse.ArgumentParser(description="Group mean and median bullseyes of the wall thickness change.")
    parser.add_argument("--fwhm-mm", type=float, default=FWHM_MM, help="Smoothing of the mesh maps (0 = none).")
    parser.add_argument("--coverage", type=float, default=COVERAGE, help="Smallest share of a group with data.")
    parser.add_argument("--percentile", type=int, default=PERCENTILE, choices=PERCENTILES, help="Quarter percentile.")
    parser.add_argument("--vmax", type=float, help="Colour scale limit in mm (default: 99th percentile of |value|).")
    parser.add_argument("--workers", type=int, default=1, help="Number of subjects read in parallel.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root, _ = load_environment()
    mesh_folder = Path(run_output_dir(root, RUN_NAME, AORTA_EXCLUSION_MM))
    quarter_folder = Path(run_output_dir(root, QUARTER_RUN_NAME, AORTA_EXCLUSION_MM))
    links = read_ids(Path(root) / "df" / LINKS_FILE, list(DATASETS))
    ht = read_ids(Path(root) / HT_FOLDER / HT_FILE, ["patient_id"])
    summaries = {
        dataset: read_ids(Path(run_output_dir(root, dataset, AORTA_EXCLUSION_MM)) / PATIENT_SUMMARY, ["patient_id"])
        for dataset in DATASETS
    }
    classes = subject_classes(links, summaries, ht)
    unclassified = classes["HT"].isna() | classes["sex"].isna()
    if unclassified.any():
        print(f"[WARN] {int(unclassified.sum())} linked subjects without an HT class or sex are left out.")
    classes = classes[~unclassified]

    name = polar_name(args.fwhm_mm)
    with_mesh = classes[[(mesh_folder / f"{a}_{b}" / f"{name}.npz").is_file() for a, b in pair_list(classes)]]
    print(f"{len(with_mesh)} of {len(classes)} classified subjects have a {name}.npz (bullseye_smooth_delta.py --all).")
    task = partial(subject_image, run_folder=mesh_folder, name=name)
    images = {
        (row["CGPS1"], row["CGPS10"]): row["image"]
        for row in run_subjects(task, pair_list(with_mesh), args.workers, lambda result: "read")
    }
    quarter_table = read_ids(quarter_folder / QUARTER_TABLE, list(DATASETS))
    column = f"P{args.percentile}_change_mm"

    mesh_maps, quarter_maps, mesh_counts, quarter_counts, rows = {}, {}, {}, {}, []
    for ht_class, sex in PANELS:
        title = group_title(ht_class, sex)
        group = in_group(classes, ht_class, sex)
        stack = np.array([images[pair] for pair in pair_list(group) if pair in images])
        mesh_maps[title] = summarise_stack(stack, args.coverage)
        mesh_counts[title] = len(stack)
        quarters = quarter_stack(quarter_table, group, column)
        quarter_maps[title] = summarise_stack(quarters, args.coverage)
        quarter_counts[title] = len(quarters)
        rows.append(
            {
                "group": title,
                "HT": ht_class,
                "sex": sex,
                "n_classified": len(group),
                "n_mesh": len(stack),
                "n_quarter": len(quarters),
            }
        )
        print(f"{title}: {len(group)} subjects, {len(stack)} with a mesh map, {len(quarters)} with quarters.")

    shown = [
        summary[statistic]
        for maps in (mesh_maps, quarter_maps)
        for summary in maps.values()
        for statistic in STATISTICS
    ]
    vmax = args.vmax or float(np.nanpercentile(np.abs(np.concatenate([values.ravel() for values in shown])), 99))
    print(f"Colour scale: -{vmax:.2f} to {vmax:.2f} mm")

    figure_folder = Path(root) / FIGURE_FOLDER
    figure_folder.mkdir(parents=True, exist_ok=True)
    theta_edges = np.linspace(-np.pi, np.pi, GROUP_ANGLES + 1)
    radius_edges = np.linspace(0, 1, GROUP_RADII + 1)
    smoothing = "no smoothing" if args.fwhm_mm == 0 else f"Gaussian FWHM {args.fwhm_mm:g} mm on the mesh"
    figures = {
        f"{FIGURE_NAME}_mesh_fwhm{args.fwhm_mm:g}mm": group_figure(
            mesh_maps,
            mesh_counts,
            lambda ax, values, norm, title: draw_image(ax, theta_edges, radius_edges, values, norm, title),
            vmax,
            f"Change in wall thickness (mm), CGPS10 − CGPS1, mesh method ({smoothing}; grey = data in < "
            f"{100 * args.coverage:g}% of the group)",
        ),
        f"{FIGURE_NAME}_quarter_P{args.percentile}": group_figure(
            quarter_maps,
            quarter_counts,
            lambda ax, values, norm, title: draw_quarters(ax, values, norm, CMAP, title),
            vmax,
            f"Change of P{args.percentile} of the wall thickness per segment quarter (mm), CGPS10 − CGPS1, no "
            f"registration (grey = data in < {100 * args.coverage:g}% of the group)",
        ),
    }
    for figure_stem, figure in figures.items():
        figure.savefig(figure_folder / f"{figure_stem}.png", dpi=200, bbox_inches="tight")
        plt.close(figure)
        print(f"Figure saved to: {figure_folder / figure_stem}.png")

    np.savez_compressed(
        figure_folder / f"{FIGURE_NAME}_mesh_fwhm{args.fwhm_mm:g}mm.npz",
        theta_edges=theta_edges,
        radius_edges=radius_edges,
        **{f"{title}|{key}": values for title, summary in mesh_maps.items() for key, values in summary.items()},
    )
    quarter_rows = [
        {
            "group": title,
            "segment": segment,
            "quarter": quarter,
            "n": int(summary["n"][segment - 1, quarter]),
            f"mean_{column}": summary["mean"][segment - 1, quarter],
            f"median_{column}": summary["median"][segment - 1, quarter],
        }
        for title, summary in quarter_maps.items()
        for segment in range(1, 17)
        for quarter in range(QUARTERS)
    ]
    pd.DataFrame(quarter_rows).to_csv(figure_folder / f"{FIGURE_NAME}_quarter_P{args.percentile}.csv", index=False)
    pd.DataFrame(rows).to_csv(figure_folder / f"{FIGURE_NAME}_counts.csv", index=False)
    print(f"Group maps, quarter table and counts saved in: {figure_folder}")

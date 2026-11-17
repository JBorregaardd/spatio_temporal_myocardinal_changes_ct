#################################################################################################################################################
# This script plots the change (delta = CGPS10 - CGPS1) of the LV wall thickness per AHA segment as bullseye plots, for the subjects
# scanned in both CGPS1 (baseline) and CGPS10 (10-year follow-up). For each subject and segment the delta of the mean and of the median
# thickness is computed, and the deltas are averaged over the women and over the men. The figure has four bullseyes in one row: mean
# thickness for women and men, then median thickness for women and men, on one shared, symmetric colour scale. The apex (segment 17) is
# greyed out, as it is left out of the wall thickness elsewhere in the analysis.
# The per-segment thickness is read from <run folder>/statistics/statistics_thickness_per_segment.csv and the sex from
# <run folder>/statistics/statistics_patient_summary.csv of each dataset, so run main_statistics_df.py for both datasets first.
# The subjects are linked by <df>/CGPS1_CGPS10_links.csv, made by comparative_analysis/compare.py.
# The figure and the table of the group deltas per segment are saved in <PROJECT_ROOT>/reports/figures/repeated_measures.
# e.g. uv run repeated_measures/bullseye_delta.py
#################################################################################################################################################

import os
import sys
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "measurements"))
sys.path.insert(0, os.path.join(REPO_ROOT, "LVM_thickness"))
sys.path.insert(0, os.path.join(REPO_ROOT, "repeated_measures"))

from main_all_df import SEX_CODES  # noqa: E402
from pipeline import AORTA_EXCLUSION_MM, load_environment, run_output_dir  # noqa: E402
from spaghetti_plot import (  # noqa: E402
    DATASETS,
    LINKS_FILE,
    OUTPUT_FOLDER,
    SEX_TITLES,
    patient_summary_path,
    read_ids,
)

SEGMENT_STATISTICS = os.path.join("statistics", "statistics_thickness_per_segment.csv")
STATISTICS = {"mean_thickness": "Mean thickness", "median_thickness": "Median thickness"}
SEX_ORDER = ["female", "male"]
SEGMENTS = list(range(1, 18))
WALL_SEGMENTS = list(range(1, 17))
RINGS = [(0, 6), (6, 12), (12, 16)]
RADII = np.linspace(0.2, 1, 4)
APEX_COLOUR = "#d9d9d9"
CMAP = "RdBu_r"
FIGURE_NAME = "bullseye_delta_thickness"


def segment_statistics_path(
    root: str, dataset: str, aorta_exclusion_mm: float, output_dir: str | None = None
) -> Path:
    """Per-segment thickness table written by main_statistics_df.py for one dataset.

    Args:
        root: Project root.
        dataset: Dataset name, e.g. ``CGPS1``.
        aorta_exclusion_mm: Aortic exclusion distance of the run.
        output_dir: Parent of the run folders; defaults to ``<root>/output``.

    Returns:
        ``<run folder>/statistics/statistics_thickness_per_segment.csv``.
    """
    return Path(run_output_dir(root, dataset, aorta_exclusion_mm, output_dir)) / SEGMENT_STATISTICS


def segment_deltas(
    segments: dict[str, pd.DataFrame],
    summaries: dict[str, pd.DataFrame],
    links: pd.DataFrame,
    statistics: list[str],
) -> pd.DataFrame:
    """The change of the thickness statistics of each segment, one row per linked subject and segment.

    Args:
        segments: Dataset name (``CGPS1``, ``CGPS10``) to its per-segment table, with ``patient_id``, ``segment`` and
            the statistic columns.
        summaries: Dataset name to its patient summary, with ``patient_id`` and ``sex``.
        links: One row per subject with the patient ids in the columns ``CGPS1`` and ``CGPS10``.
        statistics: Statistic columns to compute the change of, e.g. ``mean_thickness``.

    Returns:
        One row per subject and segment present at both examinations with ``CGPS1``, ``CGPS10`` (the ids), ``sex``
        (0 = man, 1 = woman; from CGPS1, else CGPS10), ``segment`` and ``delta_<statistic>`` (CGPS10 - CGPS1).
    """
    table = links[list(DATASETS)].drop_duplicates()
    keys: list[str] = []
    for dataset, segment_table in segments.items():
        values = segment_table[["patient_id", "segment", *statistics]].drop_duplicates(["patient_id", "segment"])
        values = values.rename(columns={statistic: f"{statistic}_{dataset}" for statistic in statistics})
        table = table.merge(
            values, left_on=[dataset, *keys], right_on=["patient_id", *keys], how="inner"
        ).drop(columns="patient_id")
        keys = ["segment"]

    sex = pd.Series(pd.NA, index=table.index, dtype="Float64")
    for dataset, summary in summaries.items():
        codes = summary.drop_duplicates("patient_id").set_index("patient_id")["sex"].astype("Float64")
        sex = sex.fillna(table[dataset].map(codes).astype("Float64"))
    table["sex"] = sex

    columns = [*DATASETS, "sex", "segment"]
    for statistic in statistics:
        table[f"delta_{statistic}"] = table[f"{statistic}_CGPS10"] - table[f"{statistic}_CGPS1"]
        columns.append(f"delta_{statistic}")
    return table[columns].sort_values([*DATASETS, "segment"]).reset_index(drop=True)


def group_deltas(deltas: pd.DataFrame, statistics: list[str]) -> pd.DataFrame:
    """Number of subjects and mean ± sd of the change per sex, statistic and segment.

    Args:
        deltas: Table from ``segment_deltas``.
        statistics: Statistic columns of the table, without the ``delta_`` prefix.

    Returns:
        One row per sex, statistic and segment 1-17 with ``n``, ``mean_delta`` and ``sd_delta``; segments without
        subjects have ``n`` 0 and missing values.
    """
    rows = []
    for sex in SEX_ORDER:
        group = deltas[deltas["sex"] == SEX_CODES[sex]]
        for statistic in statistics:
            summary = group.groupby("segment")[f"delta_{statistic}"].agg(["count", "mean", "std"]).reindex(SEGMENTS)
            rows.append(
                pd.DataFrame(
                    {
                        "sex": sex,
                        "statistic": statistic,
                        "segment": SEGMENTS,
                        "n": summary["count"].fillna(0).astype(int).to_numpy(),
                        "mean_delta": summary["mean"].to_numpy(),
                        "sd_delta": summary["std"].to_numpy(),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def draw_bullseye(ax: plt.Axes, values: np.ndarray, norm: Normalize, cmap: str, title: str) -> None:
    """Draw an AHA 17-segment bullseye of segments 1-16 on a polar axis, with the apex (segment 17) greyed out.

    The layout matches ``plot_mean_bullseye`` in ``LVM_thickness/utils/bullseye.py``.

    Args:
        ax: Polar axis to draw on.
        values: Values of segments 1-16, in segment order; missing values are drawn white.
        norm: Colour normalisation shared by all bullseyes.
        cmap: Name of the colormap.
        title: Title of the axis.
    """
    colormap = plt.get_cmap(cmap)
    outline = [pe.withStroke(linewidth=2.5, foreground="white")]
    ax.set(ylim=[0, 1], xticklabels=[], yticklabels=[])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)

    for (start, stop), r_in, r_out in zip(RINGS, RADII[2::-1], RADII[:0:-1]):
        n = stop - start
        dtheta = 2 * np.pi / n
        ring = values[start:stop]
        colours = [colormap(norm(value)) if np.isfinite(value) else "white" for value in ring]
        thetas = np.arange(n) * dtheta + np.pi / 2
        ax.bar(thetas, r_out - r_in, dtheta, r_in, color=colours, edgecolor="k", linewidth=1.5, clip_on=False)
        for theta, value in zip(thetas, ring):
            label = f"{value:.2f}" if np.isfinite(value) else "–"
            ax.text(theta, (r_in + r_out) / 2, label, ha="center", va="center", fontsize=10, fontweight="bold",
                    path_effects=outline)

    ax.bar(0, RADII[0], 2 * np.pi, 0, color=APEX_COLOUR, edgecolor="none")
    ax.plot(np.linspace(0, 2 * np.pi, 100), np.full(100, RADII[0]), "k", linewidth=1.5)
    ax.text(0, 0, "17", ha="center", va="center", fontsize=10, color="dimgrey")
    ax.set_title(title, fontsize=13, pad=12)


def bullseye_figure(summary: pd.DataFrame, statistics: list[str], vmax: float, cmap: str = CMAP) -> plt.Figure:
    """One row of bullseyes of the mean change per segment: each statistic for women and men, in that order.

    Args:
        summary: Table from ``group_deltas``.
        statistics: Statistics to plot, in order.
        vmax: Largest absolute change of the shared colour scale, which runs from ``-vmax`` to ``vmax``.
        cmap: Name of a diverging colormap.

    Returns:
        The figure.
    """
    panels = [(statistic, sex) for statistic in statistics for sex in SEX_ORDER]
    norm = Normalize(vmin=-vmax, vmax=vmax)
    fig, axes = plt.subplots(
        1, len(panels), figsize=(4.6 * len(panels), 5.6), subplot_kw=dict(polar=True), layout="constrained"
    )
    for ax, (statistic, sex) in zip(np.atleast_1d(axes), panels):
        panel = summary[(summary["statistic"] == statistic) & (summary["sex"] == sex)].set_index("segment")
        values = panel["mean_delta"].reindex(WALL_SEGMENTS).to_numpy(dtype=float)
        n = int(panel["n"].reindex(WALL_SEGMENTS).max())
        draw_bullseye(ax, values, norm, cmap, f"{SEX_TITLES[sex]}: {STATISTICS.get(statistic, statistic).lower()}"
                                              f"\n(n = {n})")
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation="horizontal", shrink=0.4,
                            aspect=40, pad=0.04)
    colorbar.set_label("Mean change in wall thickness (mm), CGPS10 − CGPS1")
    fig.suptitle("Change in LV wall thickness per AHA segment from baseline to 10-year follow-up", fontsize=15)
    return fig


if __name__ == "__main__":
    root, _ = load_environment()

    parser = argparse.ArgumentParser(description="Bullseye plots of the change in wall thickness per AHA segment.")
    parser.add_argument("--links", default=None, help=f"Links table (default: <PROJECT_ROOT>/df/{LINKS_FILE}).")
    parser.add_argument("--aorta-exclusion-mm", type=float, default=AORTA_EXCLUSION_MM,
                        help=f"Aortic exclusion distance of the runs (default: {AORTA_EXCLUSION_MM}).")
    parser.add_argument("--output-dir", default=None,
                        help="Parent of the run folders (default: <PROJECT_ROOT>/output).")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Largest absolute change of the colour scale in mm (default: largest group mean).")
    parser.add_argument("--figure-dir", default=None,
                        help=f"Folder of the figure and the table (default: <PROJECT_ROOT>/{OUTPUT_FOLDER}).")
    args = parser.parse_args()

    links_path = Path(args.links) if args.links else Path(root) / "df" / LINKS_FILE
    if not links_path.is_file():
        raise SystemExit(f"Links table not found: {links_path}\nRun comparative_analysis/compare.py first.")
    links = read_ids(links_path, list(DATASETS))

    statistics = list(STATISTICS)
    segments, summaries = {}, {}
    for dataset in DATASETS:
        segment_path = segment_statistics_path(root, dataset, args.aorta_exclusion_mm, args.output_dir)
        summary_path = patient_summary_path(root, dataset, args.aorta_exclusion_mm, args.output_dir)
        for path in (segment_path, summary_path):
            if not path.is_file():
                raise SystemExit(f"Statistics not found: {path}\nRun main_statistics_df.py for {dataset} first.")
        print(f"{dataset}: {segment_path}")
        segments[dataset] = read_ids(segment_path, ["patient_id"])
        summaries[dataset] = read_ids(summary_path, ["patient_id"])
        missing = [column for column in ["segment", *statistics] if column not in segments[dataset].columns]
        if missing or "sex" not in summaries[dataset].columns:
            raise SystemExit(f"Columns {missing or ['sex']} missing for {dataset}.")

    deltas = segment_deltas(segments, summaries, links, statistics)
    n_subjects = deltas[list(DATASETS)].drop_duplicates().shape[0]
    print(f"{n_subjects} of {len(links)} linked subjects have segment thickness at both examinations.")
    missing_sex = deltas.loc[deltas["sex"].isna(), list(DATASETS)].drop_duplicates().shape[0]
    if missing_sex:
        print(f"[WARN] {missing_sex} subjects without sex are left out of the plots.")
    if deltas.empty:
        raise SystemExit("No subjects with segment thickness at both examinations.")

    summary = group_deltas(deltas, statistics)
    walls = summary[summary["segment"].isin(WALL_SEGMENTS)]
    vmax = args.vmax if args.vmax is not None else float(np.nanmax(np.abs(walls["mean_delta"])))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    print(f"Colour scale: -{vmax:.2f} to {vmax:.2f} mm")

    output_folder = Path(args.figure_dir) if args.figure_dir else Path(root) / OUTPUT_FOLDER
    output_folder.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_folder / f"{FIGURE_NAME}.csv", index=False)

    figure = bullseye_figure(summary, statistics, vmax)
    figure_path = output_folder / f"{FIGURE_NAME}.png"
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Table saved to: {output_folder / f'{FIGURE_NAME}.csv'}")
    print(f"Figure saved to: {figure_path}")

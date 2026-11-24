#################################################################################################################################################
# This script plots the relative change of the median LV wall thickness per AHA segment as bullseye plots, for the subjects scanned in
# both CGPS1 (baseline) and CGPS10 (10-year follow-up). For each subject and segment the change is 100 * (CGPS10 - CGPS1) / CGPS1 of
# the median thickness, and the changes are averaged per group. The figure has four bullseyes in one row: women with HT, men with HT,
# women without HT and men without HT, on one shared, symmetric colour scale. The apex (segment 17) is greyed out, as it is left out of
# the wall thickness elsewhere in the analysis.
# The per-segment thickness is read from <run folder>/statistics/statistics_thickness_per_segment.csv and the sex from
# <run folder>/statistics/statistics_patient_summary.csv of each dataset, so run main_statistics_df.py for both datasets first.
# The HT class of each subject's CGPS1 scan is read from <PROJECT_ROOT>/df/repeated_measures/CGPS1_HT.csv, made by ht_classifier.py.
# The runs with 0 mm aortic exclusion are used, e.g. output/CGPS10_0mm. The subjects are linked by <PROJECT_ROOT>/df/CGPS1_CGPS10_links.csv,
# made by comparative_analysis/compare.py.
# The figure and the table of the group changes per segment are saved in <PROJECT_ROOT>/reports/figures/repeated_measures.
# e.g. uv run repeated_measures/bullseye/bullseye_segment_delta.py
#################################################################################################################################################

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "measurements"))
sys.path.insert(0, os.path.join(REPO_ROOT, "LVM_thickness"))
sys.path.insert(0, os.path.join(REPO_ROOT, "repeated_measures"))

from main_all_df import SEX_CODES  # noqa: E402
from ht_classifier import HT_FILE, HT_FOLDER  # noqa: E402
from pipeline import load_environment, run_output_dir  # noqa: E402
from spaghetti_plot import DATASETS, LINKS_FILE, OUTPUT_FOLDER, PATIENT_SUMMARY, SEX_TITLES, read_ids  # noqa: E402

AORTA_EXCLUSION_MM = 0
SEGMENT_STATISTICS = os.path.join("statistics", "statistics_thickness_per_segment.csv")
STATISTIC = "median_thickness"
PANELS = [(True, "female"), (True, "male"), (False, "female"), (False, "male")]
HT_TITLES = {True: "with HT", False: "without HT"}
SEGMENTS = list(range(1, 18))
WALL_SEGMENTS = list(range(1, 17))
RINGS = [(0, 6), (6, 12), (12, 16)]
RADII = np.linspace(0.2, 1, 4)
APEX_COLOUR = "#d9d9d9"
CMAP = "RdBu_r"
FIGURE_NAME = "bullseye_segment_delta_median_thickness"


def segment_changes(
    segments: dict[str, pd.DataFrame], summaries: dict[str, pd.DataFrame], links: pd.DataFrame, ht: pd.DataFrame
) -> pd.DataFrame:
    """The relative change of the median thickness of each segment, one row per linked subject and segment.

    Args:
        segments: Dataset name (``CGPS1``, ``CGPS10``) to its per-segment table, with ``patient_id``, ``segment`` and
            ``median_thickness``.
        summaries: Dataset name to its patient summary, with ``patient_id`` and ``sex``.
        links: One row per subject with the patient ids in the columns ``CGPS1`` and ``CGPS10``.
        ht: One row per CGPS1 patient with ``patient_id`` and ``HT``, from ht_classifier.py.

    Returns:
        One row per subject and segment present at both examinations with ``CGPS1``, ``CGPS10`` (the ids), ``sex``
        (0 = man, 1 = woman; from CGPS1, else CGPS10), ``HT`` (of the CGPS1 scan), ``segment`` and ``change_pct``
        (100 * (CGPS10 - CGPS1) / CGPS1).
    """
    table = links[list(DATASETS)].drop_duplicates()
    keys: list[str] = []
    for dataset, segment_table in segments.items():
        values = segment_table[["patient_id", "segment", STATISTIC]].drop_duplicates(["patient_id", "segment"])
        values = values.rename(columns={STATISTIC: dataset + "_value"})
        table = table.merge(
            values, left_on=[dataset, *keys], right_on=["patient_id", *keys], how="inner"
        ).drop(columns="patient_id")
        keys = ["segment"]

    sex = pd.Series(pd.NA, index=table.index, dtype="Float64")
    for dataset, summary in summaries.items():
        codes = summary.drop_duplicates("patient_id").set_index("patient_id")["sex"].astype("Float64")
        sex = sex.fillna(table[dataset].map(codes).astype("Float64"))
    table["sex"] = sex
    classes = ht.drop_duplicates("patient_id").set_index("patient_id")["HT"].astype("boolean")
    table["HT"] = table["CGPS1"].map(classes).astype("boolean")

    table["change_pct"] = 100 * (table["CGPS10_value"] - table["CGPS1_value"]) / table["CGPS1_value"]
    columns = [*DATASETS, "sex", "HT", "segment", "change_pct"]
    return table[columns].sort_values([*DATASETS, "segment"]).reset_index(drop=True)


def group_changes(changes: pd.DataFrame) -> pd.DataFrame:
    """Number of subjects and mean ± sd of the relative change per HT class, sex and segment.

    Args:
        changes: Table from ``segment_changes``.

    Returns:
        One row per panel (HT class and sex) and segment 1-17 with ``HT``, ``sex``, ``segment``, ``n``,
        ``mean_change_pct`` and ``sd_change_pct``; segments without subjects have ``n`` 0 and missing values.
    """
    rows = []
    for ht, sex in PANELS:
        group = changes[(changes["HT"] == ht) & (changes["sex"] == SEX_CODES[sex])]
        summary = group.groupby("segment")["change_pct"].agg(["count", "mean", "std"]).reindex(SEGMENTS)
        rows.append(
            pd.DataFrame(
                {
                    "HT": ht,
                    "sex": sex,
                    "segment": SEGMENTS,
                    "n": summary["count"].fillna(0).astype(int).to_numpy(),
                    "mean_change_pct": summary["mean"].to_numpy(),
                    "sd_change_pct": summary["std"].to_numpy(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def draw_bullseye(ax: plt.Axes, values: np.ndarray, norm: Normalize, cmap: str, title: str) -> None:
    """Draw an AHA 17-segment bullseye of segments 1-16 on a polar axis, with the apex (segment 17) greyed out.

    The layout matches ``plot_mean_bullseye`` in ``LVM_thickness/utils/bullseye.py``.

    Args:
        ax: Polar axis to draw on.
        values: Changes in % of segments 1-16, in segment order; missing values are drawn white.
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
            label = f"{value:+.1f}%" if np.isfinite(value) else "–"
            ax.text(theta, (r_in + r_out) / 2, label, ha="center", va="center", fontsize=10, fontweight="bold",
                    path_effects=outline)

    ax.bar(0, RADII[0], 2 * np.pi, 0, color=APEX_COLOUR, edgecolor="none")
    ax.plot(np.linspace(0, 2 * np.pi, 100), np.full(100, RADII[0]), "k", linewidth=1.5)
    ax.set_title(title, fontsize=13, pad=12)


def bullseye_figure(summary: pd.DataFrame, vmax: float, cmap: str = CMAP) -> plt.Figure:
    """One row of bullseyes of the mean relative change per segment, one per panel in ``PANELS``.

    Args:
        summary: Table from ``group_changes``.
        vmax: Largest absolute change in % of the shared colour scale, which runs from ``-vmax`` to ``vmax``.
        cmap: Name of a diverging colormap.

    Returns:
        The figure.
    """
    norm = Normalize(vmin=-vmax, vmax=vmax)
    fig, axes = plt.subplots(
        1, len(PANELS), figsize=(4.6 * len(PANELS), 5.6), subplot_kw=dict(polar=True), layout="constrained"
    )
    for ax, (ht, sex) in zip(axes, PANELS):
        panel = summary[(summary["HT"] == ht) & (summary["sex"] == sex)].set_index("segment")
        values = panel["mean_change_pct"].reindex(WALL_SEGMENTS).to_numpy(dtype=float)
        n = int(panel["n"].reindex(WALL_SEGMENTS).max())
        draw_bullseye(ax, values, norm, cmap, f"{SEX_TITLES[sex]} {HT_TITLES[ht]}\n(n = {n})")
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation="horizontal", shrink=0.4,
                            aspect=40, pad=0.04)
    colorbar.set_label("Mean change in median wall thickness (%), (CGPS10 − CGPS1) / CGPS1")
    fig.suptitle("Change in LV wall thickness per AHA segment from baseline to 10-year follow-up", fontsize=15)
    return fig


if __name__ == "__main__":
    root, _ = load_environment()
    links = read_ids(Path(root) / "df" / LINKS_FILE, list(DATASETS))
    ht = read_ids(Path(root) / HT_FOLDER / HT_FILE, ["patient_id"])

    segments, summaries = {}, {}
    for dataset in DATASETS:
        run_folder = Path(run_output_dir(root, dataset, AORTA_EXCLUSION_MM))
        print(f"{dataset}: {run_folder / SEGMENT_STATISTICS}")
        segments[dataset] = read_ids(run_folder / SEGMENT_STATISTICS, ["patient_id"])
        summaries[dataset] = read_ids(run_folder / PATIENT_SUMMARY, ["patient_id"])

    changes = segment_changes(segments, summaries, links, ht)
    subjects = changes[list(DATASETS)].drop_duplicates()
    print(f"{len(subjects)} of {len(links)} linked subjects have segment thickness at both examinations.")
    unclassified = changes.loc[changes["HT"].isna() | changes["sex"].isna(), list(DATASETS)].drop_duplicates()
    if len(unclassified):
        print(f"[WARN] {len(unclassified)} subjects without an HT class or sex are left out of the plots.")

    summary = group_changes(changes)
    vmax = float(np.nanmax(np.abs(summary.loc[summary["segment"].isin(WALL_SEGMENTS), "mean_change_pct"])))
    print(f"Colour scale: -{vmax:.1f}% to {vmax:.1f}%")

    output_folder = Path(root) / OUTPUT_FOLDER
    output_folder.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_folder / f"{FIGURE_NAME}.csv", index=False)

    figure = bullseye_figure(summary, vmax)
    figure_path = output_folder / f"{FIGURE_NAME}.png"
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Table saved to: {output_folder / f'{FIGURE_NAME}.csv'}")
    print(f"Figure saved to: {figure_path}")

#################################################################################################################################################
# This script makes a spaghetti plot of one measurement, by default the LV myocardial mass (lvm_g), for the subjects scanned in both CGPS1
# (baseline) and CGPS10 (10-year follow-up), one plot for women and one for men. Each subject is a line from the baseline to the follow-up
# value, with a small dot at each value, coloured by the HT class of the subject's CGPS1 scan in df/repeated_measures/CGPS1_HT.csv (made
# by ht_classifier.py; CGPS10 is not used for the colour). Only subjects with a value at both examinations and an HT class are plotted,
# and the script prints how many of the linked subjects are left out and why.
# The per-patient values are read from <run folder>/statistics/statistics_patient_summary.csv of each dataset, e.g.
# output/CGPS10_0mm/statistics/statistics_patient_summary.csv, so run main_statistics_df.py for both datasets first.
# The subjects are linked by <df>/CGPS1_CGPS10_links.csv, made by comparative_analysis/compare.py.
# The plots and the table of the linked values are saved in <PROJECT_ROOT>/reports/figures/repeated_measures.
# e.g. uv run repeated_measures/spaghetti_plot.py --measurement lvm_g
#################################################################################################################################################

import os
import sys
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "measurements"))
sys.path.insert(0, os.path.join(REPO_ROOT, "LVM_thickness"))

from main_all_df import ROW_LABELS, SEX_CODES  # noqa: E402
from ht_classifier import HT_FILE, HT_FOLDER  # noqa: E402
from pipeline import AORTA_EXCLUSION_MM, load_environment, run_output_dir  # noqa: E402

DATASETS = {"CGPS1": "Baseline", "CGPS10": "10-year follow-up"}
LINKS_FILE = "CGPS1_CGPS10_links.csv"
PATIENT_SUMMARY = os.path.join("statistics", "statistics_patient_summary.csv")
OUTPUT_FOLDER = os.path.join("reports", "figures", "repeated_measures")
SEX_TITLES = {"male": "Men", "female": "Women"}
HT_STYLES = {False: ("No HT", "#1f77b4"), True: ("HT", "#d62728")}


def patient_summary_path(
    root: str, dataset: str, aorta_exclusion_mm: float, output_dir: str | None = None
) -> Path:
    """Patient summary written by main_statistics_df.py for one dataset.

    Args:
        root: Project root.
        dataset: Dataset name, e.g. ``CGPS1``.
        aorta_exclusion_mm: Aortic exclusion distance of the run.
        output_dir: Parent of the run folders; defaults to ``<root>/output``.

    Returns:
        ``<run folder>/statistics/statistics_patient_summary.csv``, e.g. ``output/CGPS10_0mm/statistics/...``.
    """
    return Path(run_output_dir(root, dataset, aorta_exclusion_mm, output_dir)) / PATIENT_SUMMARY


def read_ids(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read a csv file with the given id columns as strings, so the leading zeros of the patient ids are kept."""
    return pd.read_csv(path, dtype={column: str for column in columns})


def linked_wide(summaries: dict[str, pd.DataFrame], links: pd.DataFrame, measurement: str) -> pd.DataFrame:
    """The links table with the measurement and sex of each subject at both examinations, one row per subject.

    Args:
        summaries: Dataset name to its patient summary, with ``patient_id``, ``sex`` and the measurement column.
        links: One row per subject with the patient ids in the columns ``CGPS1`` and ``CGPS10``.
        measurement: Column of the measurement, e.g. ``lvm_g``.

    Returns:
        One row per link with ``subject`` (``<CGPS1 id>/<CGPS10 id>``), the ids, ``in_<dataset>`` (the id is in the
        patient summary), ``value_<dataset>`` and ``sex`` (0 = man, 1 = woman; from CGPS1, else CGPS10).
    """
    wide = links[list(DATASETS)].drop_duplicates().reset_index(drop=True)
    wide["subject"] = wide["CGPS1"] + "/" + wide["CGPS10"]
    sex = pd.Series(pd.NA, index=wide.index, dtype="Float64")
    for dataset, summary in summaries.items():
        summary = summary.drop_duplicates("patient_id").set_index("patient_id")
        wide[f"in_{dataset}"] = wide[dataset].isin(summary.index)
        wide[f"value_{dataset}"] = wide[dataset].map(summary[measurement])
        if "sex" in summary.columns:
            sex = sex.fillna(wide[dataset].map(summary["sex"]).astype("Float64"))
    wide["sex"] = sex
    return wide


def link_report(wide: pd.DataFrame) -> pd.Series:
    """Counts of why linked subjects are or are not in the plot.

    Args:
        wide: Table from ``linked_wide``.

    Returns:
        Number of links, of links missing from each patient summary, of links missing the value at each examination
        while being in the summary, and of links with a value at both examinations.
    """
    counts = {"links": len(wide)}
    for dataset in DATASETS:
        counts[f"not in {dataset} patient summary"] = int((~wide[f"in_{dataset}"]).sum())
    for dataset in DATASETS:
        counts[f"in {dataset} summary but value missing"] = int(
            (wide[f"in_{dataset}"] & wide[f"value_{dataset}"].isna()).sum()
        )
    both = wide[[f"value_{dataset}" for dataset in DATASETS]].notna().all(axis=1)
    counts["value at both examinations"] = int(both.sum())
    return pd.Series(counts)


def paired_values(summaries: dict[str, pd.DataFrame], links: pd.DataFrame, measurement: str) -> pd.DataFrame:
    """Values of one measurement of each subject at both examinations, in long format.

    Args:
        summaries: Dataset name to its patient summary, with ``patient_id``, ``sex`` and the measurement column.
        links: One row per subject with the patient ids in the columns ``CGPS1`` and ``CGPS10``.
        measurement: Column of the measurement, e.g. ``lvm_g``.

    Returns:
        Two rows per subject with ``subject``, ``sex``, ``dataset``, ``patient_id`` and ``value``. Subjects without a
        value at both examinations are left out, so both examinations have the same subjects.
    """
    wide = linked_wide(summaries, links, measurement)
    wide = wide.dropna(subset=[f"value_{dataset}" for dataset in DATASETS])
    rows = [
        pd.DataFrame(
            {
                "subject": wide["subject"],
                "sex": wide["sex"],
                "dataset": dataset,
                "patient_id": wide[dataset],
                "value": wide[f"value_{dataset}"],
            }
        )
        for dataset in DATASETS
    ]
    return pd.concat(rows, ignore_index=True)


def add_ht(data: pd.DataFrame, ht: pd.DataFrame) -> pd.DataFrame:
    """Add the HT class of each subject's CGPS1 scan to the long table; CGPS10 is not used.

    Args:
        data: Long table from ``paired_values``.
        ht: One row per CGPS1 patient with ``patient_id`` and ``HT``, from ht_classifier.py.

    Returns:
        ``data`` with the column ``HT`` on both rows of each subject, missing if the CGPS1 scan has no class.
    """
    classes = ht.drop_duplicates("patient_id").set_index("patient_id")["HT"].astype("boolean")
    cgps1_ids = data["subject"].map(data.loc[data["dataset"] == "CGPS1"].set_index("subject")["patient_id"])
    return data.assign(HT=cgps1_ids.map(classes).astype("boolean"))


def spaghetti_plot(
    data: pd.DataFrame, label: str, group: str = "", ylim: tuple[float, float] | None = None
) -> plt.Figure:
    """Spaghetti plot with one line per subject from baseline to follow-up, a dot at each value and coloured by HT.

    Args:
        data: Long table from ``add_ht`` with a ``True``/``False`` HT class for every subject.
        label: Axis label of the measurement, e.g. ``Mass (g)``.
        group: Name of the subjects in the title, e.g. ``Women``; empty for all subjects.
        ylim: Limits of the y axis, e.g. the same for the plots of women and men; automatic if not given.

    Returns:
        The figure.
    """
    positions = {dataset: x for x, dataset in enumerate(DATASETS)}
    n_subjects = data["subject"].nunique()

    fig, ax = plt.subplots(figsize=(5, 6))
    for ht, (name, colour) in HT_STYLES.items():
        subjects = data[data["HT"] == ht]
        wide = subjects.pivot(index="subject", columns="dataset", values="value")[list(DATASETS)]
        if wide.empty:
            continue
        alpha = min(0.9, max(0.3, 60 / len(wide)))
        lines = ax.plot(
            list(positions.values()), wide.to_numpy().T, color=colour, linewidth=0.7, alpha=alpha, marker="o",
            markersize=2,
        )
        lines[0].set_label(f"{name} (n = {len(wide)})")
    ax.set_xticks(list(positions.values()), [f"{dataset}\n{DATASETS[dataset]}" for dataset in DATASETS])
    ax.set_xlim(-0.3, len(DATASETS) - 0.7)
    ax.set_ylabel(label)
    ax.set_title(f"{group or 'All'}: {label} (n = {n_subjects})")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    root, _ = load_environment()

    parser = argparse.ArgumentParser(description="Spaghetti plot of one measurement from CGPS1 to CGPS10 per subject.")
    parser.add_argument("--measurement", default="lvm_g", choices=list(ROW_LABELS),
                        help="Column of the patient summary to plot (default: lvm_g).")
    parser.add_argument("--links", default=None, help=f"Links table (default: <PROJECT_ROOT>/df/{LINKS_FILE}).")
    parser.add_argument("--aorta-exclusion-mm", type=float, default=AORTA_EXCLUSION_MM,
                        help=f"Aortic exclusion distance of the runs (default: {AORTA_EXCLUSION_MM}).")
    parser.add_argument("--output-dir", default=None,
                        help="Parent of the run folders (default: <PROJECT_ROOT>/output).")
    parser.add_argument("--ht", default=None,
                        help=f"HT classes of the CGPS1 patients (default: <PROJECT_ROOT>/{HT_FOLDER}/{HT_FILE}).")
    parser.add_argument("--figure-dir", default=None,
                        help=f"Folder of the plots and the table (default: <PROJECT_ROOT>/{OUTPUT_FOLDER}).")
    args = parser.parse_args()

    links_path = Path(args.links) if args.links else Path(root) / "df" / LINKS_FILE
    if not links_path.is_file():
        raise SystemExit(f"Links table not found: {links_path}\nRun comparative_analysis/compare.py first.")
    links = read_ids(links_path, list(DATASETS))

    summaries = {}
    for dataset in DATASETS:
        summary_path = patient_summary_path(root, dataset, args.aorta_exclusion_mm, args.output_dir)
        if not summary_path.is_file():
            raise SystemExit(
                f"Patient summary not found: {summary_path}\nRun main_statistics_df.py for {dataset} first."
            )
        print(f"{dataset}: {summary_path}")
        summaries[dataset] = read_ids(summary_path, ["patient_id"])
        if args.measurement not in summaries[dataset].columns:
            raise SystemExit(f"Column {args.measurement!r} not in {summary_path}.")

    print(link_report(linked_wide(summaries, links, args.measurement)).to_string())
    data = paired_values(summaries, links, args.measurement)
    if data.empty:
        raise SystemExit("No subjects with a value at both examinations.")

    ht_path = Path(args.ht) if args.ht else Path(root) / HT_FOLDER / HT_FILE
    if not ht_path.is_file():
        raise SystemExit(f"HT classes not found: {ht_path}\nRun repeated_measures/ht_classifier.py first.")
    data = add_ht(data, read_ids(ht_path, ["patient_id"]))
    missing_ht = data.loc[data["HT"].isna(), "subject"].nunique()
    if missing_ht:
        print(f"[WARN] {missing_ht} subjects without an HT class for their CGPS1 scan are left out of the plots.")

    output_folder = Path(args.figure_dir) if args.figure_dir else Path(root) / OUTPUT_FOLDER
    output_folder.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_folder / f"spaghetti_{args.measurement}.csv", index=False)

    padding = 0.05 * (data["value"].max() - data["value"].min())
    ylim = (data["value"].min() - padding, data["value"].max() + padding)
    missing_sex = data.loc[data["sex"].isna(), "subject"].nunique()
    if missing_sex:
        print(f"[WARN] {missing_sex} subjects without sex are left out of the plots.")
    for sex, code in SEX_CODES.items():
        group = data[(data["sex"] == code) & data["HT"].notna()]
        if group.empty:
            print(f"[WARN] No {sex} subjects, no plot made.")
            continue
        figure = spaghetti_plot(group, ROW_LABELS[args.measurement], SEX_TITLES[sex], ylim)
        figure_path = output_folder / f"spaghetti_{args.measurement}_{sex}.png"
        figure.savefig(figure_path, dpi=300)
        plt.close(figure)
        print(f"{SEX_TITLES[sex]}: n = {group['subject'].nunique()}, plot saved to: {figure_path}")

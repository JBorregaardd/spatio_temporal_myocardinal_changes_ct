#################################################################################################################################################
# This script makes a spaghetti plot of one measurement, by default the LV myocardial mass (lvm_g), for the subjects scanned in both CGPS1
# (baseline) and CGPS10 (10-year follow-up), one plot for women and one for men. Each subject is a line from the baseline to the follow-up
# value, and the mean ± sd is drawn on top. Only subjects with a value at both examinations are plotted, and the script prints how many of
# the linked subjects are left out and why.
# The per-patient values are read from the patient summary statistics_patient_summary.csv of each dataset. 
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
from pipeline import load_environment  # noqa: E402

DATASETS = {"CGPS1": "Baseline", "CGPS10": "10-year follow-up"}
LINKS_FILE = "CGPS1_CGPS10_links.csv"
PATIENT_SUMMARY = "statistics_patient_summary.csv"
OUTPUT_FOLDER = os.path.join("reports", "figures", "repeated_measures")
SEX_TITLES = {"male": "Men", "female": "Women"}


def dataset_folders(parent: Path, dataset: str) -> list[Path]:
    """Folders in ``parent`` of one dataset: the name starts with ``<dataset>_`` in any case, e.g. ``CGPS1_0mm_n1772``.

    The ``_`` keeps ``CGPS1`` from matching the ``CGPS10`` folders.

    Args:
        parent: Folder to look in, e.g. ``<PROJECT_ROOT>/df``.
        dataset: Dataset name, e.g. ``CGPS1``.

    Returns:
        The matching folders sorted by name, empty if ``parent`` does not exist.
    """
    if not parent.is_dir():
        return []
    prefix = f"{dataset.lower()}_"
    return sorted(path for path in parent.iterdir() if path.is_dir() and path.name.lower().startswith(prefix))


def find_patient_summary(root: Path, dataset: str) -> Path:
    """Patient summary of one dataset, looked for first under ``<root>/df`` and then under ``<root>/output``.

    Args:
        root: Project root.
        dataset: Dataset name, e.g. ``CGPS1``.

    Returns:
        Path of the first ``statistics_patient_summary.csv`` found in a folder of the dataset.

    Raises:
        FileNotFoundError: If no folder of the dataset has a patient summary.
    """
    searched = []
    for parent in (root / "df", root / "output"):
        for folder in dataset_folders(parent, dataset):
            searched.append(folder)
            matches = sorted(folder.rglob(PATIENT_SUMMARY))
            if matches:
                return matches[0]
    where = "\n  ".join(str(folder) for folder in searched) or "no folders of the dataset in df/ or output/"
    raise FileNotFoundError(f"No {PATIENT_SUMMARY} for {dataset}. Searched:\n  {where}")


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
    counts["value at both examinations"] = int(wide[[f"value_{dataset}" for dataset in DATASETS]].notna().all(axis=1).sum())
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


def spaghetti_plot(
    data: pd.DataFrame, label: str, group: str = "", ylim: tuple[float, float] | None = None
) -> plt.Figure:
    """Spaghetti plot with one line per subject from baseline to follow-up and the mean ± sd on top.

    Args:
        data: Long table from ``paired_values``.
        label: Axis label of the measurement, e.g. ``Mass (g)``.
        group: Name of the subjects in the title, e.g. ``Women``; empty for all subjects.
        ylim: Limits of the y axis, e.g. the same for the plots of women and men; automatic if not given.

    Returns:
        The figure.
    """
    positions = {dataset: x for x, dataset in enumerate(DATASETS)}
    wide = data.pivot(index="subject", columns="dataset", values="value")[list(DATASETS)]
    n_subjects = len(wide)
    alpha = min(0.6, max(0.05, 30 / max(n_subjects, 1)))

    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(list(positions.values()), wide.to_numpy().T, color="0.5", linewidth=0.6, alpha=alpha)
    means, sds = wide.mean(), wide.std()
    ax.errorbar(
        list(positions.values()), means.to_numpy(), yerr=sds.to_numpy(), color="C3", linewidth=2.5, marker="o",
        markersize=7, capsize=6, label="Mean ± sd", zorder=3,
    )
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
    root = Path(root)

    parser = argparse.ArgumentParser(description="Spaghetti plot of one measurement from CGPS1 to CGPS10 per subject.")
    parser.add_argument("--measurement", default="lvm_g", choices=list(ROW_LABELS),
                        help="Column of the patient summary to plot (default: lvm_g).")
    parser.add_argument("--links", default=None, help=f"Links table (default: <PROJECT_ROOT>/df/{LINKS_FILE}).")
    parser.add_argument("--output-dir", default=None,
                        help=f"Folder of the outputs (default: <PROJECT_ROOT>/{OUTPUT_FOLDER}).")
    args = parser.parse_args()

    links_path = Path(args.links) if args.links else root / "df" / LINKS_FILE
    if not links_path.is_file():
        raise SystemExit(f"Links table not found: {links_path}\nRun comparative_analysis/compare.py first.")
    links = read_ids(links_path, list(DATASETS))

    summaries = {}
    for dataset in DATASETS:
        try:
            summary_path = find_patient_summary(root, dataset)
        except FileNotFoundError as error:
            raise SystemExit(str(error))
        print(f"{dataset}: {summary_path}")
        summaries[dataset] = read_ids(summary_path, ["patient_id"])
        if args.measurement not in summaries[dataset].columns:
            raise SystemExit(f"Column {args.measurement!r} not in {summary_path}.")

    data = paired_values(summaries, links, args.measurement)
    if data.empty:
        raise SystemExit("No subjects with a value at both examinations.")
    print(link_report(linked_wide(summaries, links, args.measurement)).to_string())

    output_folder = Path(args.output_dir) if args.output_dir else root / OUTPUT_FOLDER
    output_folder.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_folder / f"spaghetti_{args.measurement}.csv", index=False)

    padding = 0.05 * (data["value"].max() - data["value"].min())
    ylim = (data["value"].min() - padding, data["value"].max() + padding)
    missing_sex = data.loc[data["sex"].isna(), "subject"].nunique()
    if missing_sex:
        print(f"[WARN] {missing_sex} subjects without sex are left out of the plots.")
    for sex, code in SEX_CODES.items():
        group = data[data["sex"] == code]
        if group.empty:
            print(f"[WARN] No {sex} subjects, no plot made.")
            continue
        figure = spaghetti_plot(group, ROW_LABELS[args.measurement], SEX_TITLES[sex], ylim)
        figure_path = output_folder / f"spaghetti_{args.measurement}_{sex}.png"
        figure.savefig(figure_path, dpi=300)
        plt.close(figure)
        print(f"{SEX_TITLES[sex]}: n = {group['subject'].nunique()}, plot saved to: {figure_path}")

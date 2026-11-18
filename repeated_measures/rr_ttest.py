#################################################################################################################################################
# This script tests whether the CGPS1 (baseline) and CGPS10 (10-year follow-up) scans of the same subject are taken at the same point in the
# R-R interval, with a paired t-test of CGPS1_rr_percent and CGPS10_rr_percent in <df>/CGPS1_CGPS10_links.csv, made by
# comparative_analysis/compare.py with CGPS1_FILELIST_CSV and CGPS10_FILELIST_CSV set in the .env.
# Only subjects with a percentage at both examinations are used. It prints n, the mean ± sd at each examination, the mean difference
# (CGPS10 - CGPS1) with its 95% confidence interval, t and p, and saves them in <PROJECT_ROOT>/df/repeated_measures/rr_ttest.csv.
# H0: The mean difference is zero (the scans are taken at the same point in the R-R interval).
# e.g. uv run repeated_measures/rr_ttest.py
#################################################################################################################################################

import os
import sys
import argparse
from pathlib import Path

import pandas as pd
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "LVM_thickness"))
sys.path.insert(0, os.path.join(REPO_ROOT, "repeated_measures"))

from ht_classifier import HT_FOLDER  # noqa: E402
from pipeline import load_environment  # noqa: E402
from spaghetti_plot import DATASETS, LINKS_FILE, read_ids  # noqa: E402

RR_COLUMNS = {dataset: f"{dataset}_rr_percent" for dataset in DATASETS}
TTEST_FILE = "rr_ttest.csv"


def paired_rr_ttest(links: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """Paired t-test of the point in the R-R interval at CGPS10 against CGPS1.

    Args:
        links: Links table with the columns in ``RR_COLUMNS``.
        confidence: Level of the confidence interval of the mean difference.

    Returns:
        One row with ``n``, ``mean_<dataset>`` and ``sd_<dataset>`` per dataset, ``mean_difference`` and
        ``sd_difference`` (CGPS10 - CGPS1), ``ci_low``, ``ci_high``, ``t`` and ``p``.

    Raises:
        ValueError: If a column in ``RR_COLUMNS`` is missing or fewer than two subjects have both percentages.
    """
    missing = [column for column in RR_COLUMNS.values() if column not in links.columns]
    if missing:
        raise ValueError(f"Links table is missing the columns {missing}.")
    pairs = links[list(RR_COLUMNS.values())].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pairs) < 2:
        raise ValueError(f"Only {len(pairs)} subjects have the R-R percentage at both examinations.")
    baseline, follow_up = pairs[RR_COLUMNS["CGPS1"]], pairs[RR_COLUMNS["CGPS10"]]
    result = stats.ttest_rel(follow_up, baseline)
    interval = result.confidence_interval(confidence)
    difference = follow_up - baseline
    row = {"n": len(pairs)}
    for dataset, column in RR_COLUMNS.items():
        row[f"mean_{dataset}"] = pairs[column].mean()
        row[f"sd_{dataset}"] = pairs[column].std()
    row.update(
        {
            "mean_difference": difference.mean(),
            "sd_difference": difference.std(),
            "ci_low": interval.low,
            "ci_high": interval.high,
            "t": result.statistic,
            "p": result.pvalue,
        }
    )
    return pd.DataFrame([row])


if __name__ == "__main__":
    root, _ = load_environment()

    parser = argparse.ArgumentParser(description="Paired t-test of the R-R percentage of the CGPS1 and CGPS10 scans.")
    parser.add_argument("--links", default=None, help=f"Links table (default: <PROJECT_ROOT>/df/{LINKS_FILE}).")
    args = parser.parse_args()

    links_path = Path(args.links) if args.links else Path(root) / "df" / LINKS_FILE
    if not links_path.is_file():
        raise SystemExit(f"Links table not found: {links_path}\nRun comparative_analysis/compare.py first.")
    links = read_ids(links_path, list(DATASETS))

    try:
        result = paired_rr_ttest(links)
    except ValueError as error:
        raise SystemExit(f"{error}\nRun comparative_analysis/compare.py with the file lists set in the .env.")
    row = result.iloc[0]
    print(f"{row['n']} of {len(links)} linked subjects have the R-R percentage at both examinations.")
    for dataset in DATASETS:
        print(f"{dataset}: {row[f'mean_{dataset}']:.1f} ± {row[f'sd_{dataset}']:.1f} %")
    print(
        f"Difference (CGPS10 - CGPS1): {row['mean_difference']:.2f} ± {row['sd_difference']:.2f} %, "
        f"95% CI [{row['ci_low']:.2f}, {row['ci_high']:.2f}], t = {row['t']:.2f}, p = {row['p']:.3g}"
    )

    output_folder = Path(root) / HT_FOLDER
    output_folder.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_folder / TTEST_FILE, index=False)
    print(f"Result saved to: {output_folder / TTEST_FILE}")

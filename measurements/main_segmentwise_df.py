#################################################################################################################################################
# This script makes one table per AHA segment from the per-segment statistics written by main_statistics_df.py. Each table has a row for n
# and for each measurement given as mean ± sd over the patients, and a column per group of patients. For now there is one group, "All";
# the hypertension groups will be added later.
# It reads <run folder>/statistics/statistics_thickness_per_segment.csv and saves the tables in
# <df>/<dataset>_<aorta-exclusion-mm>mm_n<number of patients>/segments/segment_<n>.csv.
# Run main_statistics_df.py for the same --dataset and --aorta-exclusion-mm first.
# e.g. uv run measurements/main_segmentwise_df.py --dataset CGPS10 --aorta-exclusion-mm 0
#################################################################################################################################################

import os
import sys
import argparse
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "measurements"))
sys.path.insert(0, os.path.join(REPO_ROOT, "LVM_thickness"))

from pipeline import AORTA_EXCLUSION_MM, load_environment, run_output_dir, table_output_dir  # noqa: E402
from thickness import SEGMENTS  # noqa: E402

SEGMENT_STATISTICS = os.path.join("statistics", "statistics_thickness_per_segment.csv")
ROW_LABELS = {
    "median_thickness": "Median thickness (mm)",
    "mean_thickness": "Mean thickness (mm)",
    "volume_ml": "Volume (mL)",
    "mass_g": "Mass (g)",
    "SLVDVI": "SLVDVI (mL/m^2)",
}


def segment_table(groups: dict[str, pd.DataFrame], segment: int) -> pd.DataFrame:
    """Make the table of one segment: n and each measurement as mean ± sd over the patients of each group.

    Measurements without any values in a group, e.g. SLVDVI for a dataset without demographics, are left empty.

    Args:
        groups: Group name to that group's rows, one row per patient and segment with the columns in ``ROW_LABELS``.
        segment: AHA segment (1-17).

    Returns:
        A table with one row for n and per measurement, and one column per group.
    """
    columns = {}
    for name, data in groups.items():
        rows = data[data["segment"] == segment]
        column = {"n": str(rows["patient_id"].nunique())}
        for key, label in ROW_LABELS.items():
            values = rows[key]
            column[label] = f"{values.mean():.2f} ± {values.std():.2f}" if values.notna().any() else ""
        columns[name] = column
    return pd.DataFrame(columns).rename_axis(f"Segment {segment}")


if __name__ == "__main__":
    root, _ = load_environment()

    parser = argparse.ArgumentParser(description="Create one table per AHA segment from the per-segment statistics.")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name of the pipeline run, as given to main_statistics_df.py.")
    parser.add_argument("--aorta-exclusion-mm", type=float, default=AORTA_EXCLUSION_MM,
                        help=f"Aortic exclusion distance of the run (default: {AORTA_EXCLUSION_MM}).")
    parser.add_argument("--output-dir", default=None,
                        help="Parent of the run folders (default: <PROJECT_ROOT>/output).")
    args = parser.parse_args()

    segment_csv = os.path.join(
        run_output_dir(root, args.dataset, args.aorta_exclusion_mm, args.output_dir), SEGMENT_STATISTICS
    )
    if not os.path.isfile(segment_csv):
        raise SystemExit(f"Segment statistics not found: {segment_csv}\nRun main_statistics_df.py first.")

    data = pd.read_csv(segment_csv)
    groups = {"All": data}
    output_folder = os.path.join(
        table_output_dir(root, args.dataset, args.aorta_exclusion_mm, data["patient_id"].nunique()), "segments"
    )
    os.makedirs(output_folder, exist_ok=True)

    for segment in SEGMENTS:
        table = segment_table(groups, segment)
        print(f"\n{table}")
        table.to_csv(os.path.join(output_folder, f"segment_{segment}.csv"), encoding="utf-8-sig")
    print(f"\nSegment tables saved to: {output_folder}")

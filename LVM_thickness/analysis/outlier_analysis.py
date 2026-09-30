#################################################################################################################################################
# This script takes as input the statistics_thickness_per_segment.csv and identifies outliers based on the median thickness values for each segment across all patients.
# It reads <output>/<dataset>_<aorta-exclusion-mm>/statistics/ (written by statistics_analysis.py) and saves the outliers to its outliers/ subfolder.
# e.g. uv run LVM_thickness/analysis/outlier_analysis.py --dataset ImageCAS_1-200 --aorta-exclusion-mm 2.5
#################################################################################################################################################

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import AORTA_EXCLUSION_MM, run_output_dir  # noqa: E402


def detect_outliers_per_segment(data):
    """
    Detect outliers separately for each segment using the IQR method.

    For each segment:
        lower bound = Q1 - 1.5 * IQR
        upper bound = Q3 + 1.5 * IQR

    Outliers are patients whose thickness is outside these bounds.
    """

    results = []

    for segment, segment_data in data.groupby("segment"):

        q1 = segment_data["median_thickness"].quantile(0.25)
        q3 = segment_data["median_thickness"].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = segment_data[
            (segment_data["median_thickness"] < lower_bound) |
            (segment_data["median_thickness"] > upper_bound)
        ].copy()

        outliers["lower_bound"] = lower_bound
        outliers["upper_bound"] = upper_bound

        results.append(outliers)

    if results:
        return pd.concat(results, ignore_index=True)

    return pd.DataFrame()


def plot_segment(data, segment, output_path):
    """
    Create a boxplot for one segment across patients.
    """

    segment_data = data[
        data["segment"] == segment
    ]

    plt.figure(figsize=(8, 6))

    plt.boxplot(
        segment_data["median_thickness"],
        vert=True
    )

    # Plot individual patient values
    x = [1] * len(segment_data)

    plt.scatter(
        x,
        segment_data["median_thickness"],
        alpha=0.5
    )

    plt.ylabel("Median thickness (mm)")
    plt.title(f"Median myocardial thickness - Segment {segment}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Find per-segment thickness outliers in one pipeline run.")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name of the pipeline run to analyse, as given to main_pipeline.py.")
    parser.add_argument("--aorta-exclusion-mm", type=float, default=AORTA_EXCLUSION_MM,
                        help=f"Aortic exclusion distance of the run to analyse (default: {AORTA_EXCLUSION_MM}). "
                             f"Together with --dataset it selects <output-dir>/<dataset>_<aorta-exclusion-mm>.")
    parser.add_argument("--output-dir", default=None,
                        help="Parent of the run folders (default: <PROJECT_ROOT>/output).")
    return parser.parse_args()


def main():

    args = parse_args()

    # Find project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    load_dotenv(os.path.join(project_root, ".env"))
    root = os.environ["PROJECT_ROOT"]

    statistics_folder = os.path.join(
        run_output_dir(root, args.dataset, args.aorta_exclusion_mm, args.output_dir), "statistics"
    )
    path = os.path.join(statistics_folder, "statistics_thickness_per_segment.csv")
    if not os.path.isfile(path):
        raise SystemExit(f"{path} not found; run statistics_analysis.py with the same --dataset first.")

    data = pd.read_csv(path)
    print(f"Number of patients: {data['patient_id'].nunique()}")
    output_folder = os.path.join(statistics_folder, "outliers")
    os.makedirs(output_folder, exist_ok=True)

    # --------------------------------------------------
    # Detect outliers
    # --------------------------------------------------
    outliers = detect_outliers_per_segment(data)

    print("\nOutlier detection")
    print("-----------------")
    print(f"Total number of outliers: {len(outliers)}")

    outlier_csv = os.path.join(output_folder,"outliers.csv")
    outliers.to_csv(outlier_csv,index=False)
    print(f"Outliers saved to: {outlier_csv}")

    # --------------------------------------------------
    # Print summary
    # --------------------------------------------------

    print("\nOutliers per segment:")

    if len(outliers) > 0:
        summary = (
            outliers
            .groupby("segment")
            .size()
            .reset_index(name="number_of_outliers")
        )

        print(summary.to_string(index=False))

    else:
        print("No outliers detected.")

    # --------------------------------------------------
    # Create plots for each segment
    # --------------------------------------------------

    plt.figure(figsize=(14, 7))

    data.boxplot(
        column="median_thickness",
        by="segment"
    )

    plt.xlabel("Segment")
    plt.ylabel("Median thickness (mm)")
    plt.title("Median myocardial thickness across patients by segment")

    plt.suptitle("")  # Remove pandas' automatic title

    plt.tight_layout()

    plot_path = os.path.join(
        output_folder,
        "thickness_by_segment.png"
    )

    plt.savefig(
        plot_path,
        dpi=300
    )

    plt.close()

    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
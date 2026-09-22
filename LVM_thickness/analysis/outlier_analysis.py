#################################################################################################################################################
# This script takes as input the statistics_thickness_per_segment.csv and identifies outliers based on the median thickness values for each segment across all patients.
#################################################################################################################################################

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv


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



def main():

    # Find project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    load_dotenv(os.path.join(project_root, ".env"))
    root = os.environ["PROJECT_ROOT"]

    path = os.path.join(root,"output","statistics","statistics_thickness_per_segment.csv")


    data = pd.read_csv(path)
    print(f"Number of patients: {data['patient_id'].nunique()}")
    output_folder = os.path.join(root,"output","statistics","outliers")
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
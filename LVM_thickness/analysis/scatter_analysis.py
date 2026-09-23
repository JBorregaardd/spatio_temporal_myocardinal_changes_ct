#################################################################################################################################################
# This script takes as input the statistics_thickness_per_segment.csv and creates scatter plots for
# the mean/median thickness and the volume/median thickness of each segment across all patients
#################################################################################################################################################

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from dotenv import load_dotenv


def plot_volume_vs_thickness(data, output_path):
    """
    Create a figure with two scatter plots: mean vs. median myocardial
    thickness, and volume vs. median myocardial thickness.
    Each segment is shown with a color from a gradient colormap,
    ordered from segment 1 (start color) to segment 17 (end color).
    """

    fig, (ax_mean, ax_volume) = plt.subplots(1, 2, figsize=(18, 7))

    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=1, vmax=17)

    for segment, segment_data in data.groupby("segment"):
        color = cmap(norm(segment))

        ax_mean.scatter(
            segment_data["mean_thickness"],
            segment_data["median_thickness"],
            label=f"Segment {segment}",
            color=color,
            alpha=0.7
        )

        ax_volume.scatter(
            segment_data["volume_ml"],
            segment_data["median_thickness"],
            label=f"Segment {segment}",
            color=color,
            alpha=0.7
        )

    ax_mean.set_xlabel("Mean thickness (mm)")
    ax_mean.set_ylabel("Median thickness (mm)")
    ax_mean.set_title("Mean vs. median myocardial thickness")

    ax_volume.set_xlabel("Volume (mL)")
    ax_volume.set_ylabel("Median thickness (mm)")
    ax_volume.set_title("Volume vs. median myocardial thickness")

    ax_volume.legend(
        title="Segment",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():

    # Find project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    load_dotenv(os.path.join(project_root, ".env"))
    root = os.environ["PROJECT_ROOT"]

    # Input CSV
    path = os.path.join(
        root,
        "output",
        "statistics",
        "statistics_thickness_per_segment.csv"
    )

    data = pd.read_csv(path)

    print(f"Number of patients: {data['patient_id'].nunique()}")

    # Output folder
    output_folder = os.path.join(
        root,
        "output",
        "statistics"
    )

    os.makedirs(output_folder, exist_ok=True)

    # Create plot
    plot_path = os.path.join(
        output_folder,
        "volume_vs_median_thickness.png"
    )

    plot_volume_vs_thickness(
        data,
        plot_path
    )

    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
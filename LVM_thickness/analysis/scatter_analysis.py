#################################################################################################################################################
# This script takes as input the statistics_thickness_per_segment.csv and creates scatter plots for
# the mean/median thickness and the volume/median thickness of each segment across all patients.
# It reads <output>/<dataset>_<aorta-exclusion-mm>/statistics/ (written by statistics_analysis.py) and saves the plot there.
# e.g. uv run LVM_thickness/analysis/scatter_analysis.py --dataset ImageCAS_1-200 --aorta-exclusion-mm 2.5
#################################################################################################################################################

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import AORTA_EXCLUSION_MM, run_output_dir  # noqa: E402


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

    ax_mean.set_xlabel("Mean thickness (mm)", fontsize=16)
    ax_mean.set_ylabel("Median thickness (mm)", fontsize=16)
    ax_mean.set_title("Mean vs. median myocardial thickness", fontsize=18)
    ax_mean.tick_params(axis="both", labelsize=14)

    ax_volume.set_xlabel("Volume (mL)", fontsize=16)
    ax_volume.set_ylabel("Median thickness (mm)", fontsize=16)
    ax_volume.set_title("Volume vs. median myocardial thickness", fontsize=18)
    ax_volume.tick_params(axis="both", labelsize=14)

    ax_volume.legend(
        title="Segment",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=14,
        title_fontsize=15
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Scatter plots of per-segment thickness and volume for one run.")
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

    # Input CSV
    output_folder = os.path.join(
        run_output_dir(root, args.dataset, args.aorta_exclusion_mm, args.output_dir),
        "statistics"
    )
    path = os.path.join(output_folder, "statistics_thickness_per_segment.csv")
    if not os.path.isfile(path):
        raise SystemExit(f"{path} not found; run statistics_analysis.py with the same --dataset first.")

    data = pd.read_csv(path)

    print(f"Number of patients: {data['patient_id'].nunique()}")

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
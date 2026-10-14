#################################################################################################################################################
# This script plots the volumetric sphericity index of the left ventricle (LV) cavity against the LVDVI (mL/m^2), one dot per patient.
# Both are read from <output>/<dataset>_<aorta-exclusion-mm>/statistics/statistics_patient_summary.csv written by
# measurements/main_statistics_df.py. The plot is saved in <output>/<dataset>_<aorta-exclusion-mm>/sphericity.
# e.g. uv run initial_data_investigation/spherical_scatter.py --dataset CGPS10 --aorta-exclusion-mm 0
#################################################################################################################################################

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "LVM_thickness"))

from pipeline import AORTA_EXCLUSION_MM, load_environment, run_output_dir  # noqa: E402

STATISTICS_CSV = os.path.join("statistics", "statistics_patient_summary.csv")


def plot_sphericity_vs_lvdvi(data: pd.DataFrame, output_path: str) -> None:
    """Scatter plot of the sphericity index against LVDVI, one dot per patient.

    Args:
        data: One row per patient with ``LVDVI`` and ``Sphericity_index``.
        output_path: Where to save the figure.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(data["LVDVI"], data["Sphericity_index"], color="tab:blue", alpha=0.7)

    ax.set_xlabel("LVDVI (mL/m²)", fontsize=16)
    ax.set_ylabel("Volumetric sphericity index", fontsize=16)
    ax.set_title(f"LV sphericity vs. LVDVI (n = {len(data)})", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    root, _ = load_environment()

    parser = argparse.ArgumentParser(description="Plot the LV sphericity index against LVDVI.")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name of the pipeline run to plot, as given to main_pipeline.py.")
    parser.add_argument("--aorta-exclusion-mm", type=float, default=AORTA_EXCLUSION_MM,
                        help=f"Aortic exclusion distance of the run to plot (default: {AORTA_EXCLUSION_MM}).")
    parser.add_argument("--output-dir", default=None,
                        help="Parent of the run folders (default: <PROJECT_ROOT>/output).")
    args = parser.parse_args()

    run_folder = run_output_dir(root, args.dataset, args.aorta_exclusion_mm, args.output_dir)
    csv_path = os.path.join(run_folder, STATISTICS_CSV)
    data = pd.read_csv(csv_path, dtype={"patient_id": str}).dropna(subset=["LVDVI", "Sphericity_index"])
    if data.empty:
        raise SystemExit(f"No patients with both LVDVI and Sphericity_index in {csv_path}.")

    output_folder = os.path.join(run_folder, "sphericity")
    os.makedirs(output_folder, exist_ok=True)
    plot_path = os.path.join(output_folder, "sphericity_index_vs_LVDVI.png")
    plot_sphericity_vs_lvdvi(data, plot_path)
    print(f"Plot saved to: {plot_path}")

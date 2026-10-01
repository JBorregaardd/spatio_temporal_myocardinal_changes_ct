#################################################################################################################################################
# This script takes as input the raw and processed thickness data for each segment across all patients, and creates two boxplots for the
# 6 first segments (basal) for the raw and processed data. The boxplots are saved in the output folder.
# Raw is the pipeline run without aortic exclusion (<dataset>_0mm), processed the run with it (<dataset>_2.5mm); both need
# statistics/statistics_thickness_per_segment.csv from statistics_analysis.py. Only patients present in both runs are compared.
# e.g. uv run LVM_thickness/analysis/compareRawAndProcessed.py --dataset ImageCAS_1-200
#################################################################################################################################################

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import AORTA_EXCLUSION_MM, run_output_dir  # noqa: E402

BASAL_SEGMENTS = {
    1: "Basal anterior",
    2: "Basal anteroseptal",
    3: "Basal inferoseptal",
    4: "Basal inferior",
    5: "Basal inferolateral",
    6: "Basal anterolateral",
}
STATISTICS_CSV = os.path.join("statistics", "statistics_thickness_per_segment.csv")


def load_basal_thickness(run_dir: str) -> pd.DataFrame:
    """Per-patient median thickness of the basal segments of one run.

    Args:
        run_dir: Run folder containing ``statistics/statistics_thickness_per_segment.csv``.

    Returns:
        Rows of the statistics CSV for segments 1-6.
    """
    path = os.path.join(run_dir, STATISTICS_CSV)
    if not os.path.isfile(path):
        raise SystemExit(f"{path} not found; run statistics_analysis.py for this run first.")
    data = pd.read_csv(path, dtype={"patient_id": str})
    return data[data["segment"].isin(BASAL_SEGMENTS)]


def plot_raw_vs_processed(raw: pd.DataFrame, processed: pd.DataFrame, raw_title: str, processed_title: str,
                          output_path: str) -> None:
    """Two side-by-side boxplots of the basal median thickness, raw and processed, on a shared y-axis.

    Args:
        raw: Basal rows of the raw run.
        processed: Basal rows of the processed run.
        raw_title: Title of the raw panel.
        processed_title: Title of the processed panel.
        output_path: Where the figure is saved.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

    for ax, data, title in [(axes[0], raw, raw_title), (axes[1], processed, processed_title)]:
        values = [data.loc[data["segment"] == segment, "median_thickness"].dropna() for segment in BASAL_SEGMENTS]
        ax.boxplot(values)
        ax.set_xticks(range(1, len(BASAL_SEGMENTS) + 1))
        labels = [f"{segment}\n" + name.replace(" ", "\n") for segment, name in BASAL_SEGMENTS.items()]
        ax.set_xticklabels(labels, fontsize=13)
        ax.set_title(title, fontsize=18)
        ax.set_xlabel("Segment", fontsize=16)
        ax.tick_params(axis="y", labelsize=14)

    axes[0].set_ylabel("Median thickness (mm)", fontsize=16)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compare basal-segment thickness with and without aortic exclusion.")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name of the pipeline runs to compare, as given to main_pipeline.py.")
    parser.add_argument("--raw-mm", type=float, default=0.0,
                        help="Aortic exclusion distance of the raw run (default: 0).")
    parser.add_argument("--processed-mm", type=float, default=AORTA_EXCLUSION_MM,
                        help=f"Aortic exclusion distance of the processed run (default: {AORTA_EXCLUSION_MM}).")
    parser.add_argument("--output-dir", default=None,
                        help="Parent of the run folders (default: <PROJECT_ROOT>/output).")
    return parser.parse_args()


def main():

    args = parse_args()

    # Find project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    load_dotenv(os.path.join(project_root, ".env"))
    root = os.environ["PROJECT_ROOT"]

    raw_dir = run_output_dir(root, args.dataset, args.raw_mm, args.output_dir)
    processed_dir = run_output_dir(root, args.dataset, args.processed_mm, args.output_dir)
    raw = load_basal_thickness(raw_dir)
    processed = load_basal_thickness(processed_dir)

    # Compare the same patients in both runs
    shared = set(raw["patient_id"]) & set(processed["patient_id"])
    if not shared:
        raise SystemExit(f"No patients in both {raw_dir} and {processed_dir}.")
    only_one = (set(raw["patient_id"]) | set(processed["patient_id"])) - shared
    if only_one:
        print(f"[WARN] {len(only_one)} patient(s) are only in one of the runs and are left out.")
    raw = raw[raw["patient_id"].isin(shared)]
    processed = processed[processed["patient_id"].isin(shared)]
    print(f"Number of patients: {len(shared)}")

    # Print summary
    summary = pd.DataFrame({
        "segment": list(BASAL_SEGMENTS),
        "name": list(BASAL_SEGMENTS.values()),
        "raw_median": raw.groupby("segment")["median_thickness"].median().reindex(list(BASAL_SEGMENTS)).values,
        "processed_median": processed.groupby("segment")["median_thickness"].median()
        .reindex(list(BASAL_SEGMENTS)).values,
    })
    summary["difference"] = summary["processed_median"] - summary["raw_median"]
    print("\nMedian over patients of the median thickness (mm):")
    print(summary.round(2).to_string(index=False))

    # Create plot
    raw_name = os.path.basename(raw_dir)
    processed_name = os.path.basename(processed_dir)
    output_folder = os.path.dirname(processed_dir)
    os.makedirs(output_folder, exist_ok=True)
    plot_path = os.path.join(output_folder, f"basal_thickness_{raw_name}_vs_{processed_name}.png")

    plot_raw_vs_processed(
        raw,
        processed,
        f"Raw ({args.raw_mm:g} mm aortic exclusion)",
        f"Processed ({args.processed_mm:g} mm aortic exclusion)",
        plot_path,
    )

    summary_path = plot_path.replace(".png", ".csv")
    summary.to_csv(summary_path, index=False)

    print(f"\nPlot saved to: {plot_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

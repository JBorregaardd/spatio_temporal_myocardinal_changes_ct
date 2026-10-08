#################################################################################################################################################
# This script performs a PCA analysis of the 1-17 segments median thickness and 1-17 volume for each patient.
# It is a 17x2 vector for each patient, which is then reduced.
# It takes as input a CSV file with the median thickness and volume data for each segment.
# It reads <output>/<dataset>_<aorta-exclusion-mm>/statistics/ (written by statistics_analysis.py)
# output is a PCA plot and a CSV file with the PCA results.
# e.g. uv run pca/pca_analyse.py --dataset ImageCAS_1-200 --aorta-exclusion-mm 2.5
#################################################################################################################################################

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv


sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "LVM_thickness"))

from pipeline import AORTA_EXCLUSION_MM, run_output_dir  # noqa: E402

def pca_analysis(data, output_folder):
    """
    Perform PCA analysis on the median thickness and volume data for each segment.
    """

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Select the columns for PCA
    features = [f"segment_{i}_thickness" for i in range(1, 18)] + [f"segment_{i}_volume" for i in range(1, 18)]
    x = data[features].values

    # Standardize the features
    x = StandardScaler().fit_transform(x)

    # Perform PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(x)

    # Create a DataFrame with the PCA results
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['patient_id'] = data['patient_id']

    # Save the PCA results to a CSV file
    pca_df.to_csv(os.path.join(output_folder, "pca_results.csv"), index=False)

    # Plot the PCA results
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_df['PC1'], pca_df['PC2'])
    plt.title('PCA of Median Thickness and Volume per Segment')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid()
    plt.savefig(os.path.join(output_folder, "pca_plot.png"))


def main():

    parser = argparse.ArgumentParser(description="PCA analysis of median thickness and volume per segment.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--aorta-exclusion-mm", type=float, default=AORTA_EXCLUSION_MM, help="Aorta exclusion in mm")
    args = parser.parse_args()

    # Find project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(project_root, ".env"))
    root = os.environ["PROJECT_ROOT"]

    statistics_folder = os.path.join(
        run_output_dir(root, args.dataset, args.aorta_exclusion_mm, "output"), "statistics"
    )
    path = os.path.join(statistics_folder, "statistics_thickness_per_segment.csv")
    if not os.path.isfile(path):
        raise SystemExit(f"{path} not found; run statistics_analysis.py with the same --dataset first.")

    data = pd.read_csv(path)
    print(f"Number of patients: {data['patient_id'].nunique()}")
    output_folder = os.path.join("output")
    os.makedirs(output_folder, exist_ok=True)

    pca_analysis(data, output_folder)


if __name__ == "__main__":
    main()

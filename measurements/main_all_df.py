#################################################################################################################################################
# This script generates a dataframe for all the patients in the dataset.
# The rows is the median Volume (mL), median Mass (g) and median Sphericity Index of the whole LV myocardium.
# The CSV is saved in <df>/df_myocardium.
# e.g. uv run measurements/main_all_df.py
#################################################################################################################################################

import os
import sys
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "measurements"))

from sphericity_index import sphericity_index  # noqa: E402
from volume import myocardium_volume  # noqa: E402

SPHERICITY = "si_vol" # si_vol or si_dl_mid # Choose which sphericity index to plot against volume
MYOCARDIAL_DENSITY_G_PER_ML = 1.055  # Fuchs et al. 2016, EHJ-CVI 17:1009
WORKERS = max(1, (os.cpu_count() or 2) - 1)
SPHERICITY_LABELS = {
    "si_vol": "Volumetric sphericity index",
    "si_dl_mid": "Sphericity index D_mid / L",
}


def measure_patient(segmentation_path: str) -> dict | None:
    """Volume, mass and sphericity of the LV myocardium of one patient.

    Args:
        segmentation_path: TotalSegmentator heart-chambers segmentation ``<patient_id>.heart.nii.gz``.

    Returns:
        One row with ``patient_id`` and the outputs of ``myocardium_volume``, MYOCARDIAL_DENSITY_G_PER_ML *``myocardium_volume`` and ``sphericity_index``,
        or ``None`` if the patient could not be measured.
    """
    patient_id = os.path.basename(segmentation_path).removesuffix(".heart.nii.gz")
    try:
        row = {"patient_id": patient_id, 
        "lvm_volume_ml": myocardium_volume(segmentation_path)["lvm_volume_ml"],
        "lvm_mass_g": MYOCARDIAL_DENSITY_G_PER_ML * myocardium_volume(segmentation_path)["lvm_volume_ml"],
        "si_vol": sphericity_index(segmentation_path)["si_vol"],
        "si_dl_mid": sphericity_index(segmentation_path)["si_dl_mid"]
        }
        print(f"Patient {patient_id} done")
        return row
    except Exception as error:
        print(f"Patient {patient_id} failed: {error}")
        return None


def generate_df(data: pd.DataFrame, sphericity: str, output_path: str) -> None:
    """Generate a dataframe with the measurements for all patients.

    Args:
        data: One row per patient with volume, mass and sphericity measurements.
        output_path: Where to save the dataframe.
    """

    # compute median values for volume, mass and sphericity
    median_volume = data["lvm_volume_ml"].median()
    median_mass = data["lvm_mass_g"].median()
    median_sphericity = data[sphericity].median()

    # save the dataframe to a CSV file
    data.to_csv(output_path, index=False)
    print(f"Measurements saved to: {output_path}")    



if __name__ == "__main__":
    load_dotenv(os.path.join(REPO_ROOT, ".env"))
    root = os.environ["PROJECT_ROOT"]
    segmentation_folder = os.environ.get("SEGMENTATION_FOLDER", os.path.join(root, "data", "TotalSegmentator"))

    patient_ids = sorted(
        (name.removesuffix(".heart.nii.gz") for name in os.listdir(segmentation_folder) if name.endswith(".heart.nii.gz")),
        key=lambda pid: int(pid) if pid.isdigit() else float("inf"),
    )

    segmentation_paths = [os.path.join(segmentation_folder, f"{pid}.heart.nii.gz") for pid in patient_ids]
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        rows = [row for row in executor.map(measure_patient, segmentation_paths) if row is not None]

    data = pd.DataFrame(rows)
    output_folder = os.path.join(root, "df")
    os.makedirs(output_folder, exist_ok=True)

    csv_path = os.path.join(output_folder, "df_myocardium.csv")
    data.to_csv(csv_path, index=False)
    print(f"Measurements saved to: {csv_path}")


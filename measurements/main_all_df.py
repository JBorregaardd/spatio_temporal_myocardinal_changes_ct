#################################################################################################################################################
# This script measures the LV myocardium for all patients in the dataset and makes a 4x1 table with the medians:
# median Volume (mL), median Mass (g), median volumetric Sphericity Index and median Sphericity Index D_mid / L.
# The median table is saved in <df>/df_myocardium_median.csv.
# e.g. uv run measurements/main_all_df.py
#################################################################################################################################################

import os
import sys
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "measurements"))

from sphericity_index import sphericity_index 
from volume import myocardium_volume  

MYOCARDIAL_DENSITY_G_PER_ML = 1.055  # Fuchs et al. 2016
WORKERS = max(1, (os.cpu_count() or 2) - 1)
ROW_LABELS = {
    "lvm_volume_ml": "Median volume (mL)",
    "lvm_mass_g": "Median mass (g)",
    "si_vol": "Median volumetric sphericity index",
    "si_dl_mid": "Median sphericity index D_mid / L",
}


def measure_patient(segmentation_path: str) -> dict | None:
    """Volume, mass and sphericity of the LV myocardium of one patient.

    Args:
        segmentation_path: TotalSegmentator heart-chambers segmentation ``<patient_id>.heart.nii.gz``.

    Returns:
        One row with ``patient_id``, volume, mass and the two sphericity indices,
        or ``None`` if the patient could not be measured.
    """
    patient_id = os.path.basename(segmentation_path).removesuffix(".heart.nii.gz")
    try:
        volume_ml = myocardium_volume(segmentation_path)["lvm_volume_ml"]
        sphericity = sphericity_index(segmentation_path)
        row = {
            "patient_id": patient_id,
            "lvm_volume_ml": volume_ml,
            "lvm_mass_g": MYOCARDIAL_DENSITY_G_PER_ML * volume_ml,
            "si_vol": sphericity["si_vol"],
            "si_dl_mid": sphericity["si_dl_mid"],
        }
        print(f"Patient {patient_id} done")
        return row
    except Exception as error:
        print(f"Patient {patient_id} failed: {error}")
        return None


def median_table(data: pd.DataFrame) -> pd.DataFrame:
    """Make a 4x1 table with the median of volume, mass and the two sphericity indices over all patients.

    Args:
        data: One row per patient with the columns in ``ROW_LABELS``.

    Returns:
        A table with one row per measurement and a single ``Median`` column.
    """
    medians = data[list(ROW_LABELS)].median()
    return medians.rename(index=ROW_LABELS).to_frame(name="Median")


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

    table = median_table(data)
    print(table.round(3))
    table_path = os.path.join(output_folder, "df_myocardium_median.csv")
    table.to_csv(table_path)
    print(f"Median table saved to: {table_path}")

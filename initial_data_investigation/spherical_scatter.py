#################################################################################################################################################
# This script computes, per patient, the sphericity index of the left ventricle (LV) cavity and the volume of the whole LV myocardium,
# and plots them against each other. It only reads the TotalSegmentator heart segmentations (<id>.heart.nii.gz).
# The CSV and plot are saved in <output>/sphericity.
# e.g. uv run initial_data_investigation/spherical_scatter.py
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
WORKERS = max(1, (os.cpu_count() or 2) - 1)
SPHERICITY_LABELS = {
    "si_vol": "Volumetric sphericity index",
    "si_dl_mid": "Sphericity index D_mid / L",
}


def measure_patient(segmentation_path: str) -> dict | None:
    """Sphericity of the LV cavity and volume of the whole LV myocardium of one patient.

    Args:
        segmentation_path: TotalSegmentator heart-chambers segmentation ``<patient_id>.heart.nii.gz``.

    Returns:
        One row with ``patient_id`` and the outputs of ``sphericity_index`` and ``myocardium_volume``,
        or ``None`` if the patient could not be measured.
    """
    patient_id = os.path.basename(segmentation_path).removesuffix(".heart.nii.gz")
    try:
        row = {"patient_id": patient_id, **sphericity_index(segmentation_path), **myocardium_volume(segmentation_path)}
        print(f"Patient {patient_id} done")
        return row
    except Exception as error:
        print(f"Patient {patient_id} failed: {error}")
        return None


def plot_volume_vs_sphericity(data: pd.DataFrame, sphericity: str, output_path: str) -> None:
    """Scatter plot of LV myocardial volume against LV sphericity, one dot per patient.

    Args:
        data: One row per patient with ``lvm_volume_ml`` and the ``sphericity`` column.
        sphericity: Column of ``data`` to put on the y-axis.
        output_path: Where to save the figure.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(data["lvm_volume_ml"], data[sphericity], color="tab:blue", alpha=0.7)

    ax.set_xlabel("LV myocardial volume (mL)", fontsize=16)
    ax.set_ylabel(SPHERICITY_LABELS[sphericity], fontsize=16)
    ax.set_title(f"Myocardial volume vs. LV sphericity (n = {len(data)})", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


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
    output_folder = os.path.join(root, "output", "sphericity")
    os.makedirs(output_folder, exist_ok=True)

    csv_path = os.path.join(output_folder, "sphericity_and_volume.csv")
    data.to_csv(csv_path, index=False)
    print(f"Measurements saved to: {csv_path}")

    plot_path = os.path.join(output_folder, f"volume_vs_{SPHERICITY}.png")
    plot_volume_vs_sphericity(data, SPHERICITY, plot_path)
    print(f"Plot saved to: {plot_path}")

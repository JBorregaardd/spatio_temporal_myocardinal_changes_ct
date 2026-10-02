#################################################################################################################################################
# This script takes as input the raw TotalSegmentator heart segmentations (<id>.heart.nii.gz) and computes, per patient, the sphericity
# index of the left ventricle (LV) cavity and the volume of the whole LV myocardium, and plots them against each other.
# e.g. uv run initial_data_investigation/spherical_scatter.py
#################################################################################################################################################

import os
import sys
import argparse
import traceback
from concurrent.futures import as_completed

import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "measurements"))

from sphericity_index import sphericity_index
from volume import myocardium_volume

SEGMENTATION_SUFFIX = ".heart.nii.gz"
SPHERICITY_LABELS = {
    "si_dl": "Sphericity index D_max / L",
    "si_dl_mid": "Sphericity index D_mid / L",
    "si_vol": "Volumetric sphericity index",
}


def find_patients(segmentation_folder: str) -> list[str]:
    """Patients with a TotalSegmentator heart segmentation.

    Args:
        segmentation_folder: Folder containing ``<patient_id>.heart.nii.gz``.

    Returns:
        Patient ids, numeric ids in numeric order.
    """
    ids = [name.removesuffix(SEGMENTATION_SUFFIX) for name in os.listdir(segmentation_folder)
           if name.endswith(SEGMENTATION_SUFFIX)]
    return sorted(ids, key=lambda pid: (0, int(pid), "") if pid.isdigit() else (1, 0, pid))


def measure_patient(patient_id: str, segmentation_folder: str) -> dict:
    """Sphericity of the LV cavity and volume of the whole LV myocardium of one patient.

    Args:
        patient_id: Dataset id.
        segmentation_folder: Folder containing ``<patient_id>.heart.nii.gz``.

    Returns:
        One row with ``patient_id`` and the outputs of ``sphericity_index`` and ``myocardium_volume``.
    """
    segmentation_path = os.path.join(segmentation_folder, f"{patient_id}{SEGMENTATION_SUFFIX}")
    return {
        "patient_id": patient_id,
        **sphericity_index(segmentation_path),
        **myocardium_volume(segmentation_path),
    }


def _measure_patient_safely(patient_id: str, segmentation_folder: str) -> tuple[str, dict | None, str]:
    try:
        return patient_id, measure_patient(patient_id, segmentation_folder), ""
    except Exception:
        return patient_id, None, traceback.format_exc()


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


def main():


    load_dotenv(os.path.join(REPO_ROOT, ".env"))
    root = os.environ["PROJECT_ROOT"]
    segmentation_folder = 'data/TotalSegmentator'
    patient_ids = find_patients(segmentation_folder)

    if not patient_ids:
        raise SystemExit(f"No *{SEGMENTATION_SUFFIX} files found in {segmentation_folder}.")

    rows = []
    futures = [(_measure_patient_safely, pid, segmentation_folder) for pid in patient_ids]
    for i, future in enumerate(futures, start=1):
        patient_id, row, error = future.result()
        if row is None:
            print(f"[{i}/{len(patient_ids)}] Patient {patient_id} failed:\n{error}")
        else:
            rows.append(row)
            print(f"[{i}/{len(patient_ids)}] Patient {patient_id} done")

    if not rows:
        raise SystemExit("No patients could be measured.")

    rows.sort(key=lambda row: patient_ids.index(row["patient_id"]))
    data = pd.DataFrame(rows)

    output_folder = os.path.join(args.output_dir or os.path.join(root, "output"), "sphericity")
    os.makedirs(output_folder, exist_ok=True)

    csv_path = os.path.join(output_folder, "sphericity_and_volume.csv")
    data.to_csv(csv_path, index=False)
    print(f"Measurements saved to: {csv_path}")

    plot_path = os.path.join(output_folder, f"volume_vs_{args.sphericity}.png")
    plot_volume_vs_sphericity(data, args.sphericity, plot_path)
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()

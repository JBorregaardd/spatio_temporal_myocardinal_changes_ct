import os
import json
import multiprocessing
import logging
import traceback
from functools import partial
from tqdm import tqdm

from lv_generate_segments import wrap_lv_segments
from thickness_estimation import (
    create_meshes_from_segmentation, 
    mesh_vector_allignment, 
    thickness_in_17_seg, 
    plot_thickness_histograms
)
from create_medial_sheet import create_medial_sheet
from utils import utils, path_handler, bullseye

DATASET_NAME = "imageCAS"
SCAN_TYPE = "ED"


def main(root, folder, patient_id, exists_ok=False):
    logging.info(f"Processing patient_id: {patient_id}")

    current_save_dir = os.path.join(root, "output", patient_id)
    thickness_path = os.path.join(current_save_dir, "thickness.json")
    print(f"\n[START] Saving results to: {current_save_dir}")

    if os.path.exists(thickness_path) and exists_ok:
        print("[SKIP] thickness.json already exists, exiting.")
        return

    # Paths to the image and segmentation
    image_path = os.path.join(root, "data", "1_200", f"{patient_id}.img.nii.gz")
    segmentation_path = os.path.join(folder, f"{patient_id}.heart.nii.gz")

    os.makedirs(current_save_dir, exist_ok=True)

    try:
        # 1. Create meshes from TS segmentation 
        print("[STEP 1] Creating meshes from segmentation...")
        mesh_inner, mesh_outer = create_meshes_from_segmentation(segmentation_path, current_save_dir)
        print("  -> Step 1 completed.")

        # 2. Get smooth vector correspondence and measure lengths
        print("[STEP 2] Vector alignment...")
        mesh_thick_path = os.path.join(current_save_dir, "surfaces", "dist_source.vtk")
        if os.path.exists(mesh_thick_path):
            mesh_thick = utils.read_vtk_mesh(mesh_thick_path)
        else:
            mesh_thick, _, _ = mesh_vector_allignment(
                current_save_dir, mesh_inner, mesh_outer, num_iterations=3, num_closest_vectors=20, exists_ok=True
            )
        print("  -> Step 2 completed.")

        # 3. Map points to 17 segment model
        print("[STEP 3] Preparing 17-segment model...")
        atlas_path = os.path.join(current_save_dir, "misc", "atlas_points.txt")
        if not os.path.exists(atlas_path):  
            wrap_lv_segments(current_save_dir, segmentation_path, image_path, individual_transforms=True, verbose=True)
        print("  -> Step 3 completed.")

        # 4. Measure average thickness in each segment - save to dict
        print("[STEP 4] Calculating thickness per segment...")
        mesh_path = os.path.join(current_save_dir, "surfaces", "myocardium_17.vtk")
        mesh_17 = utils.read_vtk_mesh(mesh_path)
        thickness = thickness_in_17_seg(mesh_thick, mesh_17)
        print("  -> Step 4 completed.")

        # 5. Save thickness dict to file
        print("[STEP 5] Saving thickness.json and histograms...")
        with open(thickness_path, 'w') as f:
            json.dump(dict(thickness), f)

        plot_thickness_histograms(current_save_dir, name=patient_id)
        print("  -> Step 5 completed.")

        # 6. Medial sheet
        print("[STEP 6] Creating medial sheet...")
        create_medial_sheet(current_save_dir)
        print("  -> Step 6 completed.")

        # 7. Bullseye plot
        print("[STEP 7] Generating bullseye plot...")
        clean_patient_name = patient_id.split('_')[-1]
        bullseye.create_single_bs_from_mesh(
            current_save_dir, 
            mesh_thick_path, 
            segmentation_path, 
            global_min=0, 
            global_max=20, 
            savename=f"thickness_{clean_patient_name}_{SCAN_TYPE}"
        )
        print("  -> Step 7 completed.")

    except Exception:
        print("\n[ERROR DURING EXECUTION]")
        traceback.print_exc()


if __name__ == "__main__":

    root = r"C:\Users\Jacob pc\vscode_projects\spatio_temporal_myocardinal_changes_ct"
    folder = os.path.join(root, "data", "TotalSegmentator")

    test_patient_id = "10"



    expected_img = os.path.join(root, "data", "1_200", f"{test_patient_id}.img.nii.gz")
    expected_seg = os.path.join(folder, f"{test_patient_id}.heart.nii.gz")

    print(f"Checking files for patient {test_patient_id}:")
    print(f"  Image:        {expected_img} -> Exists: {os.path.exists(expected_img)}")
    print(f"  Segmentation: {expected_seg} -> Exists: {os.path.exists(expected_seg)}")

    if not os.path.exists(expected_img) or not os.path.exists(expected_seg):
        print("\n[ERROR] One or both input files were not found. Please check the paths above before proceeding.")
    else:
        print("\nFiles found! Running pipeline on single image...")
        main(root, folder, test_patient_id, exists_ok=False)
        print("Done!")
        
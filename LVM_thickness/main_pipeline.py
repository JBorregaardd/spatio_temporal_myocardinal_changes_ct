import os
import json
import multiprocessing
import logging
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

DATASET_NAME = "CGPS" #"NSTEMI"
SCAN_TYPE = "ED" #"ES" "ED" or "CFA"

# Configure logger
LOG_FILE = f"logs/processing_errors_{DATASET_NAME}_{SCAN_TYPE}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Logs to console
        logging.FileHandler(LOG_FILE)  # Logs errors to a file
    ]
)

def main(root, folder, patient_id, exists_ok=True):
    logging.info(f"Processing patient_id: {patient_id}")

    if SCAN_TYPE == "CFA":
        current_save_dir = os.path.join(folder, f"{DATASET_NAME}_{SCAN_TYPE}", patient_id.split("_SERIES")[0], patient_id)
    else:
        # current_save_dir = os.path.join(folder, f"{DATASET_NAME}_{SCAN_TYPE}_thickness", patient_id)
        current_save_dir = os.path.join("/storage", f"LVM_thickness", patient_id)
    thickness_path = os.path.join(current_save_dir, "thickness.json")

    if os.path.exists(thickness_path) and exists_ok:
        logging.info(f"Skipping {patient_id}: Thickness file already exists.")
        return

    try:
        image_path = os.path.join(root, "NIFTI", f"{patient_id}.nii.gz")
        segmentation_path = os.path.join(
            folder, "TotalSegmentator", DATASET_NAME, patient_id, "segmentations",
            "heartchambers_highres", "heartchambers_highres.nii.gz"
        )
        # segmentation_path = os.path.join(
        #     folder, patient_id, "segmentations",
        #     "total_seg", "total_seg.nii.gz"
        # )
        os.makedirs(current_save_dir, exist_ok=True)

        # 1. Create meshes from TS segmentation 
        mesh_inner, mesh_outer = create_meshes_from_segmentation(segmentation_path, current_save_dir)

        # 2. Get smooth vector correspondence and measure lengths
        mesh_thick_path = os.path.join(current_save_dir, "surfaces", "dist_source.vtk")
        if os.path.exists(mesh_thick_path):
            mesh_thick = utils.read_vtk_mesh(mesh_thick_path)
        else:
            mesh_thick, _, _ = mesh_vector_allignment(
                current_save_dir, mesh_inner, mesh_outer, num_iterations=3, num_closest_vectors=20, exists_ok=True
            )

        # 3. Map points to 17 segment model
        atlas_path = os.path.join(current_save_dir, "misc", "atlas_points.txt")
        if not (os.path.exists(atlas_path)):  
            wrap_lv_segments(current_save_dir, segmentation_path, image_path, individual_transforms=True, verbose=False)

        # 4. Measure average thickness in each segment - save to dict
        mesh_path = os.path.join(current_save_dir, "surfaces", "myocardium_17.vtk")
        mesh_17 = utils.read_vtk_mesh(mesh_path)
        thickness = thickness_in_17_seg(mesh_thick, mesh_17)

        # 5. Save thickness dict to file
        with open(thickness_path, 'w') as f:
            json.dump(dict(thickness), f)

        plot_thickness_histograms(current_save_dir, name=patient_id)

        # 6. Create medial sheet
        create_medial_sheet(current_save_dir)

        #TODO: thickness in BS
        bullseye.create_single_bs_from_mesh(current_save_dir, mesh_thick_path, segmentation_path, global_min=0, global_max=20, savename=f"thickness_{patient_id.split('_')[1]}_{SCAN_TYPE}.png")

    except Exception as e:
        logging.exception(f"Error processing {patient_id}")
        

if __name__=="__main__":

    
    root = r"C:\Users\Jacob pc\vscode_projects\spatio_temporal_myocardinal_changes_ct"       # folder containing 'data'
    folder = r"data\TotalSegmentator"   # folder containing '1.heart.nii.gz'

    # 2. Pick the single image ID you want to test
    test_patient_id = "1"

    # 3. Verify paths before running to avoid silent failures
    # Adjust this path to match how image_path is defined in your main()
    expected_img = os.path.join(root, "data", "1-200", "NIFTI", f"{test_patient_id}.img.nii.gz")
    expected_seg = os.path.join(root, folder, f"{test_patient_id}.heart.nii.gz")

    print(f"Checking files for patient {test_patient_id}:")
    print(f"  Image:        {expected_img} -> Exists: {os.path.exists(expected_img)}")
    print(f"  Segmentation: {expected_seg} -> Exists: {os.path.exists(expected_seg)}")

    if not os.path.exists(expected_img) or not os.path.exists(expected_seg):
        print("\n[ERROR] One or both input files were not found. Please check the paths above before proceeding.")
    else:
        print("\nFiles found! Running pipeline on single image...")
        # exists_ok=False forces it to recalculate everything from scratch
        main(root, folder, test_patient_id, exists_ok=False)
        print("Done!")
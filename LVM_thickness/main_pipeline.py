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

    logging.info("Updated version 6")

    # RH, root, folder = path_handler.get_path_and_root()
    # main(root, folder)

    RH, root, folder = path_handler.get_path_and_root(DATASET_NAME)

    file = "/storage/DTU-CGPS-1/DTU-CGPS-1_meta_data_success_2025-01-05-19-40-53.csv"
    

    #patient_id = "CGPS-1_0001_SERIES0005"
    #patient_id = "CGPS-1_0002_SERIES0000"
    if SCAN_TYPE == "CFA":
        import pandas as pd
        splits = utils.read_excel_for_CFA(root, True)
        splits = pd.concat(splits, ignore_index=True)
    else:
        splits = utils.read_excel_for_best_phase(root, True, get_ES=SCAN_TYPE=="ES", file=file)
    logging.info(f"number of patients found {len(splits)}")
    #logging.info(splits.head())

    splits["filename_strip"] = splits["filename"].str.replace(".nii.gz", "", regex=False)
    patient_id_list = splits["filename_strip"].tolist()

    # patient_id = patient_id_list[0]
    # main(root, folder, patient_id)
    # exit(0)

    with multiprocessing.Pool(6) as pool: 
        process_data_with_common_arg  = partial(main, root, folder)

        for _ in tqdm(pool.imap(process_data_with_common_arg, patient_id_list), total=len(patient_id_list)):
            pass
    
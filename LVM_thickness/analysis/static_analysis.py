import os
import csv
import argparse
import sys
from pathlib import Path
import re
import numpy as np
import pandas as pd

import SimpleITK as sitk
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from skimage import morphology, measure


LVM_ID = 1
LV_ID = 3

def read_vtk_mesh(path, get_normals=False):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    output = reader.GetOutput()
    assert isinstance(output, vtk.vtkPolyData), f"Output is not a vtkPolyData object: {path}"
    assert output.GetNumberOfPoints() > 0, f"No points found in mesh: {path}"

    if get_normals and not output.GetPointData().GetNormals():
        # vtkTriangleMeshPointNormals 
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(output)
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOff()
        normals.SplittingOff()
        normals.Update()
        output = normals.GetOutput()

    return output


def get_scalar_ring_mm_coordinates(path, total_path, mesh_name, exists_ok=True, lvm=True):
    label_total = sitk.ReadImage(total_path)
    label_lv = sitk.Or(label_total == LV_ID, label_total == LVM_ID) if lvm else label_total == LV_ID

    im_bin = sitk.GetArrayFromImage(label_lv).transpose(2, 1, 0)
    im_ring = im_bin & ~morphology.erosion(im_bin)
    im_ring = im_ring.astype(np.uint8)

    mesh = read_vtk_mesh(mesh_name)

    locator = vtk.vtkPointLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()
    radius = 0.5 # mm

    ring_nnz = im_ring.nonzero()
    scalar_ring = np.zeros_like(im_ring, dtype=np.float64)
    for x, y, z in zip(*ring_nnz):
        indices = np.array([x, y, z], dtype=np.float64)
        point = label_total.TransformContinuousIndexToPhysicalPoint(indices)
        ids = vtk.vtkIdList()
        locator.FindPointsWithinRadius(radius, point, ids)
        if ids.GetNumberOfIds() == 0:
            idx = locator.FindClosestPoint(point)
            scalar_ring[x, y, z] = mesh.GetPointData().GetScalars().GetTuple1(idx)
        else:
            scalar_ring[x, y, z] = np.mean([mesh.GetPointData().GetScalars().GetTuple1(ids.GetId(i)) for i in range(ids.GetNumberOfIds())])
            # print(ids.GetNumberOfIds())
        scalar_ring[x, y, z] += 1e-10 # to avoid 0 values
        
    scalar_image = sitk.GetImageFromArray(scalar_ring.transpose(2, 1, 0))
    scalar_image.CopyInformation(label_total)
    sitk.WriteImage(scalar_image, os.path.join(path, "segmentations", f"mm_{os.path.basename(mesh_name).split('.')[0]}_ring.nii.gz"))
    return scalar_image



def calculate_mean_thickness_per_segment(folder, mesh_name, total_path):

    # Get thickness values and the 17-segment labels
    label_total = sitk.ReadImage(total_path)

    # Read thickness mesh
    mesh = read_vtk_mesh(mesh_name)

    # Get thickness values on the myocardium ring
    scalar_image = get_scalar_ring_mm_coordinates(
        folder,
        total_path,
        mesh_name
    )

    # Read the transform
    transform_path = os.path.join(folder, "lv17_transform.txt")
    transform = sitk.ReadTransform(transform_path)

    # Read the 17-segment segmentation
    lv17_path = os.path.join(
        folder,
        "segmentations",
        "lv17",
        "lv17.nii.gz"
    )

    lv17_segmentation = sitk.ReadImage(lv17_path)

    # Transform the 17-segment segmentation
    lv17_transformed = sitk.Resample(
        lv17_segmentation,
        transform,
        sitk.sitkNearestNeighbor,
        0,
        lv17_segmentation.GetPixelID()
    )

    # Transform the thickness image
    scalar_seg = sitk.Resample(
        scalar_image,
        transform,
        sitk.sitkNearestNeighbor,
        0,
        scalar_image.GetPixelID()
    )

    # Convert to numpy
    segments = sitk.GetArrayFromImage(lv17_transformed).transpose(2, 1, 0)
    thickness = sitk.GetArrayFromImage(scalar_seg).transpose(2, 1, 0)

    # Calculate mean thickness for each segment
    results = []

    for segment in range(1, 18):

        values = thickness[segments == segment]

        # Remove zero values
        values = values[values > 0]

        if len(values) > 0:
            mean_thickness = np.mean(values)
        else:
            mean_thickness = np.nan

        results.append({
            "segment": segment,
            "mean_thickness": mean_thickness
        })

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the LVM thickness pipeline for a single patient.")
    parser.add_argument(
        "--patient-id",
        default=os.environ.get("PATIENT_ID", "1"),
        help="Patient id to process (default: $PATIENT_ID from .env, else '1'). "
             "Useful for cluster array jobs, e.g. --patient-id $SLURM_ARRAY_TASK_ID.",
    )
    args = parser.parse_args()
    test_patient_id = args.patient_id


    root = r"/Users/signeolsen/Desktop/spatio_temporal_myocardinal_changes_ct/"

    folder = os.path.join(root, "output", f"{test_patient_id}")
    mesh_name = os.path.join(folder, "surfaces", "dist_source.vtk")
    total_path = os.path.join(folder, "segmentations", "lv17", "lv17.nii.gz")

    table = calculate_mean_thickness_per_segment(folder, mesh_name, total_path)

    # Save the results to a CSV file
    output_csv_path = os.path.join(folder, "mean_thickness_per_segment.csv")

    with open(output_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["segment", "mean_thickness"])
        writer.writeheader()
        writer.writerows(table)

import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import numpy as np
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import json

from collections import defaultdict
from utils import utils



def create_meshes_from_segmentation(input_path, out_path, save_name="total_seg", save=True):
    #TODO: give both paths or fix code
    save_inner = os.path.join(out_path, "surfaces", "inner_mesh.vtk") if save else None
    save_outer = os.path.join(out_path, "surfaces", "outer_mesh.vtk") if save else None

    os.makedirs(os.path.join(out_path, "surfaces"), exist_ok=True)
    os.makedirs(os.path.join(out_path, "segmentations"), exist_ok=True)
    
    # if os.path.exists(save_inner) and os.path.exists(save_outer):
    #     mesh_inner = utils.read_vtk_mesh(save_inner)
    #     mesh_outer = utils.read_vtk_mesh(save_outer)
    #     return mesh_inner, mesh_outer
    #input_path = os.path.join(in_path, "segmentations", save_name, save_name+".nii.gz")
    image = sitk.ReadImage(input_path)
    im = sitk.GetArrayFromImage(image)

    #TODO: Make sure the segmentation ids are correct
    #lab_outer = im > 0
    #lab_inner = np.logical_or(im==1, im==2)
    # lab_inner = binary_dilation(lab_inner)
    lab_outer = np.logical_or(im==1, im==3)
    lab_inner = im == 3

    lab_inner = sitk.GetImageFromArray(lab_inner.astype(np.uint8))
    lab_outer = sitk.GetImageFromArray(lab_outer.astype(np.uint8))

    lab_inner.CopyInformation(image)
    lab_outer.CopyInformation(image)
    inner_path = os.path.join(out_path, "segmentations", "inner.nii.gz")
    outer_path = os.path.join(out_path, "segmentations", "outer.nii.gz")
    sitk.WriteImage(lab_inner, inner_path)
    sitk.WriteImage(lab_outer, outer_path)
   
    utils.convert_label_map_to_surface(inner_path, save_inner)
    utils.convert_label_map_to_surface(outer_path, save_outer)
    mesh_inner = utils.read_vtk_mesh(save_inner)
    mesh_outer = utils.read_vtk_mesh(save_outer)
    # mesh_inner = utils.segmentation_vtk_to_mesh_vtk(inner_path,  save_name=save_inner)
    # mesh_outer = utils.segmentation_vtk_to_mesh_vtk(outer_path,  save_name=save_outer)

    return mesh_inner, mesh_outer

def save_points_to_mesh(path, source_in, polydata_in, name=None, use_source=True):
    # make copy of input
    source = vtk.vtkPolyData()
    source.DeepCopy(source_in)
    polydata = vtk.vtkPolyData()
    polydata.DeepCopy(polydata_in)

    if use_source:
        # Get the cell array from the mesh
        cell_array = source.GetPolys()
        polydata.SetPolys(cell_array)
    else:
        # Do triangulation
        surf = vtk.vtkSurfaceReconstructionFilter()
        surf.SetInputData(polydata)
        surf.SetNeighborhoodSize(20)
        surf.SetSampleSpacing(0.1)
        surf.Update()

        polydata = surf.GetOutput()

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(polydata)
    writer.SetFileName(os.path.join(path,r"surfaces\mesh_{name}.vtk"))
    writer.Write()

def get_distance_mesh(path, mesh_inner, mesh_outer, save=True):

    distance_filter = vtk.vtkDistancePolyDataFilter()
    distance_filter.SetInputData(0, mesh_inner)
    distance_filter.SetInputData(1, mesh_outer)
    distance_filter.Update()

    # Get the distance values
    distances = distance_filter.GetOutput().GetPointData().GetScalars()

    # Access the distance values as a numpy array
    distances_array = vtk.util.numpy_support.vtk_to_numpy(distances)

    # Print the minimum, maximum, and mean distance values
    print("Minimum distance:", distances_array.min())
    print("Maximum distance:", distances_array.max())
    print("Mean distance:", distances_array.mean())

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(distance_filter.GetOutput())
    writer.SetFileName(os.path.join(path, "surfaces", "dist_mesh.vtk"))
    writer.Write()

    # Compute the normals of mesh_inner
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(mesh_inner)
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    normals.Update()
    normals_array = normals.GetOutput().GetPointData().GetNormals()

    # Compute the distance from each point in mesh_inner to the closest point on mesh_outer along the normal direction of mesh_inner
    distance_filter = vtk.vtkImplicitPolyDataDistance()
    distance_filter.SetInput(mesh_outer)
    distances = vtk.vtkFloatArray()
    distances.SetNumberOfComponents(1)
    distances.SetName("Distance")
    for i in range(mesh_inner.GetNumberOfPoints()):
        point = mesh_inner.GetPoint(i)
        normal = normals_array.GetTuple(i)
        normal = [normal[j] for j in range(3)]
        # normal = vtk.vtkMath.Normalize(normal)
        closest_point = [point[j] + normal[j] for j in range(3)]
        distance = distance_filter.EvaluateFunction(closest_point)
        distances.InsertNextValue(distance)

    # Add the distance values to mesh_inner
    mesh_inner.GetPointData().SetScalars(distances)


    # Save the mesh with the distance values
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(mesh_inner)
    writer.SetFileName(os.path.join(path, "surfaces", "dist_normals.vtk"))
    writer.Write()


def mesh_vector_allignment(path, mesh_source, mesh_target, num_iterations=5, num_closest_vectors=20, exists_ok=True):

    # Check if the mesh has already been smoothed
    if os.path.exists(os.path.join(path, "misc", "vectors.npz")) and exists_ok:
        loaded = np.load(os.path.join(path, "misc", "vectors.npz"))
        points = loaded["points"]
        vecs = loaded["vectors"]
        if len(points) == mesh_source.GetNumberOfPoints():
            vectors = [(i, points[i], vecs[i]) for i in range(len(points))]
            print("Loaded vectors from file")
            distance_scalars = vtk.vtkFloatArray()
            for i in range(mesh_source.GetNumberOfPoints()):
                distance = np.linalg.norm(vectors[i][2])
                distance_scalars.InsertNextValue(distance)
                
            mesh_source.GetPointData().SetScalars(distance_scalars)
            return vectors

    # Find closest point per vertex in target mesh
    locator_target = vtk.vtkPointLocator()
    locator_target.SetDataSet(mesh_target)
    locator_target.BuildLocator()
    closest_points = vtk.vtkIdList()
    closest_points.SetNumberOfIds(mesh_source.GetNumberOfPoints())
    for i in range(mesh_source.GetNumberOfPoints()):
        point = mesh_source.GetPoint(i)
        closest_point_id = locator_target.FindClosestPoint(point)
        closest_points.SetId(i, closest_point_id)

    # Find the closest point from source to the smoothed endpoints of the vectors
    distance_filter = vtk.vtkImplicitPolyDataDistance()
    distance_filter.SetInput(mesh_source)
    distances = vtk.vtkFloatArray()
    distances.SetNumberOfComponents(1)
    distances.SetName("Distance")

    # Construct a point locator for the smoothed endpoints of the vectors
    mesh_vectors = vtk.vtkPolyData()
    end_points = vtk.vtkPoints()
    vectors = []
    for i in range(mesh_source.GetNumberOfPoints()):
        point1 = mesh_source.GetPoint(i)
        point2 = mesh_target.GetPoint(closest_points.GetId(i))
        vector = np.array(point2) - np.array(point1)
        vectors.append((i, point1, vector))
        end_points.InsertNextPoint(point2)
    mesh_vectors.SetPoints(end_points)

    locator_source = vtk.vtkPointLocator()
    locator_source.SetDataSet(mesh_source)
    locator_source.BuildLocator()
    # Repeat smoothing
    for iteration in range(num_iterations):
        # Construct a point locator for the smoothed endpoints of the vectors
        mesh_vectors = vtk.vtkPolyData()
        end_points = vtk.vtkPoints()
        for i in range(mesh_source.GetNumberOfPoints()):
            end_points.InsertNextPoint(vectors[i][1] + vectors[i][2])
        mesh_vectors.SetPoints(end_points)

        for i in range(mesh_source.GetNumberOfPoints()):
            if np.linalg.norm(vectors[i][2]) < 1e-6:
                continue
            end_point = vectors[i][1] + vectors[i][2]
            closest_vector_ids = vtk.vtkIdList()
            locator_source.FindClosestNPoints(num_closest_vectors, vectors[i][1], closest_vector_ids)
            closest_vectors = [vectors[closest_vector_ids.GetId(j)] for j in range(closest_vector_ids.GetNumberOfIds())
                                if closest_vector_ids.GetId(j) != i] #closest_vector_ids.GetId(j) != i and
            #if  closest_vector_ids.GetId(j) < len(vectors)] #closest_vector_ids.GetId(j) != i and

            weights = [1.0 / norm if (norm := np.linalg.norm(v[1]+vectors[i][1])) > 1e-6 else 0.0 for v in closest_vectors ]
            weights_sum = sum(weights)
            assert weights_sum > 0.0, f"Weights sum is zero: {weights_sum}"

            smoothed_vector = np.array([weights[j] * closest_vectors[j][2] for j in range(len(closest_vectors))]).sum(axis=0) / weights_sum \
                            if weights_sum > 1e-6 else vectors[i][2]
            assert not np.isnan(smoothed_vector).any(), f"Smoothed vector is nan: {smoothed_vector}"
            vectors[i] = (i, vectors[i][1], smoothed_vector)

        # Construct a point locator for the smoothed endpoints of the vectors
        mesh_vectors = vtk.vtkPolyData()
        end_points = vtk.vtkPoints()
        for i in range(mesh_source.GetNumberOfPoints()):
            end_points.InsertNextPoint(vectors[i][1] + vectors[i][2])
        mesh_vectors.SetPoints(end_points)
        # print("Iteration", iteration+1, "done")
        # save_points_to_mesh(path, mesh_source, mesh_vectors, name=f"{iteration+1}", use_source=True)

    # trace vectors from source to target
    obbTree = vtk.vtkOBBTree()
    obbTree.SetDataSet(mesh_target)
    obbTree.BuildLocator()
    mesh_vectors = vtk.vtkPolyData()
    end_points = vtk.vtkPoints()
    for i in range(mesh_source.GetNumberOfPoints()):
        line = vtk.vtkLineSource()
        line.SetPoint1(mesh_source.GetPoint(i))
        line.SetPoint2(mesh_source.GetPoint(i) + 2*vectors[i][2])
        line.Update()
        intersection_point = vtk.vtkPoints()
        does_intersect = obbTree.IntersectWithLine(mesh_source.GetPoint(i), 
                                                   mesh_source.GetPoint(i) + 2*vectors[i][2], 
                                                   intersection_point, 
                                                   None)
        if not does_intersect:
            end_points.InsertNextPoint(mesh_source.GetPoint(i)+vectors[i][2])
            continue
        end_points.InsertNextPoint(intersection_point.GetPoint(0))
        vec = np.array(intersection_point.GetPoint(0)) - np.array(mesh_source.GetPoint(i))
        vectors[i] = (i, mesh_source.GetPoint(i), vec)
    mesh_vectors.SetPoints(end_points)
    distance_scalars = vtk.vtkFloatArray()
    distance_scalars.SetNumberOfComponents(1)
    distance_scalars.SetName("Distance")
    for i in range(mesh_source.GetNumberOfPoints()):
        distance = np.linalg.norm(vectors[i][2])
        distance_scalars.InsertNextValue(distance)
        
    mesh_source.GetPointData().SetScalars(distance_scalars)
    utils.write_vtk_mesh(mesh_source, os.path.join(path, "surfaces", "dist_source.vtk"))
    utils.points_to_mesh(mesh_vectors, os.path.join(path, "surfaces", "vectors.vtk"))
    locator_end_points = vtk.vtkPointLocator()
    locator_end_points.SetDataSet(mesh_vectors)
    locator_end_points.BuildLocator()
    distance_scalars_target = vtk.vtkFloatArray()
    distance_scalars_target.SetNumberOfComponents(1)
    distance_scalars_target.SetName("Distance")
    for i in range(mesh_target.GetNumberOfPoints()):
        point = mesh_target.GetPoint(i)
        closest_point_id = locator_end_points.FindClosestPoint(point)
        distance = np.linalg.norm(vectors[closest_point_id][2])
        distance_scalars_target.InsertNextValue(distance)
    mesh_target.GetPointData().SetScalars(distance_scalars_target)
    utils.write_vtk_mesh(mesh_target, os.path.join(path, "surfaces", "avg_dist_target.vtk"))

    # Save the vectors
    points = np.asarray([v[1] for v in vectors])
    vecs = np.asarray([v[2] for v in vectors])
    os.makedirs(os.path.join(path, "misc"), exist_ok=True)
    np.savez(os.path.join(path, "misc", "vectors.npz"), points=points, vectors=vecs)
    return mesh_source, mesh_target, vectors

def thickness_in_17_seg(mesh_thick, mesh_17):
    locater = vtk.vtkPointLocator()
    locater.SetDataSet(mesh_17)
    locater.BuildLocator()
    thickness = defaultdict(list)
    for i in range(mesh_thick.GetNumberOfPoints()):
        point = mesh_thick.GetPoint(i)
        closest_point_id = locater.FindClosestPoint(point)
        distance = mesh_thick.GetPointData().GetScalars().GetTuple(i)[0]
        segment = mesh_17.GetPointData().GetScalars().GetTuple(closest_point_id)[0]
        thickness[segment].append(distance)
    return thickness

def calculate_myocardium_thickness_single(path, exists_ok=True):
    mesh_inner, mesh_outer = create_meshes_from_segmentation(path)
    mesh_thick, _, _ = mesh_vector_allignment(path, mesh_inner, mesh_outer, num_iterations=5, num_closest_vectors=20)

    if os.path.exists(mesh_path := os.path.join(path, "surfaces", "myocardium_17.vtk")):
        mesh_17 = utils.read_vtk_mesh(mesh_path)
        thickness = thickness_in_17_seg(mesh_thick, mesh_17)
        return thickness


def calculate_myocardium_thickness_all(splits, folder, exists_ok=True):
    
    for i in range(len(splits)):
        data = splits.iloc[i]
        path = os.path.join(folder, data.pseudonymized_id)
        calculate_myocardium_thickness_single(path, exists_ok=exists_ok)

def plot_thickness_histograms(path, thickness_gt=None, name=None):
    """
    Plot histograms of thickness values for each segment in the provided path.

    Args:
        path (str): Path to the directory containing the thickness data.
        thickness_gt (list, optional): Ground truth thickness values for comparison.
    """
    if thickness_gt is not None and np.isnan(thickness_gt).any():
        print(f"GT thickness contains NaN or Inf valuesin {path}")
        thickness_gt = None

    with open(os.path.join(path, "thickness.json"), 'r') as f:
        data = json.load(f)    

    # Convert keys to floats, filter out "-1.0", and sort numerically
    sorted_keys = sorted(
        [k for k in data.keys() if k != "-1.0"],  # Skip "-1.0"
        key=lambda x: float(x)  # Ensure numerical sorting
    )

    # Define subplot layout (e.g., 4 rows × 4 columns for 16 histograms)
    num_plots = len(sorted_keys)
    cols = 5
    rows = (num_plots // cols) + (num_plots % cols > 0)  # Ensure enough rows
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(12, rows * 3))

    # Flatten axes array for easier indexing (if it's 2D)
    axes = axes.flatten()

    # Plot histograms in order
    for i, key in enumerate(sorted_keys):
        values = data[key]
        axes[i].hist(values, bins=25, edgecolor="black", alpha=0.7)
        axes[i].set_title(f"Segment {key}")
        axes[i].set_xlim(0, 20)
        #axes[i].set_xlabel("Value")
        #axes[i].set_ylabel("Frequency")

        # Add ground truth line if provided
        if (thickness_gt is not None) and (i<16):
            gt_value = thickness_gt[int(float(key)) - 1]
            axes[i].axvline(gt_value, color='red', linestyle='--', label='Ground Truth')
            # axes[i].legend()
    
    # Hide any unused subplots (if number of keys < subplot grid size)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout to prevent overlap
    plt.tight_layout()

    ax_combined = fig.add_subplot(rows, 2, 8)
    ax_combined.hist([v for key in sorted_keys for v in data[key]], bins=50, edgecolor="black", alpha=0.7)
    ax_combined.set_title("All Data Combined")
    ax_combined.set_xlim(0, 20)

    if thickness_gt is not None:
        gt_combined = np.mean(thickness_gt)
        ax_combined.axvline(gt_combined, color='red', linestyle='--', label='Ground Truth')
        ax_combined.legend()

    

    #plt.show()
    filename = os.path.join(path, f"{name+'_'}thickness_histograms{'_with_gt' if thickness_gt else ''}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
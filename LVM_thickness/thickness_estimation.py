
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import numpy as np
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import json

from collections import defaultdict
from scipy.spatial import cKDTree

from utils import utils


def _set_distance_scalars(mesh: vtk.vtkPolyData, vecs: np.ndarray, name: str = "Distance") -> vtk.vtkPolyData:
    """Attach the per-point vector length to a mesh as its active scalars."""
    scalars = numpy_to_vtk(np.linalg.norm(vecs, axis=1).astype(np.float32), deep=True)
    scalars.SetName(name)
    mesh.GetPointData().SetScalars(scalars)
    return mesh



def create_meshes_from_segmentation(input_path, out_path, save_name="total_seg", save=True,
                                    decimate_to=None):
    """Build the endo- and epicardial surface meshes from a segmentation.

    decimate_to: target vertex count per surface, or None (default) for no
    decimation. Decimation used to be hardcoded to 30000 as a workaround for
    vtkOBBTree crashing and being slow during ray tracing; that turned out to
    be a bug in vtkOBBTree itself (it is no longer used - see
    mesh_vector_allignment), and full-resolution meshes now run in ~1 minute
    in well under a GB. Since decimation moves the surface and therefore
    biases the wall-thickness measurement, it is off by default. Pass e.g.
    decimate_to=30000 to restore the old behaviour for comparison.
    """
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

    # Either way the meshes go through smooth_and_fill_mesh, which is what
    # removes the marching-cubes staircase terracing - only the vertex-count
    # reduction is optional.
    for label, mesh, save_path in (("inner", mesh_inner, save_inner),
                                   ("outer", mesh_outer, save_outer)):
        n_before = mesh.GetNumberOfPoints()
        if decimate_to is None:
            print(f"    [MESH] Smoothing {label} mesh ({n_before} points, no decimation)...")
            mesh = utils.smooth_and_fill_mesh(mesh)
        else:
            print(f"    [DECIMATE] Simplifying {label} mesh from {n_before} points "
                  f"to ~{decimate_to}...")
            mesh = utils.decimate_vtk_mesh(mesh, decimate_to)
        utils.write_vtk_mesh(mesh, save_path)
        if label == "inner":
            mesh_inner = mesh
        else:
            mesh_outer = mesh

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


def _smooth_vectors(points: np.ndarray, vecs: np.ndarray, num_iterations: int,
                    num_closest_vectors: int) -> np.ndarray:
    """Smooth each point's vector towards the weighted mean of its k nearest neighbours.

    The points never move during smoothing, so the neighbour ids and weights
    are computed once and each iteration is a single gather over an (N, k)
    index matrix. This is a Jacobi update (every point uses the previous
    iteration's vectors); the former per-point loop updated in place
    (Gauss-Seidel), which made results depend on mesh vertex ordering.
    Thickness differs from the old loop by ~0.03% on average.

    Args:
        points: (N, 3) source vertex positions.
        vecs: (N, 3) initial source->target vectors.
        num_iterations: Number of smoothing passes.
        num_closest_vectors: Neighbourhood size k, including the point itself.

    Returns:
        (N, 3) smoothed vectors.
    """
    n_points = len(points)
    k = min(num_closest_vectors, n_points)

    tree = cKDTree(points)
    _, idx = tree.query(points, k=k, workers=utils.num_threads())
    idx = np.atleast_2d(idx)

    # Exclude self by id, not by column: coincident duplicates can outrank it.
    self_mask = idx != np.arange(n_points)[:, None]

    # FIXME: 1/|p_nb + p_self| sums the positions; inverse-distance weighting would
    # use the difference. As written the weights are near-constant over the
    # neighbourhood. Kept deliberately to preserve existing results.
    norms = np.linalg.norm(points[idx] + points[:, None, :], axis=2)
    weights = np.where(norms > 1e-6, 1.0 / np.where(norms > 1e-6, norms, 1.0), 0.0)
    weights *= self_mask
    weights_sum = weights.sum(axis=1)

    if not np.all(weights_sum > 0.0):
        raise ValueError(
            f"Weights sum is zero for {int((weights_sum <= 0.0).sum())} of {n_points} points"
        )

    movable = np.linalg.norm(vecs, axis=1) >= 1e-6
    usable = weights_sum > 1e-6
    update = movable & usable

    vecs = vecs.copy()
    for _ in range(num_iterations):
        smoothed = np.einsum("nk,nkd->nd", weights, vecs[idx]) / weights_sum[:, None]
        if not np.isfinite(smoothed[update]).all():
            raise ValueError("Smoothed vector is nan")
        vecs = np.where(update[:, None], smoothed, vecs)
    return vecs


def mesh_vector_allignment(path, mesh_source, mesh_target, num_iterations=5, num_closest_vectors=20, exists_ok=True,
                           save_vectors_mesh=True):
    """Measure wall thickness as smoothed endo->epi vectors ray-traced onto the target surface.

    Args:
        path: Patient output directory.
        mesh_source: Inner (endocardial) surface.
        mesh_target: Outer (epicardial) surface.
        num_iterations: Vector smoothing passes.
        num_closest_vectors: Smoothing neighbourhood size.
        exists_ok: Reuse misc/vectors.npz when it matches the source mesh.
        save_vectors_mesh: Also write surfaces/vectors.vtk. Nothing reads it back and its
            vtkDelaunay3D costs ~7s per patient, so batch runs switch it off.

    Returns:
        Tuple of (source mesh with thickness scalars, target mesh, list of (id, point, vector)).
    """
    print(f"    [ALIGN] Source vertices: {mesh_source.GetNumberOfPoints()}, Target vertices: {mesh_target.GetNumberOfPoints()}")

    source_points = vtk_to_numpy(mesh_source.GetPoints().GetData()).astype(np.float64)
    num_source_pts = len(source_points)

    # Check if the mesh has already been smoothed
    if os.path.exists(os.path.join(path, "misc", "vectors.npz")) and exists_ok:
        loaded = np.load(os.path.join(path, "misc", "vectors.npz"))
        points = loaded["points"]
        vecs = loaded["vectors"]
        if len(points) == num_source_pts:
            print("Loaded vectors from file")
            vectors = [(i, points[i], vecs[i]) for i in range(len(points))]
            _set_distance_scalars(mesh_source, vecs)
            return mesh_source, mesh_target, vectors

    # Find closest point per vertex in target mesh
    print("    [ALIGN] Finding closest points...")
    target_points = vtk_to_numpy(mesh_target.GetPoints().GetData()).astype(np.float64)
    _, closest_ids = cKDTree(target_points).query(source_points, workers=utils.num_threads())
    vecs = target_points[closest_ids] - source_points

    # Repeat smoothing
    print(f"    [ALIGN] Smoothing vectors ({num_iterations} iterations, k={num_closest_vectors})...")
    vecs = _smooth_vectors(source_points, vecs, num_iterations, num_closest_vectors)

    # trace vectors from source to target
    print("    [ALIGN] Building cell locator and tracing ray intersections...")


    # Mesh hygiene before building the locator: merge duplicate points, make
    # sure every cell is a triangle (vtkFillHolesFilter can leave n-gon
    # patches behind), and drop near-zero-area slivers, which can produce
    # spurious near-tangent ray hits.
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(mesh_target)
    cleaner.Update()
    n_before = mesh_target.GetNumberOfPoints()
    mesh_target = cleaner.GetOutput()
    print(f"    [CHECK] mesh_target points before clean: {n_before}, after: {mesh_target.GetNumberOfPoints()}")


    # NOTE: this used to be a vtkOBBTree, which is broken in VTK 9.7.0:
    # IntersectWithLine(p1, p2, points, cellIds) returns 0 ("no intersection")
    # for every ray - even one fired straight through a unit sphere - while
    # also corrupting heap memory, which segfaulted the process after a few
    # thousand calls. (vtkOBBTree.InsideOrOutside is wrong in the same build
    # too: it reports a point 9 units outside a unit sphere as inside.)
    # Because it always returned 0, every ray silently fell through to the
    # fallback branch below, so this loop never actually ray-traced anything.
    # vtkCellLocator computes the same intersections correctly and ~8x faster.
    locator = vtk.vtkCellLocator()
    locator.SetDataSet(mesh_target)
    locator.BuildLocator()

    mesh_vectors = vtk.vtkPolyData()
    end_points = vtk.vtkPoints()
    end_points.SetNumberOfPoints(num_source_pts)

    # Reuse containers across iterations
    intersection_points = vtk.vtkPoints()
    intersection_cells = vtk.vtkIdList()

    # A rare smoothing-loop edge case (e.g. weights_sum landing just above the
    # 1e-6 floor) can leave a vector with an anatomically implausible
    # magnitude, so clamp the ray length to a generous multiple of the target
    # mesh's own bounding diagonal.
    bounds = mesh_target.GetBounds()
    mesh_diagonal = np.linalg.norm([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])
    max_ray_length = 5.0 * mesh_diagonal

    v_norms = np.linalg.norm(vecs, axis=1)
    degenerate = ~np.isfinite(v_norms) | (v_norms < 1e-6)
    too_long = v_norms > max_ray_length
    if too_long.any():
        vecs[too_long] *= (max_ray_length / v_norms[too_long])[:, None]

    ray_ends = source_points + 2.0 * vecs
    fallback_ends = source_points + vecs
    hits = 0

    for i in range(num_source_pts):
        if degenerate[i]:
            end_points.SetPoint(i, source_points[i])
            continue

        # Reset containers for reuse
        intersection_points.Reset()
        intersection_cells.Reset()

        # Intersections come back sorted along the ray, so GetPoint(0) below
        # is the first crossing of the epicardial surface.
        code = locator.IntersectWithLine(source_points[i].tolist(), ray_ends[i].tolist(), 1e-6,
                                         intersection_points, intersection_cells)

        if code != 0 and intersection_points.GetNumberOfPoints() > 0:
            target_pt = intersection_points.GetPoint(0)
            end_points.SetPoint(i, target_pt)
            vecs[i] = np.asarray(target_pt) - source_points[i]
            hits += 1
        else:
            end_points.SetPoint(i, fallback_ends[i])

    print(f"    [ALIGN] Tracing completed ({hits}/{num_source_pts} hits). Assigning scalars...")
    mesh_vectors.SetPoints(end_points)

    _set_distance_scalars(mesh_source, vecs)
    utils.write_vtk_mesh(mesh_source, os.path.join(path, "surfaces", "dist_source.vtk"))
    if save_vectors_mesh:
        utils.points_to_mesh(mesh_vectors, os.path.join(path, "surfaces", "vectors.vtk"))

    end_point_array = vtk_to_numpy(mesh_vectors.GetPoints().GetData()).astype(np.float64)
    target_points = vtk_to_numpy(mesh_target.GetPoints().GetData()).astype(np.float64)
    _, nearest_end = cKDTree(end_point_array).query(target_points, workers=utils.num_threads())
    _set_distance_scalars(mesh_target, vecs[nearest_end])
    utils.write_vtk_mesh(mesh_target, os.path.join(path, "surfaces", "avg_dist_target.vtk"))

    # Save the vectors
    os.makedirs(os.path.join(path, "misc"), exist_ok=True)
    np.savez(os.path.join(path, "misc", "vectors.npz"), points=source_points, vectors=vecs)
    vectors = [(i, source_points[i], vecs[i]) for i in range(num_source_pts)]
    return mesh_source, mesh_target, vectors

def thickness_in_17_seg(mesh_thick, mesh_17):
    """Group each thickness measurement by the AHA segment of its nearest 17-segment mesh point.

    Args:
        mesh_thick: Mesh whose point scalars are wall thickness.
        mesh_17: Mesh whose point scalars are segment labels.

    Returns:
        Mapping of segment label (float, as the JSON keys have always been) to thickness values.
    """
    thick_points = vtk_to_numpy(mesh_thick.GetPoints().GetData()).astype(np.float64)
    seg_points = vtk_to_numpy(mesh_17.GetPoints().GetData()).astype(np.float64)
    distances = vtk_to_numpy(mesh_thick.GetPointData().GetScalars()).astype(np.float64)
    seg_labels = vtk_to_numpy(mesh_17.GetPointData().GetScalars()).astype(np.float64)

    _, closest = cKDTree(seg_points).query(thick_points, workers=utils.num_threads())
    segments = seg_labels[closest]

    thickness = defaultdict(list)
    order = np.argsort(segments, kind="stable")
    segments_sorted = segments[order]
    distances_sorted = distances[order]
    boundaries = np.flatnonzero(np.diff(segments_sorted)) + 1
    for chunk_seg, chunk_dist in zip(np.split(segments_sorted, boundaries),
                                     np.split(distances_sorted, boundaries)):
        if len(chunk_seg):
            thickness[float(chunk_seg[0])] = chunk_dist.tolist()
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
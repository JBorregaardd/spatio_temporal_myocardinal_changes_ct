import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import tqdm
import SimpleITK as sitk
import json
from skimage import morphology, measure
from scipy.spatial import cKDTree
import shutil

from utils import utils

LVM_ID = 1
LV_ID = 3

def get_scalar_ring_mm_coordinates(path, total_path, mesh_name, exists_ok=True, lvm=True):
    label_total = sitk.ReadImage(total_path)
    label_lv = sitk.Or(label_total == LV_ID, label_total == LVM_ID) if lvm else label_total == LV_ID

    im_bin = sitk.GetArrayFromImage(label_lv).transpose(2, 1, 0)
    im_ring = im_bin & ~morphology.binary_erosion(im_bin)
    im_ring = im_ring.astype(np.uint8)

    mesh = utils.read_vtk_mesh(mesh_name)

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

def get_scalar_ring_points_ids(total_path, mesh_name, exists_ok=True):
    label_total = sitk.ReadImage(total_path)
    label_lv = sitk.Or(label_total == LV_ID, label_total == LVM_ID)

    im_bin = sitk.GetArrayFromImage(label_lv).transpose(2, 1, 0)
    im_ring = im_bin & ~morphology.binary_erosion(im_bin)
    im_ring = im_ring.astype(np.uint8)

    mesh = utils.read_vtk_mesh(mesh_name)

    locator = vtk.vtkPointLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()
    radius = 0.25 # mm

    ring_nnz = im_ring.nonzero()
    scalar_ring_ids = {} #np.zeros_like(im_ring, dtype=np.float64)
    for x, y, z in zip(*ring_nnz):
        indices = np.array([x, y, z], dtype=np.float64)
        point = label_total.TransformContinuousIndexToPhysicalPoint(indices)
        ids = vtk.vtkIdList()
        locator.FindPointsWithinRadius(radius, point, ids)
        if ids.GetNumberOfIds() == 0:
            idx = locator.FindClosestPoint(point)
            scalar_ring_ids[(x, y, z)] = [idx]
        else:
            scalar_ring_ids[(x, y, z)] = [ids.GetId(i) for i in range(ids.GetNumberOfIds())]

    return scalar_ring_ids

def get_scalar_values_from_ring_points(path, mesh_name, scalar_ring_ids, reference):
    
    mesh = utils.read_vtk_mesh(mesh_name)
    scalar_ring = np.zeros(len(scalar_ring_ids), dtype=np.float64)
    ref = sitk.GetArrayFromImage(reference).transpose(2, 1, 0)
    scalar_ring = np.zeros_like(ref, dtype=np.float64)
    
    for i, (x, y, z) in enumerate(scalar_ring_ids.keys()):
        ids = scalar_ring_ids[(x, y, z)]
        scalar_ring[x, y, z] = np.mean([mesh.GetPointData().GetScalars().GetTuple1(i) for i in ids])
        scalar_ring[x, y, z] += 1e-10
        
    scalar_image = sitk.GetImageFromArray(scalar_ring.transpose(2, 1, 0))
    scalar_image.CopyInformation(reference)
    return scalar_image
    
            
def generate_polar_values(path,
                          mesh_name,
                          total_path,
                          scalar_ring_ids=None,
                          return_xy=True,
                          exists_ok=True,
                          lvm=True):
    """
      1. read dist_mesh with vtk
      2. convert mesh to segmentation with one ring of values
      3. read transform
      4. resample segmentation with transform
      5. generate values from polar coordinates
    """    
        
    # input_path = os.path.join(path, "segmentations/total_seg/total_seg.nii.gz")
    label_total = sitk.ReadImage(total_path)
    lab_total = sitk.GetArrayFromImage(label_total).transpose(2, 1, 0)
    ignore_ring = morphology.binary_dilation(lab_total == (LVM_ID if lvm else LV_ID)) #outer_ring & inner_ring
    
    if scalar_ring_ids is not None:
        scalar_image = get_scalar_values_from_ring_points(total_path, mesh_name, scalar_ring_ids, label_total)
    else:
        scalar_image = get_scalar_ring_mm_coordinates(path, total_path, mesh_name, exists_ok=exists_ok, lvm=lvm)

    sitk.WriteImage(scalar_image, os.path.join(path, "segmentations", f"mm_{os.path.basename(mesh_name).split('.')[0]}_ring.nii.gz"))
    # read transform
    transform_path = os.path.join(path, "lv17_transform.txt")
    transform = sitk.ReadTransform(transform_path)

     # read json
    json_path = os.path.join(path, "segmentations", "lv17", "lv17.json")
    with open(json_path, "r") as f:
        output_constants = json.load(f)
    ranges = output_constants["ranges"]
    angle_offset = output_constants["theta0"] -2/3*np.pi #- np.pi/6 -np.pi/24 #- 2/3 * np.pi
    y_0, x_0 = output_constants["com_yx_short_long_axis"]

    lv17_segmentation = sitk.ReadImage(os.path.join(path, "segmentations", "lv17", "lv17.nii.gz"))
    lv17_transformed = sitk.Resample(lv17_segmentation,
                             transform,
                             sitk.sitkNearestNeighbor,
                             0,
                             lv17_segmentation.GetPixelID())
    label_lv17 = sitk.GetArrayFromImage(lv17_transformed).transpose(2, 1, 0)

    label_lv = sitk.Or(label_total == LV_ID, label_total == LVM_ID) if lvm else label_total == LV_ID
    
    lv_transformed = sitk.Resample(label_lv,
                             transform,
                             sitk.sitkNearestNeighbor,
                             0,
                             scalar_image.GetPixelID())
    com = measure.centroid(sitk.GetArrayFromImage(lv_transformed).transpose(2, 1, 0))
    x_0, y_0 = com[0], com[1]

    # resample segmentation with transform
    scalar_im = sitk.GetArrayFromImage(scalar_image).transpose(2, 1, 0)
    scalar_im[~ignore_ring] = 0 
    scalar_image = sitk.GetImageFromArray(scalar_im.transpose(2, 1, 0))
    scalar_image.CopyInformation(label_total)
    scalar_seg = sitk.Resample(scalar_image,
                             transform,
                             sitk.sitkNearestNeighbor,
                             0,
                             scalar_image.GetPixelID())

    # generate values from polar coordinates
    scalar_np = sitk.GetArrayFromImage(scalar_seg).transpose(2, 1, 0)
    slices_nnz = scalar_np.nonzero()[2]
    slices_ = np.unique(slices_nnz)
    # slices_ = slices_[slices_ < ranges[-1]]

    angles, radii, polar_vals = [], [], []

    for n in slices_: 
        
        label_slice = scalar_np[:,:,n]
        loc_x, loc_y = np.where(label_slice)

        theta = -np.arctan2(loc_y - y_0, loc_x - x_0) - angle_offset

        if (label_lv17[:,:,n]==-1).any():
            break

        vals = label_slice[loc_x, loc_y]
        angles.extend(theta)
        radii.extend([n]*len(theta))
        polar_vals.extend(vals)
        
        # plt.figure()
        # plt.scatter(loc_x, loc_y, c=vals, cmap="plasma")
        # plt.plot(x_0, y_0, 'ro')
        # plt.savefig(f"slice_{n}.png")
        
    if return_xy:
        radii = np.array(radii) - min(radii)
        x, y = utils.polar2cartesian(radii, np.array(angles))
        sort_radii = np.argsort(radii)[::-1]
        x, y, polar_vals = x[sort_radii], y[sort_radii], np.array(polar_vals)[sort_radii]
        return x, y, polar_vals
    else:
        return angles, radii, polar_vals
    
    
def fill_holes_in_bullseye(x, y, polar_vals, n_points_per_ring=500, n_neighbors=5):
    # make default grid
    radius = np.max(np.sqrt(x**2 + y**2))
    theta = np.linspace(0, 2*np.pi, n_points_per_ring)
    # make meshgrid
    Th, Rad = np.meshgrid(theta, np.linspace(0, radius, n_points_per_ring))
    # loop through grid nad find n closest points
    # using kd tree

    # r, t = utils.cartesian2polar(x, y)
    # points = np.array([r, t]).T
    points = np.array([x, y]).T
    tree = cKDTree(points)
    polar_output = np.zeros_like(Th)
    X, Y = utils.polar2cartesian(Rad.ravel(), Th.ravel())
    points_regular = np.array([X, Y]).T
    dist, ind = tree.query(points_regular, k=n_neighbors)
    polar_output = np.median(polar_vals[ind], axis=1)

    #  = utils.polar2cartesian(Rad.flatten(), Th.flatten())
    mask = ~ (dist.mean(axis=1) > 4)
    X, Y, polar_output = X[mask], Y[mask], polar_output[mask]
    return X, Y, polar_output

def create_single_bs_from_mesh(folder, mesh_path, total_path, scalar_ring_ids=None, global_min=None, global_max=None, savename=None, show_plot=False):
    x, y, polar_vals = generate_polar_values(folder, mesh_path, total_path, scalar_ring_ids=scalar_ring_ids, lvm=False)
    x, y, polar_vals = fill_holes_in_bullseye(x, y, polar_vals, 750, 5)
    # x = -x
    rad, theta = utils.cartesian2polar(x, y)
    # theta -= np.pi/2-np.pi/10
    # x, y = utils.polar2cartesian(rad, theta)
    
    cmap = "plasma" #if mesh_path.find("squeez")!=-1 else plt.cm.tab20 if mesh_path.find("17")!=-1 else "seismic"
    darkened_cmap = cmap #utils.darken_cmap(cmap, factor=0.85)
    global_min = np.quantile(polar_vals, 0.001) if global_min is None else global_min
    global_max = np.quantile(polar_vals, 0.999) if global_max is None else global_max
    norm = Normalize(vmin=global_min, vmax=global_max, clip=True)
    if show_plot:
        fig, ax = plt.subplots(figsize=(6, 5), subplot_kw=dict(polar=True))
    else:
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
    # sc = ax.scatter(x, y, c=polar_vals, marker='.', linewidths=1, norm=norm)
    sc = ax.scatter(theta, rad, c=polar_vals, marker='.', linewidths=1, norm=norm)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.ax.tick_params(labelsize=15)
    sc.set_cmap(darkened_cmap)
    ax.set_aspect('equal')
    ax.axis('off')
    # ax.set_title("ES - Squeez")
    fig.tight_layout()
    filename = os.path.join(folder, f"{os.path.basename(mesh_path.split('.')[0])}.png") if savename is None\
                         else os.path.join("/",*mesh_path.split(os.sep)[:-2], f"{savename.replace('.png', '')}.png") 
    # plt.savefig(filename)
    r = np.linspace(0.2, 1, 4)*(rad.max()+0.25)
    for start, stop, r_in, r_out in [
            (0, 6, r[2], r[3]),
            (6, 12, r[1], r[2]),
            (12, 16, r[0], r[1]),
    ]:
        n = stop - start
        dtheta = 2*np.pi / n
        ax.bar(np.arange(n) * dtheta + np.pi/2, r_out - r_in, dtheta, r_in,
               clip_on=False, color="none", edgecolor="k", linewidth=2)
    # Draw edge of segment 17 -- here; the edge needs to be drawn differently,
    # using plot().
    ax.plot(np.linspace(0, 2*np.pi), np.linspace(r[0], r[0]), "k",
            linewidth=2)
    plt.savefig(filename.replace(".png", "_w_border.png"), dpi=300, transparent=True, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()

def generate_bs_gif(split, folder, save_folder, mesh_name, gif_name=None, global_min=None, global_max=None):
    plot_folder = os.path.join(save_folder, "plots")
    os.makedirs(plot_folder, exist_ok=True)
    scalar_ring_ids = get_scalar_ring_points_ids(os.path.join(folder, split.pseudonymized_id.iloc[0]), os.path.join(save_folder, "meshes", f"{mesh_name}_00.vtk"))
    for i in range(len(split)):
        mesh_name_i = os.path.join(save_folder, "meshes", f"{mesh_name}_{i*5:02d}.vtk")
        create_single_bs_from_mesh(split, folder, mesh_name_i, plot_folder, i, scalar_ring_ids=scalar_ring_ids, global_min=global_min, global_max=global_max)
    
    _, ES = utils.get_ED_ES(folder, split)
    shutil.copy2(os.path.join(plot_folder, f"{mesh_name.split('.')[0]}_{ES*5:02d}.png"),
                 os.path.join(plot_folder, f"{mesh_name.split('.')[0]}_ES.png"))
    
    filenames = [os.path.join(plot_folder, f"{mesh_name.split('.')[0]}_{i*5:02d}.png") for i in range(len(split))]
    gif_name = gif_name if gif_name is not None else f"{mesh_name.split('.')[0]}_bs.gif"
    utils.create_gif(filenames, os.path.join(save_folder, "gifs"), gif_name, delete_files=True)
    

def plot_mean_bullseye(data, save_folder=None, save_name=None, seg_bold=None, norm=None, cmap=plt.cm.seismic):
    if isinstance(data, tuple):
        data, std = data
    else:
        std = None
    data = np.ravel(data)
    if seg_bold is None:
        seg_bold = []
    if norm is None:
        norm = mpl.colors.Normalize(vmin=-0.3, vmax=0.3)
        
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    r = np.linspace(0.2, 1, 4)

    ax.set(ylim=[0, 1], xticklabels=[], yticklabels=[])
    ax.grid(False)  # Remove grid

    # Fill segments 1-6, 7-12, 13-16.
    for start, stop, r_in, r_out in [
            (0, 6, r[2], r[3]),
            (6, 12, r[1], r[2]),
            (12, 16, r[0], r[1]),
            (16, 17, 0, r[0]),
    ]:
        n = stop - start
        dtheta = 2*np.pi / n
        bars = ax.bar(np.arange(n) * dtheta + np.pi/2, r_out - r_in, dtheta, r_in,
               color=cmap(norm(data[start:stop])))
        
        for i, bar in enumerate(bars):
            theta = (i + 0.5) * dtheta 
            theta += np.pi/3 if start < 12 else np.pi/4
            r_text = (r_in + r_out) / 2 if stop < 17 else 0
            ax.text(theta, r_text, f"{data[start + i]:.2f}"+((r'$\pm$ '+f'{std[start+i]:.2f}') if std is not None else ''), 
                    ha='center', va='center', fontsize=10 if std is not None else 14, color='black', fontweight='bold')


    # Now, draw the segment borders.  In order for the outer bold borders not
    # to be covered by inner segments, the borders are all drawn separately
    # after the segments have all been filled.  We also disable clipping, which
    # would otherwise affect the outermost segment edges.
    # Draw edges of segments 1-6, 7-12, 13-16.
    for start, stop, r_in, r_out in [
            (0, 6, r[2], r[3]),
            (6, 12, r[1], r[2]),
            (12, 16, r[0], r[1]),
    ]:
        n = stop - start
        dtheta = 2*np.pi / n
        ax.bar(np.arange(n) * dtheta + np.pi/2, r_out - r_in, dtheta, r_in,
               clip_on=False, color="none", edgecolor="k", linewidth=[
                   4 if i + 1 in seg_bold else 2 for i in range(start, stop)])
    # Draw edge of segment 17 -- here; the edge needs to be drawn differently,
    # using plot().
    ax.plot(np.linspace(0, 2*np.pi), np.linspace(r[0], r[0]), "k",
            linewidth=(4 if 17 in seg_bold else 2))
    if save_name and save_folder:
        plt.savefig(os.path.join(save_folder, "plots", f"BS_{save_name}.png"))
        plt.close()
    else:
        plt.show()
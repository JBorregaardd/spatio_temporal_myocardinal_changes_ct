import os 
import re
import numpy as np
import pandas as pd

import numpy as np
import os
import SimpleITK as sitk
from skimage.morphology import ball
from scipy.ndimage import center_of_mass
from scipy.interpolate import interpn, make_interp_spline
import pandas as pd
from PIL import Image
import re
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import vtk 
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from matplotlib import cm  # For colormap
from matplotlib.colors import Normalize

import imageio


def read_excel_for_best_phase(root, 
                              file_must_exist=True, 
                              get_ES=False, 
                              exclude_file_consisting=None, 
                              file=None, 
                              nifti_folder="NIFTI"):
    r"""
    Reads the metadata excel file and returns the best phase for each patient.
    input:
        root: path to the patient folder (e.g. r"H:\DTU-CFA-1")
        file_must_exist: if True, the nifti file must exist on disk
        get_ES: if True, return the end-systolic phase, otherwise the end-diastolic phase
        exclude_file_consisting: if not None, exclude files that contain this string in the filename
        file: if not None, use this file instead of the default metadata file
    output:
        dataframe with the best phase for each patient
    """
    if file is not None:
        files = [file]
    else:
        files = glob.glob(os.path.join(root, "Metadata", f"{os.path.basename(root)}*.csv"))
        assert len(files)==1, f"Expected 1 metadata file, but found {len(files)} in {root}"
    file_name = files[0]
    # file_name = os.path.join(root, "Metadata", os.path.basename(root)+"_meta_data.csv")
    df = pd.read_csv(file_name)
    df.rename(columns={n:n.strip() for n in df.columns}, inplace=True)
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df = df.sort_values(by="filename")

    dff = df[df.ImageType.str.contains("ORIGINAL")      &
            (df.SeriesDescription.str.contains("HALF")  | 
             df.SeriesDescription.str.contains("SEGMENT")) &
            (df.ContrastBolusAgent.str.strip() != "")   &
            ~df.SeriesDescription.str.contains("%",na=False)    &
            ~df.SeriesDescription.str.contains("3.0 CE",na=False)    &
            (df.estimated_slice_spacing < 1.1) &
             df.filename.str.endswith(".nii.gz")
            #(df.SliceThickness < 1)
            ]
    
    # exclude files with specific strings
    if exclude_file_consisting is not None:
        dff = dff[~dff.filename.str.contains(exclude_file_consisting, na=False)]
    
    patient_filter = []

    for idx in dff.patient_id.unique():
        split = dff[dff.patient_id==idx]
        ms_score = [int( re.search(r"\d+ms",val).group().replace('ms','') ) for val in split.SeriesDescription]
        file_exists = [os.path.isfile(os.path.join(root, nifti_folder, f.strip())) if file_must_exist else True for f in split.filename]
        if len(ms_score)==1:
            ms_time = ms_score[0]
            total_time = float(re.search(r"\d+\.\d+s",split.SeriesDescription.iloc[0]).group().replace('s',''))*1000
            total_time = total_time if total_time > 0 else 1000
            to_append = (ms_time < 0.5*total_time) if get_ES else (ms_time > 0.5*total_time)
            to_append = to_append and file_exists[0]
            patient_filter.append(to_append)
        else:
            total_time = float(re.search(r"\d+\.\d+s",split.SeriesDescription.iloc[0]).group().replace('s',''))*1000
            total_time = total_time if total_time > 0 else 1000
            within_time = [ms < 0.5*total_time if get_ES else ms > 0.5*total_time for ms in ms_score]
            idx_of_interest = ms_score.index(min(ms_score) if get_ES else max(ms_score))
            patient_filter.extend([(idx==idx_of_interest and exist and time_score) 
                                   for idx,exist,time_score in zip(range(len(ms_score)),file_exists,within_time)])

    return dff[patient_filter]

def read_excel_for_CFA(root, file_must_exist=True):
    r"""
    Reads the metadata excel file and returns CFA series for each patient.
    input:
        root: path to the patient folder (e.g. r"H:\DTU-CFA-1")
    output:
        list of dataframes with CFA series for each patient
    
    """
    files = glob.glob(os.path.join(root, "Metadata", f"{os.path.basename(root)}*.csv"))
    assert len(files)==1, f"Expected 1 metadata file, but found {len(files)} in {root}"
    file_name = files[0]

    df = pd.read_csv(file_name)
    # clean the dataframe
    df.rename(columns={n:n.strip() for n in df.columns}, inplace=True)
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df = df.sort_values(by="filename")
    
    # filter the dataframe
    dff = df[df.SeriesDescription.str.contains("CE")             &
           (df.SeriesDescription.str.contains("SEGMENT"  )      |
            df.SeriesDescription.str.contains("HALF"))          &
            df.SeriesDescription.str.contains("%")              &
           (df.ContrastBolusAgent.str.strip() != "")            &   
            df.filename.str.endswith(".nii.gz")                 &
            (df.SliceThickness > 1)
            ]        
    vals = []
    
    # add percentage column
    for val in dff.SeriesDescription:
        vals.append(int( re.search(r"\d+%",val).group().replace("%","")) )
    dff.loc[:,"percentage"] = vals
    # dff["percentage"] = vals

    splits = []
    
    # loop over each patient and find the CFA series
    for idx in dff.patient_id.unique():
        split = dff[dff.patient_id==idx].sort_values(by="percentage")
        
        # if 20 scans are from the same patient and all the files are present - add series
        if len(split) == 20 and \
            (all([os.path.isfile(os.path.join(root, "NIFTI", f.strip())) for f in split.filename]) if file_must_exist else True) and \
            ((split.percentage==np.arange(0,100,5)).all()): 
            
            splits.append(split)
            
        # not a whole CFA scan
        elif len(split) < 20:
            continue
        
        # if more than 20 scans are present - find the series with the most slices and the same spacing
        elif len(split) > 20:
            for s in np.sort(split.n_slices.unique())[::-1]:
                sub_split = split[split.n_slices==s].copy()
                sub_split['spacing']=sub_split.PixelSpacing.str.extract(r'\[([\d.]+)\s')
                sub_split = sub_split[sub_split['spacing']==sub_split['spacing'].mode().iloc[0]]
                
                # if 20 scans have the same spacing, n_slices and all the files are present - add series
                if len(sub_split)==20 and \
                    (all([os.path.isfile(os.path.join(root, "NIFTI", f.strip())) for f in sub_split.filename]) if file_must_exist else True) and \
                    ((sub_split.percentage==np.arange(0,100,5)).all()): 

                    splits.append(sub_split)
                    # only add one series per patient
                    break
                elif len(sub_split) > 20:
                    sub_split_sorted = sub_split.sort_values(by="SeriesTime")
                    if (all([os.path.isfile(os.path.join(root, "NIFTI", f.strip())) for f in sub_split.filename]) if file_must_exist else True) and \
                        ((sub_split_sorted.percentage.iloc[-20:]==np.arange(0,100,5)).all()): 
                            splits.append(sub_split_sorted.iloc[-20:])
                            break
                    for i in range(len(sub_split_sorted)//20):
                        if (all([os.path.isfile(os.path.join(root, "NIFTI", f.strip())) for f in sub_split.filename]) if file_must_exist else True) and \
                        ((sub_split_sorted.percentage.iloc[i*20:(i+1)*20]==np.arange(0,100,5)).all()): 
                            splits.append(sub_split_sorted.iloc[i*20:(i+1)*20])
                            break
                    
    return splits

def spline_interp(y, x=None, n_points=1000):
    if x is None:
        x = np.arange(len(y))
    spl = make_interp_spline(x, y, k=3)
    
    x_new = np.linspace(x[0], x[-1], n_points)
    y_smooth = spl(x_new)
    return x_new, y_smooth

def calculate_mean_std(im, lab_all, radii_pct=0.02):
    lab_LV = (lab_all==1).astype(np.int16)
    com = center_of_mass(lab_LV)
    radius = radii_pct * min(lab_LV.shape)
    sphere = ball(radius)
    sphere_nnz = sphere.nonzero()
    grid = (np.arange(0,lab_LV.shape[0]),
            np.arange(0,lab_LV.shape[1]),
            np.arange(0,lab_LV.shape[2]))
    sample_points = np.array([sphere_nnz[0]+com[0]-sphere.shape[0]/2,
                              sphere_nnz[1]+com[1]-sphere.shape[1]/2,
                              sphere_nnz[2]+com[2]-sphere.shape[2]/2]).T
    HU_vals = interpn(grid, im, sample_points)
    # print("\n no. HU", len(HU_vals))
    return HU_vals.mean(), HU_vals.std()



def get_com(label, as_int=False, real_coords=False):
    """
        Get centre of mass of a SimpleITK.Image

    Args:
        label (sitk.Image): Label mask image.
        as_int (bool, optional): Returns each components as int if true. Defaults to True.
        real_coords (bool, optional): Return coordinates in physical space if true. Defaults to
            False.

    Returns:
        list: List of coordinates
    """
    if isinstance(label, np.ndarray):
        com = center_of_mass(label)

    else:
        arr = sitk.GetArrayFromImage(label)
        com = center_of_mass(arr)[::-1]

        if real_coords:
            com = label.TransformContinuousIndexToPhysicalPoint(com)

    return com




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

def write_vtk_mesh(mesh, path):
    # Remove the existing file if it exists
    if os.path.isfile(path):
        os.remove(path)

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(mesh)
    writer.SetFileTypeToBinary()
    writer.SetFileTypeToASCII()
    writer.SetFileVersion(42)
    writer.SetFileName(path)
    writer.Write()

    # Check if the file was written
    assert os.path.isfile(path), f"Mesh not written to file: {path}"



def polar2cartesian(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

def cartesian2polar(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)

def calculate_blood_stats(path, split, df_name="LV_blood_stats.csv", printout=False):
    
    folder = path.split(os.sep)[1]
    dataset_name = split.patient_id.iloc[0].split("_")[0]
    all_stats = {}
    for i in range(len(split)):
        patient_id = split["pseudonymized_id"].iloc[i]
        segmentation_path = os.path.join("/",
                folder, "TotalSegmentator", dataset_name, patient_id, "segmentations",
                "heartchambers_highres", "heartchambers_highres.nii.gz"
            )
        segmentation = sitk.ReadImage(segmentation_path)
        seg = sitk.GetArrayFromImage(segmentation).transpose(2,1,0)
        vox_volume = segmentation.GetSpacing()[0] * segmentation.GetSpacing()[1] * segmentation.GetSpacing()[2]
        stats = {}
        val = 3 # Inner LV id
        # val 
        # hu_values = im[lab==val]
        
        # stats["avg_hu"] = np.average(hu_values)
        # stats["std_hu"] = np.std(hu_values)
        # # stats["med_hu"] = np.median(hu_values)
        # stats["q01_hu"] = np.percentile(hu_values, 1)
        # stats["q99_hu"] = np.percentile(hu_values, 99)
        tmp = seg[seg == val]
        stats["n_voxels"] = len(tmp)
        stats["tot_vol_mm3"] = len(tmp) * vox_volume
        all_stats[patient_id] = stats
        if printout:
            print(f"Patient {patient_id} - {stats['n_voxels']} voxels - {stats['tot_vol_mm3']:.2f} mm3")

    # save to csv
    df = pd.DataFrame.from_dict(all_stats, orient="index")
    df.index.name = "id_seg"
    df.to_csv(os.path.join(path, df_name), index=True)

def get_ED_ES(path, split, df_name="LV_blood_stats.csv", printout=False):
    df_path = os.path.join(path, df_name)
    if not os.path.isfile(df_path):
        print(f"Calculating blood stats for {split.pseudonymized_id.iloc[0]}...")
        calculate_blood_stats(path, split, df_name=df_name)
        print(f"Blood stats saved to {df_path}")
    df = pd.read_csv(df_path)
    vols = []
    for i in range(len(split)):
    
        scan = split["pseudonymized_id"].iloc[i]
        val = df[df.id_seg==scan]["tot_vol_mm3"]
        
        # print(f"scan: {scan} - {val}")
        vols.append(val.item())

    all_vols = np.array(vols)
    ED = all_vols.argmax()
    ES = all_vols.argmin()

    if printout:
        print(f"ED={ED:02d} - {split.percentage.iloc[ED]:02d}%: {split.pseudonymized_id.iloc[ED]}")
        print(f"ES={ES:02d} - {split.percentage.iloc[ES]:02d}%: {split.pseudonymized_id.iloc[ES]}")

    return ED, ES

def get_EF(path, split, df_name="LV_blood_stats.csv", printout=False):
    df_path = os.path.join(path, df_name)
    df = pd.read_csv(df_path)

    EF = (df["tot_vol_mm3"].max()-df["tot_vol_mm3"].min())/df["tot_vol_mm3"].max()

    if printout:
        print(f"EF: {EF:.2f}")

    return EF

def sitk2vtk(img, flip_for_volume_rendering=False, debugOn=False):
    """Convert a SimpleITK image to a VTK image, via numpy."""
    size = list(img.GetSize())
    origin = list(img.GetOrigin())
    spacing = list(img.GetSpacing())
    ncomp = img.GetNumberOfComponentsPerPixel()
    direction = img.GetDirection()

    # convert the SimpleITK image to a numpy array
    i2 = sitk.GetArrayFromImage(img)
    if debugOn:
        i2_string = i2.tostring()
        print("data string address inside sitk2vtk", hex(id(i2_string)))

    vtk_image = vtk.vtkImageData()

    # VTK expects 3-dimensional parameters
    if len(size) == 2:
        size.append(1)

    if len(origin) == 2:
        origin.append(0.0)

    if len(spacing) == 2:
        spacing.append(spacing[0])

    if len(direction) == 4:
        direction = [
            direction[0],
            direction[1],
            0.0,
            direction[2],
            direction[3],
            0.0,
            0.0,
            0.0,
            1.0,
        ]

    vtk_image.SetDimensions(size)
    vtk_image.SetSpacing(spacing)
    vtk_image.SetOrigin(origin)
    vtk_image.SetExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)

    if vtk.vtkVersion.GetVTKMajorVersion() < 9:
        print("Warning: VTK version <9.  No direction matrix.")
    else:
        vtk_image.SetDirectionMatrix(direction)

    # Volume rendering does not support direction matrices (27/5-2023)
    # so sometimes the volume rendering is mirrored
    # this a brutal hack to avoid that
    if flip_for_volume_rendering:
        if direction[4] < 0:
            i2 = np.fliplr(i2)

    depth_array = numpy_to_vtk(i2.ravel(), deep=True)
    depth_array.SetNumberOfComponents(ncomp)
    vtk_image.GetPointData().SetScalars(depth_array)

    vtk_image.Modified()

    return vtk_image

def resample_to_smallest_spacing(image, smallest_spacing=None):
    # Get the smallest voxel spacing in the image
    smallest_spacing = min(image.GetSpacing()) if smallest_spacing is None else smallest_spacing

    # Define the isotropic spacing
    isotropic_spacing = [smallest_spacing] * image.GetDimension()

    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, isotropic_spacing)]

    # Create a reference image with isotropic voxel spacing
    reference_image = sitk.Image(new_size, image.GetPixelIDValue())
    reference_image.SetSpacing(isotropic_spacing)
    reference_image.SetOrigin(image.GetOrigin())
    reference_image.SetDirection(image.GetDirection())
    # print(f"Reference image size: {reference_image.GetSize()}")
 
    # Resample the original image to match the spacing of the reference image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampled_image = resampler.Execute(image)
    # print(f"Resampled image size: {resampled_image.GetSize()}")

    return resampled_image

def convert_label_map_to_surface(label_name, output_file, 
                                 reset_direction_matrix=False, 
                                 segment_id=1,
                                 only_largest_component=True,
                                 resample_to_isotropic=False,
                                 exist_ok=False):
    
    if exist_ok and os.path.isfile(output_file):
        return True

    try:
        img = sitk.ReadImage(label_name)
        # Add padding to avoid edge effects
        padding = [1,1,1]
        img = sitk.ConstantPad(img, padding, padding, constant=0)
    except RuntimeError as e:
        print(f"Got an exception {str(e)}")
        print(f"Error reading {label_name}")
        return None
    if resample_to_isotropic:
        img = resample_to_smallest_spacing(img)

    vtk_img = sitk2vtk(img, flip_for_volume_rendering=False)
    if vtk_img is None:
        return False

    # Check if there is any data
    vol_np = vtk_to_numpy(vtk_img.GetPointData().GetScalars())
    if np.sum(vol_np) < 1:
        print(f"Only zeros in {label_name}")
        return False

    if reset_direction_matrix:
        direction = [1, 0, 0.0, 0, 1, 0.0, 0.0, 0.0, 1.0]
        vtk_img.SetDirectionMatrix(direction)

    # print(f"Generating: {output_file}")

    mc = vtk.vtkDiscreteMarchingCubes()
    mc.SetInputData(vtk_img)
    mc.SetNumberOfContours(1)
    mc.SetValue(0, segment_id)
    mc.Update()

    if mc.GetOutput().GetNumberOfPoints() < 10:
        print(f"No isosurface found in {label_name}")
        return False
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(mc.GetOutputPort())
    smoother.SetNumberOfIterations(200)
    smoother.BoundarySmoothingOn()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetFeatureAngle(30.0)
    smoother.SetPassBand(0.01)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update() 

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(smoother.GetOutputPort())
    normals.SetFeatureAngle(15.0)
    normals.Update()
    # Save in VTK version 4.2 and ASCII format so Elastix can read it
    if only_largest_component:
        conn = vtk.vtkConnectivityFilter()
        conn.SetInputConnection(normals.GetOutputPort())
        conn.SetExtractionModeToLargestRegion()
        conn.Update()
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputConnection(conn.GetOutputPort())
        writer.SetFileTypeToASCII()
        writer.SetFileVersion(42)
        writer.SetFileName(output_file)
        writer.Write()
    else:
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputConnection(normals.GetOutputPort())
        writer.SetFileTypeToBinary()
        writer.SetFileTypeToASCII()
        writer.SetFileVersion(42)
        writer.SetFileName(output_file)
        writer.Write()

    return True

def decimate_vtk_mesh(mesh, n_points=None, reduction_factor=None, smooth_fill=True):

    if n_points is not None:
        reduction = 1.0 - n_points / mesh.GetNumberOfPoints()
    elif reduction_factor is not None:
        reduction = reduction_factor
    else:
        raise ValueError("Either n_points or reduction_factor must be provided")

    reduction = 1 - n_points/mesh.GetNumberOfPoints()
    decimate = vtk.vtkQuadricDecimation()
    decimate.SetInputData(mesh)
    decimate.SetTargetReduction(reduction)
    decimate.Update()

    mesh_reduce = decimate.GetOutput()

    if not smooth_fill:
        return mesh_reduce
    
    # fill holes
    fill = vtk.vtkFillHolesFilter()
    fill.SetInputData(mesh_reduce)
    fill.SetHoleSize(10000)
    fill.Update()

    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(fill.GetOutputPort())
    smoother.SetNumberOfIterations(100)
    smoother.BoundarySmoothingOn()
    smoother.FeatureEdgeSmoothingOn()
    smoother.SetFeatureAngle(0.0)
    smoother.SetPassBand(0.01)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update() 

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(smoother.GetOutput())
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    # normals.SetFeatureAngle(15.0)
    normals.Update()

    return normals.GetOutput()

def decimate_vtk_mesh_paths(mesh_path, save_name, n_points=None, reduction_factor=None, smooth_fill=True):

    mesh = read_vtk_mesh(mesh_path)
    mesh_reduce = decimate_vtk_mesh(mesh, n_points, reduction_factor, smooth_fill)
    if save_name:
        write_vtk_mesh(mesh_reduce, save_name)
    return mesh_reduce

def set_scalars_on_vtk_mesh(mesh, scalars, array_name="scalars"):
    if isinstance(scalars, list):
        scalars = np.array(scalars)
    if isinstance(scalars, vtk.vtkDataArray):
        scalars_vtk = scalars
    else:
        scalars_vtk = numpy_to_vtk(scalars, deep=True)
    scalars_vtk.SetName(array_name)
    
    if len(scalars) == mesh.GetNumberOfPoints():
        mesh.GetPointData().SetScalars(scalars_vtk)
    elif len(scalars) == mesh.GetNumberOfCells():
        mesh.GetCellData().SetScalars(scalars_vtk)
        cell_to_point_filter = vtk.vtkCellDataToPointData()
        cell_to_point_filter.SetInputData(mesh)
        cell_to_point_filter.Update()
        mesh = cell_to_point_filter.GetOutput()
    else:
        raise ValueError("Number of scalars must match number of points or cells")
    return mesh

def set_scalars_on_vtk_mesh_paths(mesh_path, save_name, scalars, array_name="scalars"):
    mesh = read_vtk_mesh(mesh_path)
    mesh = set_scalars_on_vtk_mesh(mesh, scalars, array_name)
    write_vtk_mesh(mesh, save_name)
    return mesh

def print_sitk_info(image):
    print(f"Size: {image.GetSize()}")
    print(f"Spacing: {image.GetSpacing()}")
    print(f"Origin: {image.GetOrigin()}")
    direction = np.array(image.GetDirection()).reshape(3,3)
    print(f"Direction: {direction}")
    print(f"Number of components: {image.GetNumberOfComponentsPerPixel()}")
    print(f"Pixel type: {image.GetPixelIDTypeAsString()}")



def create_gif(filenames, save_folder, save_name, delete_files=False):
    save_name = save_name if save_name.endswith(".gif") else save_name + ".gif"
    with imageio.get_writer(os.path.join(save_folder, save_name), fps=8, loop=0) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            if delete_files:
                os.remove(filename)
    print(f"GIF saved as {save_name}")
    
    
def darken_cmap(cmap, factor=0.85):
    """Darken a colormap by a given factor.
    factor < 1 makes the colormap darker, factor > 1 makes it lighter."""
    cmap = plt.get_cmap(cmap)
    new_cmap = cmap(np.linspace(0, 1, cmap.N))
    # Darken each RGB color
    new_cmap[:, :3] = new_cmap[:, :3] * factor
    return mcolors.ListedColormap(new_cmap)

def points_to_mesh(points, save_name=None):
    delaunay = vtk.vtkDelaunay3D()
    delaunay.SetInputData(points)
    delaunay.Update()

    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputData(delaunay.GetOutput())
    geometry_filter.Update()

    fill_holes = vtk.vtkFillHolesFilter()
    fill_holes.SetInputData(geometry_filter.GetOutput())
    fill_holes.SetHoleSize(1.0)
    fill_holes.Update()
    polydata = fill_holes.GetOutput()

    if save_name:
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(polydata)
        writer.SetFileName(save_name)
        writer.Write()
    return polydata

import pyvista as pv
import numpy as np
from scipy.spatial import cKDTree

def smooth_mesh_scalars(mesh, radius=1.0, sigma=None, iterations=1):
    mesh = mesh.copy()
    points = mesh.points
    scalars = vtk_to_numpy(mesh.GetPointData().GetScalars())
    
    # Build spatial search tree once
    tree = cKDTree(points)
    
    # Set default sigma
    if sigma is None:
        sigma = radius / 2.0

    for _ in range(iterations):
        new_scalars = np.zeros_like(scalars)
        
        for i, pt in enumerate(points):
            idxs = tree.query_ball_point(pt, radius)
            neighbors = points[idxs]
            values = scalars[idxs]

            # Compute Gaussian weights
            dists = np.linalg.norm(neighbors - pt, axis=1)
            weights = np.exp(-0.5 * (dists / sigma)**2)

            new_scalars[i] = np.sum(weights * values) / np.sum(weights)

        scalars = new_scalars

    mesh.GetPointData().SetScalars(numpy_to_vtk(scalars, deep=True))
    return mesh

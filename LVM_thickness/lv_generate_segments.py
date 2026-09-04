import copy
from email.mime import image
from math import inf
import os
import re
import numpy as np
import SimpleITK as sitk
import json
from scipy.spatial import KDTree
from skimage.morphology import binary_dilation
from scipy.ndimage import center_of_mass
from platipy.imaging.label.utils import get_com
from platipy.imaging.utils.crop import crop_to_roi, label_to_roi
from platipy.imaging.utils.geometry import vector_angle
from platipy.imaging.utils.valve import generate_valve_using_cylinder

from utils import utils

def check_360(thetas, tolerance=0.01):
    
    ref_array = np.arange(0, 2*np.pi, 2*np.pi/360)
    for i in range(len(ref_array)):
        # find closest value in thetas
        if (np.abs(thetas - ref_array[i])).min() > tolerance:
            return False

    return True

def get_com_with_z_mask(image, z_mask):
    arr = sitk.GetArrayViewFromImage(image)[z_mask]
    return center_of_mass(arr)

def get_com_with_z_range(image, lb, ub):
    arr = sitk.GetArrayViewFromImage(image)[lb:ub]
    return center_of_mass(arr)

def get_intersection(arr_slice, current_angle, angles, radii, loc_x, loc_y, angle_offset=np.pi/16):
    
    template_slice = np.zeros_like(arr_slice)
    # find angles closest to angle
    angle_diff = np.abs(angles - current_angle)
    close_idx = np.where(angle_diff < angle_offset)[0]
    if not close_idx.size:
        return None, None
    template_slice[loc_y[close_idx], loc_x[close_idx]] = 1
    x, y = center_of_mass(template_slice)

    x = loc_x[close_idx].mean()
    y = loc_y[close_idx].mean()
    return x, y

def extract(
    template_img,
    angles,
    radii,
    angle_min,
    angle_max,
    loc_x,
    loc_y,
    cw=False,
):
    """
    Utility function to extract relevant voxels from a mask based on polar coordinates
    """
    # Get template array
    template_arr = sitk.GetArrayViewFromImage(template_img)
    # Get the segment array
    segment_arr = np.zeros_like(template_arr)

    # Define the condition list

    if cw:
        in_segment_condition = (angles <= angle_min) | (angles >= angle_max)
        # in_segment_condition &= radii >= radius_min
    else:
        in_segment_condition = (angles <= angle_max) & (angles >= angle_min)
        # in_segment_condition &= radii >= radius_min

    # Extract matching voxels
    segment_arr[loc_y[in_segment_condition], loc_x[in_segment_condition]] = 1

    # Convert to image
    segment_img = sitk.GetImageFromArray(segment_arr)
    segment_img.CopyInformation(template_img)

    # Make sure area exceeds lower bound
    # area = segment_arr.sum() * np.prod(segment_img.GetSpacing())
    # if area < min_area_mm2:
    #     segment_img *= 0

    return sitk.Cast(segment_img, template_img.GetPixelID())


def generate_lv_segments(
    contours,
    myo_seg,
    lv_seg,
    label_left_ventricle="Ventricle_L",
    label_left_atrium="Atrium_L",
    label_right_ventricle="Ventricle_R",
    label_heart="Heart",
    myocardium_thickness_mm=10,
    hole_fill_mm=3,
    optimiser_tol_degrees=1,
    optimiser_max_iter=9,
    min_area_mm2=5,
    verbose=False,
    orig_img=None,
    transform_name=None,
):
    """
    Generates the 17 segments of the left vetricle

    This functions works as follows:
        1.  Heart volume is rotated to align the long axis to the z Cartesian (physical) space.
            Usually means it aligns with the axial axis (for normal simulation CT)
        2.  An optimiser adjusts the orientation to refine this alignment to the vector defined by
            MV COM - LV apex axis (long axis)
        3.  Left ventricle is divided into thirds along the long axis
        4.  Myocardium is defined as the outer 10mm
        5.  Geometric operations are used to define the segments
        6.  Everything is rotated back to the normal orientation
        7.  Some post-processing *magic*

    Args:
        contours (dict): A dictionary containing strings (label names) as keys and SimpleITK.Image
            (masks) as values. Must contain at least the LV, RV, MV, and whole heart.
        label_left_ventricle (str, optional): The name for the left ventricle mask (contour).
            Defaults to Ventricle_L.
        label_left_atrium (str, optional): The name for the left atrium mask (contour). Defaults to
            Atrium_L.
        label_right_ventricle (str, optional): The name for the right ventricle mask (contour).
            Defaults to Ventricle_R.
        label_heart (str, optional): The name for the heart mask (contour). Defaults to Heart.
        myocardium_thickness_mm (float, optional): Moycardial thickness, in millimetres.
            Defaults to 10.
        hole_fill_mm (float, optional): Holes smaller than this get filled in. Defaults to 3.
        optimiser_tol_degrees (float, optional): Optimiser tolerance (change in angle per iter).
            Defaults to 1, which typically requires 3-4 iterations.
        optimiser_max_iter (int, optional): Maximum optimiser iterations. Defaults to 10
        verbose (bool, optional): Print of information for debugging. Defaults to False.

    Returns:
        dict : The left ventricle segment dictionary, with labels (int) as keys and the binary
        label defining the segment (SimpleITK.Image) as values.
    """

    if verbose:
        print("Beginning LV segmentation algorithm.")

    # Initial set up
    label_mitral_valve = "MITRALVALVE"

    label_list = [label_left_ventricle, label_left_atrium, label_right_ventricle, label_heart]
    working_contours = copy.deepcopy({s: contours[s] for s in label_list})

    label_list.append(label_mitral_valve)

    output_contours = {}
    overall_transform_list = []

    # we add an automatically generated MV contour
    # TODO: find this from overlap of LV and LA
    working_contours[label_mitral_valve] = generate_valve_using_cylinder(
        working_contours[label_left_atrium],
        working_contours[label_left_ventricle],
        radius_mm=15,
        height_mm=10,
    )



    """
    Module 1 - Preparation
    Crop the images
    Rotate to the cardiac axis
    """
    # Crop to the smallest volume possible to make it FAST
    cb_size, cb_index = label_to_roi(
        working_contours[label_heart] > 0,
        expansion_mm=(30, 30, 60),  # Better to make it a bit bigger to be safe
    )

    for label in label_list:
        working_contours[label] = crop_to_roi(working_contours[label], cb_size, cb_index)

    if verbose:
        print("Module 1: Cropping and initial alignment.")
        vol_before = np.prod(contours[label_heart].GetSize())
        vol_after = np.prod(working_contours[label_heart].GetSize())
        print(f"  Images cropped. Volume reduction: {vol_before/vol_after:.3f}")

    # Initially we should reorient based on the cardiac axis
    label_orient = (
        working_contours[label_left_ventricle] + working_contours[label_left_atrium]
    ) > 0
    spacing = label_orient.GetSpacing()

    lsf = sitk.LabelShapeStatisticsImageFilter()  # this will be used throughout
    
    # load transform if it exists
    transform_exists = False #os.path.isfile(transform_name) if transform_name is not None else False
    if transform_exists:
        rotation_transform = sitk.ReadTransform(transform_name)
        label_orient_rotation = sitk.Resample(
            label_orient,
            rotation_transform,
            sitk.sitkNearestNeighbor,
            0,
            working_contours[label].GetPixelID(),
        )
        lsf.Execute(label_orient_rotation)
        cardiac_axis = np.array(lsf.GetPrincipalAxes(1)[:3])  # First principal axis approx. long axis
        if cardiac_axis[2] < 0:
            cardiac_axis = -1 * cardiac_axis
        rotation_angle = vector_angle(cardiac_axis[::-1], (0, 0, 1))

    else:
        lsf.Execute(label_orient)
        cardiac_axis = np.array(lsf.GetPrincipalAxes(1)[:3])  # First principal axis approx. long axis

        # The principal axis isn't guaranteed to point from base to apex
        # If is points apex to base, we have to invert it
        # So check that here
        if cardiac_axis[2] < 0:
            cardiac_axis = -1 * cardiac_axis

        rotation_angle = vector_angle(cardiac_axis[::-1], (0, 0, 1))
        rotation_axis = np.cross(cardiac_axis[::-1], (0, 0, 1))
        rotation_centre = get_com(label_orient, real_coords=True)

        
        rotation_transform = sitk.VersorRigid3DTransform()
        rotation_transform.SetCenter(rotation_centre)
        rotation_transform.SetRotation(rotation_axis, rotation_angle)
        # rotation_matrix = np.eye(3)
        # rotation_matrix[:,-1] = cardiac_axis[::-1]
        # rotation_transform = sitk.AffineTransform(3)
        # rotation_transform.SetMatrix(rotation_matrix.flatten())

    if verbose:
        print("  Alignment computed.")
        print("    Cardiac axis:    ", cardiac_axis)
        # print("    Rotation axis:   ", rotation_axis)
        print("    Rotation angle:  ", rotation_angle)
        # print("    Rotation centre: ", rotation_centre)
    overall_transform_list.append(rotation_transform)

    for label in label_list:
        working_contours[label] = sitk.Resample(
            working_contours[label],
            rotation_transform,
            sitk.sitkNearestNeighbor,
            0,
            working_contours[label].GetPixelID(),
        )
    
    """
    Module 2 - LV orientation alignment
    We use a very simple optimisation regime to enable robust computation of the LV apex
    We compute the vector from the MV COM to the LV apex
    This will be used for orientation (i.e. the long axis)
    """
    optimiser_tol_radians = optimiser_tol_degrees * np.pi / 180

    n = 0
    if verbose:
        print("Module 2: LV orientation alignment.")
        print("  Optimiser tolerance (degrees) =", optimiser_tol_degrees)
        print("  Beginning alignment process")

    while n < optimiser_max_iter and np.abs(rotation_angle) > optimiser_tol_radians:
        n += 1

        # Find the LV apex
        # lv_locations = np.where(sitk.GetArrayViewFromImage(working_contours[label_left_ventricle]))
        # lv_apex_z = lv_locations[0].min()
        # lv_apex_y = lv_locations[1][lv_locations[0] == lv_apex_z].mean()
        # lv_apex_x = lv_locations[2][lv_locations[0] == lv_apex_z].mean()
        # lv_apex_loc = np.array([lv_apex_x, lv_apex_y, lv_apex_z])

        # Get the MV COM
        mv_com = np.array(get_com(working_contours[label_mitral_valve], real_coords=True))
        
        lv_points = np.array(np.where(sitk.GetArrayViewFromImage(working_contours[label_left_ventricle]).transpose(2,1,0))).T
        # print(lv_points.shape)
        # print(lv_points[0].tolist())
        lv_points_physical = np.array([working_contours[label_left_ventricle].TransformContinuousIndexToPhysicalPoint(p.tolist()) for p in lv_points])
        distances = np.linalg.norm(lv_points_physical - mv_com, axis=1)
        lv_apex_loc_img = lv_points_physical[np.argmax(distances)]
        lv_apex_loc = working_contours[label_left_ventricle].TransformPhysicalPointToContinuousIndex(lv_apex_loc_img)


        # Define the LV axis
        # lv_apex_loc_img = np.array(
        #     working_contours[label_left_ventricle].TransformContinuousIndexToPhysicalPoint(
        #         lv_apex_loc.tolist()
        #     )
        # )
        lv_axis = lv_apex_loc_img - mv_com

        
        # Compute the rotation parameters
        rotation_axis = np.cross(lv_axis, (0, 0, 1))
        rotation_angle = vector_angle(lv_axis, (0, 0, 1))
        rotation_centre = 0.5 * (
            mv_com + lv_apex_loc_img
        )  # get_com(working_contours[label_left_ventricle], real_coords=True)

        rotation_transform = sitk.VersorRigid3DTransform()
        rotation_transform.SetCenter(rotation_centre)
        rotation_transform.SetRotation(rotation_axis, rotation_angle)

        overall_transform_list.append(rotation_transform)

        if verbose:
            
            print("    N:               ", n)
            print("    LV apex:         ", lv_apex_loc_img)
            print("    MV COM:          ", mv_com)
            print("    LV axis:         ", lv_axis)
            print("    Rotation axis:   ", rotation_axis)
            print("    Rotation centre: ", rotation_centre)
            print("    Rotation angle:  ", rotation_angle)

        tmp_lv = sitk.Resample(
            working_contours[label_left_ventricle],
            rotation_transform,
            sitk.sitkNearestNeighbor,
            0,
            working_contours[label_left_ventricle].GetPixelID(),
        )
        tmp_mv = sitk.Resample(
            working_contours[label_mitral_valve],
            rotation_transform,
            sitk.sitkNearestNeighbor,
            0,
            working_contours[label_mitral_valve].GetPixelID(),
        )
        # if lv is moved to 0th slice add translation
        if sitk.GetArrayFromImage(tmp_lv)[0].sum() > 0:
            if verbose:
                print("    Adding translation to move LV to 0th slice")
            translation_transform = sitk.TranslationTransform(3, [0, 0, -spacing[0]*10])
            composite_transform = sitk.CompositeTransform([rotation_transform, translation_transform])
            overall_transform_list.append(translation_transform)
            current_transform = composite_transform
        elif sitk.GetArrayFromImage(tmp_lv).sum() == 0:
            break
        elif sitk.GetArrayFromImage(tmp_mv).sum() < 1000:
            break
        elif sitk.GetArrayFromImage(tmp_lv)[-1].sum() > 0:
            if verbose:
                print("    Adding translation to move LV to last slice")
            translation_transform = sitk.TranslationTransform(3, [0, 0, spacing[0]*10])
            composite_transform = sitk.CompositeTransform([rotation_transform, translation_transform])
            overall_transform_list.append(translation_transform)
            current_transform = composite_transform
        
        else:
            current_transform = rotation_transform
        # print(f"slice -1: {sitk.GetArrayFromImage(tmp_mv)[-1].sum()}")
        # print(f"sum: {sitk.GetArrayFromImage(tmp_mv).sum()}")
        # Apply the transformation within the bounding box
        for label in label_list:
            working_contours[label] = sitk.Resample(
                working_contours[label],
                current_transform,
                sitk.sitkNearestNeighbor,
                0,
                working_contours[label].GetPixelID()
            )
        

    # allign rv and lv in short axis view
    com_lv_ = get_com(working_contours[label_left_ventricle], real_coords=True)
    com_rv_ = get_com(working_contours[label_right_ventricle], real_coords=True)
    com_lv = np.array(com_lv_)[:-1]
    com_rv = np.array(com_rv_)[:-1]

    angle = np.arctan2(com_lv[1]-com_rv[1], com_lv[0]-com_rv[0])
    rotation_centre = com_lv_
    rotation_angle = angle
    rotation_axis = [0, 0, 1]
    rotation_transform = sitk.VersorRigid3DTransform()
    rotation_transform.SetCenter(rotation_centre)
    rotation_transform.SetRotation(rotation_axis, rotation_angle)

    for label in label_list:
        working_contours[label] = sitk.Resample(
            working_contours[label],
            rotation_transform,
            sitk.sitkNearestNeighbor,
            0,
            working_contours[label].GetPixelID()
        )

    overall_transform_list.append(rotation_transform)
    # Compute the total transform
    overall_transform = sitk.CompositeTransform(overall_transform_list)
    inverse_transform = overall_transform.GetInverse()

    if (not transform_exists) and (transform_name is not None):
        sitk.WriteTransform(overall_transform, transform_name)
    """
    Module 3 - Compute the myocardium for the whole LV volume

    Divide this volume into thirds (from MV COM -> LV apex)        
    """

    if verbose:
        print("Module 3: Myocardium generation.")

    # First, let's just extract the myocardium
    # label_lv_inner2 = sitk.BinaryErode(working_contours[label_left_ventricle], erode_img)
    # label_lv_myo2 = working_contours[label_left_ventricle] - label_lv_inner2

    # # Mask the myo to a dilation of the blood pool
    # # This helps improve shape consistency
    # label_lv_myo_mask2 = sitk.BinaryDilate(label_lv_inner2, erode_img)
    # label_lv_myo2 = sitk.Mask(label_lv_myo2, label_lv_myo_mask2)

    # Computing limits for division into thirds
    # [xstart, ystart, zstart, xsize, ysize, zsize]
    # For the limits, we will use the centre of mass of the MV to the LV apex
    # The inner limit is used to assign the top portion (basal) of the LV to the anterior segment
    myo_cropped = crop_to_roi(myo_seg, cb_size, cb_index)
    lv_cropped = crop_to_roi(lv_seg, cb_size, cb_index)
    
    label_lv_myo =  sitk.Resample(
                myo_cropped,
                overall_transform,
                sitk.sitkNearestNeighbor,
                0,
                myo_cropped.GetPixelID(),
            )
    label_lv =  sitk.Resample(
                lv_cropped,
                overall_transform,
                sitk.sitkNearestNeighbor,
                0,
                lv_cropped.GetPixelID(),
            )
    label_lv_inner = label_lv
    lsf.Execute(label_lv_inner)
    _, _, inf_limit_lv, _, _, extent = lsf.GetRegion(1)
    inf_limit_lv += 3

    if sitk.GetArrayFromImage(working_contours[label_mitral_valve]).sum() > 0:
        com_mv, _, _ = get_com(working_contours[label_mitral_valve])
    else:
        com_mv = sitk.GetArrayFromImage(label_lv_myo).nonzero()[0].max()
    com_mv = int(com_mv)
    
    label_combined = label_lv+label_lv_myo
    com_ = get_com(label_combined)
    y_0, x_0 = com_[1:]
    
    good_slices = []
    for n in range(inf_limit_lv, com_mv):
        label_lv_myo_slice = label_lv_myo[:, :, n]
        arr_lv_myo_slice = sitk.GetArrayFromImage(label_lv_myo_slice)
        loc_y, loc_x = np.where(arr_lv_myo_slice)
        theta = -np.arctan2(loc_y - y_0, loc_x - x_0)
        # Convert to [0,2*np.pi]
        theta[theta < 0] += 2 * np.pi
        if check_360(theta):
            good_slices.append(n)
    com_mv_360 = good_slices[-1]
    inf_limit_lv_360 = good_slices[0]
    
    extent = com_mv_360 - inf_limit_lv_360
    dc = int(extent / 3)

    # Define limits (cut LV into thirds)
    apical_extent = inf_limit_lv_360 + dc
    mid_extent = inf_limit_lv_360 + 2 * dc
    
    # TODO: basal extent moved to top of lvm for coverage, but can be changed to com_mv, to only cover up to there
    basal_extent = com_mv_360 #sitk.GetArrayFromImage(label_lv_myo).nonzero()[0].max().item() #com_mv  # more complete coverage
    output_contours["ranges"] = [inf_limit_lv, apical_extent, mid_extent, basal_extent]

    if verbose:
        print("  Apex (long axis) slice:      ", inf_limit_lv)
        print("  Apical section extent slice: ", apical_extent)
        print("  Mid section extent slice:    ", mid_extent)
        print("  Basal section extent slice:  ", basal_extent)
        print("    DeltaCut (DC): ", dc)
        print("    Extent:        ", extent)

    # Segment 17
    label_lv_myo_apex = label_lv_myo * 1  # make a copy
    label_lv_myo_apex[:, :, inf_limit_lv_360:] = 0

    # The apical segment
    label_lv_myo_apical = label_lv_myo * 1  # make a copy
    label_lv_myo_apical[:, :, :inf_limit_lv_360] = 0
    label_lv_myo_apical[:, :, apical_extent:] = 0

    # The mid segment
    label_lv_myo_mid = label_lv_myo * 1  # make a copy
    label_lv_myo_mid[:, :, :apical_extent] = 0
    label_lv_myo_mid[:, :, mid_extent:] = 0

    # The basal segment
    label_lv_myo_basal = label_lv_myo * 1  # make a copy
    label_lv_myo_basal[:, :, :mid_extent] = 0
    label_lv_myo_basal[:, :, basal_extent:] = 0
    
    # out of basal extent
    label_lv_myo_out = label_lv_myo * 1  # make a copy
    label_lv_myo_out[:, :, :basal_extent] = 0


    """
    Module 4 - Generate 17 segments

        1. Find the basal (anterior) insertion of the RV
            This defines theta_0
        2. Find the baseline angle for the apical section
            This defines thera_0_apical
        3. Iterate though each section (apical, mid, basal):
            a. Convert each myocardium label loc to polar coords
            b. Assign each label to the appropriate LV segments
    """

    if verbose:
        print("Module 4: Segment generation.")

    # We need the angle for the basal RV insertion
    # This is the most counter-clockwise RV location
    # First, retrieve the most basal 5 slices
    loc_rv_z, loc_rv_y, loc_rv_x = np.where(
        sitk.GetArrayViewFromImage(working_contours[label_right_ventricle])
    )
    loc_rv_z_basal = np.arange(mid_extent, mid_extent + 5)

    if verbose:
        print("  RV basal slices: ", loc_rv_z_basal)

    theta_rv_insertion = []
    for z in loc_rv_z_basal:
        # Now get all the x and y positions
        loc_rv_basal_x = loc_rv_x[np.where(np.in1d(loc_rv_z, z))]
        loc_rv_basal_y = loc_rv_y[np.where(np.in1d(loc_rv_z, z))]

        # Now define the LV COM on each slice
        lv_com = get_com(working_contours[label_left_ventricle][:, :, int(z)])
        lv_com_basal_x = lv_com[1]
        lv_com_basal_y = lv_com[0]

        # Compute the angle
        theta_rv = np.arctan2(lv_com_basal_y - loc_rv_basal_y, loc_rv_basal_x - lv_com_basal_x)
        theta_rv[theta_rv < 0] += 2 * np.pi
        theta_rv_insertion.append(theta_rv.min())

    theta_0 = np.median(theta_rv_insertion)
    if theta_0 > np.pi: # This is a bit of a hack
        theta_0 -= 2 * np.pi



    if verbose:
        print("  RV insertion angle (basal section): ", theta_0)

    # We also need the angle in the apical section for accurate segmentation
    
    #TODO: finetune by adding some value
    theta_0_apical = theta_0 + np.pi/3
    if verbose:
        print(" Apical LV-RV COM angle: ", theta_0_apical)

    for i in range(17):
        working_contours[i + 1] = 0 * working_contours[label_heart]

    working_contours[17] = sitk.And(label_lv_myo_apex, label_lv_myo)

    if verbose:
        print("  Computing apical segments")
    # We are now going to compute the segments in cylindical sections
    # First up - apical slices
    

    atlas_points = {}
    # com_ = get_com_with_z_range(label_combined, apical_extent, mid_extent)
    # y_0, x_0 = com_[1:]

    for n in range(inf_limit_lv, apical_extent):
        label_lv_myo_slice = label_lv_myo[:, :, n]

        # We will need numpy arrays here
        arr_lv_myo_slice = sitk.GetArrayViewFromImage(label_lv_myo_slice)
        # Get the locations of the myocardium
        loc_y, loc_x = np.where(arr_lv_myo_slice)

        # Compute the angle(s)
        theta = -np.arctan2(loc_y - y_0, loc_x - x_0) - theta_0_apical
        # Convert to [0,2*np.pi]
        theta[theta < 0] += 2 * np.pi

        # Compute the radii
        radii = np.sqrt((loc_y - y_0) ** 2 + (loc_x - x_0) ** 2)

        # Define the list of contour IDs and their corresponding angles
        contour_angles = [(13, (5*np.pi/4, 7*np.pi/4), False),
                          (14, (1*np.pi/4, 7*np.pi/4), True),
                          (15, (1*np.pi/4, 3*np.pi/4), False),
                          (16, (3*np.pi/4, 5*np.pi/4), False)]
        
        for contour_id, (angle1, angle2), cw in contour_angles:
            working_contours[contour_id][:, :, n] = extract(
                label_lv_myo_slice,
                theta,
                radii,
                angle1,
                angle2,
                loc_x,
                loc_y,
                cw=cw,
            )
        if n == inf_limit_lv:
            # find the intersection of the basal segments
            # this will be used to define the short axis
            # we will use the intersection of the anterior and inferior segments
            for i in range(len(contour_angles)):
                angle = contour_angles[i][1][0]
                x, y = get_intersection(arr_lv_myo_slice, angle, theta, radii, loc_x, loc_y)
                atlas_points[f"apical_inner_{contour_angles[i][0]}_{contour_angles[i-1][0]}"] = (x, y, n)
        elif n == apical_extent - 1:
            # find the intersection of the basal segments
            # this will be used to define the short axis
            # we will use the intersection of the anterior and inferior segments
            for i in range(len(contour_angles)):
                angle = contour_angles[i][1][1]
                x, y = get_intersection(arr_lv_myo_slice, angle, theta, radii, loc_x, loc_y)
                atlas_points[f"apical_outer_{contour_angles[i][0]}_{contour_angles[i-1][0]}"] = (x, y, n)

        
    if verbose:
        print("  Computing mid segments")
    # Second up - mid slices
    for n in range(apical_extent, mid_extent):
        label_lv_myo_slice = label_lv_myo[:, :, n]

        # We will need numpy arrays here
        arr_lv_myo_slice = sitk.GetArrayViewFromImage(label_lv_myo_slice)
        loc_y, loc_x = np.where(arr_lv_myo_slice)

        # Now the origin
        # y_0, x_0 = get_com(label_lv_myo_slice)

        # Compute the angle(s)
        theta = -np.arctan2(loc_y - y_0, loc_x - x_0) - theta_0
        # Convert to [0,2*np.pi]
        theta[theta < 0] += 2 * np.pi

        # Compute the radii
        radii = np.sqrt((loc_y - y_0) ** 2 + (loc_x - x_0) ** 2)

        # Define the list of contour IDs and their corresponding angles
        contour_angles = [(8, (0, np.pi / 3)),
                          (9, (1 * np.pi / 3, 2 * np.pi / 3)),
                          (10, (2 * np.pi / 3, np.pi)),
                          (11, (np.pi, 4 * np.pi / 3)),
                          (12, (4 * np.pi / 3, 5 * np.pi / 3)),
                          (7, (5 * np.pi / 3, 2 * np.pi))]
        
        for contour_id, (angle1, angle2) in contour_angles:
            working_contours[contour_id][:, :, n] = extract(
                label_lv_myo_slice,
                theta,
                radii,
                angle1,
                angle2,
                loc_x,
                loc_y,
                # radius_min=15,
                # min_area_mm2=min_area_mm2,
            )
        if n == apical_extent:
            # find the intersection of the basal segments
            # this will be used to define the short axis
            # we will use the intersection of the anterior and inferior segments
            for i in range(len(contour_angles)):
                angle = contour_angles[i][1][0]
                x, y = get_intersection(arr_lv_myo_slice, angle, theta, radii, loc_x, loc_y)
                atlas_points[f"mid_inner_{contour_angles[i][0]}_{contour_angles[i-1][0]}"] = (x, y, n)
        elif n == mid_extent - 1:
            # find the intersection of the basal segments
            # this will be used to define the short axis
            # we will use the intersection of the anterior and inferior segments
            for i in range(len(contour_angles)):
                angle = contour_angles[i][1][1]
                x, y = get_intersection(arr_lv_myo_slice, angle, theta, radii, loc_x, loc_y)
                atlas_points[f"mid_outer_{contour_angles[i][0]}_{contour_angles[i-1][0]}"] = (x, y, n)
    

    if verbose:
        print("  Computing basal segments")
    # Third up - basal slices
    for n in range(mid_extent, basal_extent):
        label_lv_myo_slice = label_lv_myo[:, :, n]

        # We will need numpy arrays here
        arr_lv_myo_slice = sitk.GetArrayViewFromImage(label_lv_myo_slice)
        loc_y, loc_x = np.where(arr_lv_myo_slice)

        if arr_lv_myo_slice.sum() == 0:
            continue

        # Now the origin
        # y_0, x_0 = get_com(label_lv_myo_slice)

        # Compute the angle(s)
        theta = -np.arctan2(loc_y - y_0, loc_x - x_0) - theta_0
        # Convert to [0,2*np.pi]
        theta[theta < 0] += 2 * np.pi

        # Compute the radii
        radii = np.sqrt((loc_y - y_0) ** 2 + (loc_x - x_0) ** 2)

        # Now assign to different segments
        # TODO combine into for loop with zip(list, angles)

        # Define the list of contour IDs and their corresponding angles
        contour_angles = [(2, (0, np.pi / 3)),
                          (3, (1 * np.pi / 3, 2 * np.pi / 3)),
                          (4, (2 * np.pi / 3, np.pi)),
                          (5, (np.pi, 4 * np.pi / 3)),
                          (6, (4 * np.pi / 3, 5 * np.pi / 3)),
                          (1, (5 * np.pi / 3, 2 * np.pi))]

        # Iterate over the list of contour IDs and angles
        for contour_id, (angle1, angle2) in contour_angles:
            working_contours[contour_id][:, :, n] = extract(
                label_lv_myo_slice,
                theta,
                radii,
                angle1,
                angle2,
                loc_x,
                loc_y,
            )
        """
        if n == basal_extent - 1:
            # find the intersection of the basal segments
            # this will be used to define the short axis
            # we will use the intersection of the anterior and inferior segments
            too_short_seg = []
            for i in range(len(contour_angles)):
                angle = contour_angles[i][1][0]
                x, y = get_intersection(arr_lv_myo_slice, angle, theta, radii, loc_x, loc_y)
                if x is None:
                    too_short_seg.append(i)
                    continue
                atlas_points[f"basal_outer_{contour_angles[i][0]}_{contour_angles[i-1][0]}"] = (x, y, n)
        
    for i in too_short_seg:
        seg1 = sitk.GetArrayViewFromImage(working_contours[contour_angles[i][0]])
        seg2 = sitk.GetArrayViewFromImage(working_contours[contour_angles[i-1][0]])
        intersection = np.logical_and(binary_dilation(seg1), binary_dilation(seg2))
        z_int, y_int, x_int = np.where(intersection)
        z = np.percentile(z_int, 99)
        x = np.mean(x_int[z_int == z])
        y = np.mean(y_int[z_int == z])
        atlas_points[f"basal_outer_{contour_angles[i][0]}_{contour_angles[i-1][0]}"] = (x, y, z)
        """
    """
    Module 5 - re-orientation into image space

    We perform the total inverse transformation, and paste the labels back into the image space
    """

    if verbose:
        print("  Module 5: Re-orientation.")

    image_np = np.zeros(contours[label_heart].GetSize()[::-1])
    image_crop = image_np[cb_index[2]:cb_index[2] + working_contours[1].GetSize()[2],
                          cb_index[1]:cb_index[1] + working_contours[1].GetSize()[1],
                          cb_index[0]:cb_index[0] + working_contours[1].GetSize()[0]]
    for segment in reversed(range(17)):
        new_structure = working_contours[segment + 1]
        image_seg = sitk.GetArrayFromImage(working_contours[segment + 1])
        image_crop[image_seg > 0] = segment + 1

    image_out = sitk.GetArrayFromImage(label_lv_myo_out)
    image_crop[image_out > 0] = -1
    
    combined_image = sitk.GetImageFromArray(image_np)
    combined_image.CopyInformation(contours[label_heart])
    
    
    output_rot = sitk.Resample(
        combined_image,
        inverse_transform,
        sitk.sitkNearestNeighbor,
        0,
        combined_image.GetPixelID(),
    )

    apex_nnz = np.where(sitk.GetArrayViewFromImage(label_lv_myo_apex))
    ind = np.where(apex_nnz[0]==apex_nnz[0].min())
    apex_ind = [apex_nnz[i][ind].mean() for i in range(3)]
    apex_point = label_lv_myo_apex.TransformContinuousIndexToPhysicalPoint(apex_ind[::-1])
    apex_point_orig = overall_transform.TransformPoint(apex_point)
    apex_coordinates = contours[label_heart].TransformPhysicalPointToContinuousIndex(apex_point_orig)

    output_contours["lv17"] = output_rot
    output_contours["theta0"] = theta_0
    output_contours["theta0_apical"] = theta_0_apical
    output_contours["com_yx_short_long_axis"] = (y_0, x_0)
    output_contours["apex_coordinates_np"] = (apex_coordinates[2],
                                           apex_coordinates[1], 
                                           apex_coordinates[0])
    output_contours["apex_coordinates_physical"] = apex_point_orig
    output_np = sitk.GetArrayFromImage(output_rot)
    # atlas_points = {"apex": (apex_coordinates[2],
    #                          apex_coordinates[1], 
    #                          apex_coordinates[0])}

    atlas = {"indices": {}, # stores atlas point in np indices
             "points":  {}} # stores atlas point in physical coordinates
    # for key, value in atlas_points.items():
    #     point = label_lv_myo_apex.TransformContinuousIndexToPhysicalPoint(value)
    #     # TODO: fix index-point
    #     point_transformed =  overall_transform.TransformPoint(point)
    #     coordinates = contours[label_heart].TransformPhysicalPointToContinuousIndex(point_transformed)
    #     atlas["indices"][key] = (coordinates[2], coordinates[1], coordinates[0])
    #     atlas["points"][key] = point_transformed
   


    coms = center_of_mass(np.ones_like(output_np), output_np, range(1,18))
    for segment in range(17):
        com = coms[segment]
        atlas["indices"][f"segment_com_{segment + 1}"] = (com[0], com[1], com[2])
        atlas["points"][f"segment_com_{segment + 1}"] = contours[label_heart].TransformContinuousIndexToPhysicalPoint(com[::-1])

     # Convert images to numpy arrays
    label_lv_myo_np = sitk.GetArrayFromImage(myo_seg)
    output_rot_np = sitk.GetArrayFromImage(output_rot)
    output_rot_copy = output_rot_np.copy()
    # Find the labeled pixels in the output_rot_np
    labeled_pixels_output_rot = np.argwhere(output_rot_np)
    labeled_pixels_lv_myo = np.argwhere(label_lv_myo_np)
    # Create a KDTree for efficient nearest neighbor search
    tree = KDTree(labeled_pixels_output_rot)

    for pixel in labeled_pixels_lv_myo:
        # If the corresponding pixel in output_rot_copy is not labeled
        if output_rot_copy[tuple(pixel)] == 0:
            # Find the closest labeled pixel in output_rot_copy
            dist, idx = tree.query(pixel)
            if dist > 2:
                continue
            closest_labeled_pixel = labeled_pixels_output_rot[idx]

            # Assign the label of the closest labeled pixel to the unlabeled pixel
            output_rot_copy[tuple(pixel)] = output_rot_np[tuple(closest_labeled_pixel)]

    output_contours["lv17"] = sitk.GetImageFromArray(output_rot_copy)
    output_contours["lv17"].CopyInformation(contours[label_heart])
    #     output_contours[f"Ventricle_L_Segment{segment + 1}"] = new_structure
    output_contours["atlas"] = atlas
    if orig_img is not None:
        transformed_img = sitk.Resample(
            orig_img,
            overall_transform,
            sitk.sitkNearestNeighbor,
            0,
            orig_img.GetPixelID(),
        )
        output_contours["lv_seg_axis_view"] = combined_image
        output_contours["image_axis_view"] = transformed_img
    
    if verbose:
        print("Complete!")

    return output_contours

def generate_lvm_17_mesh(path,segmentation_path):
    #label_myo_path = os.path.join(path, "segmentations", "total_seg", "total_seg.nii.gz")
    mesh_path = os.path.join(path, "surfaces", "myocardium.vtk")
    utils.convert_label_map_to_surface(segmentation_path, mesh_path, segment_id=1)

    mesh = utils.read_vtk_mesh(mesh_path)
    mesh = utils.decimate_vtk_mesh(mesh, 20000)
    lv17_path = os.path.join(path, "segmentations","lv17","lv17.nii.gz")
    lv17 = sitk.ReadImage(lv17_path)
    lv_np = sitk.GetArrayFromImage(lv17).transpose(2,1,0)
    labeled_pixels = np.argwhere(lv_np)
    # Create a KDTree for efficient nearest neighbor search
    tree = KDTree(labeled_pixels)
    # loop though points in mesh
    scalars = np.zeros(mesh.GetNumberOfPoints())
    for i in range(mesh.GetNumberOfPoints()):
        point = mesh.GetPoint(i)
        index = lv17.TransformPhysicalPointToContinuousIndex(point)
        dist, idx = tree.query(index)
        scalars[i] = lv_np[tuple(labeled_pixels[idx])]
        # scalars[i] = lv17[[idx[0], idx[1], idx[2]]]
    # sampler = utils.read_vtk_and_get_sampler(os.path.join(path,r"segmentations\lv17\lv17.nii.gz"), use_nn=True)

    utils.set_scalars_on_vtk_mesh(mesh, scalars)
    utils.write_vtk_mesh(mesh, os.path.join(path, "surfaces", "myocardium_17.vtk"))

def wrap_lv_segments(path, segmentation_path, image_path=None, individual_transforms=True, verbose=False):
    #if os.path.exists(os.path.join(path,"segmentations","lv17","lv17.nii.gz")):
    #    return None

    total_path = segmentation_path #os.path.join(path, "heartchambers_highres", "heartchambers_highres.nii.gz")
    label = sitk.ReadImage(total_path)

    #label_myo_path = os.path.join(path, "segmentations", "trabec_myocardium", "trabec_myocardium.nii.gz")
    #label_myo = sitk.ReadImage(label_myo_path)

    contours = {}
    contours["Ventricle_L"] = sitk.Or(label == 1, label == 3)
    contours["Atrium_L"] = label == 2
    contours["Ventricle_R"] = label == 5   
    contours["Heart"] = sitk.And(label > 0, label < 6)

    transform_name = os.path.join(path, r"lv17_transform.txt") if individual_transforms else \
                     os.path.join(os.sep.join(path.split(os.sep)[:-1]), r"lv_transform.txt")
    
    
    orig_img = sitk.ReadImage(image_path) if image_path is not None else None

    myocardium = label == 1
    inner_lv = label == 3

    os.makedirs(os.path.join(path, "segmentations","lv17"), exist_ok=True)

    outputs = generate_lv_segments(contours, 
                                       myocardium,
                                       inner_lv,
                                       verbose=verbose, 
                                       transform_name=transform_name,
                                       orig_img=orig_img)
    
    sitk.WriteImage(outputs["lv17"], os.path.join(path,"segmentations","lv17","lv17.nii.gz"))
    output_constants = {"theta0": outputs["theta0"], 
                        "theta0_apical": outputs["theta0_apical"], 
                        "ranges": outputs["ranges"],
                        "apex_coordinates_np": outputs["apex_coordinates_np"],
                        "apex_coordinates_physical": outputs["apex_coordinates_physical"],
                        "com_yx_short_long_axis": outputs["com_yx_short_long_axis"]}
    # save as segmentation information as json
    with open(os.path.join(path, "segmentations","lv17", "lv17.json"), "w") as f:
        json.dump(output_constants, f)

    # save atlas points
    os.makedirs(os.path.join(path, "misc"), exist_ok=True)
    with open(os.path.join(path, "misc", "atlas.json"), "w") as f:
        json.dump(outputs["atlas"], f)

    # save atlas points as vtk
    atlas_points = outputs["atlas"]["points"]
    file = os.path.join(path, "misc", "atlas_points.txt")
    with open(file, "w") as f:
        f.write("points\n")
        f.write(str(len(atlas_points))+"\n")
        for key, value in atlas_points.items():
            f.write(" ".join([str(v) for v in value])+"\n")

    generate_lvm_17_mesh(path,segmentation_path)
    return outputs


def wrap_lv_all(split, current_save_dir, exists_ok=True):

    for i in range(len(split)):
        path = os.path.join(current_save_dir, split["pseudonymized_id"].iloc[i])

        if os.path.isfile(os.path.join(path, "segmentations", "lv17", "lv17.nii.gz")) and exists_ok:
            continue

        print(f"17 segment of: {split['pseudonymized_id'].iloc[i]}", end="\r")
        wrap_lv_segments(path)
        print(f"17 segment of: {split['pseudonymized_id'].iloc[i]} -- done")


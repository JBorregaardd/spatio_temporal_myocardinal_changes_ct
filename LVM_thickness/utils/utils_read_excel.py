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


def read_excel_for_best_phase(root, file_must_exist=True, get_ES=False, exclude_file_consisting=None, file=None):
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
        file_exists = [os.path.isfile(os.path.join(root, "NIFTI", f.strip())) if file_must_exist else True for f in split.filename]
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
                    for i in range(len(sub_split_sorted)//20):
                        if (all([os.path.isfile(os.path.join(root, "NIFTI", f.strip())) for f in sub_split.filename]) if file_must_exist else True) and \
                        ((sub_split_sorted.percentage.iloc[i*20:(i+1)*20]==np.arange(0,100,5)).all()): 
                            splits.append(sub_split_sorted.iloc[i*20:(i+1)*20])
                    if (all([os.path.isfile(os.path.join(root, "NIFTI", f.strip())) for f in sub_split.filename]) if file_must_exist else True) and \
                        ((sub_split_sorted.percentage.iloc[-20:]==np.arange(0,100,5)).all()): 
                            splits.append(sub_split_sorted.iloc[-20:])
    return splits

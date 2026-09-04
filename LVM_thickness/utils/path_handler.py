import os
import socket

def get_path_and_root(dataset_name="CFA-1"):
    user = os.getlogin()
    
    computer_name = socket.gethostname()

    # GPU-4
    if computer_name.find("GPU-4") != -1:
        RH = True

        if dataset_name == "MI":
            root = "/storage/DTU-Infarct-1"
            folder = "/data/DTU-Infarct-1"
        elif dataset_name == "CFA-1": 
            root = "/storage/DTU-CFA-1"
            folder = "/data/DTU-CFA-1"
        elif dataset_name == "CFA-3":
            root = "/storage/DTU-CFA-3"
            folder = "/data/DTU-CFA-3"
        elif dataset_name == "CGPS":
            root = "/storage/DTU-CGPS-1"
            folder = "/data"
        elif dataset_name == "NSTEMI":
            root = "/storage/DTU-NSTEMI"
            folder = "/data"
            
    # local user
    elif user == 'lowes':
        RH = False
        root = r"C:\Users\lowes\OneDrive\Skrivebord\DTU\phd\DTU-CFA-PILOT-1"
        folder = os.path.join(root, "computations")
    # DTU user
    elif user == 'mmilo':
        RH = False
        if dataset_name == "NSTEMI":
            root = r"C:\phd\data\DTU-NSTEMI"
            folder = os.path.join(root, "computations")
        elif dataset_name == "CGPS":
            root = r"C:\phd\data\DTU-CGPS-1"
            folder = os.path.join(root, "computations")
        elif dataset_name == "PILOT":
            root = r"C:\phd\data\DTU-CFA-PILOT-1"
            folder = os.path.join(root, "computations_best_phase")
        else:
            root = r"C:\phd\data\DTU-CFA-PILOT-1"
            folder = os.path.join(root, "computations")
    # RH user
    else: 
        RH = True
        if dataset_name == "MI":
            root = r"I:\DTU-Infarct-1"
            folder = r"E:\DTUTeams\mml\MI_data"
        elif dataset_name == "CFA-1": 
            root = r"H:\DTU-CFA-1"
            folder = r"E:\DTUTeams\mml\CFA_1"
        elif dataset_name == "CFA-3":
            root = r"J:\DTU-CFA-3"
            folder = r"E:\DTUTeams\mml\CFA_3"
        elif dataset_name == "CGPS":
            root = "/storage/DTU-CGPS-1"
            folder = "/data/"

    assert os.path.exists(root), f"Root path does not exist: {root}\nMake sure CFA-1 is mounted on H: and MI_data is mounted on G:"

    return RH, root, folder


import glob
import pandas as pd
import numpy as np
import re

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
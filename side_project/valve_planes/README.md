# Valve planes

Side project for identifying the cardiac phase from CT. Method 1: decide whether the aortic and mitral valves are open
or closed from en-face images of each valve.

`valve_planes.py` produces, per scan, one image of the aortic valve and one of the mitral valve, viewed along the
direction of blood flow:

1. **Segmentation** – TotalSegmentator `heartchambers_highres` (multi-label, `--ml`). Skipped if the segmentation
   file already exists. Labels used: 2 = left atrium, 3 = left ventricle, 6 = aorta.
2. **Contact surface** – Euclidean distance transforms (in mm) to each of the two structures; voxels within 1.5 mm of
   both form the contact surface. The threshold is raised (2, 3, 4, 5 mm) if fewer than 50 voxels are found.
   Aortic valve: LV/aorta. Mitral valve: LA/LV.
3. **Centre** – mean of the contact voxels, weighted by `exp(-(d_up + d_down) / threshold)`.
4. **Normal** – weighted PCA of the contact voxels; the axis of least variance, flipped so it points from the upstream
   to the downstream structure (centroids within 20 mm of the centre).
5. **Sampling** – three 60 × 60 mm views at 0.5 mm resolution (120 × 120 px) through the valve centre, chosen with
   `--views`:
   - `orthogonal` (default): `en_face` (perpendicular to the normal, looking along the flow), `long_axis_a` (contains
     the normal and the direction towards the other valve, so both valves are in view, similar to a three-chamber view)
     and `long_axis_b` (contains the normal, perpendicular to `long_axis_a`). Blood flows towards the top of the
     long-axis views. En-face shows the opening area; the long-axis views show the leaflets themselves (across the
     annulus when closed, along the wall when open).
   - `parallel`: en-face slices at −2, 0 and +2 mm along the normal (`--slab-offset-mm`).
6. **Windowing** – HU clipped to a window and scaled to 0–255. With `--window adaptive` (default) the window is
   `[myocardium − c, blood + c/2]`, where myocardium and blood are the scan's median HU in the myocardium and in LA/LV/aorta
   and `c` is their difference, so myocardium and blood get the same grey level in every scan regardless of the amount
   of contrast agent (ImageCAS blood pools range from ~400 to ~800 HU). `--window fixed` uses `--hu-min`/`--hu-max`
   (default −200 to 600 HU). The window used is stored in the JSON file.
7. **Output** – per valve `<id>_<valve>.png` (the three views as R, G, B channels) and `<id>_<valve>_<view>.png` (each
   view in grayscale), plus `<id>_valves.json` (centre, normal, in-plane axes, threshold used, settings and channel
   order) and `valve_planes.log`.
8. **Batch** – every scan in the folder is processed; failures are logged and the run continues. Scans that already
   have a JSON file are skipped unless `--overwrite` is given.

## Usage

```bash
# ImageCAS layout (data/1_200/<id>.img.nii.gz, data/TotalSegmentator/<id>.heart.nii.gz), first 10 scans
uv run side_project/valve_planes/valve_planes.py data/1_200 data/TotalSegmentator side_project/valve_planes/output --limit 10

# Other naming, e.g. TotalSegmentator output folders per patient
uv run side_project/valve_planes/valve_planes.py <ct_folder> <seg_folder> <out_folder> \
    --image-pattern "{id}.nii.gz" --segmentation-pattern "{id}/heartchambers_highres.nii.gz"
```

See `--help` for field of view, resolution, slab offset and HU window. TotalSegmentator is not a project dependency;
it must be on `PATH` (or passed with `--totalsegmentator`) only if segmentations are missing.

## CGPS end-diastolic / end-systolic dataset

`make_phase_dataset.py` builds a balanced dataset from the CGPS 10-year file lists in
`/storage/Data/DTU-CGPS-10y-august-2026/Filelists` (`..._ED_CE_fullpath.txt`: `volume_1`, end-diastolic;
`..._ES_CE_fullpath.txt`: `volume_0`, end-systolic). It draws `--n-patients` random patients (`--seed`, default 42)
that are in both lists and symlinks (or with `--copy`, copies) their scans to `<output>/ED/<id>.nii.gz` and
`<output>/ES/<id>.nii.gz`, with `dataset.csv` (`patient_id,phase,path,source_path`). The default output is
`/data/Data/JERI/data/CGPS10_ED_ES`.

```bash
uv run side_project/valve_planes/make_phase_dataset.py --n-patients 100
uv run side_project/valve_planes/valve_planes.py /data/Data/JERI/data/CGPS10_ED_ES/ED \
    /data/Data/JERI/data/CGPS10_ED_ES/ED/segmentations /data/Data/JERI/data/CGPS10_ED_ES/valve_planes/ED \
    --image-pattern "{id}.nii.gz"
```

Every patient appears in both phases, so split train and test sets by patient, not by scan.

## Labelling

`label_valves.py` serves a local page that shows the three views of both valves per patient and writes each label
straight to `<output>/labels.csv` (`patient_id,aortic,mitral,updated`, values `open`/`closed`/`unclear`):

```bash
uv run side_project/valve_planes/label_valves.py side_project/valve_planes/output
# open http://localhost:8765
```

Keys: Q/W/E aortic open/closed/unclear, A/S/D mitral open/closed/unclear, ←/→ previous/next, N next unlabelled. Once
both valves of a patient are labelled the page moves on to the next patient. Click an active button again to clear it.

## Feeding the views to a model

In `orthogonal` mode the RGB channels are not spatially aligned (only the centre pixel is shared), which breaks the
assumption of a pretrained network that the three channels describe the same point. Options, roughly in order of
preference:

1. **Shared backbone, multi-view** – pass each grayscale view (repeated to three channels) through the same pretrained
   CNN, concatenate or pool the features, and classify.
2. **Montage** – place the three views side by side in one image.
3. **RGB stack** – the `<id>_<valve>.png` files; a simple baseline.

## What to check

Look through the images of the first scans: the valve should sit roughly in the centre and an open and a closed valve
should be distinguishable by eye. If not, adjust `--field-of-view-mm`, the HU window (`--hu-min`, `--hu-max`) or the
contact thresholds (`CONTACT_THRESHOLDS_MM`).

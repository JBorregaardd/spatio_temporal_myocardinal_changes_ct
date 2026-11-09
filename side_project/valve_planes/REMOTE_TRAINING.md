# Training the valve open/closed classifier on the remote

Step-by-step guide for training an nnFoundation classifier on the labelled CGPS valve crops on `dtu-gpu-7`
(RTX 5090, 32 GB). It also explains the choices behind each step. The commands assume Linux/bash.

## Overview

```
CGPS CT + TotalSegmentator ──valve_planes.py──► <id>_valves.json (valve centre + axes) + PNG views
                                                 │
                       label_valves.py ──────────┼──► labels.csv (open / closed / unclear)
                                                 ▼
                                   export_crops.py ──► 3D crops (.b2nd) + labels.json + splits.json
                                                 ▼
                         GloViTa glovita_train ──► fine-tuned nnFoundation ViT, one model per valve
```

| Step | Where | Already done? |
| --- | --- | --- |
| 1. Build the ED/ES dataset, fetch the segmentations, blind it | this repo | yes |
| 2. `valve_planes.py` + labelling in `label_valves.py` | this repo | yes |
| 3. Export the 3D crops | this repo | **to do** |
| 4. Set up GloViTa + the pretrained weights | separate clone | **to do** |
| 5. Train, then evaluate on the test split | GloViTa | **to do** |

## Step 0: Paths used below

| What | Path on the remote |
| --- | --- |
| Data (CT, outputs, crops) | `/data/Data/JERI/data/CGPS10_ED_ES` |
| This repo | `~/spatio_temporal_myocardinal_changes_ct` |
| GloViTa clone | `~/image_classification` |

The setup commands (steps 1–3) use these paths directly. Training and testing (steps 4–6) go through `valve.py`, which
reads the data and GloViTa paths from the repo-root `.env`. Add these two lines to
`~/spatio_temporal_myocardinal_changes_ct/.env` once (see `.env.example`):

```bash
VALVE_DATA_DIR=/data/Data/JERI/data/CGPS10_ED_ES
GLOVITA_DIR=~/image_classification
```

## Step 1: Update this repo

```bash
cd ~/spatio_temporal_myocardinal_changes_ct
git pull
uv sync
uv run pytest tests/test_export_crops.py tests/test_valve_cli.py
```

`uv sync` installs `blosc2`, which writes the `.b2nd` files GloViTa reads. Run `uv sync` after every `git pull`, also
when `UV_NO_SYNC=1` is set in the shell.

## Step 2: Export the 3D crops

```bash
cd ~/spatio_temporal_myocardinal_changes_ct
uv run side_project/valve_planes/export_crops.py /data/Data/JERI/data/CGPS10_ED_ES/valve_planes /data/Data/JERI/data/CGPS10_ED_ES/crops --key-path /data/Data/JERI/data/CGPS10_ED_ES/blinded_key.csv
```

The script ends with a line per valve such as `mitral: 160 labelled scans {'closed': 70, 'open': 90}`, followed by the
closed/open counts of the validation part of each fold and of the test set. Write these counts down; they decide whether
the training makes sense (see "What to watch out for").

Output:

```
/data/Data/JERI/data/CGPS10_ED_ES/crops/mitral/images/S001.b2nd   float32 array, shape (1, 96, 96, 96)
/data/Data/JERI/data/CGPS10_ED_ES/crops/mitral/dataset.json       {"num_classes": 2, "subtask": "multiclass"}
/data/Data/JERI/data/CGPS10_ED_ES/crops/mitral/labels.json        {"S001": 1, ...}   0 = closed, 1 = open
/data/Data/JERI/data/CGPS10_ED_ES/crops/mitral/splits.json        [{"train": [...], "val": [...], "test": [...]}, ...]  one per fold
/data/Data/JERI/data/CGPS10_ED_ES/crops/aortic/...                the same for the aortic valve
```

### From 2D images to 3D crops

The labelling used three 2D views per valve (en face plus two long-axis views, each 50 × 50 mm). These PNGs are **not**
the input to the model. nnFoundation is pretrained on 3D CT/MR volumes, so the model gets a 3D cube cut straight from
the original CT:

- **Location and orientation**: the same valve centre and axes as the PNG views, read from `<id>_valves.json`. The cube
  is resampled with its axes along `in_plane_u`, `in_plane_v` and the valve normal. Every crop therefore has the same
  orientation: the valve plane lies across the first two axes, and blood flows along the last axis (from LV to aorta,
  from LA to LV).
- **Size**: 96 × 96 × 96 voxels at 0.5 mm, i.e. a 48 mm cube, close to the 50 mm field of view of the PNG views. 96 is
  divisible by the patch and downsampling sizes of both nnFoundation encoders.
- **Intensities**: normalised per scan so the median myocardium is 0 and the median blood pool is 1. This is the same
  idea as the adaptive window of the PNGs (it is computed from the stored `hu_window`), so scans with different amounts
  of contrast agent look alike. Nothing is clipped, so no information is lost. GloViTa does not normalise intensities
  itself, so this has to happen in the export.
- **No slab averaging or denoising**: the slab only helped the human reading the PNGs. The 3D network sees all slices
  and can learn its own filtering.

Why 3D rather than the PNG views: the views are not spatially aligned (only the centre pixel is shared), so stacking
them as RGB channels breaks what a pretrained network assumes. A 3D crop contains all three views and everything in
between.

### Which scans are included

- Only scans labelled `open` or `closed` go into `labels.json` and `splits.json`. `unclear` and unlabelled scans are left
  out (their crop is still written to `images/` but never used).
- Each valve is its own dataset with its own labels and splits; a scan can have a usable mitral label and an unclear
  aortic label.
- The phase (ED/ES) is not used anywhere. It is only in `blinded_key.csv`, which is read solely to know which scans
  belong to the same patient.

### Test set and 5-fold cross-validation

- **20 % test, by patient**, held out once and shared by all folds. Images were drawn independently from the ED and ES
  lists, so some patients have two scans. Both scans of such a patient always land in the same part; otherwise the model
  could recognise the patient rather than the valve state, and the scores would be too optimistic.
- **The other 80 % is split into 5 folds** (`--folds`, default 5). Fold `k` trains on the patients of the other four
  folds and validates on its own, so every non-test scan is used for validation exactly once. `--folds 1` writes the
  older single 60/20/20 train/validation/test split instead.
- **Why cross-validation**: the dataset is small, so one validation split of ~15–30 scans gives a noisy number, where
  one or two scans move balanced accuracy by about 0.05. Five folds give the mean and spread over all non-test scans,
  which is the validation result to report. It does not make a single model better; the gain on the test set comes from
  averaging the five fold models (step 6).
- **Stratified**: patients are grouped by their set of labels (only closed, only open, or one of each), and every group
  is cut and dealt over the folds separately, so the open/closed ratio is about the same in the test set and every fold.
- **Fixed seed** (`--seed`, default 42): rerunning the export gives the same split. With the same seed, the test set is
  exactly the one of the older single split, so test results stay comparable.
- **Validation** is used during training to pick each fold's best checkpoint. **Test** is not touched during training;
  it is evaluated once at the end (step 6).

## Step 3: Set up GloViTa (once)

GloViTa is the training code from MIC-DKFZ that the supervisor pointed to. It is kept as its own clone next to this
repo, not copied into it, so it can be updated from upstream and this repo stays free of a large training framework.
Pin the commit that was tested locally:

```bash
cd ~
git clone https://github.com/MIC-DKFZ/image_classification.git
cd ~/image_classification
git checkout 025e5942f5b6471889e5b8d1e6355ab4a331e402
uv venv
uv pip install -e .
uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line must print `True` and `NVIDIA GeForce RTX 5090`.

- **cu128**: the RTX 5090 (Blackwell) needs a PyTorch build for CUDA 12.8 or newer. The `cu121` in the supervisor's
  guide does not support this GPU.
- **`uv run --no-sync`**: a plain `uv run` re-syncs the environment and may replace the CUDA build of torch with the
  default one. Always use `--no-sync` in the GloViTa folder.

### Pretrained weights

```bash
cd ~/image_classification
uv pip install huggingface_hub
uv run --no-sync hf download MIC-DKFZ/nnFoundationViT checkpoint_final.pth --local-dir checkpoints/nnFoundationViT
uv run --no-sync hf download MIC-DKFZ/nnFoundationCNN checkpoint_final.pth adaptation_plan.json \
    --local-dir checkpoints/nnFoundationCNN
```

**Why the ViT**: the nnFoundation paper finds the CNN (ResEnc, 102M parameters) best for segmentation and detection,
and the ViT (PRIMUS, 674M parameters) best for classification, which is our task. The CNN is kept as a comparison and
as a fallback if the ViT does not fit in memory. GloViTa reads the architecture from the checkpoint, so the same
command works for both; only `--model.encoder.checkpoint_path` changes.

## Step 4: Check the setup and run a smoke test

All training and testing goes through `valve.py` in this repo, run from the repo root. It builds the long GloViTa
commands from `valve_config.yaml` and the `.env` paths, and runs GloViTa with the Python of its own `.venv` directly (not
through `uv run`), so the CUDA build of torch is never re-synced away and an active conda environment does not matter.

```bash
cd ~/spatio_temporal_myocardinal_changes_ct
uv run side_project/valve_planes/valve.py check
uv run side_project/valve_planes/valve.py smoke mitral
```

- `check` verifies the paths, that GloViTa's torch sees the GPU, that the pretrained weights are downloaded, that
  `splits.json` has at least `cv_folds` folds, and prints the closed/open counts of the test set and of every fold. It
  warns about a fold whose validation part holds only one class.
- `smoke` trains 2 short epochs of fold 0 only in the terminal (the `smoke` section of `valve_config.yaml`).
- While it runs, check the memory use in a second terminal with `nvidia-smi`. If it is far below 32 GB, the batch size
  can be raised (`batch_size` in `valve_config.yaml`, or `--batch-size` for one run).
- Every command has `--help`, and `train`/`smoke` accept `--dry-run` to only print the GloViTa command.

### Why mirroring only on axes 0 and 1

Flipping the crop is a standard augmentation. Axes 0 and 1 lie in the valve plane, and a mirrored valve is still a
valid valve. Axis 2 is the flow direction: flipping it would put the aorta below the LV (or the LV below the LA), so the
leaflets of an open valve would appear to point against the flow. That is an image that never occurs in reality and
blurs the open/closed difference, so axis 2 is not mirrored.

`glovita_train` cannot set this itself: `--data.augmentation.train_kwargs` does not exist on its command line, because
the field defaults to an empty dict and the CLI library only creates options for keys that already exist. `valve.py`
therefore starts training through `train_glovita.py`, which takes the `glovita_train` arguments plus `--mirror_axes`
(from `mirror_axes` in `valve_config.yaml`) and sets it in the config.

## Step 5: Train

```bash
uv run side_project/valve_planes/valve.py train mitral
uv run side_project/valve_planes/valve.py logs mitral       # follow the log, Ctrl+C to stop following
uv run side_project/valve_planes/valve.py runs              # all runs with settings and validation metrics
```

- `train` starts in the background and keeps running after logging out. The log is written to
  `~/image_classification/valve_logs/`. Add `--foreground` to run in the terminal instead.
- One `train` trains all 5 folds one after another (`cv_folds` in `valve_config.yaml`), so it takes about 5 times as
  long as one model. `--folds 1` trains fold 0 only, e.g. for a quick look.
- Start with the **mitral** valve, then `train aortic`.
- Settings come from `valve_config.yaml`. To try something for one run, override it on the command line and add a note
  that `runs` shows, e.g. `train mitral --lr 1e-5 --note "lower lr"`. Options: `--lr`, `--epochs`, `--folds`,
  `--batch-size`, `--encoder vit|cnn`, `--linear-probe`, `--precision`.
- `bf16` is the GloViTa default and is more stable than `fp16` on the 5090; the local RTX 2070 needs `--precision fp16`.
- Each epoch is one pass over the training set.
- **The best checkpoint is chosen by validation balanced accuracy**, separately in each fold. Not AUROC: AUROC ignores
  the decision threshold, so a model that ranks the validation scans correctly but calls them all closed already scores
  1.0. That happened in the first aortic run at epoch 1, and as GloViTa only replaces the best checkpoint on a strictly
  higher value, that useless checkpoint was kept. Validation loss would be smoother, but GloViTa can only select on its
  logged metrics. There is no early stopping; the best checkpoint has the same effect.
- Full fine-tuning (the GloViTa default) updates all weights. With only ~100 training scans this can overfit. If
  validation balanced accuracy rises in the first epochs and then falls, compare with `--linear-probe`, which only
  trains the classification head on top of the frozen pretrained encoder.
- The number of epochs and the learning rate are reasonable starting points, not tuned values. Compare settings on
  validation (`runs`), never on the test split.

Results land in `~/image_classification/experiments/generic_3d_dataset/<timestamp>_nnfoundationencoder_.../`, with one
folder per fold (`0/` … `4/`) holding `metrics.json` (per epoch), `final_val_metrics.json` (the best checkpoint on that
fold's validation patients), `checkpoints/` and the full `resolved_config.json`. `checkpoints/` holds only `best.pt`
and the final `last.pt`, each ~2.7 GB of weights without the optimizer state: GloViTa itself would save weights plus
optimizer state (~8 GB) for every epoch and never delete them, so `train_glovita.py` replaces its checkpoint saving. `runs` shows one row per run with the
mean ± standard deviation of the folds' best validation values and the mean final F1 and balanced accuracy.

## Step 6: Evaluate on the test split

Only once the settings are final (no more changes based on test results):

```bash
uv run side_project/valve_planes/valve.py test mitral
uv run side_project/valve_planes/valve.py errors mitral
```

- `test` evaluates the latest mitral run (`--run <part of the run name>` for another run) and prints accuracy, balanced
  accuracy, macro F1, AUROC and the confusion matrix. It saves `test_metrics.json`, `test_predictions.pt` and
  `test_predictions.csv` in the run folder. A run that was already tested shows its stored results instead of
  evaluating again (`--force` to override), so the test split is not used by accident.
- **Ensemble of the folds**: every fold's best checkpoint predicts P(open) for each test scan, and the five
  probabilities are averaged; the scan is called open when the mean is above 0.5 (GloViTa's `infer` does this when
  given the run folder). Each fold model learned from a slightly different 80 % of the patients, so their errors partly
  cancel. `test` refuses to run before all folds have a best checkpoint.
- `glovita_infer` itself only computes accuracy and F1 and ignores `balanced_acc` and `auroc`; `valve.py test`
  therefore computes all metrics from the saved per-scan predictions.
- `errors` lists the misclassified test scans sorted by predicted probability of open (`--all` for every scan). A
  probability near 0.5 points to a borderline scan; a confident error points to a label or crop worth checking in the
  labelling PNGs.
- `errors --show-phase` unblinds the scans with `blinded_key.csv` (patient and phase per scan) and prints the metrics
  within ED and within ES, for the phase-shortcut check below.

### Plots

```bash
uv run side_project/valve_planes/valve.py plots mitral
```

Writes PNGs to `<run folder>/plots/` (open them in VS Code over SSH):

- `training_curves_fold<k>.png`, one per fold (`training_curves.png` without cross-validation): train and validation
  loss per epoch, and below it validation AUROC, balanced accuracy and F1, with the best checkpoint marked. Works during
  and after training, before `test`.
- `confusion_matrix.png`: the test confusion matrix with counts and the share of each true class (after `test`).
- `misclassified_scans.png`: one row per misclassified test scan with the three labelling views (as shown in
  `label_valves.py`) and the same three central slices of the 3D crop the model saw, plus the manual label, the
  prediction and P(open). The most confidently correct closed and open test scans are added as references
  (`--no-reference` to leave them out). Use it to judge whether the label was wrong, the scan is genuinely unclear, or
  the model failed. `--all` shows every test scan (`test_scans.png`); `--show-phase` adds patient and phase.

The labelling PNGs are read from `<VALVE_DATA_DIR>/valve_planes`; set `VALVE_PLANES_DIR` in `.env` if they are
elsewhere. The model-input slices are single slices, while the long-axis labelling views are averaged over a thin slab,
so they look slightly noisier.

## What to watch out for

- **Class balance**: look at the counts from step 2. If one class has fewer than ~20 scans, the test split holds only
  ~4 of them, and its metrics are mostly noise. Report balanced accuracy and macro F1 (and AUROC next to them), not
  plain accuracy.
- **A model that always predicts one class** has a balanced accuracy of 0.5. That is what the 2-epoch local smoke test
  on ImageCAS did, as expected with 2 open aortic valves out of 195.
- **Phase as a shortcut**: open/closed correlates with phase (aortic open in ES, mitral open in ED), and the ES scans
  tend to be grainier (lower dose outside the target phase). The model could learn "grainy → systole" instead of the
  leaflets. After unblinding, compare the test performance within ED and within ES scans separately using
  `blinded_key.csv`.
- **Unclear labels**: report how many scans were `unclear` per valve and per phase, as they are excluded from training
  and testing.
- **ImageCAS** is not usable for training this classifier (aortic 193 closed / 2 open); it was only used to test the
  pipeline locally.

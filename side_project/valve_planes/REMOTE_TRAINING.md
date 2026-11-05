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

The commands below use these paths directly.

## Step 1: Update this repo

```bash
cd ~/spatio_temporal_myocardinal_changes_ct
git pull
uv sync
uv run pytest tests/test_export_crops.py
```

`uv sync` installs `blosc2`, which writes the `.b2nd` files GloViTa reads.

## Step 2: Export the 3D crops

```bash
cd ~/spatio_temporal_myocardinal_changes_ct
uv run side_project/valve_planes/export_crops.py /data/Data/JERI/data/CGPS10_ED_ES/valve_planes /data/Data/JERI/data/CGPS10_ED_ES/crops --key-path /data/Data/JERI/data/CGPS10_ED_ES/blinded_key.csv
```

The script ends with a line per valve such as `mitral: 160 labelled scans {'closed': 70, 'open': 90}`. Write these
counts down; they decide whether the training makes sense (see "What to watch out for").

Output:

```
/data/Data/JERI/data/CGPS10_ED_ES/crops/mitral/images/S001.b2nd   float32 array, shape (1, 96, 96, 96)
/data/Data/JERI/data/CGPS10_ED_ES/crops/mitral/dataset.json       {"num_classes": 2, "subtask": "multiclass"}
/data/Data/JERI/data/CGPS10_ED_ES/crops/mitral/labels.json        {"S001": 1, ...}   0 = closed, 1 = open
/data/Data/JERI/data/CGPS10_ED_ES/crops/mitral/splits.json        {"train": [...], "val": [...], "test": [...]}
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

### Train / validation / test split

- **60 % train, 20 % validation, 20 % test**, by patient. Images were drawn independently from the ED and ES lists, so
  some patients have two scans. Both scans of such a patient always land in the same split; otherwise the model could
  recognise the patient rather than the valve state, and the test score would be too optimistic.
- **Stratified**: patients are grouped by their set of labels (only closed, only open, or one of each) and every group
  is split 60/20/20 separately, so the open/closed ratio is about the same in all three splits.
- **Fixed seed** (`--seed`, default 42): rerunning the export gives the same split.
- **Validation** is used during training to pick the best checkpoint. **Test** is not touched during training; it is
  evaluated once at the end (step 6).

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

## Step 4: Smoke test (2 epochs)

Check that everything runs before starting a long job:

```bash
cd ~/image_classification
uv run --no-sync python ~/spatio_temporal_myocardinal_changes_ct/side_project/valve_planes/train_glovita.py \
    --task.metrics f1 balanced_acc auroc \
    --training.epochs 2 --training.train_steps_per_epoch 10 --training.val_steps_per_epoch 10 \
    --dataloading.batch_size 2 --dataloading.num_workers 4 \
    data:generic3d_dataset_config --data.data_root_dir /data/Data/JERI/data/CGPS10_ED_ES/crops/mitral \
    model.encoder:nnfoundation_encoder_config \
    --model.encoder.checkpoint_path checkpoints/nnFoundationViT/checkpoint_final.pth \
    --model.encoder.input_channels 1 --model.encoder.input_shape 96 96 96 \
    model.head:classification_head_config \
    logger:no_logger_config
```

`train_glovita.py` takes exactly the `glovita_train` arguments and only adds `mirror_axes = [0, 1]` (see below).
`glovita_train` itself cannot set this: `--data.augmentation.train_kwargs` does not exist on its command line, because
the field defaults to an empty dict and the CLI library only creates options for keys that already exist. Check in
`experiments/generic_3d_dataset/<run>/0/resolved_config.json` that `train_kwargs` is `{"mirror_axes": [0, 1]}`.

While it runs, check the memory use in a second terminal with `nvidia-smi`. If it is far below 32 GB, the batch size
can be raised in step 5.

### Why mirroring only on axes 0 and 1

Flipping the crop is a standard augmentation. Axes 0 and 1 lie in the valve plane, and a mirrored valve is still a
valid valve. Axis 2 is the flow direction: flipping it would put the aorta below the LV (or the LV below the LA), so the
leaflets of an open valve would appear to point against the flow. That is an image that never occurs in reality and
blurs the open/closed difference, so axis 2 is not mirrored.

## Step 5: The real training run

```bash
cd ~/image_classification
VALVE=mitral
nohup uv run --no-sync python ~/spatio_temporal_myocardinal_changes_ct/side_project/valve_planes/train_glovita.py \
    --task.metrics f1 balanced_acc auroc \
    --optimizer.name AdamW --optimizer.lr 1e-4 --optimizer.weight_decay 0.01 --optimizer.warmup_epochs 5 \
    --training.epochs 100 --training.precision bf16 --training.seed 42 \
    --training.best_checkpoint_metric val_AUROC \
    --dataloading.batch_size 2 --dataloading.num_workers 8 \
    data:generic3d_dataset_config --data.data_root_dir /data/Data/JERI/data/CGPS10_ED_ES/crops/$VALVE \
    model.encoder:nnfoundation_encoder_config \
    --model.encoder.checkpoint_path checkpoints/nnFoundationViT/checkpoint_final.pth \
    --model.encoder.input_channels 1 --model.encoder.input_shape 96 96 96 \
    model.head:classification_head_config \
    logger:no_logger_config \
    > train_$VALVE.log 2>&1 &
tail -f train_$VALVE.log
```

- Start with the **mitral** valve, then run the same with `VALVE=aortic`.
- `nohup ... &` keeps the job running after logging out (or use `tmux`/`screen`).
- `bf16` is the GloViTa default and is more stable than `fp16` on the 5090; the local RTX 2070 only supports `fp16`.
- `train_steps_per_epoch` is left out, so each epoch is one pass over the training set.
- **The best checkpoint is chosen by validation AUROC**, not accuracy: AUROC does not depend on a decision threshold
  and stays meaningful when the classes are unbalanced.
- Full fine-tuning (the GloViTa default) updates all weights. With only ~100 training scans this can overfit. If
  validation AUROC rises in the first epochs and then falls, compare with linear probing (add
  `--peft.method linear_probing`), which only trains the classification head on top of the frozen pretrained encoder.
- The number of epochs and the learning rate are reasonable starting points, not tuned values.

Results land in `~/image_classification/experiments/generic_3d_dataset/<timestamp>_nnfoundationencoder_.../0/`:
`metrics.json` (per epoch), `final_val_metrics.json`, `checkpoints/` and the full `resolved_config.json`.

## Step 6: Evaluate on the test split

Only once the settings are final (no more changes based on test results):

```bash
cd ~/image_classification
RUN=experiments/generic_3d_dataset/<run folder>
uv run --no-sync glovita_infer --exp_dir $RUN --checkpoint best \
    --metrics f1 balanced_acc auroc --pred_output $RUN/test_predictions.pt \
    data:generic3d_dataset_config --data.data_root_dir /data/Data/JERI/data/CGPS10_ED_ES/crops/$VALVE
```

- `glovita_infer`'s own options (`--metrics`, `--checkpoint`, `--pred_output`) must come **before**
  `data:generic3d_dataset_config`; options after a subcommand are applied to that subcommand.
- `infer` uses the `test` entry of `splits.json` automatically.
- `--pred_output` saves the per-scan predictions, probabilities and case ids, to look up the misclassified scans.

## What to watch out for

- **Class balance**: look at the counts from step 2. If one class has fewer than ~20 scans, the test split holds only
  ~4 of them, and its metrics are mostly noise. Report AUROC and balanced accuracy, not plain accuracy.
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

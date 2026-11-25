> Guidance for autonomous coding agents
> Read this before writing, editing, or executing anything in this repo.

# Relevant commands

* The project uses `uv` for management of virtual environments. This means:
  * To install packages, use `uv add <package-name>`.
  * To run Python scripts, use `uv run <script-name>.py`.
  * To run other commands related to Python, prefix them with `uv run `, e.g., `uv run <command>`.
* The project uses `pytest` for testing. To run tests, use `uv run pytest tests/`.
* The project uses `ruff` for linting and formatting:
    * To format code, use `uv run ruff format .`.
    * To lint code, use `uv run ruff check . --fix`.
* The project uses `invoke` for task management. To see available tasks, use `uv run invoke --list` or refer to the
    `tasks.py` file.
* The valve open/closed classifier side project is trained and tested with GloViTa (a separate clone) through
    `uv run side_project/valve_planes/valve.py <command>` (`check`, `smoke`, `train`, `logs`, `runs`, `test`,
    `errors`, `plots`, `review`, `classify`). `classify --images <ct folder> --segmentations <seg folder>
    --run <name> --workers 8` classifies both valves of a new unlabelled dataset in parallel (defaults: `IMAGE_FOLDER`,
    `SEGMENTATION_FOLDER`, `IMAGE_PATTERN`, `SEGMENTATION_PATTERN` in `.env`) and writes
    `<VALVE_DATA_DIR>/classify/<run>/<run>.csv` with an open/closed column per valve. `review` lists the labelled scans where the model disagrees with `labels.csv`; relabel
    them with `uv run side_project/valve_planes/label_valves.py <planes folder> --only <ids file>`. Settings are in `side_project/valve_planes/valve_config.yaml`, paths in `.env`
    (`VALVE_DATA_DIR`, `GLOVITA_DIR`, optional `VALVE_PLANES_DIR`). See `side_project/valve_planes/REMOTE_TRAINING.md`.
* Repeated measures statistics between CGPS1 (baseline) and CGPS10 (follow-up) live in `repeated_measures/`. The
    spaghetti plots (one for women, one for men) are made with
    `uv run repeated_measures/spaghetti_plot.py --measurement lvm_g`; it needs
    `df/CGPS1_CGPS10_links.csv` (from `comparative_analysis/compare.py`, which also adds the point in the R-R interval
    of each scan from the series descriptions in `COMPARISON_DATA_FILE`; a paired t-test of the two
    R-R percentages is run with `uv run repeated_measures/rr_ttest.py`, saved in
    `df/repeated_measures/rr_ttest.csv`) and each dataset's
    `output/<dataset>_<mm>mm/statistics/statistics_patient_summary.csv` (from `main_statistics_df.py`). The lines are
    coloured by `df/repeated_measures/CGPS1_HT.csv` (with the mean and 95% CI of each HT class on top), made with `uv run repeated_measures/ht_classifier.py
    --threshold-mm 15` (HT when the CGPS1 `lvm_average_thickness_mm` is above the threshold). The change
    (CGPS10 - CGPS1) of every patient summary feature is computed with `uv run repeated_measures/deltas.py`, saved
    in `df/repeated_measures/deltas.csv` (together with the CGPS1 and CGPS10 values). The correlation matrices of the
    CGPS1 values, the CGPS10 values and the deltas of the linked subjects are made in one run in R with
    `Rscript repeated_measures/correlation_matrix.R pearson` (or `spearman`; needs the R package `corrplot`). The
    bullseye plots of the change in median wall thickness (%) per AHA segment (1x4: women and men with HT, then
    without HT; apex greyed out) are made with `uv run repeated_measures/bullseye/bullseye_segment_delta.py`; it needs the links table,
    `CGPS1_HT.csv` and each dataset's `output/<dataset>_0mm/statistics/statistics_thickness_per_segment.csv` and
    `statistics_patient_summary.csv`. The change per subject on the mesh itself is made in three steps in
    `repeated_measures/bullseye/`, which share `--n 20 --seed 0` (the same random subjects), `--subjects`, `--all` and
    `--workers`: `uv run repeated_measures/bullseye/mesh_delta.py --all --workers 8` registers CGPS10 rigidly onto
    CGPS1 with ICP, moves the CGPS10 thickness to the nearest CGPS1 endocardial vertex and saves each subject's
    difference mesh and the registration QC in `output/CGPS1_CGPS10_mesh_delta_0mm/` (needs the links table and both
    datasets' `output/<dataset>_0mm/<id>/` pipeline folders); `uv run repeated_measures/bullseye/bullseye_mesh_delta.py`
    projects the raw difference meshes into the CGPS1 bullseye and plots them (`--keep-nifti` keeps the projection's
    NIfTI images); `uv run repeated_measures/bullseye/bullseye_smooth_delta.py --fwhm-mm 0 5 10 15` smooths the
    thickness on the mesh with a Gaussian (heat kernel along the wall) and plots one figure per FWHM plus a side-by-side
    comparison. Without registration, `uv run repeated_measures/bullseye/bullseye_quarter_distribution.py --n 20` projects
    each scan's thickness into its own bullseye, splits every AHA segment into four quarters (radius half x angle half)
    and plots per subject the change of the percentiles P10-P90 and the Wasserstein distance per quarter (64 cells;
    table in `output/CGPS1_CGPS10_quarters_0mm/quarter_distribution.csv`). The bias between the mesh and segment
    methods is diagnosed per segment (a third, registration-free method from the quarter points; correlation of the
    difference with the size of the change, the unmatched share and the segment's share of the wall) with
    `uv run repeated_measures/bullseye/diagnose_segment_bias.py --workers 8` (after `compare_segment_methods.py`,
    `bullseye_quarter_distribution.py --all` and, optionally, `match_distance.py`). The group maps (mean and median per
    cell over the subjects of women and men with and without HT; cells with data in less than `--coverage` of a group
    are grey) of the smoothed mesh change and of a quarter percentile change are made with
    `uv run repeated_measures/bullseye/bullseye_group_mean.py --fwhm-mm 10 --coverage 0.5 --percentile 50 --workers 8`
    (after `bullseye_smooth_delta.py --all --fwhm-mm 10` and `bullseye_quarter_distribution.py --all`; needs
    `CGPS1_HT.csv` and the patient summaries). How far the CGPS1 vertices had to reach for their CGPS10
    partner per AHA segment (share above a threshold and share unmatched, over the subjects processed so far) is
    summarised with `uv run repeated_measures/bullseye/match_distance.py --threshold-mm 3`. The mesh change is checked
    against the registration-free segment method (change of each scan's own segment medians; Bland-Altman, correlation
    and mean difference per segment) with `uv run repeated_measures/bullseye/compare_segment_methods.py`. A linear mixed model, `value ~ time * HT * sex + age_c + (1 | subject)`, of one
    patient summary measurement is fitted in R (package `nlme`) with
    `Rscript repeated_measures/LMM.R --measurement lvm_average_thickness_mm` (add `--dropouts` to also use the CGPS1
    subjects without follow-up; needs only `nlme`, which comes with R); it prints and saves the change per HT class and sex, the HT and sex interactions and an
    ANCOVA (`CGPS10 ~ CGPS1 + HT * sex + age_c`) as sensitivity analysis in `df/repeated_measures/LMM_<measurement>_*.csv`.
* The multiphase CFA-2 scans (20 phases 0-95% of the R-R interval plus the scanner's high-res best phase, in `CFA_DIR`
    in `.env`, default `/storage/DTU-CFA-2_v2`) live in `cfa/`. `uv run cfa/volume_curves.py --workers 8` measures the
    chamber volumes of every phase from the TotalSegmentator heartchambers_highres segmentations and saves
    `df/cfa/volume_curves.csv` (one row per scan) and `df/cfa/patient_curves.csv` (EDV/ESV as max/min LV volume, their
    phases, the EF and the best phase's high-res and interpolated grid LV volume). The point in the R-R interval and the
    heart rate of scans are read from the series descriptions in a metadata file with `uv run cfa/scan_phases.py`
    (default: the CGPS10 ED and ES file lists with `META_DATA_FILE`; other scans with `--metadata <csv>
    --folder NAME=<nifti folder>` or `--filelist NAME=<txt>`), saved in `df/cfa/scan_phases.csv`. Our own conversion of
    the CGPS10 proxy EF (best phase to ES at ~35% R-R) to the true EF is fitted on CFA-2 with
    `uv run cfa/ef_calibration.py` (optional `--min-ef 50`; after `volume_curves.py` and `scan_phases.py`, whose CGPS10
    ES phases weight the CFA-2 rows): it compares the paper's formula with refitted linear, linear + ES phase and
    EDV/ESV models by grouped cross-validation and saves `performance.csv`, `coefficients.csv` (bootstrap CI),
    `predictions.csv` and `ef_calibration.png` in `df/cfa/ef_calibration/`.
* The project uses `pre-commit` for managing pre-commit hooks. To run all hooks on all files, use
    `uv run pre-commit run --all-files`. For more information, refer to the `.pre-commit-config.yaml` file.

# Code style

* Follow existing code style.
* Keep line length within 120 characters.
* Use f-strings for formatting.
* Use type hints
* Do not add inline comments unless absolutely necessary.

# Documentation

* If the project has a `docs/` folder, update documentation there as needed.
* In this case the project will be using `mkdocs` for documentation. To build the docs locally, use
    `uv run mkdocs serve`
* Use existing docstring style.
* Ensure all functions and classes have docstrings.
* Use Google style for docstrings.
* Update this `AGENTS.md` file if any new tools or commands are added to the project.

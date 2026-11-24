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
    `statistics_patient_summary.csv`. The change per subject on the mesh itself (CGPS10 rigidly registered onto CGPS1
    with ICP, CGPS10 thickness moved to the nearest CGPS1 endocardial vertex, difference in mm projected into the CGPS1
    bullseye; no smoothing, no grouping) is plotted for random subjects with
    `uv run repeated_measures/bullseye/bullseye_mesh_delta.py --n 20 --seed 0`; it needs the links table and both
    datasets' `output/<dataset>_0mm/<id>/` pipeline folders, and saves each subject's difference mesh and the
    registration QC in `output/CGPS1_CGPS10_mesh_delta_0mm/`. How far the CGPS1 vertices had to reach for their CGPS10
    partner per AHA segment (share above a threshold and share unmatched, over the subjects processed so far) is
    summarised with `uv run repeated_measures/bullseye/match_distance.py --threshold-mm 3`. A linear mixed model, `value ~ time * HT * sex + age_c + (1 | subject)`, of one
    patient summary measurement is fitted in R (package `nlme`) with
    `Rscript repeated_measures/LMM.R --measurement lvm_average_thickness_mm` (add `--dropouts` to also use the CGPS1
    subjects without follow-up; needs only `nlme`, which comes with R); it prints and saves the change per HT class and sex, the HT and sex interactions and an
    ANCOVA (`CGPS10 ~ CGPS1 + HT * sex + age_c`) as sensitivity analysis in `df/repeated_measures/LMM_<measurement>_*.csv`.
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

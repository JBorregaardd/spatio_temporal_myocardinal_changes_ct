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
    `errors`, `plots`). Settings are in `side_project/valve_planes/valve_config.yaml`, paths in `.env`
    (`VALVE_DATA_DIR`, `GLOVITA_DIR`, optional `VALVE_PLANES_DIR`). See `side_project/valve_planes/REMOTE_TRAINING.md`.
* Repeated measures statistics between CGPS1 (baseline) and CGPS10 (follow-up) live in `repeated_measures/`. The
    spaghetti plots (one for women, one for men) are made with
    `uv run repeated_measures/spaghetti_plot.py --measurement lvm_g`; it needs
    `df/CGPS1_CGPS10_links.csv` (from `comparative_analysis/compare.py`) and each dataset's
    `output/<dataset>_<mm>mm/statistics/statistics_patient_summary.csv` (from `main_statistics_df.py`). The lines are
    coloured by `df/repeated_measures/CGPS1_HT.csv`, made with `uv run repeated_measures/ht_classifier.py
    --threshold-mm 15` (HT when the CGPS1 `lvm_average_thickness_mm` is above the threshold). The change
    (CGPS10 - CGPS1) of every patient summary feature is computed with `uv run repeated_measures/deltas.py`, saved
    in `df/repeated_measures/deltas.csv`. Their correlation matrix is made in R with
    `Rscript repeated_measures/delta_correlation.R pearson` (or `spearman`; needs the R package `corrplot`).
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

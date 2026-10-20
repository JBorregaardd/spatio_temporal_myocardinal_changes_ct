"""Checks that the metadata and translation table files in the repo-root .env can be found and read.

Run on the machine with the CGPS data before main_statistics_df.py, e.g. uv run pytest tests/test_env_metadata.py -v -rs
The tests are skipped when the variables are not set (e.g. locally with only ImageCAS).
"""
import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "measurements"))
sys.path.insert(0, str(REPO_ROOT / "LVM_thickness"))

from BSA import (  # noqa: E402
    SEX_CODES,
    TRANSLATION_ID_COLUMNS,
    body_surface_area,
    gender_and_age,
    load_metadata_file,
    load_translationtable_file,
)

load_dotenv(REPO_ROOT / ".env")

DATASET = "CGPS10"
FILE_VARIABLES = ["META_DATA_FILE", "TRANSLATION_DATA_FILE"]


def require_variable(name: str) -> str:
    """Value of an environment variable, skipping the test if it is not set.

    Args:
        name: Name of the environment variable.

    Returns:
        The value of the variable.
    """
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} is not set in {REPO_ROOT / '.env'} or the environment.")
    return value


def test_env_file_exists() -> None:
    assert (REPO_ROOT / ".env").is_file(), f"No .env file in the repo root {REPO_ROOT}."


@pytest.mark.parametrize("name", FILE_VARIABLES)
def test_file_in_env_exists(name: str) -> None:
    path = require_variable(name)
    assert path == path.strip().strip("\"'"), f"{name} has surrounding spaces or quotes: {path!r}"
    assert os.path.isfile(path), f"{name} points to a file that does not exist: {path!r}"
    assert os.access(path, os.R_OK), f"{name} points to a file that cannot be read: {path!r}"


def test_loaders_return_env_paths() -> None:
    metadata_file = require_variable("META_DATA_FILE")
    translation_file = require_variable("TRANSLATION_DATA_FILE")
    assert load_metadata_file() == metadata_file
    assert load_translationtable_file() == translation_file


def test_metadata_gives_bsa() -> None:
    metadata = pd.read_csv(require_variable("META_DATA_FILE"))
    missing = {"pseudonymized_id", "PatientSize", "PatientWeight"} - set(metadata.columns)
    assert not missing, f"Metadata file is missing the columns {sorted(missing)}."

    patients = body_surface_area(metadata)
    assert len(patients) > 0
    assert patients["BSA"].notna().any(), "No patient has a valid BSA."


def test_translation_table_gives_sex_and_age() -> None:
    transdata = pd.read_csv(require_variable("TRANSLATION_DATA_FILE"), dtype=str)
    id_column = TRANSLATION_ID_COLUMNS[DATASET]
    expected = {id_column, f"{id_column}-PatientSex", f"{id_column}-PatientAgeCompletedYears"}
    missing = expected - set(transdata.columns)
    assert not missing, f"Translation table is missing the columns {sorted(missing)}."

    sex_and_age = gender_and_age(transdata, DATASET)
    assert len(sex_and_age) > 0
    raw_sex = transdata[f"{id_column}-PatientSex"].dropna().str.strip().str.upper()
    unknown = sorted(set(raw_sex) - set(SEX_CODES))
    assert not unknown, f"Sex values in the translation table that are not in SEX_CODES: {unknown}"
    assert sex_and_age["sex"].notna().any(), "No patient has a sex."
    assert set(sex_and_age["sex"].dropna()) <= {0, 1}


def test_load_demographics_as_in_statistics_script() -> None:
    for name in FILE_VARIABLES:
        require_variable(name)
    from main_statistics_df import load_demographics

    demographics = load_demographics(DATASET)
    assert len(demographics) > 0
    assert demographics["bsa_m2"].notna().any(), "No patient has a BSA after merging with the translation table."
    assert demographics["age"].notna().any(), "No patient has an age after merging with the translation table."

"""Body surface area of each patient from the height and weight in the DICOM metadata file.

Run directly to save the per-patient table to <repo>/output/BSA.csv, e.g. uv run measurements/BSA.py
"""
import os

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def calculate_bsa(weight: float | pd.Series, height: float | pd.Series) -> float | pd.Series:
    """Body surface area (m^2) with the Du Bois formula.

    Args:
        weight: Weight in kg.
        height: Height in cm.

    Returns:
        Body surface area in m^2.
    """
    return 0.007184 * (weight**0.425) * (height**0.725)


def body_surface_area(metadata: pd.DataFrame) -> pd.DataFrame:
    """Height, weight and BSA of each patient in a metadata table with one row per scan series.

    The patient id is the number before ``_SERIES`` in ``pseudonymized_id``, e.g. ``0001`` in
    ``CGPS-10y-august-2026_0001_SERIES0013``. A height or weight of 0 counts as missing, and each patient gets the
    first non-missing height and weight over their series.

    Args:
        metadata: Metadata table with the columns ``pseudonymized_id``, ``PatientSize`` (m) and ``PatientWeight`` (kg).

    Returns:
        One row per patient with ``patient_id``, ``PatientSize``, ``PatientWeight``, ``height_cm`` and ``BSA`` (m^2).
    """
    df = metadata[["pseudonymized_id", "PatientSize", "PatientWeight"]].copy()
    df["patient_id"] = df["pseudonymized_id"].str.extract(r"_(\d+)_SERIES", expand=False)
    df[["PatientSize", "PatientWeight"]] = df[["PatientSize", "PatientWeight"]].replace(0, np.nan)

    patients = df.groupby("patient_id")[["PatientSize", "PatientWeight"]].first().reset_index()
    patients["height_cm"] = patients["PatientSize"] * 100
    patients["BSA"] = calculate_bsa(patients["PatientWeight"], patients["height_cm"])
    return patients


def load_metadata_file() -> str:
    """Path of the metadata file in ``META_DATA_FILE`` from the repo-root .env or the environment."""
    from dotenv import load_dotenv

    load_dotenv(os.path.join(REPO_ROOT, ".env"))
    metadata_file = os.environ.get("META_DATA_FILE")
    if not metadata_file:
        raise SystemExit("META_DATA_FILE is not set. Add it to the .env file in the repo root (see .env.example).")
    return metadata_file


def load_translationtable_file() -> str:
    """Path of the translationtable file in ``TRANSLATION_DATA_FILE`` from the repo-root .env or the environment."""
    from dotenv import load_dotenv

    load_dotenv(os.path.join(REPO_ROOT, ".env"))
    transdata_file = os.environ.get("TRANSLATION_DATA_FILE")
    if not transdata_file:
        raise SystemExit(
            "TRANSLATION_DATA_FILE is not set. Add it to the .env file in the repo root (see .env.example)."
        )
    return transdata_file


SEX_CODES = {"M": 0, "F": 1, "0": 0, "1": 1, "0.0": 0, "1.0": 1}
TRANSLATION_ID_COLUMNS = {"CGPS10": "CGPS-10y-august-2026", "CGPS1": "CGPS-1-august-2026"}


def gender_and_age(transdata: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Sex and age of each patient in the translation table at the examination of the dataset.

    ``CGPS10`` uses the 10-year follow-up and ``CGPS1`` the baseline examination. For ``CGPS10`` the id column is
    ``CGPS-10y-august-2026`` and the sex and age columns are ``CGPS-10y-august-2026-PatientSex`` and
    ``CGPS-10y-august-2026-PatientAgeCompletedYears``, and likewise with ``CGPS-1-august-2026`` for ``CGPS1``. The
    patient id is the number after the last ``_`` in the id column, e.g. ``0001`` in ``CGPS-10y-august-2026_0001``.
    Rows without an id are dropped. Sex is coded 0 for men and 1 for women, and may be given in the table either as
    ``M``/``F`` or already as ``0``/``1``.

    Args:
        transdata: Translation table with the id, sex (``M``/``F`` or ``0``/``1``) and age columns of the dataset.
        dataset: Dataset name, ``CGPS10`` or ``CGPS1``.

    Returns:
        One row per patient with ``patient_id``, ``sex`` (0 = man, 1 = woman) and ``age`` (years).

    Raises:
        ValueError: If the dataset has no columns in the translation table.
    """
    if dataset not in TRANSLATION_ID_COLUMNS:
        raise ValueError(f"No sex and age for dataset {dataset!r}, expected one of {list(TRANSLATION_ID_COLUMNS)}.")
    id_column = TRANSLATION_ID_COLUMNS[dataset]
    sex_column = f"{id_column}-PatientSex"
    age_column = f"{id_column}-PatientAgeCompletedYears"

    df = transdata[[id_column, sex_column, age_column]].dropna(subset=[id_column])
    patients = pd.DataFrame(
        {
            "patient_id": df[id_column].astype(str).str.extract(r"_(\d+)$", expand=False),
            "sex": df[sex_column].astype(str).str.strip().str.upper().map(SEX_CODES).astype("Int64"),
            "age": pd.to_numeric(df[age_column], errors="coerce"),
        }
    )
    return patients.dropna(subset=["patient_id"]).groupby("patient_id").first().reset_index()


if __name__ == "__main__":
    patients = body_surface_area(pd.read_csv(load_metadata_file()))
    output_dir = os.path.join(REPO_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "BSA.csv")
    patients.to_csv(output_file, index=False)
    print(f"Saved BSA for {len(patients)} patients to {output_file}")

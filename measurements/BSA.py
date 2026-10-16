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
        raise SystemExit("TRANSLATION_DATA_FILE is not set. Add it to the .env file in the repo root (see .env.example).")
    return transdata_file

def gender_and_age(transdata: pd.DataFrame) -> pd.DataFrame:
    df = transdata[["CGPS-10y-august-2026", "CGPS-1-august-2026-PatientSex", "CGPS-1-august-2026-PatientAgeCompletedYears"]].copy()
    



if __name__ == "__main__":
    patients = body_surface_area(pd.read_csv(load_metadata_file()))
    output_dir = os.path.join(REPO_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "BSA.csv")
    patients.to_csv(output_file, index=False)
    print(f"Saved BSA for {len(patients)} patients to {output_file}")

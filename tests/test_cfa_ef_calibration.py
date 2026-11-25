"""Tests for the CFA-2 ejection fraction calibration."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cfa"))

from ef_calibration import (  # noqa: E402
    MODELS,
    coefficient_table,
    cross_validate,
    design_table,
    es_phase_weights,
    fit_models,
    performance,
    qc_patients,
)

PHASES = np.arange(0, 100, 5.0)


def synthetic_curves(n: int = 30, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cosine-shaped LV volume curves with ED at 0%, ES at 35% and a best phase at 75%."""
    rng = np.random.default_rng(seed)
    curve_rows, patient_rows = [], []
    for i in range(n):
        edv, esv = rng.uniform(120, 200), rng.uniform(40, 90)
        volume = esv + (edv - esv) * (0.5 + 0.5 * np.cos(np.pi * np.clip(PHASES, 0, 35) / 35))
        volume = np.where(PHASES > 35, esv + (edv - esv) * 0.85 * np.clip((PHASES - 35) / 30, 0, 1), volume)
        volume[0] = edv
        curve_rows += [
            {"patient_id": f"p{i}", "series": "grid", "phase_pct": p, "lv_volume_ml": v} for p, v in zip(PHASES, volume)
        ]
        patient_rows.append(
            {
                "patient_id": f"p{i}",
                "n_grid_phases": 20,
                "edv_ml": volume.max(),
                "ed_phase_pct": 0.0,
                "esv_ml": volume.min(),
                "es_phase_pct": float(PHASES[volume.argmin()]),
                "ef_pct": (volume.max() - volume.min()) / volume.max() * 100,
                "best_phase_pct": 75.0,
                "best_lv_volume_ml": float(np.interp(75, PHASES, volume)),
            }
        )
    return pd.DataFrame(curve_rows), pd.DataFrame(patient_rows)


def test_qc_flags_implausible_curves() -> None:
    _, patients = synthetic_curves(3)
    patients.loc[0, "es_phase_pct"] = 95
    patients.loc[1, "ed_phase_pct"] = 50
    flagged = qc_patients(patients)["exclusion"]
    assert flagged[0].startswith("ES") and flagged[1].startswith("ED") and flagged[2] == ""


def test_es_phase_weights_bin_valid_cgps10_es_scans() -> None:
    scan_phases = pd.DataFrame(
        {
            "group": ["ES", "ES", "ES", "ES", "ED"],
            "rr_percent": [33.0, 36.0, 41.0, 0.9, 75.0],
            "heart_rate_bpm": [60.0, 60.0, 60.0, 60.0, 60.0],
        }
    )
    weights = es_phase_weights(scan_phases)
    assert weights.to_dict() == pytest.approx({35.0: 2 / 3, 40.0: 1 / 3})
    assert es_phase_weights(None).sum() == pytest.approx(1)


def test_design_table_interpolates_the_es_volume() -> None:
    curves, patients = synthetic_curves(2)
    table = design_table(curves, patients, pd.Series([0.5, 0.5], index=[35.0, 37.5]))
    first = table[table["patient_id"] == "p0"].set_index("es_phase_pct")
    assert first.loc[35.0, "v_es_ml"] == pytest.approx(patients.loc[0, "esv_ml"])
    assert first.loc[37.5, "v_es_ml"] > first.loc[35.0, "v_es_ml"]
    assert first.loc[35.0, "ef_proxy"] < first.loc[35.0, "ef_true"]


def test_fitted_linear_model_recovers_an_exact_relation() -> None:
    table = pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(6)],
            "es_phase_pct": 35.0,
            "weight": 1.0,
            "ef_proxy": [30.0, 35.0, 40.0, 45.0, 50.0, 55.0],
            "v_best_ml": 100.0,
            "v_es_ml": 50.0,
            "edv_ml": 120.0,
            "esv_ml": 50.0,
        }
    )
    table["ef_true"] = 0.8 * table["ef_proxy"] + 20
    assert fit_models(table)["linear"] == pytest.approx([20, 0.8])


def test_fitted_models_beat_the_uncorrected_proxy_in_cross_validation() -> None:
    curves, patients = synthetic_curves()
    table = design_table(curves, patients, es_phase_weights(None))
    results = performance(cross_validate(table, folds=5)).set_index("model")
    assert list(results.index) == list(MODELS)
    assert results.loc["linear", "rmse"] < results.loc["uncorrected", "rmse"]
    assert abs(results.loc["linear", "bias"]) < abs(results.loc["uncorrected", "bias"])
    coefficients = coefficient_table(table, n_bootstrap=20, seed=0)
    assert (coefficients["ci_low"] <= coefficients["estimate"] + 1e-9).all()

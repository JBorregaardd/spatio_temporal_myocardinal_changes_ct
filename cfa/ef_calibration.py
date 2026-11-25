#################################################################################################################################################
# This script fits our own conversion of the CGPS10 ejection fraction proxy to the true ejection fraction on the multiphase CFA-2 scans,
# like Isma'eel et al. 2009 (LVEF = 0.782 * LVEF(75_35) + 20.76), which measurements/volume.py uses now.
# In CGPS10 the "ED" scan is the scanner's high-res best diastolic phase (~75% R-R) and the "ES" scan a phase at ~35% R-R (median 35%,
# 5-95% 27-46%), so the proxy EF = (V_best - V_es) / V_best underestimates the EF. In CFA-2 the same best phase exists, and the 20 grid
# phases give the true EDV and ESV (maximum and minimum LV volume) and the LV volume at any ES phase p (interpolated).
# Every CFA-2 patient is used at each grid phase p in the CGPS10 ES range, and the rows are weighted by the share of CGPS10 ES scans at
# that phase (df/cfa/scan_phases.csv from scan_phases.py; uniform over 25-45% without it), so the fit and its evaluation mimic CGPS10.
# Models (all weighted least squares):
#   uncorrected   EF = EF_proxy
#   paper         EF = 0.782 * EF_proxy + 20.76
#   linear        EF = a + b * EF_proxy
#   linear_phase  EF = a + b * EF_proxy + c * p
#   volumes       EDV = a + b * V_best, ESV = c + d * V_es + e * p, EF from the predicted EDV and ESV
# The performance is from 10-fold cross-validation grouped by patient (bias, limits of agreement, RMSE, MAE, r of the predicted vs true
# EF) and the coefficients are fitted on all patients with a 95% CI from a patient bootstrap.
# Patients whose curve is implausible (ES phase outside 15-60% or ED phase within 20-75%) are excluded; --min-ef excludes low true EF.
# Needs df/cfa/volume_curves.csv and patient_curves.csv from volume_curves.py. Saves in <PROJECT_ROOT>/df/cfa/ef_calibration/:
# performance.csv, coefficients.csv, predictions.csv (cross-validated, one row per patient and ES phase), excluded.csv and
# ef_calibration.png (true vs proxy EF and Bland-Altman of the cross-validated predictions).
# e.g. uv run cfa/ef_calibration.py
#################################################################################################################################################

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "LVM_thickness"))
sys.path.insert(0, os.path.join(REPO_ROOT, "measurements"))
sys.path.insert(0, os.path.join(REPO_ROOT, "cfa"))

from pipeline import load_environment  # noqa: E402
from volume import MID_DIASTOLIC_EF_INTERCEPT, MID_DIASTOLIC_EF_SLOPE  # noqa: E402
from volume_curves import CURVES_FILE, OUTPUT_FOLDER, PATIENTS_FILE  # noqa: E402

SCAN_PHASES_FILE = "scan_phases.csv"
RESULT_FOLDER = "ef_calibration"
DEFAULT_ES_PHASES = (25.0, 30.0, 35.0, 40.0, 45.0)
ES_PHASE_LIMITS = (15.0, 60.0)
ED_PHASE_GAP = (20.0, 75.0)
HEART_RATE_LIMITS = (30.0, 150.0)
GRID_STEP = 5.0
MODELS = ("uncorrected", "paper", "linear", "linear_phase", "volumes")
COLORS = {
    "uncorrected": "#52514e",
    "paper": "#eb6834",
    "linear": "#2a78d6",
    "linear_phase": "#1baf7a",
    "volumes": "#6250d6",
}


def qc_patients(patients: pd.DataFrame) -> pd.DataFrame:
    """Flag the patients whose volume curve cannot be used.

    Args:
        patients: Table from ``volume_curves.patient_summary``.

    Returns:
        ``patients`` with ``exclusion`` (empty when the patient is kept): no best phase, an ES phase (minimum LV
        volume) outside 15-60% or an ED phase (maximum) within 20-75% of the R-R interval.
    """
    reasons = pd.Series("", index=patients.index)
    es_out = ~patients["es_phase_pct"].between(*ES_PHASE_LIMITS)
    ed_in = patients["ed_phase_pct"].between(*ED_PHASE_GAP)
    reasons[ed_in] = f"ED phase within {ED_PHASE_GAP[0]:.0f}-{ED_PHASE_GAP[1]:.0f}%"
    reasons[es_out] = f"ES phase outside {ES_PHASE_LIMITS[0]:.0f}-{ES_PHASE_LIMITS[1]:.0f}%"
    reasons[patients["best_lv_volume_ml"].isna() | patients["n_grid_phases"].lt(2)] = "no best phase or curve"
    return patients.assign(exclusion=reasons)


def es_phase_weights(scan_phases: pd.DataFrame | None) -> pd.Series:
    """Weight of each grid phase as ES phase: the share of CGPS10 ES scans nearest to it.

    Args:
        scan_phases: Table from ``scan_phases.py`` with ``group``, ``rr_percent`` and ``heart_rate_bpm``, or ``None``.
            ES scans with an R-R percentage outside 15-60% or a heart rate outside 30-150 bpm are ignored.

    Returns:
        Weights summing to 1, indexed by the grid phase; uniform over 25-45% without ``scan_phases``.
    """
    if scan_phases is None:
        return pd.Series(1 / len(DEFAULT_ES_PHASES), index=pd.Index(DEFAULT_ES_PHASES, name="es_phase_pct"))
    es = scan_phases[
        scan_phases["group"].eq("ES")
        & scan_phases["rr_percent"].between(*ES_PHASE_LIMITS)
        & scan_phases["heart_rate_bpm"].between(*HEART_RATE_LIMITS)
    ]
    if es.empty:
        raise ValueError("No valid ES scans in the scan phases table.")
    phases = (es["rr_percent"] / GRID_STEP).round() * GRID_STEP
    weights = phases.value_counts(normalize=True).sort_index()
    weights.index = weights.index.astype(float).rename("es_phase_pct")
    return weights.rename("weight")


def design_table(curves: pd.DataFrame, patients: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """One row per patient and ES phase with the proxy and true volumes and ejection fractions.

    Args:
        curves: Table from ``volume_curves.volume_curves``.
        patients: Kept patients from ``qc_patients``.
        weights: ES phase weights from ``es_phase_weights``.

    Returns:
        Rows with ``patient_id``, ``es_phase_pct``, ``weight``, ``v_best_ml``, ``v_es_ml`` (grid LV volume at the ES
        phase, interpolated), ``edv_ml``, ``esv_ml``, ``ef_true`` and ``ef_proxy`` (percent).
    """
    grid = curves[(curves["series"] == "grid") & curves["lv_volume_ml"].notna()]
    rows = []
    for patient in patients.itertuples():
        curve = grid[grid["patient_id"] == patient.patient_id]
        v_es = np.interp(weights.index.to_numpy(), curve["phase_pct"], curve["lv_volume_ml"], period=100)
        for phase, weight, volume in zip(weights.index, weights.to_numpy(), v_es):
            rows.append(
                {
                    "patient_id": patient.patient_id,
                    "es_phase_pct": phase,
                    "weight": weight,
                    "v_best_ml": patient.best_lv_volume_ml,
                    "v_es_ml": volume,
                    "edv_ml": patient.edv_ml,
                    "esv_ml": patient.esv_ml,
                }
            )
    table = pd.DataFrame(rows)
    table["ef_true"] = (table["edv_ml"] - table["esv_ml"]) / table["edv_ml"] * 100
    table["ef_proxy"] = (table["v_best_ml"] - table["v_es_ml"]) / table["v_best_ml"] * 100
    return table


def fit_wls(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted least squares with an intercept.

    Args:
        x: Predictors, shape (n, k).
        y: Outcome, shape (n,).
        weights: Row weights, shape (n,).

    Returns:
        Coefficients, the intercept first, shape (k + 1,).
    """
    design = np.column_stack([np.ones(len(y)), x]) * np.sqrt(weights)[:, None]
    return np.linalg.lstsq(design, y * np.sqrt(weights), rcond=None)[0]


def fit_models(table: pd.DataFrame) -> dict[str, np.ndarray]:
    """Fit the coefficients of the fitted models.

    Args:
        table: Rows from ``design_table``.

    Returns:
        ``linear``: (a, b); ``linear_phase``: (a, b, c); ``volumes``: (a, b, c, d, e), see the script header.
    """
    w = table["weight"].to_numpy()
    edv = fit_wls(table[["v_best_ml"]].to_numpy(), table["edv_ml"].to_numpy(), w)
    esv = fit_wls(table[["v_es_ml", "es_phase_pct"]].to_numpy(), table["esv_ml"].to_numpy(), w)
    return {
        "linear": fit_wls(table[["ef_proxy"]].to_numpy(), table["ef_true"].to_numpy(), w),
        "linear_phase": fit_wls(table[["ef_proxy", "es_phase_pct"]].to_numpy(), table["ef_true"].to_numpy(), w),
        "volumes": np.concatenate([edv, esv]),
    }


def predict(model: str, coefficients: dict[str, np.ndarray], table: pd.DataFrame) -> np.ndarray:
    """Predicted true ejection fraction (percent) of each row.

    Args:
        model: One of ``MODELS``.
        coefficients: Output of ``fit_models``; unused by ``uncorrected`` and ``paper``.
        table: Rows with ``ef_proxy``, ``es_phase_pct``, ``v_best_ml`` and ``v_es_ml``.

    Returns:
        The predictions.
    """
    proxy, phase = table["ef_proxy"].to_numpy(), table["es_phase_pct"].to_numpy()
    if model == "uncorrected":
        return proxy
    if model == "paper":
        return MID_DIASTOLIC_EF_SLOPE * proxy + MID_DIASTOLIC_EF_INTERCEPT
    c = coefficients[model]
    if model == "linear":
        return c[0] + c[1] * proxy
    if model == "linear_phase":
        return c[0] + c[1] * proxy + c[2] * phase
    edv = c[0] + c[1] * table["v_best_ml"].to_numpy()
    esv = c[2] + c[3] * table["v_es_ml"].to_numpy() + c[4] * phase
    return (edv - esv) / edv * 100


def cross_validate(table: pd.DataFrame, folds: int) -> pd.DataFrame:
    """Cross-validated predictions of every model, grouped by patient.

    Args:
        table: Rows from ``design_table``.
        folds: Number of folds (at most the number of patients).

    Returns:
        ``table`` with a column ``pred_<model>`` per model.
    """
    result = table.copy()
    splitter = GroupKFold(n_splits=min(folds, table["patient_id"].nunique()))
    for train, test in splitter.split(table, groups=table["patient_id"]):
        coefficients = fit_models(table.iloc[train])
        for model in MODELS:
            result.loc[result.index[test], f"pred_{model}"] = predict(model, coefficients, table.iloc[test])
    return result


def performance(predictions: pd.DataFrame) -> pd.DataFrame:
    """Weighted agreement of each model's prediction with the true EF.

    Args:
        predictions: Output of ``cross_validate``.

    Returns:
        One row per model with ``n_patients``, ``bias`` (mean of predicted - true), ``sd``, ``loa_low``, ``loa_high``
        (bias -/+ 1.96 sd), ``rmse``, ``mae`` and ``r`` (weighted Pearson correlation), all in EF percentage points.
    """
    w = predictions["weight"].to_numpy() / predictions["weight"].sum()
    truth = predictions["ef_true"].to_numpy()
    rows = []
    for model in MODELS:
        pred = predictions[f"pred_{model}"].to_numpy()
        diff = pred - truth
        bias = float(np.sum(w * diff))
        sd = float(np.sqrt(np.sum(w * (diff - bias) ** 2)))
        cov = np.cov(pred, truth, aweights=w)
        rows.append(
            {
                "model": model,
                "n_patients": predictions["patient_id"].nunique(),
                "bias": bias,
                "sd": sd,
                "loa_low": bias - 1.96 * sd,
                "loa_high": bias + 1.96 * sd,
                "rmse": float(np.sqrt(np.sum(w * diff**2))),
                "mae": float(np.sum(w * np.abs(diff))),
                "r": float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])),
            }
        )
    return pd.DataFrame(rows)


def coefficient_table(table: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    """Coefficients of the fitted models on all patients with a 95% CI from resampling the patients.

    Args:
        table: Rows from ``design_table``.
        n_bootstrap: Bootstrap resamples (0 skips the CI).
        seed: Random seed.

    Returns:
        One row per model and term with ``estimate``, ``ci_low`` and ``ci_high``.
    """
    names = {
        "linear": ["intercept", "ef_proxy"],
        "linear_phase": ["intercept", "ef_proxy", "es_phase_pct"],
        "volumes": ["edv_intercept", "edv_v_best", "esv_intercept", "esv_v_es", "esv_es_phase_pct"],
    }
    estimates = fit_models(table)
    rng = np.random.default_rng(seed)
    by_patient = {patient: rows for patient, rows in table.groupby("patient_id")}
    patients = np.array(list(by_patient))
    samples = {model: [] for model in names}
    for _ in range(n_bootstrap):
        resample = pd.concat([by_patient[p] for p in rng.choice(patients, len(patients))], ignore_index=True)
        for model, coefficients in fit_models(resample).items():
            samples[model].append(coefficients)
    rows = []
    for model, terms in names.items():
        low, high = (
            np.percentile(samples[model], q, axis=0) if n_bootstrap else np.full(len(terms), np.nan)
            for q in (2.5, 97.5)
        )
        rows += [
            {"model": model, "term": term, "estimate": estimates[model][i], "ci_low": low[i], "ci_high": high[i]}
            for i, term in enumerate(terms)
        ]
    return pd.DataFrame(rows)


def plot(predictions: pd.DataFrame, coefficients: dict[str, np.ndarray], phase: float, path: Path) -> None:
    """Plot the true vs proxy EF at one ES phase with the lines of the linear models, and Bland-Altman plots.

    Args:
        predictions: Output of ``cross_validate``.
        coefficients: Output of ``fit_models`` on all patients.
        phase: ES phase of the points.
        path: Output PNG.
    """
    rows = predictions[predictions["es_phase_pct"] == phase]
    models = [model for model in MODELS if model != "uncorrected"]
    fig, axes = plt.subplots(1, 1 + len(models), figsize=(4.2 * (1 + len(models)), 4.2))
    ax = axes[0]
    ax.scatter(rows["ef_proxy"], rows["ef_true"], s=14, color="#2a78d6", alpha=0.6, linewidths=0)
    x = np.linspace(rows["ef_proxy"].min(), rows["ef_proxy"].max(), 50)
    line = pd.DataFrame({"ef_proxy": x, "es_phase_pct": phase})
    for model in ("paper", "linear", "linear_phase"):
        ax.plot(x, predict(model, coefficients, line), color=COLORS[model], lw=2, label=model)
    ax.plot(x, x, color="#52514e", lw=1, ls="--", label="identity")
    ax.set(xlabel=f"Proxy EF, best phase to {phase:.0f}% (%)", ylabel="True EF, max to min volume (%)")
    ax.legend(frameon=False, fontsize=8)
    for ax, model in zip(axes[1:], models):
        pred = rows[f"pred_{model}"]
        mean, diff = (pred + rows["ef_true"]) / 2, pred - rows["ef_true"]
        bias, sd = diff.mean(), diff.std()
        ax.scatter(mean, diff, s=14, color=COLORS[model], alpha=0.6, linewidths=0)
        for value, style in ((bias, "-"), (bias - 1.96 * sd, "--"), (bias + 1.96 * sd, "--")):
            ax.axhline(value, color="#52514e", lw=1, ls=style)
        ax.axhline(0, color="#c3c2b7", lw=0.8)
        ax.set(
            title=f"{model}: bias {bias:.1f}, LoA ±{1.96 * sd:.1f}",
            xlabel="Mean of predicted and true EF (%)",
            ylabel="Predicted - true EF (%)",
        )
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"CFA-2, ES phase {phase:.0f}% (cross-validated predictions)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    root, _ = load_environment()

    parser = argparse.ArgumentParser(description="Fit the conversion of the CGPS10 proxy EF to the true EF on CFA-2.")
    parser.add_argument(
        "--input-dir", default=None, help=f"Folder of volume_curves.py (default: <PROJECT_ROOT>/{OUTPUT_FOLDER})."
    )
    parser.add_argument(
        "--scan-phases", default=None, help=f"scan_phases.py output (default: <input dir>/{SCAN_PHASES_FILE})."
    )
    parser.add_argument(
        "--min-ef", type=float, default=None, help="Exclude patients with a lower true EF (default: none)."
    )
    parser.add_argument("--folds", type=int, default=10, help="Cross-validation folds.")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap resamples for the coefficient CIs.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed of the bootstrap.")
    parser.add_argument("--plot-phase", type=float, default=35.0, help="ES phase of the plotted points.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir) if args.input_dir else Path(root) / OUTPUT_FOLDER
    for name in (CURVES_FILE, PATIENTS_FILE):
        if not (input_dir / name).is_file():
            raise SystemExit(f"{input_dir / name} not found. Run cfa/volume_curves.py first.")
    curves = pd.read_csv(input_dir / CURVES_FILE)
    patients = qc_patients(pd.read_csv(input_dir / PATIENTS_FILE))
    if args.min_ef is not None:
        patients.loc[
            (patients["exclusion"] == "") & (patients["ef_pct"] < args.min_ef), "exclusion"
        ] = f"true EF below {args.min_ef:g}%"
    excluded = patients[patients["exclusion"] != ""]
    kept = patients[patients["exclusion"] == ""]
    print(f"{len(kept)} of {len(patients)} patients kept")
    if len(excluded):
        print(excluded[["patient_id", "ed_phase_pct", "es_phase_pct", "ef_pct", "exclusion"]].to_string(index=False))

    scan_phases_path = Path(args.scan_phases) if args.scan_phases else input_dir / SCAN_PHASES_FILE
    scan_phases = pd.read_csv(scan_phases_path) if scan_phases_path.is_file() else None
    weights = es_phase_weights(scan_phases)
    source = scan_phases_path if scan_phases is not None else "uniform, no scan phases table"
    print(f"ES phase weights ({source}):\n{weights.round(3).to_string()}")

    table = design_table(curves, kept, weights)
    predictions = cross_validate(table, args.folds)
    results = performance(predictions)
    coefficients = coefficient_table(table, args.bootstrap, args.seed)

    output_dir = input_dir / RESULT_FOLDER
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "performance.csv", index=False)
    coefficients.to_csv(output_dir / "coefficients.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    excluded.to_csv(output_dir / "excluded.csv", index=False)
    if args.plot_phase in weights.index:
        plot(predictions, fit_models(table), args.plot_phase, output_dir / "ef_calibration.png")

    print(
        f"\nCross-validated agreement with the true EF (percentage points):\n{results.round(2).to_string(index=False)}"
    )
    print(f"\nCoefficients (all patients, 95% bootstrap CI):\n{coefficients.round(3).to_string(index=False)}")
    print(f"\nSaved in {output_dir}")

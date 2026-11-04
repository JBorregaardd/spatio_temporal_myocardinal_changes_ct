"""Image quality checks of contrast-enhanced cardiac CT scans, run over the whole dataset in parallel.

For every patient with both a CT image and a TotalSegmentator heart-chambers segmentation two checks are computed:

- k: the number of t-tests, out of n_iter, in which the myocardium and RV HU-values were not significantly
  different. k = 0 means there is probably contrast between the RV and the myocardium; k > 0 is evidence of
  limited contrast, and the larger k the stronger the signal towards no contrast. In HØ1 even k = 1 indicated
  non-existing or very limited contrast (with n_iter = 100). The group k = 0 may still contain some scans
  with very limited contrast.
- cnr: the contrast-to-noise ratio between the aorta core and the myocardium. Low CNR means a "bad" quality scan,
  high CNR a "good" one. Plot a histogram of the CNR distribution of the dataset to choose a cut-off; in HØ1 CNR < 5
  was used to indicate a low quality scan.

The results are written to <PROJECT_ROOT>/df/preprocessing/<dataset>_ct_quality.csv

Examples: uv run preprocessing/ct_quality_checks.py --dataset ImageCAS_1-200 --workers 4
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import edt
import numpy as np
import pandas as pd
import scipy.stats as stats
import SimpleITK as sitk

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "LVM_thickness"))

from pipeline import discover_patients, load_environment  # noqa: E402

MYO_ID = 1
RV_ID = 5
AORTA_ID = 6
SAMPLE_SIZE = 100
AORTA_CORE_MM = 5


def is_RV_contrast_image(image: np.ndarray, whs: np.ndarray, n_iter: int = 100) -> int:
    """Count how often random myocardium and RV samples have HU-values that are not significantly different.
    Each iteration draws SAMPLE_SIZE HU-values from the myocardium and from the RV and compares them with
    Welch's t-test.

    Args:
        image: 3D CT image in HU.
        whs: Whole heart segmentation with the same shape as image.
        n_iter: Number of t-tests.

    Returns:
        k, the number of t-tests with p > 0.05 out of n_iter.
    """
    myo_huvs = image[whs == MYO_ID]
    rv_huvs = image[whs == RV_ID]
    if min(myo_huvs.size, rv_huvs.size) < SAMPLE_SIZE:
        raise ValueError(f"Too few voxels to sample: myocardium {myo_huvs.size}, RV {rv_huvs.size}")

    N_accept = 0
    for _ in range(n_iter):
        sample_myo = np.random.choice(a=myo_huvs, size=SAMPLE_SIZE, replace=False)
        sample_rv = np.random.choice(a=rv_huvs, size=SAMPLE_SIZE, replace=False)
        ttest = stats.ttest_ind(a=sample_myo, b=sample_rv, equal_var=False)
        if ttest.pvalue > 0.05:
            N_accept += 1

    return N_accept


def check_quality(image: np.ndarray, whs: np.ndarray, spacing: tuple[float, float, float], parallel: int = 8) -> tuple[float, float, float]:
    """Signal-to-noise ratio, contrast-to-noise ratio and coefficient of variation of the aorta core.

    The aorta core is the part of the aorta more than AORTA_CORE_MM mm from its wall, which avoids partial volume
    voxels at the wall. Its mean HU is the signal and its standard deviation the noise; the median HU of the
    myocardium is the background. CNR is thus how big the difference between blood and muscle is in "noise units",
    e.g. (400 - 100) / 30 = 10.

    Args:
        image: 3D CT image in HU, indexed (z, y, x).
        whs: Whole heart segmentation with the same shape as image.
        spacing: Voxel spacing in mm in the array's axis order (z, y, x).
        parallel: Threads used for the distance transform.

    Returns:
        Tuple of (snr, cnr, cv).
    """
    sdf_arota = -edt.sdf(whs == AORTA_ID, anisotropy=spacing, parallel=parallel)
    slim_aorta = sdf_arota < -AORTA_CORE_MM
    aorta_huvs = image[slim_aorta]

    signal_bg = np.median(image[whs == MYO_ID])
    signal = np.mean(aorta_huvs, dtype=np.float64)
    noise = np.std(aorta_huvs, dtype=np.float64)

    snr = signal / noise
    cnr = (signal - signal_bg) / noise
    cv = noise / signal

    return snr, cnr, cv


def process_patient(patient_id: str, image_path: str, segmentation_path: str, n_iter: int) -> dict[str, object]:
    """Run both quality checks for one patient.

    Args:
        patient_id: Dataset id.
        image_path: CT image path.
        segmentation_path: TotalSegmentator heart-chambers segmentation path.
        n_iter: Number of t-tests in is_RV_contrast_image.

    Returns:
        One result row; error holds the error message if the patient failed.
    """
    row: dict[str, object] = {"patient_id": patient_id, "k": None, "snr": None, "cnr": None, "cv": None, "error": None}
    try:
        image_sitk = sitk.ReadImage(image_path)
        spc_x, spc_y, spc_z = image_sitk.GetSpacing()
        image = sitk.GetArrayFromImage(image_sitk)
        whs = sitk.GetArrayFromImage(sitk.ReadImage(segmentation_path))
        if image.shape != whs.shape:
            raise ValueError(f"Image shape {image.shape} differs from segmentation shape {whs.shape}")

        row["k"] = is_RV_contrast_image(image, whs, n_iter=n_iter)
        row["snr"], row["cnr"], row["cv"] = check_quality(image, whs, spacing=(spc_z, spc_y, spc_x), parallel=1)
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compute RV contrast and CNR for every scan in the dataset.")
    parser.add_argument(
        "--dataset", required=True, help="Dataset name; results go to df/preprocessing/<dataset>_ct_quality.csv."
    )
    parser.add_argument("--workers", type=int, default=None, help="Patients processed at once (default: CPU count).")
    parser.add_argument("--n-iter", type=int, default=100, help="Number of t-tests per patient (default: 100).")
    return parser.parse_args()


def main() -> int:
    """Run the quality checks over the dataset and write the result table.

    Returns:
        Process exit code: 0 if every patient succeeded, else 1.
    """
    args = parse_args()
    root, data = load_environment()

    patients = discover_patients(data)
    if not patients:
        raise SystemExit(f"[ERROR] No patients found in {data.image_folder} and {data.segmentation_folder}")

    workers = max(1, min(args.workers or os.cpu_count() or 1, len(patients)))
    print(f"Found {len(patients)} patients, running with {workers} workers", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(process_patient, pid, data.image_path(pid), data.segmentation_path(pid), args.n_iter)
            for pid in patients
        ]
        for i, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            if row["error"]:
                print(f"[{i}/{len(patients)}] patient {row['patient_id']}: FAILED - {row['error']}", flush=True)
            else:
                print(f"[{i}/{len(patients)}] patient {row['patient_id']}: k = {row['k']}, CNR = {row['cnr']:.2f}",
                      flush=True)

    order = {pid: i for i, pid in enumerate(patients)}
    results = pd.DataFrame(rows).sort_values("patient_id", key=lambda ids: ids.map(order))
    out_dir = os.path.join(root, "df", "preprocessing")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.dataset}_ct_quality.csv")
    results.to_csv(out_path, index=False)

    failed = results["error"].notna().sum()
    print(f"Finished {len(results) - failed}/{len(results)} patients. Results: {out_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

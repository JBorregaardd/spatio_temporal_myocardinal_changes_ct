"""Run the LVM thickness pipeline for a single patient.

Also the per-patient worker that ``main_pipeline.py`` launches for every patient in the dataset.
"""
import argparse
import os
import sys
import traceback

from dotenv import load_dotenv

from pipeline import patient_paths, process_patient


def load_environment() -> tuple[str, str]:
    """Load PROJECT_ROOT and SEGMENTATION_FOLDER from the repo-root .env or the environment.

    Returns:
        Tuple of (project root, segmentation folder).
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(repo_root, ".env"))
    try:
        return os.environ["PROJECT_ROOT"], os.environ["SEGMENTATION_FOLDER"]
    except KeyError as missing:
        raise SystemExit(
            f"\n[ERROR] Missing required environment variable {missing}.\n"
            f"Set PROJECT_ROOT and SEGMENTATION_FOLDER in a .env file at the repo root "
            f"(see .env.example), or export them directly in your shell/job script."
        )


def main() -> int:
    """Parse arguments and process one patient.

    Returns:
        Process exit code: 0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(description="Run the LVM thickness pipeline for a single patient.")
    parser.add_argument(
        "--patient-id",
        default=os.environ.get("PATIENT_ID", "1"),
        help="Patient id to process (default: $PATIENT_ID from .env, else '1'). "
             "Useful for cluster array jobs, e.g. --patient-id $SLURM_ARRAY_TASK_ID.",
    )
    parser.add_argument("--output-dir", default=None, help="Parent output folder (default: <PROJECT_ROOT>/output).")
    parser.add_argument("--no-plots", action="store_true", help="Skip the histogram and bullseye figures.")
    parser.add_argument("--no-debug-meshes", action="store_true", help="Skip writing the QA-only vectors.vtk mesh.")
    args = parser.parse_args()

    root, folder = load_environment()
    paths = patient_paths(root, folder, args.patient_id, args.output_dir)

    print(f"Checking files for patient {paths.patient_id}:")
    print(f"  Image:        {paths.image} -> Exists: {os.path.exists(paths.image)}")
    print(f"  Segmentation: {paths.segmentation} -> Exists: {os.path.exists(paths.segmentation)}")
    if not (os.path.exists(paths.image) and os.path.exists(paths.segmentation)):
        print("\n[ERROR] One or both input files were not found. Please check the paths above before proceeding.")
        return 1

    print(f"\n[START] Saving results to: {paths.save_dir}", flush=True)
    try:
        timings = process_patient(
            paths,
            plots=not args.no_plots,
            save_debug_meshes=not args.no_debug_meshes,
            log=lambda msg: print(msg, flush=True),
        )
    except Exception:
        print("\n[ERROR DURING EXECUTION]")
        traceback.print_exc()
        return 1
    print(f"Done! Total {sum(timings.values()):.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

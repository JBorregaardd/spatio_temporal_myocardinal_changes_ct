"""Run the LVM thickness pipeline over the whole dataset in parallel.

Each patient runs in its own ``main_single.py`` subprocess: a native crash in VTK/ITK then fails
only that patient instead of the batch, memory is fully released between patients, and every
patient gets its own log file. Finished patients (``done.json`` present) are skipped, so an
interrupted batch resumes where it stopped.

Examples:
    uv run LVM_thickness/main_pipeline.py                       # everything, workers sized to free RAM
    uv run LVM_thickness/main_pipeline.py --patients 1 2 3 --workers 2
    uv run LVM_thickness/main_pipeline.py --shard 0/8           # cluster: this job takes every 8th patient
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

from tqdm import tqdm

from main_single import load_environment
from pipeline import PatientPaths, discover_patients, patient_paths

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER_SCRIPT = os.path.join(HERE, "main_single.py")
SUMMARY_FIELDS = ["patient_id", "status", "wall_seconds", "peak_memory_gb", "return_code", "log"]


def physical_cores() -> int:
    """Physical core count, falling back to half the logical count."""
    try:
        import psutil

        return psutil.cpu_count(logical=False) or max(1, (os.cpu_count() or 2) // 2)
    except ImportError:
        return max(1, (os.cpu_count() or 2) // 2)


def available_memory_gb() -> float | None:
    """Currently available system memory in GB, or None if psutil is unavailable."""
    try:
        import psutil

        return psutil.virtual_memory().available / 1e9
    except ImportError:
        return None


def default_workers(memory_per_worker_gb: float) -> int:
    """Largest worker count that fits both the physical cores and currently free RAM.

    Args:
        memory_per_worker_gb: Expected peak memory of one patient.

    Returns:
        Worker count, at least 1.
    """
    cores = physical_cores()
    free = available_memory_gb()
    by_memory = cores if free is None else int((free - 1.0) // memory_per_worker_gb)
    return max(1, min(cores, by_memory))


def select_patients(args: argparse.Namespace, root: str, folder: str) -> list[str]:
    """Apply --patients, --shard and --limit to the discovered patient list."""
    discovered = discover_patients(root, folder)
    if args.patients:
        missing = sorted(set(args.patients) - set(discovered))
        if missing:
            raise SystemExit(f"[ERROR] No image+segmentation pair for patient(s): {', '.join(missing)}")
        wanted = set(args.patients)
        selected = [pid for pid in discovered if pid in wanted]
    else:
        selected = discovered
    if args.shard:
        index, count = (int(x) for x in args.shard.split("/"))
        if not 0 <= index < count:
            raise SystemExit(f"[ERROR] --shard index must be in [0, {count}), got {index}")
        selected = selected[index::count]
    if args.limit:
        selected = selected[: args.limit]
    return selected


def worker_environment(threads: int) -> dict[str, str]:
    """Environment for a patient subprocess, capped at ``threads`` threads per native library."""
    env = os.environ.copy()
    t = str(threads)
    env.update(
        OMP_NUM_THREADS=t,
        OPENBLAS_NUM_THREADS=t,
        MKL_NUM_THREADS=t,
        ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=t,
        VTK_SMP_MAX_THREADS=t,
        LVM_THREADS=t,
        MPLBACKEND="Agg",
        PYTHONUNBUFFERED="1",
    )
    return env


class BatchRunner:
    """Runs patient subprocesses concurrently and records each outcome.

    Args:
        args: Parsed command-line arguments.
        output_dir: Parent folder of the per-patient output folders.
        threads: Native threads allowed per patient subprocess.
    """

    def __init__(self, args: argparse.Namespace, output_dir: str, threads: int) -> None:
        self.args = args
        self.output_dir = output_dir
        self.env = worker_environment(threads)
        self.running: set[subprocess.Popen] = set()
        self.lock = threading.Lock()
        self.stopping = False

    def command(self, patient_id: str) -> list[str]:
        """Command line that processes one patient."""
        cmd = [sys.executable, WORKER_SCRIPT, "--patient-id", patient_id, "--output-dir", self.output_dir]
        if self.args.no_plots:
            cmd.append("--no-plots")
        if not self.args.debug_meshes:
            cmd.append("--no-debug-meshes")
        return cmd

    def run_one(self, paths: PatientPaths) -> dict[str, object]:
        """Process one patient in a subprocess, logging to ``<save_dir>/pipeline.log``.

        Returns:
            One summary row.
        """
        os.makedirs(paths.save_dir, exist_ok=True)
        log_path = os.path.join(paths.save_dir, "pipeline.log")
        start = time.perf_counter()
        return_code: int | None = None
        status = "failed"
        with open(log_path, "w", encoding="utf-8", errors="replace") as log:
            with self.lock:
                if self.stopping:
                    return {"patient_id": paths.patient_id, "status": "cancelled", "log": log_path}
                proc = subprocess.Popen(
                    self.command(paths.patient_id), stdout=log, stderr=subprocess.STDOUT, env=self.env, cwd=HERE
                )
                self.running.add(proc)
            try:
                return_code = proc.wait(timeout=self.args.timeout_min * 60)
                status = "ok" if return_code == 0 and paths.is_done() else "failed"
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                status = "timeout"
            finally:
                with self.lock:
                    self.running.discard(proc)
        if self.stopping and status != "ok":
            status = "cancelled"
        return {
            "patient_id": paths.patient_id,
            "status": status,
            "wall_seconds": round(time.perf_counter() - start, 1),
            "peak_memory_gb": self._peak_memory(paths) if status == "ok" else None,
            "return_code": return_code,
            "log": log_path,
        }

    def stop(self) -> None:
        """Stop launching patients and kill the ones in flight."""
        with self.lock:
            self.stopping = True
            for proc in self.running:
                proc.kill()

    @staticmethod
    def _peak_memory(paths: PatientPaths) -> float | None:
        try:
            with open(paths.done_marker) as f:
                return json.load(f).get("peak_memory_gb")
        except (OSError, ValueError):
            return None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the LVM thickness pipeline over the whole dataset.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Patients processed at once (default: as many as fit in free RAM and physical cores).")
    parser.add_argument("--threads-per-worker", type=int, default=None,
                        help="Native threads per patient (default: physical cores / workers).")
    parser.add_argument("--memory-per-worker-gb", type=float, default=5.0,
                        help="Peak RAM budgeted per patient when choosing --workers (default: 5; ~3.7 GB measured "
                             "on the smallest scan, ~5 GB expected on the median 512x512x275 scan).")
    parser.add_argument("--patients", nargs="+", default=None, help="Only these patient ids.")
    parser.add_argument("--shard", default=None, metavar="INDEX/COUNT",
                        help="Take every COUNT-th patient starting at INDEX, e.g. $SLURM_ARRAY_TASK_ID/8.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many patients.")
    parser.add_argument("--output-dir", default=None, help="Parent output folder (default: <PROJECT_ROOT>/output).")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess patients that already have done.json.")
    parser.add_argument("--no-plots", action="store_true", help="Skip the histogram and bullseye figures (~25s each).")
    parser.add_argument("--debug-meshes", action="store_true", help="Also write the QA-only vectors.vtk mesh (~7s).")
    parser.add_argument("--timeout-min", type=float, default=30.0,
                        help="Kill a patient that runs longer than this (default: 30).")
    parser.add_argument("--dry-run", action="store_true", help="List what would run and exit.")
    return parser.parse_args()


def main() -> int:
    """Run the batch.

    Returns:
        Process exit code: 0 if every attempted patient succeeded, else 1.
    """
    args = parse_args()
    root, folder = load_environment()
    output_dir = os.path.abspath(args.output_dir or os.path.join(root, "output"))

    selected = select_patients(args, root, folder)
    all_paths = [patient_paths(root, folder, pid, output_dir) for pid in selected]
    todo = [p for p in all_paths if args.overwrite or not p.is_done()]
    skipped = len(all_paths) - len(todo)

    workers = args.workers or default_workers(args.memory_per_worker_gb)
    workers = max(1, min(workers, len(todo))) if todo else 0
    threads = args.threads_per_worker or max(1, physical_cores() // max(workers, 1))

    free = available_memory_gb()
    print(f"Patients selected: {len(all_paths)}  already done (skipped): {skipped}  to run: {len(todo)}")
    print(f"Workers: {workers}  threads/worker: {threads}  "
          f"free RAM: {'unknown' if free is None else f'{free:.1f} GB'}  output: {output_dir}")
    if workers and free is not None and workers * args.memory_per_worker_gb > free:
        print(f"[WARN] {workers} workers x {args.memory_per_worker_gb:.1f} GB exceeds free RAM; expect swapping.")
    if args.dry_run or not todo:
        for p in todo:
            print(f"  would run {p.patient_id}")
        return 0

    os.makedirs(output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(output_dir, f"batch_summary_{stamp}.csv")
    runner = BatchRunner(args, output_dir, threads)
    results: list[dict[str, object]] = []
    start = time.perf_counter()

    with open(summary_path, "w", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        pool = ThreadPoolExecutor(max_workers=workers)
        pending: set[Future] = {pool.submit(runner.run_one, p) for p in todo}
        progress = tqdm(total=len(todo), unit="patient", dynamic_ncols=True)
        try:
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    row = future.result()
                    results.append(row)
                    writer.writerow(row)
                    summary_file.flush()
                    failed = sum(r["status"] != "ok" for r in results)
                    progress.set_postfix(ok=len(results) - failed, failed=failed, last=row["patient_id"])
                    progress.update()
                    if row["status"] != "ok":
                        tqdm.write(f"[{row['status'].upper()}] patient {row['patient_id']} - see {row['log']}")
        except KeyboardInterrupt:
            tqdm.write("\nInterrupted: stopping workers. Finished patients are kept; rerun to resume.")
            runner.stop()
            for future in pending:
                future.cancel()
        finally:
            progress.close()
            pool.shutdown(wait=True, cancel_futures=True)

    elapsed = time.perf_counter() - start
    ok = [r for r in results if r["status"] == "ok"]
    bad = [r for r in results if r["status"] != "ok"]
    peaks = [r["peak_memory_gb"] for r in ok if r.get("peak_memory_gb")]
    print(f"\nFinished {len(ok)}/{len(todo)} in {elapsed / 60:.1f} min"
          + (f" ({elapsed / len(ok):.0f}s wall per patient)" if ok else "")
          + (f", peak memory per patient {min(peaks):.1f}-{max(peaks):.1f} GB" if peaks else ""))
    if bad:
        print(f"Not ok ({len(bad)}): {' '.join(str(r['patient_id']) for r in bad)}")
        print(f"Retry with: --patients {' '.join(str(r['patient_id']) for r in bad)}")
    print(f"Summary: {summary_path}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())

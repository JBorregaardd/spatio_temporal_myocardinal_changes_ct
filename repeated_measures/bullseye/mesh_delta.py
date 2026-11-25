#################################################################################################################################################
# This script computes the change in LV wall thickness (CGPS10 - CGPS1, in mm) of single subjects on the mesh. 
# For each subject the CGPS10 wall (surfaces/myocardium.vtk) is rigidly registered onto the CGPS1 wall with ICP, started both from the raw
# scanner frames and from the long-axis alignment of the two lv17 transforms, keeping the start with the smallest residual. The
# CGPS10 thickness (surfaces/dist_source.vtk) is then moved onto each CGPS1 endocardial vertex by its nearest registered CGPS10 vertex
# (no value when that is further than --max-distance-mm away), and the difference is taken per CGPS1 vertex.
# Each subject's difference mesh (delta_mesh.vtk, binary) and registration (registration.json) are saved in
# <PROJECT_ROOT>/output/CGPS1_CGPS10_mesh_delta_0mm/<CGPS1 id>_<CGPS10 id>/, and the registration quality of all of them in
# registration_qc.csv in that run folder. Saved registrations are reused (unless --overwrite) and the difference mesh is only remade
# for another --max-distance-mm. Without --all only --n random subjects (or --subjects) are processed; the same --n and --seed pick
# the same subjects in the two bullseye scripts. Subjects run in parallel with --workers.
# It needs the links table <PROJECT_ROOT>/df/CGPS1_CGPS10_links.csv (comparative_analysis/compare.py) and the finished pipeline runs
# with 0 mm aortic exclusion of both datasets, e.g. output/CGPS1_0mm and output/CGPS10_0mm.
# e.g. uv run repeated_measures/bullseye/mesh_delta.py --all --workers 8
#      uv run repeated_measures/bullseye/mesh_delta.py --n 20 --seed 0
#################################################################################################################################################

import argparse
import json
import os
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import vtk
from scipy.spatial import cKDTree
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "LVM_thickness"))
sys.path.insert(0, os.path.join(REPO_ROOT, "repeated_measures"))

from pipeline import DONE_MARKER, load_environment, run_output_dir  # noqa: E402
from spaghetti_plot import DATASETS, LINKS_FILE, read_ids  # noqa: E402
from utils import utils  # noqa: E402

AORTA_EXCLUSION_MM = 0
RUN_NAME = "CGPS1_CGPS10_mesh_delta"
THICKNESS_MESH = os.path.join("surfaces", "dist_source.vtk")
REGISTRATION_MESH = os.path.join("surfaces", "myocardium.vtk")
TRANSFORM_FILE = "lv17_transform.txt"
DELTA_MESH = "delta_mesh.vtk"
REGISTRATION_FILE = "registration.json"
QC_FILE = "registration_qc.csv"
MAX_DISTANCE_MM = 5.0
ICP_ITERATIONS = 100
ICP_LANDMARKS = 5000


@dataclass
class Registration:
    """Rigid registration of a CGPS10 scan onto its CGPS1 scan.

    Attributes:
        matrix: 4x4 matrix taking CGPS10 physical points to CGPS1 physical points, as nested lists.
        start: Start of the ICP that was kept, ``scanner`` or ``lv17``.
        mean_distance_mm: Mean distance of the registered CGPS10 wall vertices to the nearest CGPS1 wall vertex.
        rotation_deg: Rotation angle of ``matrix`` in degrees.
        matched_fraction: Fraction of CGPS1 endocardial vertices with a CGPS10 vertex within the maximum distance.
        median_match_mm: Median distance of the CGPS1 endocardial vertices to their nearest CGPS10 vertex.
        max_distance_mm: Largest match distance the difference mesh was made with; a different one remakes it.
    """

    matrix: list[list[float]]
    start: str
    mean_distance_mm: float
    rotation_deg: float
    matched_fraction: float = np.nan
    median_match_mm: float = np.nan
    max_distance_mm: float = np.nan


def mesh_points(mesh: vtk.vtkPolyData) -> np.ndarray:
    """(N, 3) vertex coordinates of a mesh."""
    return vtk_to_numpy(mesh.GetPoints().GetData()).astype(np.float64)


def point_array(mesh: vtk.vtkPolyData, name: str) -> np.ndarray:
    """A named point data array of a mesh as float64."""
    return vtk_to_numpy(mesh.GetPointData().GetArray(name)).astype(np.float64)


def apply_matrix(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous matrix to (N, 3) points."""
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def rotation_angle_deg(matrix: np.ndarray) -> float:
    """Rotation angle in degrees of the rotation part of a rigid 4x4 matrix."""
    cosine = (np.trace(matrix[:3, :3]) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))


def transform_matrix(transform: sitk.Transform) -> np.ndarray:
    """4x4 matrix of a rigid (or affine) SimpleITK transform, found by mapping the origin and the unit vectors.

    Args:
        transform: Transform taking points of the aligned LV frame to physical points, as used by ``sitk.Resample``.

    Returns:
        The 4x4 homogeneous matrix of the transform.
    """
    origin = np.array(transform.TransformPoint((0.0, 0.0, 0.0)))
    matrix = np.eye(4)
    for axis in range(3):
        unit = [0.0, 0.0, 0.0]
        unit[axis] = 1.0
        matrix[:3, axis] = np.array(transform.TransformPoint(tuple(unit))) - origin
    matrix[:3, 3] = origin
    return matrix


def lv17_start(fixed_dir: Path, moving_dir: Path) -> np.ndarray:
    """Start matrix that gives the moving scan the long-axis orientation of the fixed scan, from their lv17 transforms.

    Args:
        fixed_dir: Pipeline output folder of the CGPS1 scan.
        moving_dir: Pipeline output folder of the CGPS10 scan.

    Returns:
        4x4 matrix taking moving physical points to fixed physical points.
    """
    fixed = transform_matrix(sitk.ReadTransform(str(fixed_dir / TRANSFORM_FILE)))
    moving = transform_matrix(sitk.ReadTransform(str(moving_dir / TRANSFORM_FILE)))
    return fixed @ np.linalg.inv(moving)


def transformed_mesh(mesh: vtk.vtkPolyData, matrix: np.ndarray) -> vtk.vtkPolyData:
    """Copy of a mesh with its vertices moved by a 4x4 matrix."""
    moved = vtk.vtkPolyData()
    moved.DeepCopy(mesh)
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(apply_matrix(matrix, mesh_points(mesh)), deep=True))
    moved.SetPoints(points)
    return moved


def icp(fixed: vtk.vtkPolyData, moving: vtk.vtkPolyData, start: np.ndarray) -> tuple[np.ndarray, float]:
    """Rigid ICP of ``moving`` onto ``fixed`` from a start matrix, beginning by matching the centroids.

    Args:
        fixed: Mesh to register onto.
        moving: Mesh to move.
        start: 4x4 start matrix applied to ``moving`` before the ICP.

    Returns:
        The 4x4 matrix taking the original ``moving`` points onto ``fixed`` and the mean distance in mm of the moved
        vertices to the nearest fixed vertex.
    """
    source = transformed_mesh(moving, start)
    solver = vtk.vtkIterativeClosestPointTransform()
    solver.SetSource(source)
    solver.SetTarget(fixed)
    solver.GetLandmarkTransform().SetModeToRigidBody()
    solver.StartByMatchingCentroidsOn()
    solver.SetMaximumNumberOfIterations(ICP_ITERATIONS)
    solver.SetMaximumNumberOfLandmarks(ICP_LANDMARKS)
    solver.CheckMeanDistanceOn()
    solver.SetMaximumMeanDistance(1e-4)
    solver.Update()
    refined = np.array([[solver.GetMatrix().GetElement(i, j) for j in range(4)] for i in range(4)])
    matrix = refined @ start
    distance, _ = cKDTree(mesh_points(fixed)).query(apply_matrix(matrix, mesh_points(moving)))
    return matrix, float(distance.mean())


def register(fixed_dir: Path, moving_dir: Path) -> Registration:
    """Rigidly register the CGPS10 wall onto the CGPS1 wall, keeping the better of two ICP starts.

    Args:
        fixed_dir: Pipeline output folder of the CGPS1 scan.
        moving_dir: Pipeline output folder of the CGPS10 scan.

    Returns:
        The registration.
    """
    fixed = utils.read_vtk_mesh(str(fixed_dir / REGISTRATION_MESH))
    moving = utils.read_vtk_mesh(str(moving_dir / REGISTRATION_MESH))
    starts = {"scanner": np.eye(4), "lv17": lv17_start(fixed_dir, moving_dir)}
    results = {name: icp(fixed, moving, start) for name, start in starts.items()}
    best = min(results, key=lambda name: results[name][1])
    matrix, distance = results[best]
    return Registration(matrix.tolist(), best, distance, rotation_angle_deg(matrix))


def transfer_thickness(
    fixed_points: np.ndarray, moving_points: np.ndarray, moving_values: np.ndarray, max_distance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Value of the nearest moving vertex at every fixed vertex.

    Args:
        fixed_points: (N, 3) vertices to transfer the values to.
        moving_points: (M, 3) registered vertices carrying the values.
        moving_values: (M,) values of the moving vertices, e.g. wall thickness.
        max_distance: Fixed vertices further than this from every moving vertex get no value.

    Returns:
        (N,) transferred values (NaN where unmatched) and (N,) distances to the nearest moving vertex.
    """
    distance, nearest = cKDTree(moving_points).query(fixed_points)
    values = np.where(distance <= max_distance, moving_values[nearest], np.nan)
    return values, distance


def with_arrays(mesh: vtk.vtkPolyData, arrays: dict[str, np.ndarray], scalars: str) -> vtk.vtkPolyData:
    """Copy of a mesh with point arrays added (as float32) and one of them made the active scalars.

    Args:
        mesh: Mesh to copy.
        arrays: Array name to its (N,) values.
        scalars: Name of the array to make the active scalars, which the bullseye projection reads.

    Returns:
        The new mesh.
    """
    copy = vtk.vtkPolyData()
    copy.DeepCopy(mesh)
    data = copy.GetPointData()
    for name, values in arrays.items():
        array = numpy_to_vtk(np.asarray(values, dtype=np.float32), deep=True)
        array.SetName(name)
        if name == scalars:
            data.SetScalars(array)
        else:
            data.AddArray(array)
    return copy


def delta_mesh(fixed_dir: Path, moving_dir: Path, registration: Registration, max_distance: float) -> vtk.vtkPolyData:
    """CGPS1 endocardial mesh with the thickness at both examinations and their difference per vertex.

    Args:
        fixed_dir: Pipeline output folder of the CGPS1 scan.
        moving_dir: Pipeline output folder of the CGPS10 scan.
        registration: Registration of CGPS10 onto CGPS1; its match statistics are filled in.
        max_distance: Largest distance in mm between a CGPS1 vertex and the CGPS10 vertex it takes its value from.

    Returns:
        The CGPS1 mesh with the point arrays ``thickness_CGPS1``, ``thickness_CGPS10``, ``match_distance_mm`` and
        the active scalars ``delta_mm`` (CGPS10 - CGPS1, NaN where unmatched).
    """
    fixed = utils.read_vtk_mesh(str(fixed_dir / THICKNESS_MESH))
    moving = utils.read_vtk_mesh(str(moving_dir / THICKNESS_MESH))
    fixed_values = vtk_to_numpy(fixed.GetPointData().GetScalars()).astype(np.float64)
    moving_values = vtk_to_numpy(moving.GetPointData().GetScalars()).astype(np.float64)
    moving_points = apply_matrix(np.array(registration.matrix), mesh_points(moving))
    moved_values, distance = transfer_thickness(mesh_points(fixed), moving_points, moving_values, max_distance)
    delta = moved_values - fixed_values
    registration.matched_fraction = float(np.isfinite(delta).mean())
    registration.median_match_mm = float(np.median(distance))
    arrays = {
        "thickness_CGPS1": fixed_values,
        "thickness_CGPS10": moved_values,
        "match_distance_mm": distance,
        "delta_mm": delta,
    }
    return with_arrays(fixed, arrays, "delta_mm")


def write_binary_mesh(mesh: vtk.vtkPolyData, path: Path) -> None:
    """Write a mesh as a binary legacy VTK file, roughly half the size of the ASCII files of the pipeline."""
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(mesh)
    writer.SetFileTypeToBinary()
    writer.SetFileVersion(42)
    writer.SetFileName(str(path))
    if not writer.Write():
        raise OSError(f"Mesh not written: {path}")


def process_subject(fixed_dir: Path, moving_dir: Path, work_dir: Path, max_distance: float, overwrite: bool) -> dict:
    """Register and compute the difference mesh of one subject, reusing saved results where possible.

    A saved registration is reused unless ``overwrite``; the difference mesh is remade when it is missing or was made
    with another ``max_distance``.

    Args:
        fixed_dir: Pipeline output folder of the CGPS1 scan.
        moving_dir: Pipeline output folder of the CGPS10 scan.
        work_dir: Folder for this subject's results.
        max_distance: Largest vertex match distance in mm.
        overwrite: Also redo the registration.

    Returns:
        The registration as a dictionary, without the matrix.
    """
    mesh_path = work_dir / DELTA_MESH
    registration_path = work_dir / REGISTRATION_FILE
    saved = None if overwrite or not registration_path.is_file() else json.loads(registration_path.read_text())
    if saved is not None and mesh_path.is_file() and saved.get("max_distance_mm") == max_distance:
        registration = Registration(**saved)
    else:
        work_dir.mkdir(parents=True, exist_ok=True)
        if saved is None:
            registration = register(fixed_dir, moving_dir)
        else:
            saved.pop("max_distance_mm", None)
            registration = Registration(**saved)
        registration.max_distance_mm = max_distance
        write_binary_mesh(delta_mesh(fixed_dir, moving_dir, registration, max_distance), mesh_path)
        registration_path.write_text(json.dumps(asdict(registration), indent=2))
    summary = asdict(registration)
    summary.pop("matrix")
    return summary


def mesh_task(
    cgps1: str, cgps10: str, folders: dict[str, Path], run_folder: Path, max_distance: float, overwrite: bool
) -> dict:
    """``process_subject`` for one linked subject, with the ids added to its summary."""
    work_dir = run_folder / f"{cgps1}_{cgps10}"
    summary = process_subject(folders["CGPS1"] / cgps1, folders["CGPS10"] / cgps10, work_dir, max_distance, overwrite)
    return {"CGPS1": cgps1, "CGPS10": cgps10, **summary}


def _run_safely(task: Callable[[str, str], dict], cgps1: str, cgps10: str) -> dict:
    """Run a subject task in a worker process, returning the error instead of raising it."""
    try:
        return task(cgps1, cgps10)
    except Exception as error:
        return {"CGPS1": cgps1, "CGPS10": cgps10, "error": f"{type(error).__name__}: {error}"}


def run_subjects(
    task: Callable[[str, str], dict], pairs: list[tuple[str, str]], workers: int, describe: Callable[[dict], str]
) -> list[dict]:
    """Run a task for every subject in parallel processes, printing one line per finished subject.

    Args:
        task: Picklable function (e.g. a ``functools.partial`` of a module-level function) taking the CGPS1 and
            CGPS10 ids and returning a dictionary with ``CGPS1`` and ``CGPS10``.
        pairs: (CGPS1 id, CGPS10 id) of the subjects.
        workers: Number of parallel processes.
        describe: Text printed after a subject's ids for its result.

    Returns:
        The results of the subjects that succeeded; failures are printed.
    """
    print(f"Processing {len(pairs)} subjects with {workers} worker(s).")
    results, failed = [], []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_safely, task, cgps1, cgps10) for cgps1, cgps10 in pairs]
        for number, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            label = f"[{number}/{len(pairs)}] {result['CGPS1']} / {result['CGPS10']}"
            if "error" in result:
                failed.append(result)
                print(f"{label} [WARN] skipped: {result['error']}")
            else:
                results.append(result)
                print(f"{label}: {describe(result)}")
    if failed:
        print(f"[WARN] {len(failed)} subjects failed: {', '.join(r['CGPS1'] + '/' + r['CGPS10'] for r in failed)}")
    return results


def available_subjects(links: pd.DataFrame, folders: dict[str, Path]) -> pd.DataFrame:
    """Linked subjects whose pipeline run finished at both examinations."""
    pairs = links[list(DATASETS)].drop_duplicates()
    finished = np.ones(len(pairs), dtype=bool)
    for dataset, folder in folders.items():
        finished &= pairs[dataset].map(lambda patient: (folder / patient / DONE_MARKER).is_file()).to_numpy()
    return pairs[finished].reset_index(drop=True)


def choose_subjects(subjects: pd.DataFrame, n: int, seed: int, ids: list[str] | None) -> pd.DataFrame:
    """The subjects given by CGPS1 id, or ``n`` random ones (the same for the same ``n``, ``seed`` and subjects)."""
    if ids:
        return subjects[subjects["CGPS1"].isin(ids)]
    return subjects.sample(n=min(n, len(subjects)), random_state=seed)


def add_subject_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the subject choice and parallelism arguments shared by the three mesh delta scripts."""
    parser.add_argument("--n", type=int, default=20, help="Number of random subjects (default 20).")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the random subject choice.")
    parser.add_argument("--subjects", nargs="+", help="CGPS1 ids to use instead of a random choice.")
    parser.add_argument("--all", action="store_true", help="Process every available subject.")
    parser.add_argument("--workers", type=int, default=1, help="Number of subjects processed in parallel.")


def setup_subjects(args: argparse.Namespace) -> tuple[str, dict[str, Path], Path, pd.DataFrame, pd.DataFrame]:
    """Environment, folders and subjects of a run of one of the three mesh delta scripts.

    Args:
        args: Parsed arguments with the options of ``add_subject_arguments``.

    Returns:
        Project root, dataset name to its pipeline run folder, the run folder of the difference meshes, the subjects
        to process (all with ``--all``) and the chosen ones (``--n`` random or ``--subjects``).
    """
    root, _ = load_environment()
    links = read_ids(Path(root) / "df" / LINKS_FILE, list(DATASETS))
    folders = {dataset: Path(run_output_dir(root, dataset, AORTA_EXCLUSION_MM)) for dataset in DATASETS}
    subjects = available_subjects(links, folders)
    print(f"{len(subjects)} of {len(links)} linked subjects have a finished pipeline run at both examinations.")
    chosen = choose_subjects(subjects, args.n, args.seed, args.subjects)
    run_folder = Path(run_output_dir(root, RUN_NAME, AORTA_EXCLUSION_MM))
    return root, folders, run_folder, subjects if args.all else chosen, chosen


def pair_list(table: pd.DataFrame) -> list[tuple[str, str]]:
    """(CGPS1 id, CGPS10 id) of every row of a subject table."""
    return list(table[list(DATASETS)].itertuples(index=False, name=None))


def describe_registration(result: dict) -> str:
    """One line summary of a subject's registration."""
    return (
        f"start {result['start']}, rotation {result['rotation_deg']:.1f} deg, wall distance "
        f"{result['mean_distance_mm']:.2f} mm, matched {100 * result['matched_fraction']:.0f}%"
    )


def parse_args() -> argparse.Namespace:
    """Command line arguments."""
    parser = argparse.ArgumentParser(description="Per-subject CGPS10 - CGPS1 wall thickness difference meshes.")
    add_subject_arguments(parser)
    parser.add_argument(
        "--max-distance-mm",
        type=float,
        default=MAX_DISTANCE_MM,
        help="Largest distance between matched CGPS1 and CGPS10 vertices.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Also redo the registration of saved subjects.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root, folders, run_folder, to_process, _ = setup_subjects(args)
    task = partial(
        mesh_task, folders=folders, run_folder=run_folder, max_distance=args.max_distance_mm, overwrite=args.overwrite
    )
    rows = run_subjects(task, pair_list(to_process), args.workers, describe_registration)
    if not rows:
        raise SystemExit("No subject could be processed.")

    qc_path = run_folder / QC_FILE
    qc = pd.DataFrame(rows)
    if qc_path.is_file():
        old = read_ids(qc_path, list(DATASETS))
        qc = pd.concat([old, qc]).drop_duplicates(list(DATASETS), keep="last")
    qc.to_csv(qc_path, index=False)
    print(f"Difference meshes saved in: {run_folder}")
    print(f"Registration QC saved to: {qc_path}")

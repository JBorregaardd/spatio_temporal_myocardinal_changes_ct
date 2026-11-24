#################################################################################################################################################
# This script computes the change in LV wall thickness (CGPS10 - CGPS1, in mm) of single subjects on the mesh and shows it as bullseye
# plots, without smoothing and without grouping, so the raw per-subject difference maps can be inspected.
# For each subject the CGPS10 wall (surfaces/myocardium.vtk) is rigidly registered onto the CGPS1 wall with ICP, started both from the raw
# scanner frames and from the long-axis alignment of the two lv17 transforms, keeping the start with the smallest residual. The
# CGPS10 thickness (surfaces/dist_source.vtk) is then moved onto each CGPS1 endocardial vertex by its nearest registered CGPS10 vertex
# (no value when that is further than --max-distance-mm away), and the difference is taken per CGPS1 vertex. The difference mesh is
# projected into the bullseye of the CGPS1 scan with the same code as the per-scan bullseye of the pipeline.
# Each subject's difference mesh (binary VTK), bullseye points and registration are saved in
# <PROJECT_ROOT>/output/CGPS1_CGPS10_mesh_delta_0mm/<CGPS1 id>_<CGPS10 id>/, and the registration quality of all of them in
# registration_qc.csv in that run folder. The NIfTI images of the projection are made in a temporary folder and deleted, unless
# --keep-nifti. Saved registrations are reused (unless --overwrite) and the difference mesh is only remade for another
# --max-distance-mm. Subjects run in parallel with --workers; --all processes every available subject. The figure of --n random
# subjects is saved in <PROJECT_ROOT>/reports/figures/repeated_measures/bullseye.
# It needs the links table <PROJECT_ROOT>/df/CGPS1_CGPS10_links.csv (comparative_analysis/compare.py) and the finished pipeline runs
# with 0 mm aortic exclusion of both datasets, e.g. output/CGPS1_0mm and output/CGPS10_0mm.
# e.g. uv run repeated_measures/bullseye/bullseye_mesh_delta.py --n 20 --seed 0
#      uv run repeated_measures/bullseye/bullseye_mesh_delta.py --all --workers 8
#################################################################################################################################################

import argparse
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import SimpleITK as sitk  # noqa: E402
import vtk  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "LVM_thickness"))
sys.path.insert(0, os.path.join(REPO_ROOT, "repeated_measures"))

from pipeline import DONE_MARKER, load_environment, run_output_dir  # noqa: E402
from spaghetti_plot import DATASETS, LINKS_FILE, OUTPUT_FOLDER, read_ids  # noqa: E402
from utils import bullseye, utils  # noqa: E402

AORTA_EXCLUSION_MM = 0
RUN_NAME = "CGPS1_CGPS10_mesh_delta"
THICKNESS_MESH = os.path.join("surfaces", "dist_source.vtk")
REGISTRATION_MESH = os.path.join("surfaces", "myocardium.vtk")
TRANSFORM_FILE = "lv17_transform.txt"
LV17_FOLDER = os.path.join("segmentations", "lv17")
INNER_MASK = os.path.join("segmentations", "inner.nii.gz")
DELTA_MESH = "delta_mesh.vtk"
LABEL_FILE = "lv_label.nii.gz"
POLAR_FILE = "polar_delta.npz"
REGISTRATION_FILE = "registration.json"
QC_FILE = "registration_qc.csv"
MAX_DISTANCE_MM = 5.0
ICP_ITERATIONS = 100
ICP_LANDMARKS = 5000
BULLSEYE_GRID = 750
BULLSEYE_NEIGHBOURS = 5
RADII = np.linspace(0.2, 1, 4)
RINGS = [(0, 6), (6, 12), (12, 16)]
CMAP = "RdBu_r"
MISSING_COLOUR = "#bdbdbd"
IMAGE_ANGLES = 720
IMAGE_RADII = 375
FIGURE_FOLDER = os.path.join(OUTPUT_FOLDER, "bullseye")


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

    mesh = vtk.vtkPolyData()
    mesh.DeepCopy(fixed)
    data = mesh.GetPointData()
    for name, values in [
        ("thickness_CGPS1", fixed_values),
        ("thickness_CGPS10", moved_values),
        ("match_distance_mm", distance),
    ]:
        array = numpy_to_vtk(values.astype(np.float32), deep=True)
        array.SetName(name)
        data.AddArray(array)
    scalars = numpy_to_vtk(delta.astype(np.float32), deep=True)
    scalars.SetName("delta_mm")
    data.SetScalars(scalars)
    return mesh


def polar_delta(fixed_dir: Path, mesh_path: Path, projection_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bullseye points of a difference mesh, projected with the CGPS1 scan's lv17 frame.

    The pipeline's projection reads the lv17 model from, and writes its ring image to, the folder it is given, so the
    CGPS1 lv17 files are linked into ``projection_dir`` and an LV label image is rebuilt there from ``inner.nii.gz``.

    Args:
        fixed_dir: Pipeline output folder of the CGPS1 scan.
        mesh_path: The difference mesh.
        projection_dir: Folder for the links and the NIfTI images of the projection.

    Returns:
        Angle (radians), radius (normalised so the AHA rings lie at 0.2, 0.47, 0.73 and 1) and change in mm of every
        bullseye point.
    """
    (projection_dir / "segmentations").mkdir(parents=True, exist_ok=True)
    for source, target in [
        (fixed_dir / TRANSFORM_FILE, projection_dir / TRANSFORM_FILE),
        (fixed_dir / LV17_FOLDER, projection_dir / LV17_FOLDER),
    ]:
        if not target.exists():
            target.symlink_to(source.resolve())
    inner = sitk.ReadImage(str(fixed_dir / INNER_MASK))
    label = sitk.Cast(inner > 0, sitk.sitkUInt8) * bullseye.LV_ID
    label_path = projection_dir / LABEL_FILE
    sitk.WriteImage(label, str(label_path))

    x, y, values = bullseye.generate_polar_values(
        str(projection_dir), str(mesh_path), str(label_path), lvm=False, aorta_exclusion_mm=AORTA_EXCLUSION_MM
    )
    x, y, values = bullseye.fill_holes_in_bullseye(x, y, values, BULLSEYE_GRID, BULLSEYE_NEIGHBOURS)
    radius, theta = utils.cartesian2polar(x, y)
    return theta, radius / (radius.max() + 0.25), values


def write_binary_mesh(mesh: vtk.vtkPolyData, path: Path) -> None:
    """Write a mesh as a binary legacy VTK file, roughly half the size of the ASCII files of the pipeline."""
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(mesh)
    writer.SetFileTypeToBinary()
    writer.SetFileVersion(42)
    writer.SetFileName(str(path))
    if not writer.Write():
        raise OSError(f"Mesh not written: {path}")


def remove_intermediates(work_dir: Path) -> None:
    """Remove the links and NIfTI images of a projection made directly in ``work_dir`` by an earlier version.

    The links are removed, never what they point to.
    """
    for path in [work_dir / TRANSFORM_FILE, work_dir / LV17_FOLDER, work_dir / LABEL_FILE]:
        if path.is_symlink() or path.is_file():
            path.unlink()
    segmentation_dir = work_dir / "segmentations"
    if segmentation_dir.is_dir() and not segmentation_dir.is_symlink():
        for path in segmentation_dir.glob("*.nii.gz"):
            path.unlink()
        if not any(segmentation_dir.iterdir()):
            segmentation_dir.rmdir()


def process_subject(
    fixed_dir: Path, moving_dir: Path, work_dir: Path, max_distance: float, overwrite: bool, keep_nifti: bool = False
) -> dict:
    """Register, compute the difference mesh and project it for one subject, reusing saved results where possible.

    A saved registration is reused unless ``overwrite``; the difference mesh and bullseye points are redone when they
    are missing or were made with another ``max_distance``.

    Args:
        fixed_dir: Pipeline output folder of the CGPS1 scan.
        moving_dir: Pipeline output folder of the CGPS10 scan.
        work_dir: Folder for this subject's results.
        max_distance: Largest vertex match distance in mm.
        overwrite: Also redo the registration.
        keep_nifti: Keep the NIfTI images of the projection (``lv_label.nii.gz`` and
            ``segmentations/mm_delta_mesh_ring.nii.gz``) in ``work_dir``; otherwise they are made in a temporary folder.

    Returns:
        The registration as a dictionary, without the matrix.
    """
    polar_path = work_dir / POLAR_FILE
    registration_path = work_dir / REGISTRATION_FILE
    saved = None if overwrite or not registration_path.is_file() else json.loads(registration_path.read_text())
    if saved is not None and polar_path.is_file() and saved.get("max_distance_mm") == max_distance:
        registration = Registration(**saved)
    else:
        work_dir.mkdir(parents=True, exist_ok=True)
        if saved is None:
            registration = register(fixed_dir, moving_dir)
        else:
            saved.pop("max_distance_mm", None)
            registration = Registration(**saved)
        registration.max_distance_mm = max_distance
        mesh_path = work_dir / DELTA_MESH
        write_binary_mesh(delta_mesh(fixed_dir, moving_dir, registration, max_distance), mesh_path)
        if keep_nifti:
            theta, radius, values = polar_delta(fixed_dir, mesh_path, work_dir)
        else:
            with tempfile.TemporaryDirectory(dir=work_dir) as projection_dir:
                theta, radius, values = polar_delta(fixed_dir, mesh_path, Path(projection_dir))
        np.savez_compressed(polar_path, theta=theta, radius=radius, values=values)
        registration_path.write_text(json.dumps(asdict(registration), indent=2))
    if not keep_nifti:
        remove_intermediates(work_dir)
    summary = asdict(registration)
    summary.pop("matrix")
    return summary


def process_subject_safely(
    cgps1: str,
    cgps10: str,
    folders: dict[str, Path],
    run_folder: Path,
    max_distance: float,
    overwrite: bool,
    keep_nifti: bool,
) -> dict:
    """``process_subject`` for one linked subject in a worker process, returning the error instead of raising it.

    Args:
        cgps1: CGPS1 patient id.
        cgps10: CGPS10 patient id.
        folders: Dataset name to its pipeline run folder.
        run_folder: Run folder of this script.
        max_distance: Largest vertex match distance in mm.
        overwrite: Also redo the registration.
        keep_nifti: Keep the NIfTI images of the projection.

    Returns:
        ``CGPS1``, ``CGPS10`` and either the registration summary or ``error``.
    """
    try:
        summary = process_subject(
            folders["CGPS1"] / cgps1,
            folders["CGPS10"] / cgps10,
            run_folder / f"{cgps1}_{cgps10}",
            max_distance,
            overwrite,
            keep_nifti,
        )
    except Exception as error:
        return {"CGPS1": cgps1, "CGPS10": cgps10, "error": f"{type(error).__name__}: {error}"}
    return {"CGPS1": cgps1, "CGPS10": cgps10, **summary}


def draw_outline(ax: plt.Axes) -> None:
    """Draw the AHA 17-segment borders on a polar axis with radius normalised to 1."""
    for (start, stop), r_in, r_out in zip(RINGS, RADII[2::-1], RADII[:0:-1]):
        n = stop - start
        dtheta = 2 * np.pi / n
        ax.bar(
            np.arange(n) * dtheta + np.pi / 2,
            r_out - r_in,
            dtheta,
            r_in,
            color="none",
            edgecolor="k",
            linewidth=0.8,
            clip_on=False,
        )
    ax.plot(np.linspace(0, 2 * np.pi, 100), np.full(100, RADII[0]), "k", linewidth=0.8)


def polar_image(points: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean change per cell of a regular polar grid, NaN for cells without points and inside the apex ring.

    The cells are a little larger than the spacing of the bullseye points, so every cell inside a covered area holds
    at least one point and only areas without data stay empty.

    Args:
        points: Bullseye points with ``theta`` (radians), ``radius`` (normalised to 1) and ``values``.

    Returns:
        Angle edges, radius edges and the (radius, angle) image of mean values.
    """
    theta_edges = np.linspace(-np.pi, np.pi, IMAGE_ANGLES + 1)
    radius_edges = np.linspace(0, 1, IMAGE_RADII + 1)
    finite = np.isfinite(points["values"])
    sample = (points["radius"][finite], np.mod(points["theta"][finite] + np.pi, 2 * np.pi) - np.pi)
    sums, _, _ = np.histogram2d(*sample, bins=[radius_edges, theta_edges], weights=points["values"][finite])
    counts, _, _ = np.histogram2d(*sample, bins=[radius_edges, theta_edges])
    with np.errstate(invalid="ignore", divide="ignore"):
        image = sums / counts
    image[(radius_edges[1:] <= RADII[0])] = np.nan
    return theta_edges, radius_edges, image


def bullseye_grid(polar: dict[str, dict[str, np.ndarray]], vmax: float, n_columns: int = 5) -> plt.Figure:
    """Grid of single-subject difference bullseyes on one symmetric colour scale.

    Args:
        polar: Subject label to its bullseye points, with ``theta``, ``radius`` and ``values``.
        vmax: Largest absolute change in mm of the colour scale.
        n_columns: Number of bullseyes per row.

    Returns:
        The figure.
    """
    n_rows = int(np.ceil(len(polar) / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(3.2 * n_columns, 3.4 * n_rows),
        subplot_kw=dict(polar=True),
        layout="constrained",
        squeeze=False,
    )
    norm = Normalize(vmin=-vmax, vmax=vmax, clip=True)
    for ax, (label, points) in zip(axes.flat, polar.items()):
        theta_edges, radius_edges, image = polar_image(points)
        ax.bar(0, 1, 2 * np.pi, 0, color=MISSING_COLOUR, edgecolor="none")
        ax.pcolormesh(theta_edges, radius_edges, image, cmap=CMAP, norm=norm, shading="flat", rasterized=True)
        ax.bar(0, RADII[0], 2 * np.pi, 0, color="white", edgecolor="none")
        draw_outline(ax)
        ax.set(ylim=[0, 1])
        ax.axis("off")
        ax.set_title(label, fontsize=9)
    for ax in axes.flat[len(polar) :]:
        ax.axis("off")
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=CMAP), ax=axes, orientation="horizontal", shrink=0.4, aspect=40
    )
    colorbar.set_label(
        "Change in wall thickness (mm), CGPS10 − CGPS1, per subject (no smoothing; grey = no data, apex left out)"
    )
    return fig


def available_subjects(links: pd.DataFrame, folders: dict[str, Path]) -> pd.DataFrame:
    """Linked subjects whose pipeline run finished at both examinations."""
    pairs = links[list(DATASETS)].drop_duplicates()
    finished = np.ones(len(pairs), dtype=bool)
    for dataset, folder in folders.items():
        finished &= pairs[dataset].map(lambda patient: (folder / patient / DONE_MARKER).is_file()).to_numpy()
    return pairs[finished].reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    """Command line arguments."""
    parser = argparse.ArgumentParser(description="Per-subject bullseyes of the CGPS10 - CGPS1 wall thickness change.")
    parser.add_argument("--n", type=int, default=20, help="Number of random subjects to plot.")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the random subject choice.")
    parser.add_argument("--subjects", nargs="+", help="CGPS1 ids to plot instead of a random choice.")
    parser.add_argument("--vmax", type=float, help="Colour scale limit in mm (default: 98th percentile of |change|).")
    parser.add_argument(
        "--max-distance-mm",
        type=float,
        default=MAX_DISTANCE_MM,
        help="Largest distance between matched CGPS1 and CGPS10 vertices.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Also redo the registration of saved subjects.")
    parser.add_argument("--workers", type=int, default=1, help="Number of subjects processed in parallel.")
    parser.add_argument(
        "--all", action="store_true", help="Process every available subject; the figure still shows --n random ones."
    )
    parser.add_argument(
        "--keep-nifti", action="store_true", help="Keep the NIfTI images of the projection, e.g. to view in 3D."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root, _ = load_environment()
    links = read_ids(Path(root) / "df" / LINKS_FILE, list(DATASETS))
    folders = {dataset: Path(run_output_dir(root, dataset, AORTA_EXCLUSION_MM)) for dataset in DATASETS}
    subjects = available_subjects(links, folders)
    print(f"{len(subjects)} of {len(links)} linked subjects have a finished pipeline run at both examinations.")
    if args.subjects:
        chosen = subjects[subjects["CGPS1"].isin(args.subjects)]
    else:
        chosen = subjects.sample(n=min(args.n, len(subjects)), random_state=args.seed)
    to_process = subjects if args.all else chosen

    run_folder = Path(run_output_dir(root, RUN_NAME, AORTA_EXCLUSION_MM))
    pairs = list(to_process[list(DATASETS)].itertuples(index=False, name=None))
    print(f"Processing {len(pairs)} subjects with {args.workers} worker(s).")
    settings = (folders, run_folder, args.max_distance_mm, args.overwrite, args.keep_nifti)
    rows, failed = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_subject_safely, cgps1, cgps10, *settings) for cgps1, cgps10 in pairs]
        for number, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            label = f"[{number}/{len(pairs)}] {result['CGPS1']} / {result['CGPS10']}"
            if "error" in result:
                failed.append(result)
                print(f"{label} [WARN] skipped: {result['error']}")
                continue
            rows.append(result)
            print(
                f"{label}: start {result['start']}, rotation {result['rotation_deg']:.1f} deg, wall distance "
                f"{result['mean_distance_mm']:.2f} mm, matched {100 * result['matched_fraction']:.0f}%"
            )
    if failed:
        print(f"[WARN] {len(failed)} subjects failed: {', '.join(f'{r['CGPS1']}/{r['CGPS10']}' for r in failed)}")
    if not rows:
        raise SystemExit("No subject could be processed.")

    done = {(row["CGPS1"], row["CGPS10"]) for row in rows}
    polar = {
        f"{cgps1} / {cgps10}": dict(np.load(run_folder / f"{cgps1}_{cgps10}" / POLAR_FILE))
        for cgps1, cgps10 in chosen[list(DATASETS)].itertuples(index=False, name=None)
        if (cgps1, cgps10) in done
    }

    qc_path = run_folder / QC_FILE
    qc = pd.DataFrame(rows)
    if qc_path.is_file():
        old = read_ids(qc_path, list(DATASETS))
        qc = pd.concat([old, qc]).drop_duplicates(list(DATASETS), keep="last")
    qc.to_csv(qc_path, index=False)

    if not polar:
        raise SystemExit(f"None of the {len(chosen)} subjects to plot could be processed.")
    values = np.concatenate([points["values"] for points in polar.values()])
    vmax = args.vmax or float(np.nanpercentile(np.abs(values), 98))
    print(f"Colour scale: -{vmax:.1f} to {vmax:.1f} mm")
    figure_folder = Path(root) / FIGURE_FOLDER
    figure_folder.mkdir(parents=True, exist_ok=True)
    name = "subjects" if args.subjects else f"random{len(chosen)}_seed{args.seed}"
    figure_path = figure_folder / f"bullseye_mesh_delta_{name}.png"
    figure = bullseye_grid(polar, vmax)
    figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Registration QC saved to: {qc_path}")
    print(f"Figure saved to: {figure_path}")

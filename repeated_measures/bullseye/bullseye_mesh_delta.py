#################################################################################################################################################
# This script computes the change in LV wall thickness (CGPS10 - CGPS1, in mm) of single subjects on the mesh and shows it as bullseye
# plots, without smoothing and without grouping, so the raw per-subject difference maps can be inspected.
# For each subject the CGPS10 wall (surfaces/myocardium.vtk) is rigidly registered onto the CGPS1 wall with ICP, started both from the raw
# scanner frames and from the long-axis alignment of the two lv17 transforms, keeping the start with the smallest residual. The
# CGPS10 thickness (surfaces/dist_source.vtk) is then moved onto each CGPS1 endocardial vertex by its nearest registered CGPS10 vertex
# (no value when that is further than --max-distance-mm away), and the difference is taken per CGPS1 vertex. The difference mesh is
# projected into the bullseye of the CGPS1 scan with the same code as the per-scan bullseye of the pipeline.
# Each subject's difference mesh, bullseye points and registration are saved in
# <PROJECT_ROOT>/output/CGPS1_CGPS10_mesh_delta_0mm/<CGPS1 id>_<CGPS10 id>/, and the registration quality of all of them in
# registration_qc.csv in that run folder. The figure of --n random subjects is saved in
# <PROJECT_ROOT>/reports/figures/repeated_measures/bullseye.
# It needs the links table <PROJECT_ROOT>/df/CGPS1_CGPS10_links.csv (comparative_analysis/compare.py) and the finished pipeline runs
# with 0 mm aortic exclusion of both datasets, e.g. output/CGPS1_0mm and output/CGPS10_0mm.
# e.g. uv run repeated_measures/bullseye/bullseye_mesh_delta.py --n 20 --seed 0
#################################################################################################################################################

import argparse
import json
import os
import sys
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
    """

    matrix: list[list[float]]
    start: str
    mean_distance_mm: float
    rotation_deg: float
    matched_fraction: float = np.nan
    median_match_mm: float = np.nan


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
        array = numpy_to_vtk(values, deep=True)
        array.SetName(name)
        data.AddArray(array)
    scalars = numpy_to_vtk(delta, deep=True)
    scalars.SetName("delta_mm")
    data.SetScalars(scalars)
    return mesh


def polar_delta(fixed_dir: Path, work_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bullseye points of the difference mesh in ``work_dir``, projected with the CGPS1 scan's lv17 frame.

    The pipeline's projection reads the lv17 model from, and writes its ring image to, the folder it is given, so the
    CGPS1 lv17 files are linked into ``work_dir`` and an LV label image is rebuilt there from ``inner.nii.gz``.

    Args:
        fixed_dir: Pipeline output folder of the CGPS1 scan.
        work_dir: Folder holding ``delta_mesh.vtk``.

    Returns:
        Angle (radians), radius (normalised so the AHA rings lie at 0.2, 0.47, 0.73 and 1) and change in mm of every
        bullseye point.
    """
    segmentation_dir = work_dir / "segmentations"
    segmentation_dir.mkdir(exist_ok=True)
    for source, target in [
        (fixed_dir / TRANSFORM_FILE, work_dir / TRANSFORM_FILE),
        (fixed_dir / LV17_FOLDER, work_dir / LV17_FOLDER),
    ]:
        if not target.exists():
            target.symlink_to(source.resolve())
    inner = sitk.ReadImage(str(fixed_dir / INNER_MASK))
    label = sitk.Cast(inner > 0, sitk.sitkUInt8) * bullseye.LV_ID
    label_path = work_dir / LABEL_FILE
    sitk.WriteImage(label, str(label_path))

    x, y, values = bullseye.generate_polar_values(
        str(work_dir), str(work_dir / DELTA_MESH), str(label_path), lvm=False, aorta_exclusion_mm=AORTA_EXCLUSION_MM
    )
    x, y, values = bullseye.fill_holes_in_bullseye(x, y, values, BULLSEYE_GRID, BULLSEYE_NEIGHBOURS)
    radius, theta = utils.cartesian2polar(x, y)
    return theta, radius / (radius.max() + 0.25), values


def process_subject(fixed_dir: Path, moving_dir: Path, work_dir: Path, max_distance: float, overwrite: bool) -> dict:
    """Register, compute the difference mesh and project it for one subject, reusing saved results unless asked not to.

    Args:
        fixed_dir: Pipeline output folder of the CGPS1 scan.
        moving_dir: Pipeline output folder of the CGPS10 scan.
        work_dir: Folder for this subject's results.
        max_distance: Largest vertex match distance in mm.
        overwrite: Recompute even if the bullseye points are saved.

    Returns:
        The registration as a dictionary, without the matrix.
    """
    polar_path = work_dir / POLAR_FILE
    registration_path = work_dir / REGISTRATION_FILE
    if not overwrite and polar_path.is_file() and registration_path.is_file():
        registration = Registration(**json.loads(registration_path.read_text()))
    else:
        work_dir.mkdir(parents=True, exist_ok=True)
        registration = register(fixed_dir, moving_dir)
        utils.write_vtk_mesh(delta_mesh(fixed_dir, moving_dir, registration, max_distance), str(work_dir / DELTA_MESH))
        registration_path.write_text(json.dumps(asdict(registration), indent=2))
        theta, radius, values = polar_delta(fixed_dir, work_dir)
        np.savez_compressed(polar_path, theta=theta, radius=radius, values=values)
    summary = asdict(registration)
    summary.pop("matrix")
    return summary


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
    parser.add_argument("--overwrite", action="store_true", help="Recompute subjects that are already saved.")
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

    run_folder = Path(run_output_dir(root, RUN_NAME, AORTA_EXCLUSION_MM))
    polar, rows = {}, []
    for number, (cgps1, cgps10) in enumerate(chosen[list(DATASETS)].itertuples(index=False), start=1):
        work_dir = run_folder / f"{cgps1}_{cgps10}"
        print(f"[{number}/{len(chosen)}] {cgps1} / {cgps10}")
        try:
            summary = process_subject(
                folders["CGPS1"] / cgps1, folders["CGPS10"] / cgps10, work_dir, args.max_distance_mm, args.overwrite
            )
        except Exception as error:
            print(f"  [WARN] skipped: {error}")
            continue
        rows.append({"CGPS1": cgps1, "CGPS10": cgps10, **summary})
        print(
            f"  start {summary['start']}, rotation {summary['rotation_deg']:.1f} deg, wall distance "
            f"{summary['mean_distance_mm']:.2f} mm, matched {100 * summary['matched_fraction']:.0f}%"
        )
        polar[f"{cgps1} / {cgps10}"] = dict(np.load(work_dir / POLAR_FILE))
    if not polar:
        raise SystemExit("No subject could be processed.")

    qc_path = run_folder / QC_FILE
    qc = pd.DataFrame(rows)
    if qc_path.is_file():
        old = read_ids(qc_path, list(DATASETS))
        qc = pd.concat([old, qc]).drop_duplicates(list(DATASETS), keep="last")
    qc.to_csv(qc_path, index=False)

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

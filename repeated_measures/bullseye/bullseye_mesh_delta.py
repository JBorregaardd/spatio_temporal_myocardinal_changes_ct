#################################################################################################################################################
# This script shows the per-subject change in LV wall thickness (CGPS10 - CGPS1, in mm) computed on the mesh by mesh_delta.py as bullseye
# plots, without smoothing and without grouping, so the raw difference maps can be inspected.
# Each subject's delta_mesh.vtk is projected into the bullseye of the CGPS1 scan with the same code as the per-scan bullseye of the
# pipeline, and the bullseye points are saved as polar_delta.npz next to it (remade when the difference mesh is newer). The NIfTI images
# of the projection are made in a temporary folder and deleted, unless --keep-nifti (then in projection_polar_delta/).
# The figure of the --n random subjects (the same as in mesh_delta.py for the same --n and --seed), with grey for no data and the apex
# left out, is saved in <PROJECT_ROOT>/reports/figures/repeated_measures/bullseye. With --all every subject is projected.
# It needs the difference meshes in <PROJECT_ROOT>/output/CGPS1_CGPS10_mesh_delta_0mm/, so run mesh_delta.py first.
# e.g. uv run repeated_measures/bullseye/bullseye_mesh_delta.py --n 20 --seed 0
#################################################################################################################################################

import argparse
import os
import sys
import tempfile
from collections.abc import Callable
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import SimpleITK as sitk  # noqa: E402
import vtk  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mesh_delta import (  # noqa: E402
    AORTA_EXCLUSION_MM,
    DELTA_MESH,
    TRANSFORM_FILE,
    add_subject_arguments,
    pair_list,
    run_subjects,
    setup_subjects,
    write_binary_mesh,
)
from spaghetti_plot import OUTPUT_FOLDER  # noqa: E402
from utils import bullseye, utils  # noqa: E402

LV17_FOLDER = os.path.join("segmentations", "lv17")
INNER_MASK = os.path.join("segmentations", "inner.nii.gz")
LABEL_FILE = "lv_label.nii.gz"
POLAR_NAME = "polar_delta"
BULLSEYE_GRID = 750
BULLSEYE_NEIGHBOURS = 5
RADII = np.linspace(0.2, 1, 4)
RINGS = [(0, 6), (6, 12), (12, 16)]
CMAP = "RdBu_r"
MISSING_COLOUR = "#bdbdbd"
IMAGE_ANGLES = 720
IMAGE_RADII = 375
FIGURE_FOLDER = os.path.join(OUTPUT_FOLDER, "bullseye")
COLOURBAR_LABEL = "Change in wall thickness (mm), CGPS10 − CGPS1, per subject (grey = no data, apex left out)"


def _project(fixed_dir: Path, mesh: vtk.vtkPolyData, projection_dir: Path, fill: bool) -> dict[str, np.ndarray]:
    """Project the active scalars of a CGPS1 mesh into the bullseye, writing the projection's files to a folder.

    The pipeline's projection reads the lv17 model from, and writes its ring image to, the folder it is given, so the
    CGPS1 lv17 files are linked into ``projection_dir`` and an LV label image is rebuilt there from ``inner.nii.gz``.
    """
    (projection_dir / "segmentations").mkdir(parents=True, exist_ok=True)
    for source, target in [
        (fixed_dir / TRANSFORM_FILE, projection_dir / TRANSFORM_FILE),
        (fixed_dir / LV17_FOLDER, projection_dir / LV17_FOLDER),
    ]:
        if not target.exists():
            target.symlink_to(source.resolve())
    inner = sitk.ReadImage(str(fixed_dir / INNER_MASK))
    label_path = projection_dir / LABEL_FILE
    sitk.WriteImage(sitk.Cast(inner > 0, sitk.sitkUInt8) * bullseye.LV_ID, str(label_path))
    mesh_path = projection_dir / DELTA_MESH
    write_binary_mesh(mesh, mesh_path)

    x, y, values = bullseye.generate_polar_values(
        str(projection_dir), str(mesh_path), str(label_path), lvm=False, aorta_exclusion_mm=AORTA_EXCLUSION_MM
    )
    if fill:
        x, y, values = bullseye.fill_holes_in_bullseye(x, y, values, BULLSEYE_GRID, BULLSEYE_NEIGHBOURS)
    radius, theta = utils.cartesian2polar(np.asarray(x), np.asarray(y))
    return {"theta": theta, "radius": radius / (radius.max() + 0.25), "values": np.asarray(values)}


def project_mesh(
    fixed_dir: Path, mesh: vtk.vtkPolyData, keep_dir: Path | None = None, fill: bool = True
) -> dict[str, np.ndarray]:
    """Bullseye points of the active scalars of a mesh, projected with the lv17 frame of the scan it belongs to.

    Args:
        fixed_dir: Pipeline output folder of the scan whose endocardium the mesh is (CGPS1 for difference meshes).
        mesh: Mesh on that scan's endocardium whose active scalars are projected.
        keep_dir: Folder to keep the projection's links and NIfTI images in; None uses a temporary folder.
        fill: Resample onto the regular bullseye grid, filling small holes (median of the nearest points), as for the
            plots; False returns the projected ring voxels themselves, one measurement each.

    Returns:
        ``theta`` (radians), ``radius`` (normalised so the AHA rings lie at 0.2, 0.47, 0.73 and 1) and ``values`` of
        every bullseye point.
    """
    if keep_dir is not None:
        return _project(fixed_dir, mesh, keep_dir, fill)
    with tempfile.TemporaryDirectory() as projection_dir:
        return _project(fixed_dir, mesh, Path(projection_dir), fill)


def remove_intermediates(work_dir: Path) -> None:
    """Remove the links and NIfTI images an earlier version projected directly in ``work_dir``.

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


def cached_polar(
    work_dir: Path,
    fixed_dir: Path,
    name: str,
    make_mesh: Callable[[vtk.vtkPolyData], vtk.vtkPolyData] | None,
    keep_nifti: bool,
    overwrite: bool,
) -> Path:
    """Project a mesh made from a subject's difference mesh into the bullseye, unless that is already saved.

    The saved bullseye points are reused unless ``overwrite`` or the difference mesh is newer.

    Args:
        work_dir: The subject's folder with ``delta_mesh.vtk``.
        fixed_dir: Pipeline output folder of the CGPS1 scan.
        name: Name of the saved points, ``<name>.npz``.
        make_mesh: Makes the mesh to project from the difference mesh, e.g. smoothed; None projects it as it is.
        keep_nifti: Keep the projection's NIfTI images in ``projection_<name>/``.
        overwrite: Project even if the points are saved.

    Returns:
        Path of the saved bullseye points.

    Raises:
        FileNotFoundError: If the subject has no difference mesh.
    """
    mesh_path = work_dir / DELTA_MESH
    polar_path = work_dir / f"{name}.npz"
    if not mesh_path.is_file():
        raise FileNotFoundError(f"{mesh_path} (run mesh_delta.py first)")
    if overwrite or not polar_path.is_file() or polar_path.stat().st_mtime < mesh_path.stat().st_mtime:
        mesh = utils.read_vtk_mesh(str(mesh_path))
        if make_mesh is not None:
            mesh = make_mesh(mesh)
        keep_dir = work_dir / f"projection_{name}" if keep_nifti else None
        np.savez_compressed(polar_path, **project_mesh(fixed_dir, mesh, keep_dir))
    return polar_path


def polar_task(
    cgps1: str, cgps10: str, folders: dict[str, Path], run_folder: Path, keep_nifti: bool, overwrite: bool
) -> dict:
    """Project one subject's raw difference mesh into the bullseye."""
    work_dir = run_folder / f"{cgps1}_{cgps10}"
    cached_polar(work_dir, folders["CGPS1"] / cgps1, POLAR_NAME, None, keep_nifti, overwrite)
    remove_intermediates(work_dir)
    return {"CGPS1": cgps1, "CGPS10": cgps10}


def load_polar(path: Path) -> dict[str, np.ndarray]:
    """Saved bullseye points with ``theta``, ``radius`` and ``values``."""
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


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
    """Mean value per cell of a regular polar grid, NaN for cells without points and inside the apex ring.

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


def draw_polar(ax: plt.Axes, points: dict[str, np.ndarray], norm: Normalize, title: str) -> None:
    """Draw one bullseye of bullseye points: grey where there is no data, the apex left white.

    Args:
        ax: Polar axis to draw on.
        points: Bullseye points with ``theta``, ``radius`` and ``values``.
        norm: Colour normalisation.
        title: Title of the axis.
    """
    theta_edges, radius_edges, image = polar_image(points)
    ax.bar(0, 1, 2 * np.pi, 0, color=MISSING_COLOUR, edgecolor="none")
    ax.pcolormesh(theta_edges, radius_edges, image, cmap=CMAP, norm=norm, shading="flat", rasterized=True)
    ax.bar(0, RADII[0], 2 * np.pi, 0, color="white", edgecolor="none")
    draw_outline(ax)
    ax.set(ylim=[0, 1])
    ax.axis("off")
    ax.set_title(title, fontsize=9)


def bullseye_grid(
    polar: dict[str, dict[str, np.ndarray]], vmax: float, label: str = COLOURBAR_LABEL, n_columns: int = 5
) -> plt.Figure:
    """Grid of single-subject bullseyes on one symmetric colour scale.

    Args:
        polar: Panel title to its bullseye points, with ``theta``, ``radius`` and ``values``.
        vmax: Largest absolute value of the colour scale.
        label: Colour bar label.
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
    for ax, (title, points) in zip(axes.flat, polar.items()):
        draw_polar(ax, points, norm, title)
    for ax in axes.flat[len(polar) :]:
        ax.axis("off")
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=CMAP), ax=axes, orientation="horizontal", shrink=0.4, aspect=40
    )
    colorbar.set_label(label)
    return fig


def colour_limit(polar: dict[str, dict[str, np.ndarray]], vmax: float | None) -> float:
    """The given colour scale limit, or the 98th percentile of the absolute values of all bullseye points."""
    if vmax:
        return vmax
    return float(np.nanpercentile(np.abs(np.concatenate([points["values"] for points in polar.values()])), 98))


def figure_name(args: argparse.Namespace) -> str:
    """Name part of a figure for the chosen subjects."""
    return "subjects" if args.subjects else f"random{args.n}_seed{args.seed}"


def parse_args() -> argparse.Namespace:
    """Command line arguments."""
    parser = argparse.ArgumentParser(description="Bullseyes of the per-subject CGPS10 - CGPS1 difference meshes.")
    add_subject_arguments(parser)
    parser.add_argument("--vmax", type=float, help="Colour scale limit in mm (default: 98th percentile of |change|).")
    parser.add_argument("--keep-nifti", action="store_true", help="Keep the projection's NIfTI images, e.g. for 3D.")
    parser.add_argument("--overwrite", action="store_true", help="Project again even if the bullseye is saved.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root, folders, run_folder, to_process, chosen = setup_subjects(args)
    task = partial(
        polar_task, folders=folders, run_folder=run_folder, keep_nifti=args.keep_nifti, overwrite=args.overwrite
    )
    done = {
        (row["CGPS1"], row["CGPS10"])
        for row in run_subjects(task, pair_list(to_process), args.workers, lambda result: "projected")
    }
    polar = {
        f"{cgps1} / {cgps10}": load_polar(run_folder / f"{cgps1}_{cgps10}" / f"{POLAR_NAME}.npz")
        for cgps1, cgps10 in pair_list(chosen)
        if (cgps1, cgps10) in done
    }
    if not polar:
        raise SystemExit("None of the chosen subjects could be projected.")

    vmax = colour_limit(polar, args.vmax)
    print(f"Colour scale: -{vmax:.1f} to {vmax:.1f} mm")
    figure_folder = Path(root) / FIGURE_FOLDER
    figure_folder.mkdir(parents=True, exist_ok=True)
    figure_path = figure_folder / f"bullseye_mesh_delta_{figure_name(args)}.png"
    figure = bullseye_grid(polar, vmax, f"{COLOURBAR_LABEL}; no smoothing")
    figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Figure saved to: {figure_path}")

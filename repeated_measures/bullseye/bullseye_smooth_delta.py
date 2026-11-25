#################################################################################################################################################
# This script smooths the per-subject change in LV wall thickness (CGPS10 - CGPS1, in mm) on the mesh with a Gaussian kernel and shows it
# as bullseye plots, one figure per kernel width, so the effect of the smoothing can be compared with the raw maps. It is the third of
# three steps: mesh_delta.py, bullseye_mesh_delta.py and bullseye_smooth_delta.py (this).
# The smoothing follows the wall: heat diffusion on each subject's CGPS1 endocardial mesh (cotangent Laplacian, lumped vertex areas,
# implicit steps) for a time sigma^2 / 2, which is a Gaussian with standard deviation sigma in mm along the surface, so values do not
# leak across the cavity. The CGPS1 and the moved CGPS10 thickness are smoothed with the same kernel over the vertices that have both,
# as normalised convolution (smoothed value * mask / smoothed mask), so unmatched vertices get no weight but small gaps are filled by
# their neighbours; vertices where less than half of the kernel has data stay empty. The change is the difference of the smoothed
# thicknesses. --fwhm-mm sets the kernel widths (full width at half maximum, FWHM = 2.355 sigma); 0 is the raw change.
# Each subject's bullseye points are saved as polar_smooth_fwhm<F>mm.npz next to its delta_mesh.vtk (remade when the difference mesh
# is newer; FWHM 0 is the polar_delta.npz of bullseye_mesh_delta.py). One figure per FWHM of the --n random subjects (the same as in
# the other two scripts for the same --n and --seed), all on one colour scale, and a figure comparing the FWHMs side by side for the
# first --compare of them, are saved in <PROJECT_ROOT>/reports/figures/repeated_measures/bullseye. With --all every subject is smoothed.
# It needs the difference meshes in <PROJECT_ROOT>/output/CGPS1_CGPS10_mesh_delta_0mm/, so run mesh_delta.py first.
# e.g. uv run repeated_measures/bullseye/bullseye_smooth_delta.py --fwhm-mm 0 5 10 15 --n 20 --seed 0 --workers 4
#################################################################################################################################################

import argparse
import os
import sys
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sparse  # noqa: E402
import vtk  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from scipy.sparse.linalg import factorized  # noqa: E402
from vtk.util.numpy_support import vtk_to_numpy  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bullseye_mesh_delta import (  # noqa: E402
    CMAP,
    COLOURBAR_LABEL,
    FIGURE_FOLDER,
    POLAR_NAME,
    bullseye_grid,
    cached_polar,
    colour_limit,
    draw_polar,
    figure_name,
    load_polar,
)
from mesh_delta import (  # noqa: E402
    add_subject_arguments,
    mesh_points,
    pair_list,
    point_array,
    run_subjects,
    setup_subjects,
    with_arrays,
)

FWHM_MM = [0.0, 5.0, 10.0, 15.0]
FWHM_PER_SIGMA = 2 * np.sqrt(2 * np.log(2))
HEAT_STEPS = 10
MIN_WEIGHT = 0.5
N_COMPARE = 5


def triangles(mesh: vtk.vtkPolyData) -> np.ndarray:
    """(F, 3) vertex ids of the triangles of a mesh made only of triangles."""
    polys = mesh.GetPolys()
    sizes = np.diff(vtk_to_numpy(polys.GetOffsetsArray()))
    if not np.all(sizes == 3):
        raise ValueError("The mesh has cells that are not triangles.")
    return vtk_to_numpy(polys.GetConnectivityArray()).reshape(-1, 3).astype(np.int64)


def cotangent_laplacian(points: np.ndarray, faces: np.ndarray) -> tuple[sparse.csc_matrix, np.ndarray]:
    """Cotangent Laplacian (stiffness matrix) and lumped vertex areas of a triangle mesh.

    Triangles without area are left out.

    Args:
        points: (N, 3) vertex coordinates.
        faces: (F, 3) vertex ids of the triangles.

    Returns:
        The positive semi-definite (N, N) Laplacian ``L`` and the (N,) vertex areas (a third of the area of every
        triangle around the vertex), so that ``M du/dt = -L u`` with ``M = diag(areas)`` is heat diffusion on the
        surface.
    """
    corners = points[faces]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    double_area = np.linalg.norm(normals, axis=1)
    keep = double_area > 1e-12
    faces, corners, double_area = faces[keep], corners[keep], double_area[keep]

    rows, cols, weights = [], [], []
    for corner in range(3):
        a, b = (corner + 1) % 3, (corner + 2) % 3
        u = corners[:, a] - corners[:, corner]
        v = corners[:, b] - corners[:, corner]
        cotangent = np.einsum("ij,ij->i", u, v) / double_area
        rows.extend([faces[:, a], faces[:, b]])
        cols.extend([faces[:, b], faces[:, a]])
        weights.extend([cotangent / 2, cotangent / 2])
    n = len(points)
    adjacency = sparse.coo_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))), (n, n))
    adjacency = adjacency.tocsr()
    laplacian = sparse.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    areas = np.bincount(faces.ravel(), weights=np.repeat(double_area / 6, 3), minlength=n)
    return laplacian.tocsc(), areas


class HeatSmoother:
    """Gaussian smoothing along a triangle mesh by heat diffusion with implicit Euler steps.

    Diffusing for a time ``sigma**2 / 2`` spreads a point into a Gaussian with standard deviation ``sigma`` (on a flat
    surface); several smaller implicit steps keep the kernel close to a Gaussian.
    """

    def __init__(self, points: np.ndarray, faces: np.ndarray, sigma: float, steps: int = HEAT_STEPS) -> None:
        """Factorise the diffusion step of the mesh.

        Args:
            points: (N, 3) vertex coordinates in mm.
            faces: (F, 3) vertex ids of the triangles.
            sigma: Standard deviation of the Gaussian kernel in mm.
            steps: Number of implicit steps.
        """
        laplacian, areas = cotangent_laplacian(points, faces)
        self.areas = np.maximum(areas, 1e-12)
        self.steps = steps
        step = sigma**2 / 2 / steps
        self.solve = factorized((sparse.diags(self.areas) + step * laplacian).tocsc())

    def smooth(self, values: np.ndarray) -> np.ndarray:
        """Diffuse (N,) vertex values, which must all be finite."""
        smoothed = np.asarray(values, dtype=np.float64)
        for _ in range(self.steps):
            smoothed = self.solve(self.areas * smoothed)
        return smoothed

    def smooth_masked(self, values: np.ndarray, mask: np.ndarray, min_weight: float = MIN_WEIGHT) -> np.ndarray:
        """Normalised convolution: smooth only the vertices in ``mask``, filling the others from their neighbours.

        Args:
            values: (N,) vertex values.
            mask: (N,) vertices that carry a value.
            min_weight: Vertices where less than this share of the kernel has data get NaN.

        Returns:
            (N,) smoothed values.
        """
        weight = self.smooth(mask.astype(np.float64))
        total = self.smooth(np.where(mask, values, 0.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(weight >= min_weight, total / weight, np.nan)


def smoothed_delta_mesh(mesh: vtk.vtkPolyData, fwhm: float) -> vtk.vtkPolyData:
    """Difference mesh with the thicknesses smoothed over the matched vertices and their difference as active scalars.

    Args:
        mesh: Difference mesh from mesh_delta.py with ``thickness_CGPS1`` and ``thickness_CGPS10``.
        fwhm: Full width at half maximum of the Gaussian kernel in mm.

    Returns:
        The mesh with ``thickness_CGPS1_smooth``, ``thickness_CGPS10_smooth`` and the active scalars
        ``delta_mm_smooth`` (NaN where too little of the kernel has data).
    """
    cgps1 = point_array(mesh, "thickness_CGPS1")
    cgps10 = point_array(mesh, "thickness_CGPS10")
    mask = np.isfinite(cgps1) & np.isfinite(cgps10)
    smoother = HeatSmoother(mesh_points(mesh), triangles(mesh), fwhm / FWHM_PER_SIGMA)
    smooth1 = smoother.smooth_masked(cgps1, mask)
    smooth10 = smoother.smooth_masked(cgps10, mask)
    arrays = {
        "thickness_CGPS1_smooth": smooth1,
        "thickness_CGPS10_smooth": smooth10,
        "delta_mm_smooth": smooth10 - smooth1,
    }
    return with_arrays(mesh, arrays, "delta_mm_smooth")


def polar_name(fwhm: float) -> str:
    """Name of the saved bullseye points for a kernel width; FWHM 0 is the raw change of bullseye_mesh_delta.py."""
    return POLAR_NAME if fwhm == 0 else f"polar_smooth_fwhm{fwhm:g}mm"


def smooth_task(
    cgps1: str,
    cgps10: str,
    folders: dict[str, Path],
    run_folder: Path,
    fwhms: list[float],
    keep_nifti: bool,
    overwrite: bool,
) -> dict:
    """Smooth one subject's difference mesh with every kernel width and project it into the bullseye."""
    work_dir = run_folder / f"{cgps1}_{cgps10}"
    for fwhm in fwhms:
        make_mesh = None if fwhm == 0 else partial(smoothed_delta_mesh, fwhm=fwhm)
        cached_polar(work_dir, folders["CGPS1"] / cgps1, polar_name(fwhm), make_mesh, keep_nifti, overwrite)
    return {"CGPS1": cgps1, "CGPS10": cgps10}


def fwhm_title(fwhm: float) -> str:
    """Short description of a kernel width."""
    return "no smoothing" if fwhm == 0 else f"Gaussian FWHM {fwhm:g} mm"


def comparison_figure(
    polar: dict[float, dict[str, dict[str, np.ndarray]]], subjects: list[str], vmax: float
) -> plt.Figure:
    """Bullseyes of some subjects (rows) at every kernel width (columns) on one colour scale.

    Args:
        polar: FWHM to subject label to its bullseye points.
        subjects: Subject labels, one row each.
        vmax: Largest absolute change in mm of the colour scale.

    Returns:
        The figure.
    """
    fwhms = list(polar)
    fig, axes = plt.subplots(
        len(subjects),
        len(fwhms),
        figsize=(3.2 * len(fwhms), 3.3 * len(subjects)),
        subplot_kw=dict(polar=True),
        layout="constrained",
        squeeze=False,
    )
    norm = Normalize(vmin=-vmax, vmax=vmax, clip=True)
    for row, subject in enumerate(subjects):
        for column, fwhm in enumerate(fwhms):
            title = f"{subject}\n{fwhm_title(fwhm)}" if column == 0 else fwhm_title(fwhm)
            draw_polar(axes[row, column], polar[fwhm][subject], norm, title)
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=CMAP), ax=axes, orientation="horizontal", shrink=0.5, aspect=40
    )
    colorbar.set_label(COLOURBAR_LABEL)
    return fig


def parse_args() -> argparse.Namespace:
    """Command line arguments."""
    parser = argparse.ArgumentParser(description="Bullseyes of the per-subject change after smoothing on the mesh.")
    add_subject_arguments(parser)
    parser.add_argument("--fwhm-mm", type=float, nargs="+", default=FWHM_MM, help="Kernel widths; 0 is no smoothing.")
    parser.add_argument("--compare", type=int, default=N_COMPARE, help="Subjects in the side-by-side figure.")
    parser.add_argument("--vmax", type=float, help="Colour scale limit in mm (default: 98th percentile of |change|).")
    parser.add_argument("--keep-nifti", action="store_true", help="Keep the projection's NIfTI images, e.g. for 3D.")
    parser.add_argument("--overwrite", action="store_true", help="Smooth and project again even if saved.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root, folders, run_folder, to_process, chosen = setup_subjects(args)
    fwhms = sorted(set(args.fwhm_mm))
    task = partial(
        smooth_task,
        folders=folders,
        run_folder=run_folder,
        fwhms=fwhms,
        keep_nifti=args.keep_nifti,
        overwrite=args.overwrite,
    )
    results = run_subjects(task, pair_list(to_process), args.workers, lambda result: "smoothed")
    done = {(row["CGPS1"], row["CGPS10"]) for row in results}
    shown = [(cgps1, cgps10) for cgps1, cgps10 in pair_list(chosen) if (cgps1, cgps10) in done]
    if not shown:
        raise SystemExit("None of the chosen subjects could be smoothed.")
    polar = {
        fwhm: {
            f"{cgps1} / {cgps10}": load_polar(run_folder / f"{cgps1}_{cgps10}" / f"{polar_name(fwhm)}.npz")
            for cgps1, cgps10 in shown
        }
        for fwhm in fwhms
    }

    vmax = colour_limit(
        {f"{fwhm}/{label}": points for fwhm in fwhms for label, points in polar[fwhm].items()}, args.vmax
    )
    print(f"Colour scale: -{vmax:.1f} to {vmax:.1f} mm")
    figure_folder = Path(root) / FIGURE_FOLDER
    figure_folder.mkdir(parents=True, exist_ok=True)
    for fwhm in fwhms:
        path = figure_folder / f"bullseye_smooth_delta_fwhm{fwhm:g}mm_{figure_name(args)}.png"
        figure = bullseye_grid(polar[fwhm], vmax, f"{COLOURBAR_LABEL}; {fwhm_title(fwhm)}")
        figure.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        print(f"Figure saved to: {path}")
    path = figure_folder / f"bullseye_smooth_delta_comparison_{figure_name(args)}.png"
    figure = comparison_figure(polar, list(polar[fwhms[0]])[: args.compare], vmax)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Figure saved to: {path}")

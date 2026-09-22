"""Per-patient LVM wall-thickness pipeline.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator

SCAN_TYPE = "ED"
IMAGE_SUBDIR = os.path.join("data", "1_200")
DONE_MARKER = "done.json"


@dataclass(frozen=True)
class PatientPaths:
    """Input and output locations for one patient.

    Attributes:
        patient_id: Dataset id, e.g. ``"7"``.
        image: CT image path.
        segmentation: TotalSegmentator heart-chambers segmentation path.
        save_dir: Directory all outputs for this patient are written to.
    """

    patient_id: str
    image: str
    segmentation: str
    save_dir: str

    @property
    def done_marker(self) -> str:
        """Path of the file written only after every step has succeeded."""
        return os.path.join(self.save_dir, DONE_MARKER)

    def is_done(self) -> bool:
        """Whether a previous run finished this patient."""
        return os.path.isfile(self.done_marker)


def patient_paths(root: str, segmentation_folder: str, patient_id: str, output_dir: str | None = None) -> PatientPaths:
    """Build the input/output paths for a patient.

    Args:
        root: Project root containing ``data/1_200``.
        segmentation_folder: Folder containing ``<id>.heart.nii.gz`` files.
        patient_id: Dataset id.
        output_dir: Parent of the per-patient output folders; defaults to ``<root>/output``.

    Returns:
        The patient's paths.
    """
    output_dir = output_dir or os.path.join(root, "output")
    return PatientPaths(
        patient_id=patient_id,
        image=os.path.join(root, IMAGE_SUBDIR, f"{patient_id}.img.nii.gz"),
        segmentation=os.path.join(segmentation_folder, f"{patient_id}.heart.nii.gz"),
        save_dir=os.path.join(output_dir, patient_id),
    )


def discover_patients(root: str, segmentation_folder: str) -> list[str]:
    """List patient ids that have both a CT image and a heart segmentation.

    Args:
        root: Project root containing ``data/1_200``.
        segmentation_folder: Folder containing ``<id>.heart.nii.gz`` files.

    Returns:
        Patient ids, numeric ids in numeric order.
    """
    image_ids = _ids_with_suffix(os.path.join(root, IMAGE_SUBDIR), ".img.nii.gz")
    seg_ids = _ids_with_suffix(segmentation_folder, ".heart.nii.gz")
    return sorted(image_ids & seg_ids, key=lambda pid: (0, int(pid), "") if pid.isdigit() else (1, 0, pid))


def _ids_with_suffix(folder: str, suffix: str) -> set[str]:
    return {name[: -len(suffix)] for name in os.listdir(folder) if name.endswith(suffix)}


def peak_memory_bytes() -> int | None:
    """Peak resident memory of the current process, or None if it cannot be measured."""
    try:
        import psutil

        peak = getattr(psutil.Process().memory_info(), "peak_wset", None)
        if peak:
            return int(peak)
    except ImportError:
        pass
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except ImportError:
        return None


def process_patient(
    paths: PatientPaths,
    plots: bool = True,
    save_debug_meshes: bool = True,
    log: Callable[[str], None] = print,
) -> dict[str, float]:
    """Run every pipeline step for one patient, from scratch.

    Nothing from an earlier, possibly interrupted, run is reused: steps 2 and 3
    used to be skipped whenever their output files existed, which silently kept
    stale results after code changes. ``done.json`` is written last, so its
    presence means every output in ``save_dir`` came from one complete run.

    Args:
        paths: The patient's input and output paths.
        plots: Write the thickness histograms and the bullseye plot.
        save_debug_meshes: Write ``surfaces/vectors.vtk``, a QA-only mesh (~7s).
        log: Receives one progress line per step.

    Returns:
        Seconds spent in each step.

    Raises:
        FileNotFoundError: If the image or segmentation is missing.
    """
    # Imported here so the dataset runner can discover patients without loading VTK/ITK.
    from create_medial_sheet import create_medial_sheet
    from lv_generate_segments import wrap_lv_segments
    from thickness_estimation import (
        create_meshes_from_segmentation,
        mesh_vector_allignment,
        plot_thickness_histograms,
        thickness_in_17_seg,
    )
    from utils import bullseye, utils

    for required in (paths.image, paths.segmentation):
        if not os.path.isfile(required):
            raise FileNotFoundError(required)

    save_dir = paths.save_dir
    os.makedirs(save_dir, exist_ok=True)
    if paths.is_done():
        os.remove(paths.done_marker)

    timings: dict[str, float] = {}

    @contextmanager
    def step(name: str, description: str) -> Iterator[None]:
        log(f"[{name}] {description}...")
        start = time.perf_counter()
        yield
        timings[name] = round(time.perf_counter() - start, 2)
        log(f"  -> {name} completed in {timings[name]:.1f}s")

    with step("1_meshes", "Creating meshes from segmentation"):
        mesh_inner, mesh_outer = create_meshes_from_segmentation(paths.segmentation, save_dir)

    with step("2_alignment", "Vector alignment"):
        mesh_thick, _, _ = mesh_vector_allignment(
            save_dir, mesh_inner, mesh_outer, num_iterations=3, num_closest_vectors=20, exists_ok=False,
            save_vectors_mesh=save_debug_meshes,
        )

    with step("3_lv17", "Preparing 17-segment model"):
        wrap_lv_segments(save_dir, paths.segmentation, paths.image, individual_transforms=True, verbose=False)

    with step("4_thickness", "Calculating thickness per segment"):
        mesh_17 = utils.read_vtk_mesh(os.path.join(save_dir, "surfaces", "myocardium_17.vtk"))
        thickness = thickness_in_17_seg(mesh_thick, mesh_17)
        with open(os.path.join(save_dir, "thickness.json"), "w") as f:
            json.dump(dict(thickness), f)

    if plots:
        with step("5_histograms", "Plotting thickness histograms"):
            plot_thickness_histograms(save_dir, name=paths.patient_id)

    with step("6_medial_sheet", "Creating medial sheet"):
        medial_path = os.path.join(save_dir, "surfaces", "medial_sheet.vtk")
        if os.path.exists(medial_path):
            os.remove(medial_path)
        create_medial_sheet(save_dir)
        if not os.path.isfile(medial_path):
            raise RuntimeError(f"create_medial_sheet did not write {medial_path}; see its logged error")

    if plots:
        with step("7_bullseye", "Generating bullseye plot"):
            bullseye.create_single_bs_from_mesh(
                save_dir,
                os.path.join(save_dir, "surfaces", "dist_source.vtk"),
                paths.segmentation,
                global_min=0,
                global_max=20,
                savename=f"thickness_{paths.patient_id.split('_')[-1]}_{SCAN_TYPE}",
            )

    peak = peak_memory_bytes()
    with open(paths.done_marker, "w") as f:
        json.dump(
            {
                "patient_id": paths.patient_id,
                "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                "seconds": timings,
                "total_seconds": round(sum(timings.values()), 2),
                "peak_memory_gb": round(peak / 1e9, 2) if peak else None,
            },
            f,
            indent=2,
        )
    return timings

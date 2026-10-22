"""Per-patient LVM wall-thickness pipeline.
"""
from __future__ import annotations

import glob
import json
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator

SCAN_TYPE = "ED"
DEFAULT_IMAGE_PATTERN = "{id}.img.nii.gz"
DEFAULT_SEGMENTATION_PATTERN = "{id}.heart.nii.gz"
DONE_MARKER = "done.json"
AORTA_EXCLUSION_MM = 0


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


@dataclass(frozen=True)
class DataSource:
    """Where the CT images and segmentations of a dataset live and how their files are named.

    Patterns are paths relative to their folder in which ``{id}`` marks the patient id and ``*`` matches any
    characters except ``/``, e.g. ``"CGPS-10y-august-2026_{id}_*/segmentations/heartchambers_highres.nii.gz"``.

    Attributes:
        image_folder: Folder containing the CT images.
        segmentation_folder: Folder containing the TotalSegmentator heart-chambers segmentations.
        image_pattern: Image path pattern relative to ``image_folder``.
        segmentation_pattern: Segmentation path pattern relative to ``segmentation_folder``.
    """

    image_folder: str
    segmentation_folder: str
    image_pattern: str = DEFAULT_IMAGE_PATTERN
    segmentation_pattern: str = DEFAULT_SEGMENTATION_PATTERN

    def image_path(self, patient_id: str) -> str:
        """The patient's CT image path (the unresolved pattern if no file matches)."""
        return _resolve(self.image_folder, self.image_pattern, patient_id)

    def segmentation_path(self, patient_id: str) -> str:
        """The patient's segmentation path (the unresolved pattern if no file matches)."""
        return _resolve(self.segmentation_folder, self.segmentation_pattern, patient_id)


def _resolve(folder: str, pattern: str, patient_id: str) -> str:
    """Path of the single file matching ``pattern`` for ``patient_id``.

    Raises:
        ValueError: If more than one file matches.
    """
    path = os.path.join(folder, pattern.replace("{id}", glob.escape(patient_id)))
    matches = sorted(glob.glob(path))
    if len(matches) > 1:
        raise ValueError(f"Patient {patient_id} matches {len(matches)} files for {path}: {matches}")
    return matches[0] if matches else path


def _ids_matching(folder: str, pattern: str) -> dict[str, list[str]]:
    """Map every patient id found in ``folder`` to the files matching ``pattern`` for it."""
    regex = re.compile(
        "".join(
            "(?P<id>[^/]+?)" if part == "{id}" else "[^/]*" if part == "*" else re.escape(part)
            for part in re.split(r"(\{id\}|\*)", pattern)
        )
    )
    found: dict[str, list[str]] = {}
    for path in glob.glob(os.path.join(folder, pattern.replace("{id}", "*"))):
        match = regex.fullmatch(os.path.relpath(path, folder).replace(os.sep, "/"))
        if match:
            found.setdefault(match["id"], []).append(path)
    return found


def load_environment() -> tuple[str, DataSource]:
    """Load the project root and data locations from the repo-root .env or the environment.

    ``PROJECT_ROOT`` is required. ``IMAGE_FOLDER`` and ``SEGMENTATION_FOLDER`` default to ``<PROJECT_ROOT>/data/1_200``
    and ``<PROJECT_ROOT>/data/TotalSegmentator``; ``IMAGE_PATTERN`` and ``SEGMENTATION_PATTERN`` default to the local
    ``<id>.img.nii.gz`` / ``<id>.heart.nii.gz`` naming.

    Returns:
        Tuple of (project root, data source).
    """
    from dotenv import load_dotenv

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(repo_root, ".env"))
    root = os.environ.get("PROJECT_ROOT")
    if not root:
        raise SystemExit(
            "\n[ERROR] Missing required environment variable 'PROJECT_ROOT'.\n"
            "Set it in a .env file at the repo root (see .env.example), or export it in your shell/job script."
        )
    data = DataSource(
        image_folder=os.environ.get("IMAGE_FOLDER", os.path.join(root, "data", "1_200")),
        segmentation_folder=os.environ.get("SEGMENTATION_FOLDER", os.path.join(root, "data", "TotalSegmentator")),
        image_pattern=os.environ.get("IMAGE_PATTERN", DEFAULT_IMAGE_PATTERN),
        segmentation_pattern=os.environ.get("SEGMENTATION_PATTERN", DEFAULT_SEGMENTATION_PATTERN),
    )
    return root, data


def patient_paths(root: str, data: DataSource, patient_id: str, output_dir: str | None = None) -> PatientPaths:
    """Build the input/output paths for a patient.

    Args:
        root: Project root.
        data: Location and naming of the images and segmentations.
        patient_id: Dataset id.
        output_dir: Parent of the per-patient output folders; defaults to ``<root>/output``.

    Returns:
        The patient's paths.
    """
    output_dir = output_dir or os.path.join(root, "output")
    return PatientPaths(
        patient_id=patient_id,
        image=data.image_path(patient_id),
        segmentation=data.segmentation_path(patient_id),
        save_dir=os.path.join(output_dir, patient_id),
    )


def run_output_dir(root: str, dataset: str, aorta_exclusion_mm: float, output_dir: str | None = None) -> str:
    """Folder one run writes all its patients to: ``<output_dir>/<dataset>_<aorta_exclusion_mm>``.

    Args:
        root: Project root.
        dataset: Dataset name.
        aorta_exclusion_mm: Aortic exclusion distance in mm, e.g. 2.5 gives ``<dataset>_2.5``.
        output_dir: Parent of the run folders; defaults to ``<root>/output``.

    Returns:
        Absolute path of the run folder.
    """
    parent = output_dir or os.path.join(root, "output")
    return os.path.abspath(os.path.join(parent, f"{dataset}_{aorta_exclusion_mm:g}mm")) # end with mm e.g. "2.5mm"


def table_output_dir(root: str, dataset: str, aorta_exclusion_mm: float, n_patients: int) -> str:
    """Folder the result tables of one run are written to: ``<root>/df/<dataset>_<aorta_exclusion_mm>mm_n<n_patients>``.

    Args:
        root: Project root.
        dataset: Dataset name.
        aorta_exclusion_mm: Aortic exclusion distance in mm of the run.
        n_patients: Number of patients the tables are made from.

    Returns:
        Absolute path of the table folder.
    """
    return os.path.abspath(os.path.join(root, "df", f"{dataset}_{aorta_exclusion_mm:g}mm_n{n_patients}"))


def discover_patients(data: DataSource, require_image: bool = True) -> list[str]:
    """List patient ids that have exactly one heart segmentation and, by default, exactly one CT image.

    Ids matching several files for the image or segmentation pattern are skipped with a warning.

    Args:
        data: Location and naming of the images and segmentations.
        require_image: Only return ids that also have a CT image.

    Returns:
        Patient ids, numeric ids in numeric order.
    """
    segmentations = _ids_matching(data.segmentation_folder, data.segmentation_pattern)
    images = _ids_matching(data.image_folder, data.image_pattern) if require_image else {}
    ids = set(segmentations) & set(images) if require_image else set(segmentations)
    ambiguous = {pid for pid in ids if len(images.get(pid, [])) > 1 or len(segmentations[pid]) > 1}
    for pid in sorted(ambiguous):
        print(f"[WARN] Skipping patient {pid}: several matching files {images.get(pid, []) + segmentations[pid]}")
    return sorted(ids - ambiguous, key=lambda pid: (0, int(pid), "") if pid.isdigit() else (1, 0, pid))


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
    aorta_exclusion_mm: float = AORTA_EXCLUSION_MM,
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
        aorta_exclusion_mm: Mesh vertices within this distance (mm) of the aorta are left out of the thickness
            statistics and the bullseye plot; 0 disables it.

    Returns:
        Seconds spent in each step.

    Raises:
        FileNotFoundError: If the image or segmentation is missing.
    """
    # Imported here so the dataset runner can discover patients without loading VTK/ITK.
    import SimpleITK as sitk
    from create_medial_sheet import create_medial_sheet
    from lv_generate_segments import wrap_lv_segments
    from thickness_estimation import (
        create_meshes_from_segmentation,
        mesh_vector_allignment,
        plot_thickness_histograms,
        thickness_in_17_seg,
    )
    from utils import bullseye, utils
    from vtk.util.numpy_support import vtk_to_numpy

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
        vertices = vtk_to_numpy(mesh_thick.GetPoints().GetData())
        keep = utils.aorta_keep_mask(sitk.ReadImage(paths.segmentation), vertices, aorta_exclusion_mm)
        thickness = thickness_in_17_seg(mesh_thick, mesh_17, keep=keep)
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
                aorta_exclusion_mm=aorta_exclusion_mm,
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

"""Extract en-face images of the aortic and mitral valves from cardiac CT.

For every scan the valve planes are located from the contact surfaces between TotalSegmentator
``heartchambers_highres`` structures (LV/aorta for the aortic valve, LA/LV for the mitral valve). Three parallel
slices perpendicular to each valve normal are sampled and stored as the RGB channels of a PNG, together with a JSON
file holding the valve centre and normal so the crop can be reproduced.

Example:
    uv run side_project/valve_planes/valve_planes.py data/1_200 data/TotalSegmentator side_project/valve_planes/output
"""
from __future__ import annotations

import glob
import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, replace

import numpy as np
import SimpleITK as sitk
import typer
from PIL import Image
from scipy.ndimage import distance_transform_edt, map_coordinates

MYOCARDIUM_ID = 1
LEFT_ATRIUM_ID = 2
LEFT_VENTRICLE_ID = 3
AORTA_ID = 6
BLOOD_POOL_IDS = (LEFT_ATRIUM_ID, LEFT_VENTRICLE_ID, AORTA_ID)

CONTACT_THRESHOLDS_MM = (1.5, 2.0, 3.0, 4.0, 5.0)
MIN_CONTACT_VOXELS = 50
LOCAL_RADIUS_MM = 20.0
AIR_HU = -1024.0
VIEW_NAMES = {
    "orthogonal": ("en_face", "long_axis_a", "long_axis_b"),
    "parallel": ("upstream", "valve", "downstream"),
}
WINDOW_MODES = ("adaptive", "fixed")

logger = logging.getLogger("valve_planes")


@dataclass(frozen=True)
class Valve:
    """A valve defined by the two structures it separates, in the direction of blood flow.

    Attributes:
        name: Valve name used in file names.
        upstream_id: Label of the structure blood flows from.
        downstream_id: Label of the structure blood flows into.
    """

    name: str
    upstream_id: int
    downstream_id: int


VALVES = (
    Valve("aortic", LEFT_VENTRICLE_ID, AORTA_ID),
    Valve("mitral", LEFT_ATRIUM_ID, LEFT_VENTRICLE_ID),
)


@dataclass(frozen=True)
class SliceSettings:
    """How the valve image is sampled and windowed.

    Attributes:
        field_of_view_mm: Side length of the square image in mm.
        resolution_mm: Pixel size in mm.
        slab_offset_mm: Distance of the outer slices from the valve plane in mm (``parallel`` views only).
        hu_min: Lower bound of the HU window (replaced per scan in ``adaptive`` mode).
        hu_max: Upper bound of the HU window (replaced per scan in ``adaptive`` mode).
        views: ``"orthogonal"`` for an en-face view and two long-axis views through the valve centre, or
            ``"parallel"`` for three en-face slices ``slab_offset_mm`` apart.
        window: ``"adaptive"`` to derive the HU window from the scan's myocardium and blood pool (see
            ``adaptive_window``), or ``"fixed"`` to use ``hu_min`` and ``hu_max`` as given.
    """

    field_of_view_mm: float = 60.0
    resolution_mm: float = 0.5
    slab_offset_mm: float = 2.0
    hu_min: float = -200.0
    hu_max: float = 600.0
    views: str = "orthogonal"
    window: str = "adaptive"


@dataclass(frozen=True)
class ValvePlane:
    """Location and orientation of a valve in physical (LPS) coordinates.

    Attributes:
        valve: Valve name.
        center_mm: Weighted centre of the contact surface.
        normal: Unit normal of the valve plane pointing in the direction of blood flow.
        in_plane_u: Unit vector in the valve plane; points towards the other valve once oriented.
        in_plane_v: Unit vector in the valve plane completing the right-handed basis (u, v, normal).
        contact_threshold_mm: Distance threshold that gave enough contact voxels.
        n_contact_voxels: Number of voxels in the contact surface.
    """

    valve: str
    center_mm: list[float]
    normal: list[float]
    in_plane_u: list[float]
    in_plane_v: list[float]
    contact_threshold_mm: float
    n_contact_voxels: int


def index_to_physical(image: sitk.Image, index_xyz: np.ndarray) -> np.ndarray:
    """Convert (N, 3) continuous voxel indices in (x, y, z) order to physical points."""
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    return np.asarray(image.GetOrigin()) + (index_xyz * np.asarray(image.GetSpacing())) @ direction.T


def physical_to_index(image: sitk.Image, points: np.ndarray) -> np.ndarray:
    """Convert (N, 3) physical points to continuous voxel indices in (x, y, z) order."""
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    return ((points - np.asarray(image.GetOrigin())) @ direction) / np.asarray(image.GetSpacing())


def _bounding_box(mask: np.ndarray, margin: int) -> tuple[slice, ...]:
    """Slices of the bounding box of ``mask`` grown by ``margin`` voxels."""
    coords = np.argwhere(mask)
    lo = np.maximum(coords.min(axis=0) - margin, 0)
    hi = np.minimum(coords.max(axis=0) + margin + 1, mask.shape)
    return tuple(slice(a, b) for a, b in zip(lo, hi))


def contact_surface(
    labels: np.ndarray,
    spacing_zyx: np.ndarray,
    valve: Valve,
    thresholds_mm: tuple[float, ...] = CONTACT_THRESHOLDS_MM,
    min_voxels: int = MIN_CONTACT_VOXELS,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Voxels lying close to both structures of a valve.

    The threshold is increased step by step until at least ``min_voxels`` voxels are found, since the segmentation
    sometimes leaves a thin gap between the structures.

    Args:
        labels: Label array in (z, y, x) order.
        spacing_zyx: Voxel spacing in mm in (z, y, x) order.
        valve: The valve to locate.
        thresholds_mm: Increasing distance thresholds to try.
        min_voxels: Minimum number of contact voxels required.

    Returns:
        Tuple of (contact voxel indices in (z, y, x) order, summed distance to both structures in mm, threshold used).

    Raises:
        ValueError: If a structure is missing or no threshold gives enough contact voxels.
    """
    upstream = labels == valve.upstream_id
    downstream = labels == valve.downstream_id
    for structure, label in ((upstream, valve.upstream_id), (downstream, valve.downstream_id)):
        if not structure.any():
            raise ValueError(f"Label {label} needed for the {valve.name} valve is missing")

    margin = int(np.ceil(max(thresholds_mm) / spacing_zyx.min())) + 1
    box = _bounding_box(upstream | downstream, margin)
    offset = np.array([s.start for s in box])
    dist_up = distance_transform_edt(~upstream[box], sampling=spacing_zyx)
    dist_down = distance_transform_edt(~downstream[box], sampling=spacing_zyx)

    for threshold in thresholds_mm:
        contact = (dist_up <= threshold) & (dist_down <= threshold)
        if contact.sum() >= min_voxels:
            coords = np.argwhere(contact)
            return coords + offset, dist_up[contact] + dist_down[contact], threshold
    raise ValueError(f"Fewer than {min_voxels} contact voxels for the {valve.name} valve within {thresholds_mm[-1]} mm")


def _in_plane_axes(normal: np.ndarray, towards: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors (u, v) spanning the plane perpendicular to ``normal`` with ``u x v = normal``.

    ``u`` is the projection of ``towards`` onto the plane if given and not (nearly) parallel to ``normal``; otherwise
    ``v`` is chosen as close to superior as possible.
    """
    if towards is not None:
        u = towards - (towards @ normal) * normal
        if np.linalg.norm(u) > 1e-3 * np.linalg.norm(towards):
            u /= np.linalg.norm(u)
            return u, np.cross(normal, u)
    reference = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(reference, normal)
    u /= np.linalg.norm(u)
    return u, np.cross(normal, u)


def orient_towards(plane: ValvePlane, target_mm: np.ndarray) -> ValvePlane:
    """Rotate the in-plane axes so ``in_plane_u`` points towards ``target_mm`` (e.g. the other valve's centre).

    The long-axis view spanned by ``in_plane_u`` and the normal then contains both valves, similar to a three-chamber
    view, which gives the views the same anatomical meaning in every patient.
    """
    u, v = _in_plane_axes(np.asarray(plane.normal), np.asarray(target_mm) - np.asarray(plane.center_mm))
    return replace(plane, in_plane_u=u.tolist(), in_plane_v=v.tolist())


def locate_valve(image: sitk.Image, labels: np.ndarray, valve: Valve) -> ValvePlane:
    """Find the centre and blood-flow normal of a valve from its contact surface.

    The centre is the mean of the contact voxels weighted towards those closest to both structures. The normal is the
    principal axis of least variance of the (weighted) contact voxels, flipped to point from the upstream to the
    downstream structure.

    Args:
        image: Segmentation image providing the geometry.
        labels: Label array in (z, y, x) order.
        valve: The valve to locate.

    Returns:
        The valve plane.
    """
    spacing_zyx = np.asarray(image.GetSpacing())[::-1]
    coords, summed_distance, threshold = contact_surface(labels, spacing_zyx, valve)
    points = index_to_physical(image, coords[:, ::-1].astype(float))
    weights = np.exp(-summed_distance / threshold)
    weights /= weights.sum()

    center = weights @ points
    centered = points - center
    covariance = (centered * weights[:, None]).T @ centered
    _, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, 0]

    flow = _local_centroid(image, labels, valve.downstream_id, center) - _local_centroid(
        image, labels, valve.upstream_id, center
    )
    if normal @ flow < 0:
        normal = -normal
    u, v = _in_plane_axes(normal)
    return ValvePlane(
        valve=valve.name,
        center_mm=center.tolist(),
        normal=normal.tolist(),
        in_plane_u=u.tolist(),
        in_plane_v=v.tolist(),
        contact_threshold_mm=float(threshold),
        n_contact_voxels=len(coords),
    )


def _local_centroid(image: sitk.Image, labels: np.ndarray, label: int, center: np.ndarray) -> np.ndarray:
    """Centroid of the voxels of ``label`` within ``LOCAL_RADIUS_MM`` of ``center`` (all voxels if none are)."""
    coords = np.argwhere(labels == label)[:, ::-1].astype(float)
    points = index_to_physical(image, coords)
    near = np.linalg.norm(points - center, axis=1) <= LOCAL_RADIUS_MM
    return points[near].mean(axis=0) if near.any() else points.mean(axis=0)


def sample_plane(
    ct: sitk.Image, center: np.ndarray, column_axis: np.ndarray, up_axis: np.ndarray, settings: SliceSettings
) -> np.ndarray:
    """Sample a square HU image centred on ``center``, with columns along ``column_axis`` and ``up_axis`` pointing up.

    Args:
        ct: CT image in HU.
        center: Physical image centre in mm.
        column_axis: Unit vector along increasing columns.
        up_axis: Unit vector towards the top row.
        settings: Field of view and resolution.

    Returns:
        HU array of shape (rows, columns); points outside the scan are ``AIR_HU``.
    """
    n_pixels = int(round(settings.field_of_view_mm / settings.resolution_mm))
    axis = (np.arange(n_pixels) - (n_pixels - 1) / 2) * settings.resolution_mm
    rows, cols = np.meshgrid(axis, axis, indexing="ij")
    points = (center + cols[..., None] * column_axis - rows[..., None] * up_axis).reshape(-1, 3)
    index_zyx = physical_to_index(ct, points)[:, ::-1].T
    volume = sitk.GetArrayViewFromImage(ct)
    return map_coordinates(volume, index_zyx, order=1, cval=AIR_HU).reshape(n_pixels, n_pixels)


def valve_views(ct: sitk.Image, plane: ValvePlane, settings: SliceSettings) -> dict[str, np.ndarray]:
    """Sample the three views of a valve in HU, named as in ``VIEW_NAMES``.

    ``orthogonal``: ``en_face`` (looking along the flow), ``long_axis_a`` (spanned by ``in_plane_u`` and the normal) and
    ``long_axis_b`` (spanned by ``in_plane_v`` and the normal); in the long-axis views blood flows towards the top.
    ``parallel``: en-face slices ``slab_offset_mm`` upstream of, in and downstream of the valve plane.

    Args:
        ct: CT image in HU.
        plane: Valve plane to sample.
        settings: Sampling settings including the view mode.

    Returns:
        Mapping from view name to HU array, in channel order.

    Raises:
        ValueError: If ``settings.views`` is not a known view mode.
    """
    if settings.views not in VIEW_NAMES:
        raise ValueError(f"Unknown view mode '{settings.views}', expected one of {list(VIEW_NAMES)}")
    center, normal = np.asarray(plane.center_mm), np.asarray(plane.normal)
    u, v = np.asarray(plane.in_plane_u), np.asarray(plane.in_plane_v)
    if settings.views == "orthogonal":
        planes = [(center, u, v), (center, u, normal), (center, v, normal)]
    else:
        offset = settings.slab_offset_mm * normal
        planes = [(center - offset, u, v), (center, u, v), (center + offset, u, v)]
    return {
        name: sample_plane(ct, origin, column_axis, up_axis, settings)
        for name, (origin, column_axis, up_axis) in zip(VIEW_NAMES[settings.views], planes)
    }


def adaptive_window(volume: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """HU window that maps the median myocardium and blood pool to the same grey levels in every scan.

    With ``c`` the blood-myocardium contrast the window is ``[myocardium - c, blood + c / 2]``, so myocardium lands at
    40% and blood at 80% of the grey range regardless of the amount of contrast agent. For a typical scan (myocardium
    ~100 HU, blood ~400 HU) this is close to the fixed window of -200 to 600 HU.

    Args:
        volume: CT array in HU, in the same grid as ``labels``.
        labels: Label array.

    Returns:
        Tuple of (hu_min, hu_max).

    Raises:
        ValueError: If the blood pool is not brighter than the myocardium.
    """
    blood = float(np.median(volume[np.isin(labels, BLOOD_POOL_IDS)]))
    myocardium = float(np.median(volume[labels == MYOCARDIUM_ID]))
    contrast = blood - myocardium
    if not contrast > 0:
        raise ValueError(f"Blood pool ({blood:.0f} HU) is not brighter than myocardium ({myocardium:.0f} HU)")
    return myocardium - contrast, blood + contrast / 2


def window(hu: np.ndarray, settings: SliceSettings) -> np.ndarray:
    """Clip HU values to the window in ``settings`` and scale them to uint8."""
    scaled = (np.clip(hu, settings.hu_min, settings.hu_max) - settings.hu_min) / (settings.hu_max - settings.hu_min)
    return np.round(scaled * 255).astype(np.uint8)


def sample_valve_image(ct: sitk.Image, plane: ValvePlane, settings: SliceSettings) -> np.ndarray:
    """Sample the three views of a valve (see ``valve_views``) and stack them windowed as an 8-bit RGB image.

    Returns:
        Array of shape (rows, columns, 3) and dtype uint8.
    """
    return np.stack([window(hu, settings) for hu in valve_views(ct, plane, settings).values()], axis=-1)


def run_totalsegmentator(image_path: str, segmentation_path: str, executable: str = "TotalSegmentator") -> None:
    """Segment the heart chambers with TotalSegmentator ``heartchambers_highres`` into a multi-label file.

    Raises:
        RuntimeError: If the TotalSegmentator executable cannot be found.
        subprocess.CalledProcessError: If TotalSegmentator fails.
    """
    if shutil.which(executable) is None:
        raise RuntimeError(f"'{executable}' not found; install TotalSegmentator or provide existing segmentations")
    os.makedirs(os.path.dirname(os.path.abspath(segmentation_path)), exist_ok=True)
    subprocess.run(
        [executable, "-i", image_path, "-o", segmentation_path, "-ta", "heartchambers_highres", "--ml"],
        check=True,
        capture_output=True,
    )


def find_scans(image_folder: str, image_pattern: str) -> dict[str, str]:
    """Map patient ids to CT paths for all files in ``image_folder`` matching ``image_pattern``.

    ``image_pattern`` is a file name in which ``{id}`` marks the patient id and ``*`` matches anything, e.g.
    ``"{id}.img.nii.gz"``.
    """
    regex = re.compile(
        "".join(
            "(?P<id>.+?)" if part == "{id}" else ".*" if part == "*" else re.escape(part)
            for part in re.split(r"(\{id\}|\*)", image_pattern)
        )
    )
    scans = {}
    for path in sorted(glob.glob(os.path.join(image_folder, image_pattern.replace("{id}", "*")))):
        match = regex.fullmatch(os.path.basename(path))
        if match:
            scans[match["id"]] = path
    return scans


def resolve_segmentation(segmentation_folder: str, segmentation_pattern: str, patient_id: str) -> str:
    """Path of the patient's segmentation; the first glob match if the pattern contains ``*``, else the plain path."""
    path = os.path.join(segmentation_folder, segmentation_pattern.replace("{id}", glob.escape(patient_id)))
    matches = sorted(glob.glob(path))
    return (
        matches[0] if matches else os.path.join(segmentation_folder, segmentation_pattern.replace("{id}", patient_id))
    )


def process_scan(
    patient_id: str,
    image_path: str,
    segmentation_path: str,
    output_folder: str,
    settings: SliceSettings,
    totalsegmentator: str = "TotalSegmentator",
) -> list[ValvePlane]:
    """Segment (if needed), locate both valves and save their images and metadata for one scan.

    Writes, for each valve, ``<id>_<valve>.png`` (the three views as RGB channels) and ``<id>_<valve>_<view>.png``
    (each view in grayscale), plus ``<id>_valves.json`` to ``output_folder``.

    Returns:
        The located valve planes.
    """
    if not os.path.isfile(segmentation_path):
        logger.info(f"{patient_id}: running TotalSegmentator")
        run_totalsegmentator(image_path, segmentation_path, totalsegmentator)

    ct = sitk.Cast(sitk.ReadImage(image_path), sitk.sitkFloat32)
    segmentation = sitk.ReadImage(segmentation_path)
    if segmentation.GetSize() != ct.GetSize():
        segmentation = sitk.Resample(segmentation, ct, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
    labels = sitk.GetArrayFromImage(segmentation)
    if settings.window == "adaptive":
        hu_min, hu_max = adaptive_window(sitk.GetArrayViewFromImage(ct), labels)
        settings = replace(settings, hu_min=hu_min, hu_max=hu_max)

    os.makedirs(output_folder, exist_ok=True)
    aortic, mitral = (locate_valve(segmentation, labels, valve) for valve in VALVES)
    planes = [orient_towards(aortic, mitral.center_mm), orient_towards(mitral, aortic.center_mm)]
    for plane in planes:
        views = {name: window(hu, settings) for name, hu in valve_views(ct, plane, settings).items()}
        prefix = os.path.join(output_folder, f"{patient_id}_{plane.valve}")
        Image.fromarray(np.stack(list(views.values()), axis=-1)).save(f"{prefix}.png")
        for name, view in views.items():
            Image.fromarray(view).save(f"{prefix}_{name}.png")

    metadata = {
        "patient_id": patient_id,
        "image": os.path.abspath(image_path),
        "segmentation": os.path.abspath(segmentation_path),
        "settings": asdict(settings),
        "channels": list(VIEW_NAMES[settings.views]),
        "valves": {plane.valve: asdict(plane) for plane in planes},
    }
    with open(os.path.join(output_folder, f"{patient_id}_valves.json"), "w") as file:
        json.dump(metadata, file, indent=2)
    return planes


def main(
    image_folder: str = typer.Argument(..., help="Folder with the CT scans."),
    segmentation_folder: str = typer.Argument(..., help="Folder with (or for) the TotalSegmentator segmentations."),
    output_folder: str = typer.Argument(..., help="Folder the PNGs, JSON files and log are written to."),
    image_pattern: str = typer.Option("{id}.img.nii.gz", help="CT file name pattern; {id} marks the patient id."),
    segmentation_pattern: str = typer.Option("{id}.heart.nii.gz", help="Segmentation file name pattern."),
    field_of_view_mm: float = typer.Option(60.0, help="Side length of the valve image in mm."),
    resolution_mm: float = typer.Option(0.5, help="Pixel size in mm."),
    views: str = typer.Option("orthogonal", help="View mode: 'orthogonal' (en-face + two long-axis) or 'parallel'."),
    slab_offset_mm: float = typer.Option(2.0, help="Distance of the outer slices in 'parallel' mode in mm."),
    window_mode: str = typer.Option(
        "adaptive", "--window", help="HU window: 'adaptive' (per scan from myocardium/blood) or 'fixed'."
    ),
    hu_min: float = typer.Option(-200.0, help="Lower bound of the HU window with --window fixed."),
    hu_max: float = typer.Option(600.0, help="Upper bound of the HU window with --window fixed."),
    totalsegmentator: str = typer.Option("TotalSegmentator", help="TotalSegmentator executable."),
    overwrite: bool = typer.Option(False, help="Reprocess scans that already have a metadata file."),
    limit: int = typer.Option(0, help="Only process the first N scans (0 processes all)."),
) -> None:
    """Extract aortic and mitral valve images for every scan in ``image_folder``."""
    os.makedirs(output_folder, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(output_folder, "valve_planes.log"))],
    )
    if views not in VIEW_NAMES:
        raise typer.BadParameter(f"--views must be one of {list(VIEW_NAMES)}")
    if window_mode not in WINDOW_MODES:
        raise typer.BadParameter(f"--window must be one of {list(WINDOW_MODES)}")
    settings = SliceSettings(field_of_view_mm, resolution_mm, slab_offset_mm, hu_min, hu_max, views, window_mode)
    scans = find_scans(image_folder, image_pattern)
    if limit > 0:
        scans = dict(list(scans.items())[:limit])
    logger.info(f"Found {len(scans)} scans in {image_folder}")

    failed = []
    for patient_id, image_path in scans.items():
        if not overwrite and os.path.isfile(os.path.join(output_folder, f"{patient_id}_valves.json")):
            logger.info(f"{patient_id}: already processed, skipping")
            continue
        segmentation_path = resolve_segmentation(segmentation_folder, segmentation_pattern, patient_id)
        try:
            planes = process_scan(patient_id, image_path, segmentation_path, output_folder, settings, totalsegmentator)
            summary = ", ".join(f"{p.valve} {p.n_contact_voxels} vox @ {p.contact_threshold_mm} mm" for p in planes)
            logger.info(f"{patient_id}: done ({summary})")
        except Exception as error:
            failed.append(patient_id)
            logger.error(f"{patient_id}: failed - {type(error).__name__}: {error}")
    logger.info(f"Finished: {len(scans) - len(failed)} ok, {len(failed)} failed {failed if failed else ''}")


if __name__ == "__main__":
    typer.run(main)

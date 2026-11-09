"""Rebuild the valve views and the training crop of one scan as a 3D Slicer scene, for figures.

Runs inside 3D Slicer's Python, not ``uv``. The scene shows the three views of ``valve_planes.py`` in the Red
(``en_face``), Yellow (``long_axis_a``) and Green (``long_axis_b``) slice viewers, the same slices inside the 3D view,
the valve axes ``in_plane_u``, ``in_plane_v`` and ``normal`` as arrows, and the cube that ``export_crops.py`` samples
as an ROI box in the same frame. The three views are the central orthogonal cross-sections of that cube.

Usage:
    "C:/Program Files/Slicer 5.x/Slicer.exe" --python-script side_project/valve_planes/slicer_valve_scene.py \
        side_project/valve_planes/output/1_valves.json aortic

    or in Slicer's Python console:
    import sys; sys.argv = ["", r"C:/Users/Jacob pc/vscode_projects/spatio_temporal_myocardinal_changes_ct/side_project/valve_planes/output/1_valves.json", "aortic"]; exec(open(r"C:/Users/Jacob pc/vscode_projects/spatio_temporal_myocardinal_changes_ct/side_project/valve_planes/slicer_valve_scene.py").read())
"""

import json
import os
import sys

import numpy as np
import slicer
import vtk

LPS_TO_RAS = np.array([-1.0, -1.0, 1.0])
CROP_VOXELS = 96
CROP_RESOLUTION_MM = 0.5
ARROW_MM = 30.0
VIEWS = {
    "Red": ("en_face", "in_plane_u", "in_plane_v"),
    "Yellow": ("long_axis_a", "in_plane_u", "normal"),
    "Green": ("long_axis_b", "in_plane_v", "normal"),
}
AXIS_COLORS = {"in_plane_u": (0.9, 0.2, 0.2), "in_plane_v": (0.2, 0.7, 0.2), "normal": (0.2, 0.4, 0.9)}
VIEW_COLORS = {"Red": (0.95, 0.35, 0.3), "Yellow": (0.95, 0.85, 0.3), "Green": (0.4, 0.75, 0.35)}


def to_ras(vector: list[float]) -> np.ndarray:
    """Convert an LPS point or direction from the JSON to Slicer's RAS frame."""
    return np.asarray(vector, dtype=float) * LPS_TO_RAS


def valve_frame(plane: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return the valve centre and its axes ``in_plane_u``, ``in_plane_v`` and ``normal`` in RAS."""
    return to_ras(plane["center_mm"]), {name: to_ras(plane[name]) for name in ("in_plane_u", "in_plane_v", "normal")}


def frame_transform(center: np.ndarray, axes: dict[str, np.ndarray], name: str) -> slicer.vtkMRMLLinearTransformNode:
    """Create a transform whose x, y and z axes are ``in_plane_u``, ``in_plane_v`` and ``normal`` at the centre."""
    matrix = vtk.vtkMatrix4x4()
    for column, axis in enumerate([axes["in_plane_u"], axes["in_plane_v"], axes["normal"], center]):
        for row in range(3):
            matrix.SetElement(row, column, float(axis[row]))
    node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", name)
    node.SetMatrixTransformToParent(matrix)
    return node


def set_view(
    slice_name: str,
    center: np.ndarray,
    column_axis: np.ndarray,
    up_axis: np.ndarray,
    metadata: dict,
    show_in_3d: bool = True,
) -> None:
    """Orient a slice viewer like ``sample_plane``: columns along ``column_axis`` and ``up_axis`` towards the top.

    Args:
        slice_name: Slice viewer to set, ``Red``, ``Yellow`` or ``Green``.
        center: Valve centre in RAS.
        column_axis: Unit vector along increasing columns in RAS.
        up_axis: Unit vector towards the top row in RAS.
        metadata: Contents of the ``<id>_valves.json`` file.
        show_in_3d: Also show the slice in the 3D view.
    """
    slice_node = slicer.app.layoutManager().sliceWidget(slice_name).mrmlSliceNode()
    normal = np.cross(column_axis, up_axis)
    slice_node.SetSliceToRASByNTP(*normal, *column_axis, *center, 0)
    field_of_view = metadata["field_of_view_mm"]
    slice_node.SetFieldOfView(field_of_view, field_of_view, slice_node.GetFieldOfView()[2])
    if slice_name != "Red" and hasattr(slice_node, "SetSlabReconstructionEnabled"):
        slice_node.SetSlabReconstructionEnabled(True)
        slice_node.SetSlabReconstructionThickness(metadata["slab_mm"])
        slice_node.SetSlabReconstructionType(vtk.VTK_IMAGE_SLAB_MEAN)
    logic = slicer.app.layoutManager().sliceWidget(slice_name).sliceLogic()
    logic.GetSliceModelDisplayNode().SetVisibility(show_in_3d)


def add_view_square(
    name: str,
    center: np.ndarray,
    column_axis: np.ndarray,
    up_axis: np.ndarray,
    side_mm: float,
    color: tuple[float, float, float],
    png_path: str | None,
) -> None:
    """Add a square of the view's size and pose to the 3D view, textured with its PNG and framed in ``color``.

    Args:
        name: Node name.
        center: Valve centre in RAS.
        column_axis: Unit vector along increasing columns in RAS.
        up_axis: Unit vector towards the top row in RAS.
        side_mm: Side length of the square, the field of view of the view.
        color: Frame colour, and the fill colour when there is no PNG.
        png_path: View image written by ``valve_planes.py``; ``None`` or a missing file gives a plain square.
    """
    half_column, half_up = 0.5 * side_mm * column_axis, 0.5 * side_mm * up_axis
    plane = vtk.vtkPlaneSource()
    plane.SetOrigin(*(center - half_column - half_up))
    plane.SetPoint1(*(center + half_column - half_up))
    plane.SetPoint2(*(center - half_column + half_up))
    square = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
    square.SetPolyDataConnection(plane.GetOutputPort())
    square.CreateDefaultDisplayNodes()
    display = square.GetDisplayNode()
    display.SetBackfaceCulling(False)
    display.SetAmbient(1.0)
    display.SetDiffuse(0.0)
    display.SetSpecular(0.0)
    if png_path and os.path.exists(png_path):
        reader = vtk.vtkPNGReader()
        reader.SetFileName(png_path)
        display.SetColor(1.0, 1.0, 1.0)
        display.SetTextureImageDataConnection(reader.GetOutputPort())
    else:
        display.SetColor(*color)
        display.SetOpacity(0.5)

    edges = vtk.vtkFeatureEdges()
    edges.SetInputConnection(plane.GetOutputPort())
    edges.BoundaryEdgesOn()
    edges.FeatureEdgesOff()
    edges.NonManifoldEdgesOff()
    edges.ManifoldEdgesOff()
    frame = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"{name}_frame")
    frame.SetPolyDataConnection(edges.GetOutputPort())
    frame.CreateDefaultDisplayNodes()
    frame_display = frame.GetDisplayNode()
    frame_display.SetScalarVisibility(False)
    frame_display.SetColor(*color)
    frame_display.SetLineWidth(4.0)
    frame_display.SetVisibility2D(False)


def add_arrow(name: str, center: np.ndarray, direction: np.ndarray) -> None:
    """Add a labelled line markup from the valve centre along ``direction``."""
    line = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", name)
    line.AddControlPoint(vtk.vtkVector3d(*center))
    line.AddControlPoint(vtk.vtkVector3d(*(center + ARROW_MM * direction)))
    line.SetNthControlPointLabel(0, "")
    line.SetNthControlPointLabel(1, name)
    display = line.GetDisplayNode()
    display.SetSelectedColor(AXIS_COLORS[name])
    display.SetColor(AXIS_COLORS[name])
    display.SetPropertiesLabelVisibility(False)
    line.SetLocked(True)


def add_crop_box(transform: slicer.vtkMRMLLinearTransformNode) -> None:
    """Add the ``export_crops.py`` cube as an ROI centred on the valve and aligned with its axes."""
    side_mm = CROP_VOXELS * CROP_RESOLUTION_MM
    roi = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", f"crop_{side_mm:g}mm")
    roi.SetCenter(0.0, 0.0, 0.0)
    roi.SetSize(side_mm, side_mm, side_mm)
    roi.SetAndObserveTransformNodeID(transform.GetID())
    roi.GetDisplayNode().SetFillOpacity(0.1)
    roi.SetLocked(True)


def build_scene(
    json_path: str, valve: str = "aortic", show_segmentation: bool = True, view_squares: bool = True
) -> None:
    """Load the scan of a ``<id>_valves.json`` file and set up the views, axes and crop box of one valve.

    Args:
        json_path: Path to a ``<id>_valves.json`` file written by ``valve_planes.py``.
        valve: Valve to show, ``aortic`` or ``mitral``.
        show_segmentation: Also load the TotalSegmentator heart labels as a 3D surface for context.
        view_squares: Show the views in 3D as squares textured with the PNGs next to the JSON, instead of Slicer's
            slice planes.
    """
    with open(json_path) as file:
        metadata = json.load(file)
    center, axes = valve_frame(metadata["valves"][valve])

    volume = slicer.util.loadVolume(metadata["image"])
    display = volume.GetDisplayNode()
    display.SetAutoWindowLevel(False)
    display.SetWindowLevelMinMax(*metadata["hu_window"])
    if show_segmentation:
        segmentation = slicer.util.loadSegmentation(metadata["segmentation"])
        segmentation.CreateClosedSurfaceRepresentation()
        segmentation.GetDisplayNode().SetOpacity3D(0.3)
        segmentation.GetDisplayNode().SetVisibility2D(False)

    slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
    slicer.util.setSliceViewerLayers(background=volume, fit=False)
    output_folder = os.path.dirname(os.path.abspath(json_path))
    for slice_name, (view_name, column_axis, up_axis) in VIEWS.items():
        set_view(slice_name, center, axes[column_axis], axes[up_axis], metadata, show_in_3d=not view_squares)
        if view_squares:
            png_path = os.path.join(output_folder, f"{metadata['patient_id']}_{valve}_{view_name}.png")
            add_view_square(
                f"{valve}_{view_name}",
                center,
                axes[column_axis],
                axes[up_axis],
                metadata["field_of_view_mm"],
                VIEW_COLORS[slice_name],
                png_path,
            )

    for name, direction in axes.items():
        add_arrow(name, center, direction)
    add_crop_box(frame_transform(center, axes, f"{valve}_frame"))

    view = slicer.app.layoutManager().threeDWidget(0).threeDView()
    view.resetFocalPoint()
    view.resetCamera()


if __name__ == "__main__":
    build_scene(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "aortic")

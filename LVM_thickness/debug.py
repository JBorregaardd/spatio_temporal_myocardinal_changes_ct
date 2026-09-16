"""
Debug script: visualize the myocardium mask and the (y_0, x_0) center point
used by generate_lv_segments, across a range of slices, to see why the
360-degree ring check is failing.

This does NOT modify lv_generate_segments.py. It re-derives the same
intermediate arrays (label_lv_myo, y_0, x_0, z_start, z_end) by calling into
the module directly, so it should behave identically to what wrap_lv_segments
does internally up to the point of the crash.

Usage:
    python debug_ring_coverage.py
Adjust `segmentation_path`, `image_path`, `current_save_dir` below to match
your patient before running.
"""

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

import lv_generate_segments as lvseg

# ---- adjust these to match your failing case ----
root = r"C:\Users\Jacob pc\vscode_projects\spatio_temporal_myocardinal_changes_ct"
folder = os.path.join(root, "data", "TotalSegmentator")
patient_id = "1"
current_save_dir = os.path.join(root, "output", patient_id)

segmentation_path = os.path.join(folder, f"{patient_id}.heart.nii.gz")
image_path = os.path.join(root, "data", "1_200", f"{patient_id}.img.nii.gz")

out_dir = os.path.join(current_save_dir, "debug_ring")
os.makedirs(out_dir, exist_ok=True)
# ---------------------------------------------------

# Monkey-patch generate_lv_segments to capture its internal state right
# before the crash point, instead of duplicating all the alignment logic.
# We do this by wrapping the module-level function and stopping it early
# via an injected exception, then reading back locals from the traceback.
# Simpler & more robust: just re-run wrap_lv_segments up to Module 3 by
# calling generate_lv_segments with a patched version that dumps state.

import types
import traceback

captured = {}

orig_func = lvseg.generate_lv_segments

def patched_generate_lv_segments(*args, **kwargs):
    # Run the original function but intercept via sys.settrace to grab
    # locals right after the 'good_slices' loop executes, before the crash.
    #
    # IMPORTANT: this two-tier tracer only enables line-level tracing
    # *inside* generate_lv_segments itself. The global trace function fires
    # on every 'call' event process-wide (cheap, C-level calls don't even
    # trigger it), but it returns None for every frame except
    # generate_lv_segments, so nothing it calls into (resampling, COM calcs,
    # etc.) gets the expensive per-line tracing. A naive tracer that returns
    # itself unconditionally enables full line tracing everywhere and will
    # look like the script has hung.
    import sys

    def line_tracer(frame, event, arg):
        if event == "line":
            local_vars = frame.f_locals
            if "good_slices" in local_vars and "y_0" in local_vars and "label_lv_myo" in local_vars:
                captured["good_slices"] = list(local_vars["good_slices"])
                captured["y_0"] = local_vars.get("y_0")
                captured["x_0"] = local_vars.get("x_0")
                captured["z_start"] = local_vars.get("z_start")
                captured["z_end"] = local_vars.get("z_end")
                captured["label_lv_myo"] = local_vars.get("label_lv_myo")
                captured["com_mv"] = local_vars.get("com_mv")
                captured["inf_limit_lv"] = local_vars.get("inf_limit_lv")
                mv_mask = local_vars.get("working_contours", {}).get(local_vars.get("label_mitral_valve"))
                if mv_mask is not None:
                    import SimpleITK as _sitk
                    import numpy as _np
                    mv_arr = _sitk.GetArrayFromImage(mv_mask)  # z,y,x
                    nz = mv_arr.nonzero()
                    captured["mv_voxel_count"] = int(mv_arr.sum())
                    if len(nz[0]) > 0:
                        captured["mv_z_index_range"] = (int(nz[0].min()), int(nz[0].max()))
                    lsf2 = _sitk.LabelShapeStatisticsImageFilter()
                    lsf2.Execute(mv_mask > 0)
                    if lsf2.GetLabels():
                        centroid_phys = lsf2.GetCentroid(1)
                        captured["mv_centroid_index_in_mv_frame"] = mv_mask.TransformPhysicalPointToContinuousIndex(centroid_phys)
                        lv_myo_local = local_vars.get("label_lv_myo")
                        if lv_myo_local is not None:
                            captured["mv_centroid_index_in_lv_myo_frame"] = lv_myo_local.TransformPhysicalPointToContinuousIndex(centroid_phys)
                            captured["lv_myo_size"] = lv_myo_local.GetSize()
        return line_tracer

    def call_tracer(frame, event, arg):
        if event == "call" and frame.f_code.co_name == "generate_lv_segments":
            return line_tracer
        return None  # don't trace anything else - keeps overhead near zero

    sys.settrace(call_tracer)
    try:
        return orig_func(*args, **kwargs)
    except IndexError:
        # expected - good_slices was empty, we already captured state above
        pass
    finally:
        sys.settrace(None)

lvseg.generate_lv_segments = patched_generate_lv_segments

# Now run wrap_lv_segments as normal - it will crash internally but we've
# already captured what we need via the trace hook.
try:
    lvseg.wrap_lv_segments(current_save_dir, segmentation_path, image_path,
                            individual_transforms=True, verbose=True)
except Exception:
    traceback.print_exc()

label_lv_myo = captured.get("label_lv_myo")
y_0 = captured.get("y_0")
x_0 = captured.get("x_0")
z_start = captured.get("z_start")
z_end = captured.get("z_end")

print("\n--- Mitral valve diagnostics ---")
print(f"com_mv (used): {captured.get('com_mv')}")
print(f"inf_limit_lv (used): {captured.get('inf_limit_lv')}")
print(f"MV voxel count: {captured.get('mv_voxel_count')}")
print(f"MV nonzero Z-index range (in MV mask's own array): {captured.get('mv_z_index_range')}")
print(f"MV centroid continuous index, in MV mask's own frame (x,y,z): {captured.get('mv_centroid_index_in_mv_frame')}")
print(f"MV centroid continuous index, in label_lv_myo's frame (x,y,z): {captured.get('mv_centroid_index_in_lv_myo_frame')}")
print(f"label_lv_myo size (x,y,z): {captured.get('lv_myo_size')}")
print("--- end diagnostics ---\n")

if label_lv_myo is None:
    raise RuntimeError("Could not capture internal state - check that the "
                        "line 'if len(loc_y) == 0:' still exists unchanged "
                        "in generate_lv_segments, or lower the tracer's "
                        "condition to fire earlier.")

print(f"Captured z_start={z_start}, z_end={z_end}, y_0={y_0}, x_0={x_0}")

# Pick a handful of slices spread across the range to visualize
n_slices_to_plot = 8
slice_indices = np.linspace(z_start, z_end - 1, n_slices_to_plot).astype(int)
slice_indices = sorted(set(slice_indices.tolist()))

fig, axes = plt.subplots(1, len(slice_indices), figsize=(4 * len(slice_indices), 4))
if len(slice_indices) == 1:
    axes = [axes]

for ax, n in zip(axes, slice_indices):
    label_lv_myo_slice = label_lv_myo[:, :, int(n)]
    arr = sitk.GetArrayFromImage(label_lv_myo_slice)
    ax.imshow(arr, cmap="gray", origin="lower")
    ax.plot(x_0, y_0, "r+", markersize=15, markeredgewidth=2)
    ax.set_title(f"slice {n}")
    ax.axis("off")

plt.tight_layout()
out_path = os.path.join(out_dir, "ring_coverage_debug.png")
plt.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")

# Also save the single best/worst slices at higher res for close inspection
best_n = z_start + int(np.argmax([
    (sitk.GetArrayFromImage(label_lv_myo[:, :, n]) > 0).sum()
    for n in range(z_start, z_end)
]))
fig2, ax2 = plt.subplots(figsize=(8, 8))
arr_best = sitk.GetArrayFromImage(label_lv_myo[:, :, best_n])
ax2.imshow(arr_best, cmap="gray", origin="lower")
ax2.plot(x_0, y_0, "r+", markersize=20, markeredgewidth=3)
ax2.set_title(f"Largest-area myocardium slice: {best_n} (center marked)")
out_path2 = os.path.join(out_dir, "largest_slice_debug.png")
plt.savefig(out_path2, dpi=150)
print(f"Saved: {out_path2}")

print("\nLook at these images:")
print("- If the red crosshair sits roughly in the middle of the myocardium ring "
      "in most slices -> axis alignment is probably fine, center estimate is the issue.")
print("- If the crosshair is clearly off to one side / outside the ring, or drifts "
      "away from the ring across slices -> LV long-axis alignment in Module 2 is off.")
print("- If there's no visible ring shape at all (just a blob or fragment) -> the "
      "z_start:z_end crop range itself may not be covering real mid-ventricular myocardium.")
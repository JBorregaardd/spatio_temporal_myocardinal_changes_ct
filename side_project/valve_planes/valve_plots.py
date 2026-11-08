"""Plots for the valve classifier runs: training curves, the test confusion matrix and a gallery of test scans.

Used by ``valve.py plots``. Colours follow a fixed categorical order (blue, orange, aqua) and a single-hue blue ramp
for the confusion matrix; text stays in neutral inks, and every line carries a legend entry and a direct label.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e4e3df"
CRITICAL = "#d03b3b"
GOOD = "#0ca30c"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
BLUE_RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#104281")
VIEW_TITLES = ("en face", "long axis A (flow ↑)", "long axis B (flow ↑)")
CROP_WINDOW = (-0.75, 1.5)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": INK_MUTED,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


@dataclass
class GalleryEntry:
    """One row of the scan gallery.

    Attributes:
        title: Text above the row (scan id, labels, probability).
        flag: Short status shown before the title, e.g. ``misclassified``; empty for none.
        flag_color: Colour of the status text.
        views: The labelling views, in ``VIEW_TITLES`` order; ``None`` for a missing PNG.
        crop_slices: Mid slices of the model input in the same orientation, or empty when the crop is not available.
    """

    title: str
    flag: str = ""
    flag_color: str = CRITICAL
    views: list[Optional[np.ndarray]] = field(default_factory=list)
    crop_slices: list[np.ndarray] = field(default_factory=list)


def _style_axis(axis: plt.Axes) -> None:
    """Draw a recessive horizontal grid behind the data."""
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(length=0)


def _end_labels(axis: plt.Axes, items: list[tuple[float, float, str, str]], min_gap_points: float = 12) -> None:
    """Write direct labels right of the last points of the lines, pushed apart so they never overlap.

    Args:
        axis: Axis with the lines; its limits and the figure layout must be final.
        items: ``(x, y, text, color)`` of each line's last point.
        min_gap_points: Minimum vertical distance between two labels in points.
    """
    if not items:
        return
    to_pixels = axis.transData.transform
    gap = min_gap_points * axis.figure.dpi / 72
    placed = sorted(
        ((to_pixels((x, y))[1], x, y, text, color) for x, y, text, color in items), key=lambda item: item[0]
    )
    positions: list[float] = []
    for pixel_y, *_ in placed:
        positions.append(max(pixel_y, positions[-1] + gap) if positions else pixel_y)
    for (pixel_y, x, y, text, color), label_y in zip(placed, positions):
        axis.plot(x, y, "o", color=color, markersize=4, markeredgecolor=SURFACE, markeredgewidth=1)
        axis.annotate(
            text,
            (x, y),
            xytext=(6, (label_y - pixel_y) * 72 / axis.figure.dpi),
            textcoords="offset points",
            va="center",
            color=INK_SECONDARY,
            fontsize=9,
        )


def plot_training_curves(
    history: dict[str, list[dict[str, Any]]], best_metric: str, title: str, subtitle: str, out_path: Path
) -> None:
    """Plot train and validation loss and the validation metrics per epoch, with the best epoch marked.

    Args:
        history: Content of GloViTa's ``metrics.json`` (``{"train": [...], "val": [...]}``, one dict per epoch).
        best_metric: Validation metric that selects the best checkpoint, e.g. ``val_AUROC``.
        title: Figure title.
        subtitle: Line under the title (run settings).
        out_path: PNG file to write.
    """
    figure, (loss_axis, metric_axis) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2], "hspace": 0.25}
    )
    figure.suptitle(title, x=0.07, ha="left", fontsize=13, fontweight="bold")
    figure.text(0.07, 0.928, subtitle, color=INK_SECONDARY, fontsize=9.5)

    loss_series = (("train", "train_loss", "train loss"), ("val", "val_loss", "validation loss"))
    metric_series = (("val_AUROC", "AUROC"), ("val_Balanced_Accuracy", "balanced accuracy"), ("val_F1", "F1 (macro)"))
    panels = (
        (loss_axis, [(split, key, name) for split, key, name in loss_series]),
        (metric_axis, [("val", key, name) for key, name in metric_series]),
    )
    end_points: dict[plt.Axes, list[tuple[float, float, str, str]]] = {loss_axis: [], metric_axis: []}
    for axis, series in panels:
        for color, (split, key, name) in zip(SERIES, series):
            rows = [row for row in history.get(split, []) if key in row]
            if rows:
                epochs, values = [row["epoch"] for row in rows], [row[key] for row in rows]
                axis.plot(epochs, values, color=color, linewidth=2, label=name)
                end_points[axis].append((epochs[-1], values[-1], f"{name} {values[-1]:.2f}", color))
    loss_axis.set_ylabel("cross-entropy loss")
    loss_axis.set_ylim(bottom=0)
    loss_axis.set_title("Loss", pad=22)
    metric_axis.set_ylim(0, 1.05)
    metric_axis.set_ylabel("score")
    metric_axis.set_xlabel("epoch")
    metric_axis.set_title("Validation metrics", pad=22)
    metric_axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    val_rows = history.get("val", [])
    best_rows = [row for row in val_rows if best_metric in row]
    if best_rows:
        best = max(best_rows, key=lambda row: row[best_metric])
        last_epoch = max(row["epoch"] for row in val_rows)
        on_right = best["epoch"] > 0.6 * last_epoch
        for axis in (loss_axis, metric_axis):
            axis.axvline(best["epoch"], color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)))
        metric_axis.annotate(
            f"best checkpoint: epoch {best['epoch']}\n{best_metric} {best[best_metric]:.3f}",
            (best["epoch"], 0.04),
            xytext=(-6 if on_right else 6, 0),
            textcoords="offset points",
            ha="right" if on_right else "left",
            color=INK_SECONDARY,
            fontsize=9,
        )
    for axis in (loss_axis, metric_axis):
        _style_axis(axis)
        axis.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=9, labelcolor=INK_SECONDARY,
                    borderaxespad=0.3, handlelength=1.6)  # fmt: skip
        axis.margins(x=0.02)
    figure.subplots_adjust(left=0.07, right=0.8, top=0.86, bottom=0.08)
    for axis, items in end_points.items():
        _end_labels(axis, items)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


def plot_confusion_matrix(metrics: dict[str, Any], title: str, out_path: Path) -> None:
    """Plot the test confusion matrix with counts and the share of each true class.

    Args:
        metrics: Output of ``valve.compute_metrics`` (``confusion``, ``n`` and the metric values).
        title: Figure title.
        out_path: PNG file to write.
    """
    true_names = list(metrics["confusion"])
    pred_names = list(metrics["confusion"][true_names[0]])
    counts = np.array([[metrics["confusion"][true][pred] for pred in pred_names] for true in true_names])
    shares = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)

    figure, axis = plt.subplots(figsize=(5.6, 5.2))
    colormap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    axis.imshow(shares, cmap=colormap, vmin=0, vmax=1)
    for row in range(counts.shape[0]):
        for column in range(counts.shape[1]):
            color = "#ffffff" if shares[row, column] > 0.55 else INK
            axis.text(column, row - 0.08, str(counts[row, column]), ha="center", va="center", fontsize=20,
                      fontweight="bold", color=color)  # fmt: skip
            axis.text(column, row + 0.2, f"{shares[row, column]:.0%} of true {true_names[row].split()[-1]}",
                      ha="center", va="center", fontsize=9, color=color)  # fmt: skip
    axis.set_xticks(range(len(pred_names)), [name.split()[-1] for name in pred_names])
    axis.set_yticks(range(len(true_names)), [name.split()[-1] for name in true_names])
    axis.set_xlabel("predicted")
    axis.set_ylabel("true (manual label)")
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_xticks(np.arange(-0.5, len(pred_names)), minor=True)
    axis.set_yticks(np.arange(-0.5, len(true_names)), minor=True)
    axis.grid(which="minor", color=SURFACE, linewidth=3)
    axis.tick_params(which="minor", length=0)

    values = [("F1", "f1_macro"), ("balanced acc", "balanced_accuracy"), ("AUROC", "auroc")]
    summary = " · ".join(f"{name} {metrics[key]:.3f}" for name, key in values if metrics.get(key) is not None)
    figure.suptitle(title, x=0.04, ha="left", fontsize=12, fontweight="bold")
    figure.text(0.04, 0.905, f"{summary} · {metrics['n']} scans", color=INK_SECONDARY, fontsize=9.5)
    figure.subplots_adjust(left=0.18, right=0.96, top=0.86, bottom=0.11)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


def crop_mid_slices(crop: np.ndarray) -> list[np.ndarray]:
    """Return the three central slices of a crop, oriented like the labelling views.

    Args:
        crop: Crop of shape (1, u, v, normal) or (u, v, normal).

    Returns:
        En face (u across, v up), long axis A (u across, normal up) and long axis B (v across, normal up), each
        transposed so that it is shown with ``origin="lower"``.
    """
    volume = crop[0] if crop.ndim == 4 else crop
    mid_u, mid_v, mid_n = (size // 2 for size in volume.shape)
    return [volume[:, :, mid_n].T, volume[:, mid_v, :].T, volume[mid_u, :, :].T]


def plot_scan_gallery(entries: list[GalleryEntry], title: str, subtitle: str, out_path: Path) -> None:
    """Plot one row per scan with the three labelling views and, when available, the model input.

    Args:
        entries: Rows to plot.
        title: Figure title.
        subtitle: Line under the title.
        out_path: PNG file to write.
    """
    with_crops = any(entry.crop_slices for entry in entries)
    columns = 6 if with_crops else 3
    row_height, header_height = 2.6, 1.45
    height = row_height * len(entries) + header_height
    figure, axes = plt.subplots(len(entries), columns, figsize=(2.1 * columns + 0.4, height), squeeze=False)
    figure.subplots_adjust(
        left=0.02, right=0.98, top=1 - header_height / height, bottom=0.15 / height, wspace=0.06, hspace=0.42
    )
    figure.suptitle(title, x=0.02, y=1 - 0.2 / height, ha="left", va="top", fontsize=13, fontweight="bold")
    figure.text(0.02, 1 - 0.58 / height, subtitle, color=INK_SECONDARY, fontsize=9.5, va="top")
    renderer = figure.canvas.get_renderer()
    for row_index, entry in enumerate(entries):
        top = axes[row_index, 0].get_position().y1 + 0.32 / height
        flag = figure.text(0.02, top, entry.flag, color=entry.flag_color, fontweight="bold", va="bottom")
        offset = flag.get_window_extent(renderer).width / figure.bbox.width + (0.008 if entry.flag else 0)
        figure.text(0.02 + offset, top, entry.title, color=INK, va="bottom")
        images = [(view, None, "upper") for view in entry.views]
        images += [(crop_slice, CROP_WINDOW, "lower") for crop_slice in entry.crop_slices]
        for column, axis in enumerate(axes[row_index]):
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if column < len(images) and images[column][0] is not None:
                image, window, origin = images[column]
                vmin, vmax = window if window else (None, None)
                axis.imshow(image, cmap="gray", vmin=vmin, vmax=vmax, origin=origin)
            else:
                axis.text(0.5, 0.5, "not available", ha="center", va="center", color=INK_MUTED, fontsize=9)
            prefix = "labelling: " if column < 3 else "model input: "
            axis.set_title(prefix + VIEW_TITLES[column % 3], fontsize=8, fontweight="normal", color=INK_MUTED,
                           loc="center", pad=3)  # fmt: skip
    figure.savefig(out_path, dpi=120)
    plt.close(figure)

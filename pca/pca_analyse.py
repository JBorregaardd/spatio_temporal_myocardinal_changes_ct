#################################################################################################################################################
# This script performs a PCA of the LV myocardium per AHA segment. Each patient is described by a 34-long vector: the median wall
# thickness (mm) of segments 1-17 followed by the volume (mL) of segments 1-17. The features are standardised (z-scored) before the PCA,
# since thickness and volume have different units.
# It reads median_thickness and volume_ml from <output>/<dataset>_<aorta-exclusion-mm>mm/statistics/statistics_thickness_per_segment.csv
# (written by initial_data_investigation/statistics_analysis.py).
# Outputs in pca/output/<dataset>_<aorta-exclusion-mm>mm/ (ignored by git):
#   pca_features.csv         the 34-long vector of each patient
#   pca_eigenvalues.csv      eigenvalue, explained variance and cumulative explained variance of each component
#   pca_eigenvectors.csv     eigenvectors (loadings) of each component, one row per component and one column per feature
#   pca_scores.csv           the patients projected on the components
#   pca_outliers.csv         patients outside the 95% Hotelling T^2 limit in the PC1-PC2 plane
#   pca_explained_variance.png, pca_pc1_pc2.png, pca_loadings.png, pca_loadings_bull.png
# e.g. uv run pca/pca_analyse.py --dataset ImageCAS_1-200 --aorta-exclusion-mm 2.5
#################################################################################################################################################

import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.patches import Ellipse, Wedge
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "LVM_thickness"))
sys.path.insert(0, os.path.join(REPO_ROOT, "measurements"))

from pipeline import AORTA_EXCLUSION_MM, run_output_dir  # noqa: E402
from thickness import SEGMENTS  # noqa: E402

THICKNESS_FEATURES = [f"segment_{segment}_thickness" for segment in SEGMENTS]
VOLUME_FEATURES = [f"segment_{segment}_volume" for segment in SEGMENTS]
FEATURES = THICKNESS_FEATURES + VOLUME_FEATURES
VARIANCE_THRESHOLDS = (0.8, 0.9, 0.95)
CONFIDENCE = 0.95
N_LOADING_COMPONENTS = 3
BULLSEYE_RINGS = [(1, 6, 0), (7, 12, 0), (13, 16, 45)]


def build_features(statistics_path: str) -> pd.DataFrame:
    """Make the 34-long feature vector of each patient from the per-segment statistics table.

    Args:
        statistics_path: ``statistics_thickness_per_segment.csv`` with one row per patient and segment and the columns
            ``patient_id``, ``segment``, ``median_thickness`` and ``volume_ml``.

    Returns:
        One row per patient with ``patient_id`` and the columns in ``FEATURES``.
    """
    data = pd.read_csv(statistics_path, dtype={"patient_id": str})
    thickness = data.pivot(index="patient_id", columns="segment", values="median_thickness")
    volume = data.pivot(index="patient_id", columns="segment", values="volume_ml")
    thickness.columns = [f"segment_{segment}_thickness" for segment in thickness.columns]
    volume.columns = [f"segment_{segment}_volume" for segment in volume.columns]
    features = pd.concat([thickness, volume], axis=1).reindex(columns=FEATURES).reset_index()
    order = features["patient_id"].map(lambda pid: int(pid) if pid.isdigit() else float("inf"))
    return features.iloc[order.argsort()].reset_index(drop=True)


def run_pca(features: pd.DataFrame) -> tuple[PCA, StandardScaler, pd.DataFrame]:
    """Standardise the features and fit a PCA with all components.

    Args:
        features: One row per patient with ``patient_id`` and the columns in ``FEATURES``; no missing values.

    Returns:
        The fitted PCA, the fitted scaler and the scores with one row per patient and one column per component.
    """
    scaler = StandardScaler()
    x = scaler.fit_transform(features[FEATURES].values)
    pca = PCA().fit(x)
    components = [f"PC{i}" for i in range(1, pca.n_components_ + 1)]
    scores = pd.DataFrame(pca.transform(x), columns=components)
    scores.insert(0, "patient_id", features["patient_id"].values)
    return pca, scaler, scores


def save_tables(pca: PCA, output_folder: str) -> pd.DataFrame:
    """Save the eigenvalues and eigenvectors of the PCA as CSV files.

    Args:
        pca: The fitted PCA.
        output_folder: Folder to save ``pca_eigenvalues.csv`` and ``pca_eigenvectors.csv`` in.

    Returns:
        The eigenvalue table.
    """
    components = [f"PC{i}" for i in range(1, pca.n_components_ + 1)]
    eigenvalues = pd.DataFrame(
        {
            "component": components,
            "eigenvalue": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    eigenvectors = pd.DataFrame(pca.components_, index=pd.Index(components, name="component"), columns=FEATURES)
    eigenvalues.to_csv(os.path.join(output_folder, "pca_eigenvalues.csv"), index=False)
    eigenvectors.to_csv(os.path.join(output_folder, "pca_eigenvectors.csv"))
    return eigenvalues


def plot_explained_variance(pca: PCA, output_folder: str) -> None:
    """Plot the explained variance of each component and the cumulative explained variance.

    The Kaiser criterion (eigenvalue > 1, i.e. the component explains more than one standardised feature) is shown as a
    dashed line on the individual bars.

    Args:
        pca: The fitted PCA.
        output_folder: Folder to save ``pca_explained_variance.png`` in.
    """
    n = np.arange(1, pca.n_components_ + 1)
    ratio = pca.explained_variance_ratio_ * 100
    cumulative = np.cumsum(ratio)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(n, ratio, color="tab:blue", alpha=0.6, label="Explained variance per component")
    ax.plot(n, cumulative, "o-", color="tab:red", label="Cumulative explained variance")
    ax.axhline(100 / pca.n_features_in_, color="tab:blue", linestyle="--", linewidth=1, label="Kaiser criterion")
    for threshold in VARIANCE_THRESHOLDS:
        k = int(np.searchsorted(cumulative, threshold * 100) + 1)
        ax.axhline(threshold * 100, color="grey", linestyle=":", linewidth=1)
        ax.annotate(f"{threshold:.0%}: {k} PCs", (n[-1], threshold * 100), ha="right", va="bottom", fontsize=9)
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("Explained variance of the PCA of segment thickness and volume")
    ax.set_xticks(n)
    ax.set_ylim(0, 105)
    ax.legend(loc="center right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, "pca_explained_variance.png"), dpi=200)
    plt.close(fig)


def hotelling_outliers(scores: pd.DataFrame, pca: PCA) -> tuple[pd.DataFrame, float]:
    """Hotelling T^2 of each patient in the PC1-PC2 plane and whether it exceeds the ``CONFIDENCE`` limit.

    Args:
        scores: The PCA scores with ``patient_id``, ``PC1`` and ``PC2``.
        pca: The fitted PCA.

    Returns:
        One row per patient with ``patient_id``, ``PC1``, ``PC2``, ``t2`` and ``outlier``, sorted by ``t2``, and the
        T^2 limit.
    """
    n = len(scores)
    t2 = (scores["PC1"] ** 2 / pca.explained_variance_[0]) + (scores["PC2"] ** 2 / pca.explained_variance_[1])
    limit = 2 * (n - 1) / (n - 2) * stats.f.ppf(CONFIDENCE, 2, n - 2)
    result = scores[["patient_id", "PC1", "PC2"]].assign(t2=t2, outlier=t2 > limit)
    return result.sort_values("t2", ascending=False).reset_index(drop=True), float(limit)


def plot_pc1_pc2(scores: pd.DataFrame, features: pd.DataFrame, pca: PCA, output_folder: str) -> pd.DataFrame:
    """Scatter the patients on the first two components, coloured by the total volume of the 17 segments.

    The ellipse is the ``CONFIDENCE`` Hotelling T^2 limit; patients outside it are labelled with their id.

    Args:
        scores: The PCA scores with ``patient_id``, ``PC1`` and ``PC2``.
        features: The feature table, in the same patient order as ``scores``.
        pca: The fitted PCA.
        output_folder: Folder to save ``pca_pc1_pc2.png`` in.

    Returns:
        The outlier table of ``hotelling_outliers``.
    """
    outliers, limit = hotelling_outliers(scores, pca)
    ratio = pca.explained_variance_ratio_ * 100
    total_volume = features[VOLUME_FEATURES].sum(axis=1)

    fig, ax = plt.subplots(figsize=(9, 7))
    points = ax.scatter(
        scores["PC1"], scores["PC2"], c=total_volume, cmap="viridis", s=25, edgecolor="k", linewidth=0.3
    )
    fig.colorbar(points, ax=ax, label="Total volume of segments 1-17 (mL)")
    ellipse = Ellipse(
        (0, 0),
        width=2 * np.sqrt(limit * pca.explained_variance_[0]),
        height=2 * np.sqrt(limit * pca.explained_variance_[1]),
        fill=False,
        linestyle="--",
        color="tab:red",
        label=f"{CONFIDENCE:.0%} Hotelling T$^2$",
    )
    ax.add_patch(ellipse)
    for _, row in outliers[outliers["outlier"]].iterrows():
        ax.annotate(row["patient_id"], (row["PC1"], row["PC2"]), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.set_xlabel(f"PC1 ({ratio[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({ratio[1]:.1f}%)")
    ax.set_title("Patients on the first two principal components")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, "pca_pc1_pc2.png"), dpi=200)
    plt.close(fig)
    return outliers


def plot_loadings(pca: PCA, output_folder: str) -> None:
    """Bar plot of the eigenvector of the first ``N_LOADING_COMPONENTS`` components, thickness and volume side by side.

    Args:
        pca: The fitted PCA.
        output_folder: Folder to save ``pca_loadings.png`` in.
    """
    n_segments = len(SEGMENTS)
    x = np.arange(1, n_segments + 1)
    width = 0.4
    fig, axes = plt.subplots(N_LOADING_COMPONENTS, 1, figsize=(11, 3 * N_LOADING_COMPONENTS), sharex=True, sharey=True)
    for i, ax in enumerate(axes):
        loading = pca.components_[i]
        ax.bar(x - width / 2, loading[:n_segments], width, label="Median thickness", color="tab:orange")
        ax.bar(x + width / 2, loading[n_segments:], width, label="Volume", color="tab:blue")
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_ylabel(f"PC{i + 1} ({pca.explained_variance_ratio_[i]:.1%})")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[0].set_title("Eigenvector loadings per AHA segment")
    axes[-1].set_xlabel("AHA segment")
    axes[-1].set_xticks(x)
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, "pca_loadings.png"), dpi=200)
    plt.close(fig)


def draw_bullseye(ax: plt.Axes, values: np.ndarray, vmax: float, title: str) -> plt.cm.ScalarMappable:
    """Draw values of the 17 AHA segments as a bull's-eye plot.

    Args:
        ax: Axes to draw on.
        values: One value per segment 1-17.
        vmax: Symmetric colour limit.
        title: Axes title.

    Returns:
        The colour mapping, for a colour bar.
    """
    norm = plt.Normalize(-vmax, vmax)
    cmap = plt.get_cmap("RdBu_r")
    radii = [(3, 2), (2, 1), (1, 0.4)]
    for (first, last, offset), (outer, inner) in zip(BULLSEYE_RINGS, radii):
        count = last - first + 1
        step = 360 / count
        for k, segment in enumerate(range(first, last + 1)):
            start = 90 + offset + k * step
            wedge = Wedge(
                (0, 0),
                outer,
                start,
                start + step,
                width=outer - inner,
                facecolor=cmap(norm(values[segment - 1])),
                edgecolor="k",
                linewidth=0.5,
            )
            ax.add_patch(wedge)
            angle = np.deg2rad(start + step / 2)
            r = (outer + inner) / 2
            ax.text(r * np.cos(angle), r * np.sin(angle), str(segment), ha="center", va="center", fontsize=7)
    ax.add_patch(plt.Circle((0, 0), 0.4, facecolor=cmap(norm(values[16])), edgecolor="k", linewidth=0.5))
    ax.text(0, 0, "17", ha="center", va="center", fontsize=7)
    ax.set_xlim(-3.1, 3.1)
    ax.set_ylim(-3.1, 3.1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10)
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)


def plot_loadings_bullseye(pca: PCA, output_folder: str) -> None:
    """Bull's-eye plots of the thickness and volume loadings of the first ``N_LOADING_COMPONENTS`` components.

    Args:
        pca: The fitted PCA.
        output_folder: Folder to save ``pca_loadings_bull.png`` in.
    """
    n_segments = len(SEGMENTS)
    vmax = float(np.abs(pca.components_[:N_LOADING_COMPONENTS]).max())
    fig, axes = plt.subplots(2, N_LOADING_COMPONENTS, figsize=(4 * N_LOADING_COMPONENTS, 8))
    for i in range(N_LOADING_COMPONENTS):
        loading = pca.components_[i]
        label = f"PC{i + 1} ({pca.explained_variance_ratio_[i]:.1%})"
        mappable = draw_bullseye(axes[0, i], loading[:n_segments], vmax, f"{label} - thickness")
        draw_bullseye(axes[1, i], loading[n_segments:], vmax, f"{label} - volume")
    fig.colorbar(mappable, ax=axes, shrink=0.6, label="Loading")
    fig.savefig(os.path.join(output_folder, "pca_loadings_bull.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Read the per-segment statistics, run the PCA and save the tables and plots."""
    parser = argparse.ArgumentParser(description="PCA of median thickness and volume per AHA segment.")
    parser.add_argument("--dataset", type=str, default="ImageCAS_1-200", help="Dataset name")
    parser.add_argument("--aorta-exclusion-mm", type=float, default=AORTA_EXCLUSION_MM, help="Aorta exclusion in mm")
    args = parser.parse_args()

    load_dotenv(os.path.join(REPO_ROOT, ".env"))
    root = os.environ["PROJECT_ROOT"]
    run_folder = run_output_dir(root, args.dataset, args.aorta_exclusion_mm)
    statistics_path = os.path.join(run_folder, "statistics", "statistics_thickness_per_segment.csv")
    if not os.path.isfile(statistics_path):
        raise SystemExit(f"{statistics_path} not found; run statistics_analysis.py with the same --dataset first.")

    output_folder = os.path.join(REPO_ROOT, "pca", "output", os.path.basename(run_folder))
    os.makedirs(output_folder, exist_ok=True)

    features = build_features(statistics_path)
    features.to_csv(os.path.join(output_folder, "pca_features.csv"), index=False)

    incomplete = features[features[FEATURES].isna().any(axis=1)]
    if not incomplete.empty:
        print(f"Excluding {len(incomplete)} patients with missing segments: {', '.join(incomplete['patient_id'])}")
    features = features.dropna(subset=FEATURES).reset_index(drop=True)
    print(f"Number of patients in the PCA: {len(features)}")

    pca, _, scores = run_pca(features)
    eigenvalues = save_tables(pca, output_folder)
    scores.to_csv(os.path.join(output_folder, "pca_scores.csv"), index=False)

    plot_explained_variance(pca, output_folder)
    outliers = plot_pc1_pc2(scores, features, pca, output_folder)
    outliers[outliers["outlier"]].to_csv(os.path.join(output_folder, "pca_outliers.csv"), index=False)
    plot_loadings(pca, output_folder)
    plot_loadings_bullseye(pca, output_folder)

    print(f"\n{eigenvalues.head(10).to_string(index=False)}")
    for threshold in VARIANCE_THRESHOLDS:
        k = int((eigenvalues["cumulative_explained_variance_ratio"] < threshold).sum() + 1)
        print(f"{threshold:.0%} of the variance is explained by {k} components")
    print(f"Components with eigenvalue > 1 (Kaiser): {int((eigenvalues['eigenvalue'] > 1).sum())}")
    print(f"Hotelling T^2 outliers in PC1-PC2: {', '.join(outliers.loc[outliers['outlier'], 'patient_id']) or 'none'}")
    print(f"\nPCA results saved to: {output_folder}")


if __name__ == "__main__":
    main()

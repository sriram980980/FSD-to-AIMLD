# Dependencies: scikit-learn>=1.3, numpy>=1.24, matplotlib>=3.7
# Node: 2.2.1 — Dimensionality Reduction — PCA & Clustering
# Run: python starter.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from typing import Tuple, List


# ---------------------------------------------------------------------------
# IMPLEMENTED HELPERS — do not modify these
# ---------------------------------------------------------------------------

def generate_customer_data(n_customers: int = 500, random_state: int = 42) -> np.ndarray:
    """Generate synthetic 6-feature retail customer dataset (RFM-like)."""
    X, _ = make_blobs(
        n_samples=n_customers,
        n_features=6,
        centers=4,
        cluster_std=1.8,
        random_state=random_state,
    )
    # Add realistic scale differences so standardization matters
    scale = np.array([100.0, 50.0, 5000.0, 12.0, 0.5, 200.0])
    shift = np.array([30.0, 10.0, 500.0, 1.0, 0.1, 20.0])
    X = np.abs(X * scale + shift)
    return X


def plot_elbow_curve(k_range: range, inertias: List[float]) -> None:
    """Save elbow curve of inertia vs k to elbow_curve.png."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(k_range), inertias, marker="o", linewidth=2, color="#2563eb")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method — Optimal k")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("elbow_curve.png", dpi=120)
    plt.close(fig)
    print("[Task 6] Elbow curve saved → elbow_curve.png")


def plot_clusters(X_pca: np.ndarray, labels: np.ndarray, title: str = "Customer Segments") -> None:
    """Save 2-D PCA cluster scatter to cluster_scatter.png."""
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2"]
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            label=f"Segment {label}",
            color=palette[label % len(palette)],
            alpha=0.6,
            s=30,
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("cluster_scatter.png", dpi=120)
    plt.close(fig)
    print("[Task 6] Cluster scatter saved → cluster_scatter.png")


# ---------------------------------------------------------------------------
# STUDENT TASKS — implement each function below
# ---------------------------------------------------------------------------

def standardize_features(X: np.ndarray) -> np.ndarray:
    """Fit a StandardScaler on X and return the scaled array.

    Requirements:
    - Use sklearn.preprocessing.StandardScaler
    - Returned array has same shape as X
    - Each column has mean ≈ 0 and std ≈ 1
    """
    raise NotImplementedError("TODO: implement this")


def apply_pca(X_scaled: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Fit PCA on X_scaled, print explained variance ratios, return projected array.

    Requirements:
    - Use sklearn.decomposition.PCA with n_components
    - Print each PC's explained_variance_ratio_ and the cumulative total
    - Return the transformed (projected) data of shape (n_samples, n_components)

    Expected print format:
      PC1: 0.XXXX  (cumulative: 0.XXXX)
      PC2: 0.XXXX  (cumulative: 0.XXXX)
      Total variance captured by 2 PCs: 0.XXXX
    """
    raise NotImplementedError("TODO: implement this")


def find_optimal_k(X_pca: np.ndarray, k_range: range) -> List[float]:
    """Fit K-Means for each k in k_range and return a list of inertia values.

    Requirements:
    - Use sklearn.cluster.KMeans with random_state=42 and n_init=10
    - Return inertias in the same order as k_range
    - Print each k and its inertia

    Expected print format:
      k=2: XXXXXX.XX | k=3: XXXXXX.XX | ...
    """
    raise NotImplementedError("TODO: implement this")


def run_kmeans(X_pca: np.ndarray, k: int) -> Tuple[KMeans, np.ndarray]:
    """Fit K-Means with k clusters and return (fitted_model, labels).

    Requirements:
    - Use sklearn.cluster.KMeans with random_state=42 and n_init=10
    - Return the fitted KMeans object AND the integer label array
    - Print cluster sizes (how many points per cluster)
    """
    raise NotImplementedError("TODO: implement this")


def compute_silhouette(X_pca: np.ndarray, labels: np.ndarray) -> float:
    """Compute and return the silhouette score for the clustering.

    Requirements:
    - Use sklearn.metrics.silhouette_score
    - Print the result with 4 decimal places
    - Return the float value

    Expected print format:
      Silhouette Score: 0.XXXX
    """
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# MAIN — orchestrates the full pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Customer Segmentation Pipeline ===\n")

    # Generate data
    X_raw = generate_customer_data(n_customers=500, random_state=42)
    print(f"Raw data shape: {X_raw.shape}  (500 customers, 6 features)\n")

    # Task 1 — Standardize
    X_scaled = standardize_features(X_raw)
    col_means = X_scaled.mean(axis=0)
    col_stds = X_scaled.std(axis=0)
    print(
        f"[Task 1] Standardized features: shape {X_scaled.shape}, "
        f"mean ≈ {col_means.mean():.3f}, std ≈ {col_stds.mean():.3f}\n"
    )

    # Task 2 — PCA
    print("[Task 2] PCA Explained Variance Ratio:")
    X_pca = apply_pca(X_scaled, n_components=2)
    print()

    # Task 3 — Elbow method
    k_range = range(2, 9)
    print("[Task 3] Inertias for k=2..8:")
    inertias = find_optimal_k(X_pca, k_range)
    plot_elbow_curve(k_range, inertias)
    optimal_k = 4  # update this after inspecting the elbow curve
    print(f"  Optimal k (elbow): {optimal_k}\n")

    # Task 4 — K-Means
    print(f"[Task 4] K-Means fitted with k={optimal_k}")
    model, labels = run_kmeans(X_pca, k=optimal_k)
    unique, counts = np.unique(labels, return_counts=True)
    print(f"  Cluster sizes: {dict(zip(unique.tolist(), counts.tolist()))}\n")

    # Task 5 — Silhouette
    print("[Task 5]", end=" ")
    score = compute_silhouette(X_pca, labels)
    print()

    # Task 6 — Visualize
    plot_clusters(X_pca, labels, title=f"Customer Segments (k={optimal_k})")
    print("\n=== Pipeline complete ===")


if __name__ == "__main__":
    main()

# 2.2.1 — Dimensionality Reduction — PCA & Clustering

## Hook

PCA is like gzip for feature vectors — it compresses high-dimensional data into its most informative axes, the same way a zip archive discards redundant byte patterns while preserving the content that matters for reconstruction.

## The Problem

Real-world datasets arrive with hundreds or thousands of features. Many of those features are correlated (think: `page_views` and `session_duration` almost always move together), which makes models slow to train, hard to visualise, and prone to overfitting because irrelevant noise gets as many votes as informative signal. K-Means compounds the problem: clustering in 500-dimensional space is dominated by distance distortion — every point looks equally far from every other point, so cluster assignments become meaningless. You need to reduce and denoise the feature space before you cluster, not after.

## Theory

### Principal Component Analysis (PCA)

PCA rotates your dataset so that the first axis captures the most variance, the second axis captures the next most, and so on. The axes (principal components) are orthogonal, so they are uncorrelated by construction.

**Step 1 — Mean-centre the data**

$$\tilde{X} = X - \bar{X}$$

where $X \in \mathbb{R}^{n \times d}$ is the raw data matrix ($n$ samples, $d$ features) and $\bar{X}$ is the row of column means broadcast over all rows.

**Step 2 — Compute the covariance matrix**

$$\Sigma = \frac{1}{n - 1} \tilde{X}^{\top} \tilde{X}$$

$\Sigma \in \mathbb{R}^{d \times d}$ captures how much each pair of features varies together. The diagonal entries are per-feature variances; off-diagonal entries are covariances.

**Step 3 — Eigendecomposition**

$$\Sigma \, \mathbf{v} = \lambda \, \mathbf{v}$$

$\mathbf{v} \in \mathbb{R}^d$ is an eigenvector — one principal component direction. $\lambda \in \mathbb{R}$ is its eigenvalue — the total variance of the data projected onto that direction. Sorting eigenvectors by descending $\lambda$ gives you the components in order of importance.

**Step 4 — Project**

$$Z = \tilde{X} \, W_k$$

$W_k \in \mathbb{R}^{d \times k}$ is the matrix of the top-$k$ eigenvectors (columns). $Z \in \mathbb{R}^{n \times k}$ is the compressed dataset.

**Explained Variance Ratio (EVR)**

$$\text{EVR}_i = \frac{\lambda_i}{\sum_{j=1}^{d} \lambda_j}$$

EVR tells you what fraction of the total variance component $i$ captures. Sum the top-$k$ EVRs to know how much information you preserved.

**Numeric worked example (2D → 1D)**

Start with 3 data points, 2 features:

```
X = [[2, 3],
     [4, 5],
     [6, 7]]
```

Column means: $\bar{X} = [4, 5]$. Mean-centred matrix:

```
X̃ = [[-2, -2],
      [ 0,  0],
      [ 2,  2]]
```

Covariance matrix ($n-1 = 2$):

$$\Sigma = \frac{1}{2} \begin{bmatrix} 8 & 8 \\ 8 & 8 \end{bmatrix} = \begin{bmatrix} 4 & 4 \\ 4 & 4 \end{bmatrix}$$

Eigenvalues: $\lambda_1 = 8, \, \lambda_2 = 0$. The first eigenvector is $\mathbf{v}_1 = [1/\sqrt{2},\; 1/\sqrt{2}]^{\top}$.

$$\text{EVR}_1 = \frac{8}{8 + 0} = 1.0 \quad (100\%)$$

One component captures all variance — the data is perfectly collinear along the diagonal. Projecting gives:

$$Z = \tilde{X} \, \mathbf{v}_1 = [-2\sqrt{2},\; 0,\; 2\sqrt{2}]^{\top} \approx [-2.83,\; 0,\; 2.83]$$

---

### K-Means Clustering

K-Means partitions $n$ points into $K$ clusters by minimising **inertia** — the total within-cluster squared distance to the centroid.

**Objective (inertia)**

$$J = \sum_{i=1}^{n} \left\| \mathbf{x}_i - \boldsymbol{\mu}_{c(i)} \right\|^2$$

where:
- $\mathbf{x}_i$ — the $i$-th data point
- $\boldsymbol{\mu}_{c(i)}$ — centroid of the cluster $c(i)$ assigned to point $i$
- $K$ — number of clusters (hyperparameter you choose)

**Algorithm (two alternating steps)**

1. **Assign** — each point goes to the nearest centroid: $c(i) = \arg\min_k \left\| \mathbf{x}_i - \boldsymbol{\mu}_k \right\|^2$
2. **Update** — recompute each centroid as the mean of its assigned points: $\boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{i \in C_k} \mathbf{x}_i$

Repeat until centroid positions stop changing (convergence). Inertia is guaranteed to decrease or stay the same on every iteration.

**Numeric worked example (K=2, 3 points)**

Points: $A=(1, 1)$, $B=(5, 5)$, $C=(1.5, 1.5)$. Initial centroids: $\boldsymbol{\mu}_1=(1, 1)$, $\boldsymbol{\mu}_2=(5, 5)$.

*Assign:*
- $A$: dist to $\boldsymbol{\mu}_1 = 0$, dist to $\boldsymbol{\mu}_2 = 5.66$ → Cluster 1
- $C$: dist to $\boldsymbol{\mu}_1 = 0.71$, dist to $\boldsymbol{\mu}_2 = 4.95$ → Cluster 1
- $B$: dist to $\boldsymbol{\mu}_1 = 5.66$, dist to $\boldsymbol{\mu}_2 = 0$ → Cluster 2

*Update:*

$$\boldsymbol{\mu}_1 = \frac{(1,1)+(1.5,1.5)}{2} = (1.25, 1.25), \quad \boldsymbol{\mu}_2 = (5, 5)$$

*Inertia after one iteration:*

$$J = \underbrace{(1-1.25)^2+(1-1.25)^2}_{A} + \underbrace{(1.5-1.25)^2+(1.5-1.25)^2}_{C} + \underbrace{0}_{B} = 0.125$$

Centroids have shifted, so the algorithm runs another assign-update cycle. With these 3 points it converges in one more step.

---

## Python Implementation

```python
# Dependencies: scikit-learn>=1.3, numpy>=1.24, matplotlib>=3.7
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def generate_data(
    n_samples: int = 300,
    n_features: int = 10,
    n_centers: int = 3,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic high-dimensional clustered data."""
    X, y_true = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_centers,
        cluster_std=1.5,
        random_state=random_state,
    )
    return X, y_true


def apply_pca(
    X: np.ndarray, n_components: int = 2
) -> Tuple[np.ndarray, PCA]:
    """Standardise, then reduce to n_components principal components."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X_scaled)
    return X_reduced, pca


def run_kmeans(
    X: np.ndarray, n_clusters: int = 3, random_state: int = 42
) -> KMeans:
    """Fit K-Means on the (already-reduced) data."""
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    kmeans.fit(X)
    return kmeans


def print_diagnostics(pca: PCA, kmeans: KMeans) -> None:
    """Print EVR per component and K-Means inertia."""
    print("=== PCA Diagnostics ===")
    cumulative_evr = 0.0
    for i, evr in enumerate(pca.explained_variance_ratio_):
        cumulative_evr += evr
        print(
            f"  PC{i + 1}: EVR = {evr:.4f} | "
            f"Cumulative EVR = {cumulative_evr:.4f}"
        )

    print(f"\n=== K-Means Diagnostics ===")
    print(f"  Inertia (within-cluster sum of squares): {kmeans.inertia_:.2f}")
    print(f"  Iterations to converge:                  {kmeans.n_iter_}")
    print(f"  Cluster sizes: {np.bincount(kmeans.labels_).tolist()}")


def plot_clusters(
    X_reduced: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    save_path: str = "pca_kmeans_clusters.png",
) -> None:
    """Scatter-plot the 2D PCA projection coloured by cluster label."""
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        X_reduced[:, 0],
        X_reduced[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.7,
        edgecolors="none",
        s=40,
        label="Data points",
    )
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="X",
        s=200,
        color="black",
        zorder=5,
        label="Centroids",
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("K-Means Clusters on PCA-Reduced Features")
    ax.legend()
    plt.colorbar(scatter, ax=ax, label="Cluster label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")


def main() -> None:
    # 1. Generate 10-dimensional data with 3 natural clusters
    X, y_true = generate_data(n_samples=300, n_features=10, n_centers=3)
    print(f"Raw data shape:     {X.shape}")

    # 2. Compress to 2 dimensions with PCA
    X_reduced, pca = apply_pca(X, n_components=2)
    print(f"Reduced data shape: {X_reduced.shape}")

    # 3. Cluster the compressed data with K-Means
    kmeans = run_kmeans(X_reduced, n_clusters=3)

    # 4. Diagnostics
    print_diagnostics(pca, kmeans)

    # 5. Visualise
    plot_clusters(X_reduced, kmeans.labels_, kmeans.cluster_centers_)


if __name__ == "__main__":
    main()
```

Run this script and notice the following in the output:

- **PC1 carries most variance** — in a well-separated blob dataset, the first principal component often explains 40–70 % of total variance. Verify this with the `Cumulative EVR` line.
- **Inertia is a diagnostic, not a target** — a lower number is better, but comparing inertia across different values of $K$ (the "elbow method") is how you choose $K$ in practice.
- **Cluster sizes** tell you whether K-Means is balanced. Wildly unequal sizes (e.g., 270 / 20 / 10) signal that $K$ is wrong or the data has density outliers.
- The scatter plot visually confirms whether the three PCA axes cleanly separate the blobs — if they overlap, you need more components or a different algorithm (DBSCAN, GMM).

---

## Java Implementation

```java
// Dependencies (Maven):
//   org.tribuo:tribuo-clustering-kmeans:4.3.1
//   org.tribuo:tribuo-data:4.3.1

import com.oracle.labs.mlrg.tribuo.clustering.ClusterID;
import com.oracle.labs.mlrg.tribuo.clustering.evaluation.ClusteringEvaluation;
import com.oracle.labs.mlrg.tribuo.clustering.evaluation.ClusteringEvaluator;
import com.oracle.labs.mlrg.tribuo.clustering.kmeans.KMeansModel;
import com.oracle.labs.mlrg.tribuo.clustering.kmeans.KMeansTrainer;
import com.oracle.labs.mlrg.tribuo.clustering.kmeans.KMeansTrainer.Distance;
import com.oracle.labs.mlrg.tribuo.data.columnar.ColumnarIterator;
import com.oracle.labs.mlrg.tribuo.data.columnar.ResponseProcessor;
import com.oracle.labs.mlrg.tribuo.data.columnar.RowProcessor;
import com.oracle.labs.mlrg.tribuo.data.columnar.fieldprocessors.DoubleFieldProcessor;
import com.oracle.labs.mlrg.tribuo.data.csv.CSVDataSource;
import com.oracle.labs.mlrg.tribuo.data.csv.CSVIterator;
import com.oracle.labs.mlrg.tribuo.dataset.MutableDataset;
import com.oracle.labs.mlrg.tribuo.provenance.DataSourceProvenance;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Demonstrates K-Means clustering via Tribuo on synthetic 2-feature data.
 *
 * Build with Maven:
 *   mvn compile exec:java -Dexec.mainClass="KMeansDemo"
 *
 * Tribuo does not yet ship a built-in PCA transformer; use the Python pipeline
 * for full PCA + K-Means. This example focuses on the K-Means API in Java.
 */
public class KMeansDemo {

    /** Write a tiny CSV of 2D Gaussian blobs to a temp file. */
    static Path generateSyntheticCSV() throws IOException {
        Random rng = new Random(42);
        Path csvPath = Files.createTempFile("kmeans_data", ".csv");
        try (var writer = Files.newBufferedWriter(csvPath)) {
            writer.write("feat1,feat2\n");

            // Cluster 0: centred at (1, 1)
            for (int i = 0; i < 100; i++) {
                double f1 = 1.0 + rng.nextGaussian() * 0.5;
                double f2 = 1.0 + rng.nextGaussian() * 0.5;
                writer.write(f1 + "," + f2 + "\n");
            }
            // Cluster 1: centred at (5, 5)
            for (int i = 0; i < 100; i++) {
                double f1 = 5.0 + rng.nextGaussian() * 0.5;
                double f2 = 5.0 + rng.nextGaussian() * 0.5;
                writer.write(f1 + "," + f2 + "\n");
            }
            // Cluster 2: centred at (9, 1)
            for (int i = 0; i < 100; i++) {
                double f1 = 9.0 + rng.nextGaussian() * 0.5;
                double f2 = 1.0 + rng.nextGaussian() * 0.5;
                writer.write(f1 + "," + f2 + "\n");
            }
        }
        return csvPath;
    }

    public static void main(String[] args) throws IOException {
        // --- 1. Generate data ---
        Path csvPath = generateSyntheticCSV();
        System.out.println("Synthetic CSV written to: " + csvPath);

        // --- 2. Build Tribuo data pipeline ---
        Map<String, DoubleFieldProcessor> fieldProcessors = new HashMap<>();
        fieldProcessors.put("feat1", new DoubleFieldProcessor("feat1"));
        fieldProcessors.put("feat2", new DoubleFieldProcessor("feat2"));

        // Clustering has no response column; use a dummy processor
        ResponseProcessor<ClusterID> responseProcessor =
            new ResponseProcessor.DummyResponseProcessor<>(new ClusterID(0));

        RowProcessor<ClusterID> rowProcessor = new RowProcessor<>(
            responseProcessor,
            fieldProcessors
        );

        CSVDataSource<ClusterID> dataSource = new CSVDataSource<>(
            csvPath,
            rowProcessor,
            true // has header
        );

        MutableDataset<ClusterID> dataset = new MutableDataset<>(dataSource);
        System.out.printf("Loaded dataset: %d examples, %d features%n",
            dataset.size(), dataset.getFeatureMap().size());

        // --- 3. Train K-Means (K=3, max 100 iterations, 4 parallel threads) ---
        KMeansTrainer trainer = new KMeansTrainer(
            3,    // K
            100,  // maxIterations
            Distance.EUCLIDEAN,
            4,    // numThreads
            42    // seed
        );
        KMeansModel model = trainer.train(dataset);
        System.out.println("K-Means training complete.");

        // --- 4. Evaluate ---
        ClusteringEvaluator evaluator = new ClusteringEvaluator();
        ClusteringEvaluation evaluation = evaluator.evaluate(model, dataset);

        System.out.printf("Normalised MI:     %.4f%n",
            evaluation.normalizedMI());
        System.out.printf("Adjusted Rand Idx: %.4f%n",
            evaluation.adjustedRand());

        // --- 5. Show centroid positions ---
        System.out.println("\nLearned centroid positions:");
        double[][] centroids = model.getCentroidVectors();
        List<String> featureNames = List.of("feat1", "feat2");
        for (int k = 0; k < centroids.length; k++) {
            System.out.printf("  Centroid %d:", k);
            for (int f = 0; f < centroids[k].length; f++) {
                System.out.printf("  %s=%.3f", featureNames.get(f), centroids[k][f]);
            }
            System.out.println();
        }

        // Cleanup
        Files.deleteIfExists(csvPath);
    }
}
```

The Tribuo API mirrors the Python pattern: define a trainer with hyperparameters, call `train(dataset)`, then evaluate. One gap: **Tribuo 4.3 does not include a PCA transformer**, so the full PCA → K-Means pipeline requires either (a) pre-processing the data in Python and writing the reduced CSV, or (b) using a third-party math library (Apache Commons Math) for the eigendecomposition step before passing features to Tribuo.

---

## Stack Comparison

| Dimension | Python (scikit-learn) | Java (Tribuo) |
|---|---|---|
| **PCA** | `sklearn.decomposition.PCA` — one-liner fit/transform, EVR built-in | Not natively in Tribuo; use Apache Commons Math `EigenDecomposition` |
| **K-Means** | `sklearn.cluster.KMeans` — `n_init`, `random_state`, inertia attribute | `KMeansTrainer` — same K/iterations API, multi-threaded by default |
| **Evaluation** | Silhouette score, inertia, Davies-Bouldin via `sklearn.metrics` | `ClusteringEvaluator` gives normalised MI and Adjusted Rand Index |
| **Pipeline API** | `sklearn.pipeline.Pipeline` chains scaler → PCA → KMeans in one object | Manual sequential calls; no built-in pipeline chaining |
| **Visualisation** | matplotlib scatter in 5 lines | No built-in; export to CSV and plot externally |
| **Maturity** | Production-grade, widely benchmarked | Production-grade for K-Means; PCA gap is a real limitation |

---

## Key Takeaways

- **PCA does not discard features — it rotates them.** Principal components are linear combinations of all original features, ordered by how much variance they explain. You choose how many to keep based on cumulative EVR (≥ 95 % is a common threshold).
- **Always standardise before PCA.** Features on different scales (e.g., `price` in thousands vs. `rating` 1–5) will dominate the covariance matrix and skew the components toward the highest-variance column, not the most informative one.
- **K-Means inertia always decreases with more clusters — use the elbow method to pick K.** Plot inertia vs. $K$; the point where the curve bends (the "elbow") is the sweet spot between model complexity and explained structure.

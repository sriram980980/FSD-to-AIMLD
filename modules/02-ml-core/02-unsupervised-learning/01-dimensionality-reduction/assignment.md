# Assignment 2.2.1 — Implement Customer Segmentation Pipeline

## Objective

Build an end-to-end unsupervised pipeline that standardizes synthetic retail customer data,
compresses it with PCA, finds the optimal number of clusters via the elbow method, segments
customers with K-Means, and evaluates cluster quality with silhouette score.

## Background

In the lesson you saw how PCA projects high-dimensional feature vectors onto their most
informative axes (principal components), and how K-Means partitions points in that reduced
space by minimizing within-cluster inertia. Here you chain those two steps into a production-
style segmentation pipeline: scale → compress → cluster → evaluate. Refer back to the Theory
and Python Implementation sections of `lesson.md` for the math behind explained variance ratio
and the K-Means objective function.

## Setup

```bash
pip install scikit-learn>=1.3 numpy>=1.24 matplotlib>=3.7
```

## Tasks

1. **Standardize features** — Implement `standardize_features(X)` using `StandardScaler` so
   every feature has mean 0 and unit variance before PCA is applied.

2. **Apply PCA** — Implement `apply_pca(X_scaled, n_components)` to project the scaled data
   into `n_components` dimensions. Print the explained variance ratio for each component and
   the cumulative variance captured.

3. **Find optimal k** — Implement `find_optimal_k(X_pca, k_range)` by fitting K-Means for
   each k in `k_range`, recording the inertia, and returning the list of inertias so the
   elbow plot can be drawn.

4. **Run K-Means** — Implement `run_kmeans(X_pca, k)` to fit K-Means with the chosen k and
   return the fitted model together with the cluster labels array.

5. **Evaluate with silhouette score** — Implement `compute_silhouette(X_pca, labels)` to
   return the silhouette score. A score above 0.45 on this synthetic dataset indicates
   well-separated clusters.

6. **Visualize clusters** — Run `main()` end-to-end. Two plots should appear (and be saved as
   `elbow_curve.png` and `cluster_scatter.png`): the inertia elbow curve and the 2-D PCA
   cluster scatter.

## Expected Output

```
=== Customer Segmentation Pipeline ===

[Task 1] Standardized features: shape (500, 6), mean ≈ 0.000, std ≈ 1.000

[Task 2] PCA Explained Variance Ratio:
  PC1: 0.XXXX  (cumulative: 0.XXXX)
  PC2: 0.XXXX  (cumulative: 0.XXXX)
  Total variance captured by 2 PCs: between 0.60 and 0.95

[Task 3] Inertias for k=2..8:
  k=2: XXXXXX.XX | k=3: XXXXXX.XX | k=4: XXXXXX.XX | ...
  Optimal k (elbow): 4

[Task 4] K-Means fitted with k=4
  Cluster sizes: [~125, ~125, ~125, ~125]  (roughly balanced)

[Task 5] Silhouette Score: between 0.45 and 0.80

[Task 6] Plots saved → elbow_curve.png, cluster_scatter.png
```

> **Tolerance:** Exact numbers vary with random seed. Silhouette score must fall in `[0.45, 0.80]`.
> Cumulative variance of first 2 PCs must exceed 0.60.

## Evaluation Criteria

- [ ] `standardize_features` returns an array of shape `(500, 6)` with column means within
      `±0.01` of 0 and column stds within `±0.01` of 1.
- [ ] `apply_pca` returns a 2-column array and prints per-component explained variance ratios
      that sum to a cumulative value ≥ 0.60.
- [ ] `find_optimal_k` returns a list of 7 inertia values (one per k in `range(2, 9)`) with
      inertias strictly decreasing.
- [ ] `run_kmeans` returns labels whose `np.unique` count equals the chosen k.
- [ ] `compute_silhouette` returns a float in `[0.45, 0.80]`.
- [ ] Both `elbow_curve.png` and `cluster_scatter.png` are written to disk without error.

## Extension Challenge

Replace synthetic data with the **UCI Online Retail dataset** (available at
<https://archive.ics.uci.edu/ml/datasets/Online+Retail>). Engineer three RFM features
(Recency, Frequency, Monetary value) from raw transactions, apply the same pipeline, and
determine whether the resulting segments map meaningfully onto recognizable customer archetypes
(e.g., champions, at-risk, hibernating). Add a cluster-profile table that reports the mean
RFM values per segment. No starter code provided.

# Assignment 1.1.1 — Implement NumPy Vectorization Scripts

## Objective

Build five NumPy functions that implement core linear-algebra operations from scratch, proving you can construct the math underlying every neural-network layer without relying on black-box wrappers.

## Background

The lesson showed that matrix multiplication is a batch dot product, eigenvalues measure directional stretch, and shape discipline prevents silent bugs. This assignment takes those three ideas further: you will compute cosine similarity (dot product + norms), L2-normalize a matrix row-by-row, project a feature matrix onto a weight vector, approximate the dominant eigenvector via power iteration, and derive a covariance matrix from raw data. Each function maps directly to an operation you will encounter repeatedly in ML pipelines — from embedding comparison to PCA to dense-layer projections.

See [lesson.md](../lesson.md) for the full theory walkthrough and worked numeric examples.

## Setup

```bash
pip install "numpy>=1.24"
```

## Tasks

### Task 1 — Cosine Similarity

Implement `cosine_similarity(a, b)`. Compute the cosine similarity between two 1-D vectors using only NumPy element-wise operations and `np.linalg.norm`. Do **not** call `np.dot` directly — compute the numerator as a manual element-wise multiply + sum.

Expected: `cosine_similarity([1, 2, 3], [4, 5, 6])` → `0.9746` (within ±0.0001)

### Task 2 — Row-Wise L2 Normalisation

Implement `row_normalize(X)`. Given a 2-D matrix `X` of shape `(n, d)`, return a new matrix where every row has unit L2 norm. Each row divided by its own norm. Handle the edge case where a row's norm is zero (leave that row unchanged).

Expected: rows of `row_normalize(X)` each have `np.linalg.norm(row)` ≈ 1.0

### Task 3 — Batch Feature Projection

Implement `batch_project(X, weights)`. Given feature matrix `X` of shape `(n, d)` and weight vector `weights` of shape `(d,)`, return the 1-D projection array of shape `(n,)` where each element is the dot product of that sample with `weights`. Use the `@` operator for the multiplication.

Expected: `batch_project([[0.25, 0.80], [0.60, 0.45], [0.90, 0.10]], [0.7, 0.3])` → `[0.415, 0.555, 0.660]`

### Task 4 — Power Iteration for Dominant Eigenpair

Implement `power_iteration(M, n_iter)`. Use the power iteration algorithm to approximate the dominant eigenvector of symmetric square matrix `M`:

1. Start with a random unit vector `v` (seed with `np.random.seed(42)`).
2. For `n_iter` steps: `v = M @ v`, then normalise `v` to unit length.
3. Compute the dominant eigenvalue as the Rayleigh quotient: `λ = vᵀ M v`.
4. Return `(lambda, v)` as a tuple.

Expected: on `M = [[4, 2], [2, 3]]` with `n_iter=100` → eigenvalue ≈ `5.5616` (within ±0.001)

### Task 5 — Covariance Matrix

Implement `covariance_matrix(X)`. Given data matrix `X` of shape `(n, d)`:

1. Subtract the column mean from each row (centre the data).
2. Compute the `(d, d)` covariance matrix as `(X_centred.T @ X_centred) / (n - 1)`.
3. Return the result. Do **not** call `np.cov`.

Expected: verify your output matches `np.cov(X, rowvar=False)` to within `1e-10`.

## Expected Output

Running `python starter.py` should produce output matching this pattern:

```
==================================================
Task 1 — Cosine Similarity
==================================================
  a                      : [1. 2. 3.]
  b                      : [4. 5. 6.]
  cosine_similarity(a,b) : 0.9746

==================================================
Task 2 — Row-Wise L2 Normalisation
==================================================
  Input X:
  [[3. 4.]
   [1. 0.]
   [0. 5.]]
  Normalized rows:
  [[0.6 0.8]
   [1.  0. ]
   [0.  1. ]]
  Row norms after normalisation: [1. 1. 1.]

==================================================
Task 3 — Batch Feature Projection
==================================================
  Feature matrix X (3 users × 2 features):
  [[0.25 0.8 ]
   [0.6  0.45]
   [0.9  0.1 ]]
  Weights: [0.7 0.3]
  Projections: [0.415 0.555 0.66 ]

==================================================
Task 4 — Power Iteration (dominant eigenpair)
==================================================
  M:
  [[4. 2.]
   [2. 3.]]
  Dominant eigenvalue  : 5.5616
  Dominant eigenvector : [0.8507 0.5257]
  Verification M·v ≈ λ·v : True

==================================================
Task 5 — Covariance Matrix
==================================================
  X (4 samples × 3 features):
  [[2. 0. 4.]
   [3. 1. 5.]
   [4. 2. 6.]
   [5. 3. 7.]]
  Covariance matrix:
  [[1.6667 1.6667 1.6667]
   [1.6667 1.6667 1.6667]
   [1.6667 1.6667 1.6667]]
  Matches np.cov: True
```

> Numeric values are rounded for display. Accept ±0.001 on eigenvalue, ±0.0001 on cosine similarity, exact match (within `1e-10`) for covariance.

## Evaluation Criteria

- [ ] `cosine_similarity` returns a value within ±0.0001 of `0.9746` for the given input and does not call `np.dot` directly
- [ ] `row_normalize` returns a matrix where every non-zero row has L2 norm of exactly `1.0` (within `1e-10`)
- [ ] `batch_project` returns `[0.415, 0.555, 0.660]` (within ±0.0001) and uses the `@` operator
- [ ] `power_iteration` converges to eigenvalue ≈ `5.5616` (within ±0.001) on `M = [[4,2],[2,3]]` with `n_iter=100`
- [ ] `covariance_matrix` output matches `np.cov(X, rowvar=False)` to within `1e-10` for the given test data
- [ ] `python starter.py` runs to completion without errors or modifications

## Extension Challenge

**Implement the full Power Iteration SVD (no `np.linalg.svd` allowed).**

Write a function `top_k_svd(X, k, n_iter=200)` that approximates the top-`k` left singular vectors and singular values of a rectangular matrix `X` of shape `(n, d)`:

1. Form the square symmetric matrix `C = Xᵀ X`.
2. Use deflation: find the dominant eigenpair of `C` via power iteration, record it, subtract its contribution (`λ · v · vᵀ`), then repeat to find the next eigenpair. Repeat `k` times.
3. The `i`-th singular value is `sqrt(λᵢ)` and the right singular vector is `vᵢ`.
4. Compute the left singular vectors as `u_i = X @ v_i / σ_i`.
5. Verify your `σ` values match the first `k` entries of `np.linalg.svd(X, full_matrices=False)[1]` to within `0.01`.

Test on `X = np.random.default_rng(0).standard_normal((50, 10))` with `k=3`. This is the mathematical core of PCA — you will see it again in node 2.2.1.

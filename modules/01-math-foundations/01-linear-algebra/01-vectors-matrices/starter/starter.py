# Dependencies: numpy>=1.24
# Node: 1.1.1 — Vectors & Matrices: Core Operations
# Run: python starter.py

import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Implemented helpers — provided for you, no changes needed
# ---------------------------------------------------------------------------

def print_separator(label: str) -> None:
    """Print a labelled section header."""
    print("=" * 50)
    print(f"{label}")
    print("=" * 50)


def load_sample_data() -> dict:
    """Return fixed arrays used across all tasks."""
    return {
        # Task 1
        "a": np.array([1.0, 2.0, 3.0]),
        "b": np.array([4.0, 5.0, 6.0]),
        # Task 2
        "X_norm": np.array([[3.0, 4.0],
                            [1.0, 0.0],
                            [0.0, 5.0]]),
        # Task 3
        "X_proj": np.array([[0.25, 0.80],
                            [0.60, 0.45],
                            [0.90, 0.10]]),
        "weights": np.array([0.7, 0.3]),
        # Task 4
        "M": np.array([[4.0, 2.0],
                       [2.0, 3.0]]),
        # Task 5
        "X_cov": np.array([[2.0, 0.0, 4.0],
                           [3.0, 1.0, 5.0],
                           [4.0, 2.0, 6.0],
                           [5.0, 3.0, 7.0]]),
    }


def verify_close(label: str, actual: np.ndarray, expected: np.ndarray,
                 atol: float = 1e-10) -> bool:
    """Print pass/fail for an array comparison and return the bool result."""
    match = np.allclose(actual, expected, atol=atol)
    status = "PASS" if match else "FAIL"
    print(f"  [{status}] {label}")
    return match


# ---------------------------------------------------------------------------
# Task 1 — Cosine Similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine similarity between two 1-D vectors.

    Compute the numerator as element-wise multiply then sum (no np.dot).
    Divide by the product of the L2 norms.
    """
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Task 2 — Row-Wise L2 Normalisation
# ---------------------------------------------------------------------------

def row_normalize(X: np.ndarray) -> np.ndarray:
    """Return a copy of X where every row has unit L2 norm.

    Rows with zero norm are left unchanged.
    """
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Task 3 — Batch Feature Projection
# ---------------------------------------------------------------------------

def batch_project(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Project each row of X onto weights using the @ operator.

    Args:
        X:       shape (n, d) — feature matrix, one sample per row
        weights: shape (d,)   — projection weights

    Returns:
        shape (n,) — one scalar projection per sample
    """
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Task 4 — Power Iteration for Dominant Eigenpair
# ---------------------------------------------------------------------------

def power_iteration(M: np.ndarray, n_iter: int = 100) -> Tuple[float, np.ndarray]:
    """Approximate the dominant eigenpair of symmetric square matrix M.

    Algorithm:
        1. Seed np.random with 42, draw a random starting vector v of length M.shape[0].
        2. Normalise v to unit length.
        3. For n_iter steps: v = M @ v, then re-normalise v.
        4. Compute eigenvalue as the Rayleigh quotient: lambda = v.T @ M @ v.
        5. Return (lambda, v).

    Args:
        M:      square symmetric matrix, shape (n, n)
        n_iter: number of power iteration steps

    Returns:
        (dominant_eigenvalue: float, dominant_eigenvector: np.ndarray of shape (n,))
    """
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Task 5 — Covariance Matrix
# ---------------------------------------------------------------------------

def covariance_matrix(X: np.ndarray) -> np.ndarray:
    """Compute the (d, d) sample covariance matrix from data matrix X (n, d).

    Steps:
        1. Subtract the column mean from each row to centre the data.
        2. Return (X_centred.T @ X_centred) / (n - 1).

    Do NOT call np.cov.
    """
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# main — runs all tasks in order with labelled output
# ---------------------------------------------------------------------------

def main() -> None:
    data = load_sample_data()

    # ------------------------------------------------------------------
    # Task 1
    # ------------------------------------------------------------------
    print_separator("Task 1 — Cosine Similarity")
    a, b = data["a"], data["b"]
    sim = cosine_similarity(a, b)
    print(f"  a                      : {a}")
    print(f"  b                      : {b}")
    print(f"  cosine_similarity(a,b) : {sim:.4f}")
    print()

    # ------------------------------------------------------------------
    # Task 2
    # ------------------------------------------------------------------
    print_separator("Task 2 — Row-Wise L2 Normalisation")
    X_norm = data["X_norm"]
    X_out = row_normalize(X_norm)
    print(f"  Input X:\n  {X_norm}")
    print(f"  Normalized rows:\n  {X_out}")
    row_norms = np.linalg.norm(X_out, axis=1)
    print(f"  Row norms after normalisation: {row_norms}")
    print()

    # ------------------------------------------------------------------
    # Task 3
    # ------------------------------------------------------------------
    print_separator("Task 3 — Batch Feature Projection")
    X_proj, weights = data["X_proj"], data["weights"]
    projections = batch_project(X_proj, weights)
    print(f"  Feature matrix X (3 users × 2 features):\n  {X_proj}")
    print(f"  Weights: {weights}")
    print(f"  Projections: {np.round(projections, 4)}")
    print()

    # ------------------------------------------------------------------
    # Task 4
    # ------------------------------------------------------------------
    print_separator("Task 4 — Power Iteration (dominant eigenpair)")
    M = data["M"]
    lam, v = power_iteration(M, n_iter=100)
    Mv = M @ v
    lv = lam * v
    print(f"  M:\n  {M}")
    print(f"  Dominant eigenvalue  : {lam:.4f}")
    print(f"  Dominant eigenvector : {np.round(v, 4)}")
    print(f"  Verification M·v ≈ λ·v : {np.allclose(Mv, lv, atol=1e-6)}")
    print()

    # ------------------------------------------------------------------
    # Task 5
    # ------------------------------------------------------------------
    print_separator("Task 5 — Covariance Matrix")
    X_cov = data["X_cov"]
    cov = covariance_matrix(X_cov)
    expected_cov = np.cov(X_cov, rowvar=False)
    print(f"  X (4 samples × 3 features):\n  {X_cov}")
    print(f"  Covariance matrix:\n  {np.round(cov, 4)}")
    verify_close("Matches np.cov", cov, expected_cov, atol=1e-10)
    print()


if __name__ == "__main__":
    main()

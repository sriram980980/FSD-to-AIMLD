# Dependencies: numpy>=1.24, scikit-learn>=1.3, matplotlib>=3.7
# Node: 2.1.1 — Linear Models — Regression & Classification
# Run: python starter.py

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error
from sklearn.model_selection import train_test_split
from typing import Tuple


# ── implemented helpers (do not modify) ────────────────────────────────────────


def add_bias_column(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones to X so the dot product includes a bias term."""
    return np.c_[np.ones(len(X)), X]


def load_regression_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic regression dataset and return an 80/20 train/test split."""
    X, y = make_regression(n_samples=300, n_features=3, noise=20.0, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def load_classification_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic binary classification dataset and return an 80/20 split."""
    X, y = make_classification(
        n_samples=600,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ── student implementations ────────────────────────────────────────────────────


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Squash any real-valued array into (0, 1) using σ(z) = 1 / (1 + e^{-z}).

    Args:
        z: Array of real-valued logits, shape (n,) or any broadcastable shape.

    Returns:
        Array of the same shape with all values in the open interval (0, 1).

    Expected:
        sigmoid(np.array([0.0])) → [0.5]
        sigmoid(np.array([0.8])) → [≈0.6900]
    """
    raise NotImplementedError("TODO: implement this")


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Squared Error: (1/n) * Σ (y_true_i − y_pred_i)².

    Args:
        y_true: Ground-truth continuous targets, shape (n,).
        y_pred: Model predictions, shape (n,).

    Returns:
        Scalar MSE value ≥ 0.

    Expected:
        mse_loss(np.array([3.0, 5.0]), np.array([2.5, 4.5])) → 0.25
    """
    raise NotImplementedError("TODO: implement this")


def cross_entropy_loss(
    y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12
) -> float:
    """Binary Cross-Entropy: −(1/n) Σ [y_i log(p_i) + (1−y_i) log(1−p_i)].

    Clip all probabilities into [eps, 1−eps] before taking the logarithm to
    prevent log(0) producing −inf.

    Args:
        y_true: Binary labels in {0, 1}, shape (n,).
        y_prob: Predicted probabilities in (0, 1), shape (n,).
        eps:    Small constant for numerical stability (default 1e-12).

    Returns:
        Scalar cross-entropy value ≥ 0.

    Expected:
        cross_entropy_loss(np.array([1.0]), np.array([0.6900])) → ≈0.3711
    """
    raise NotImplementedError("TODO: implement this")


def fit_linear_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit linear regression using the closed-form OLS normal equation.

    Solve w* = (X^T X)^{-1} X^T y via np.linalg.pinv for numerical stability.
    Prepend a bias column of ones to X before solving so the first element of
    the returned vector is the bias term b.

    Hint: use add_bias_column(X) before computing.

    Args:
        X: Feature matrix, shape (n, d).
        y: Continuous targets, shape (n,).

    Returns:
        Weight vector of shape (d+1,) where index 0 is the bias.
    """
    raise NotImplementedError("TODO: implement this")


def fit_logistic_gd(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.1,
    epochs: int = 200,
) -> np.ndarray:
    """Train binary logistic regression via batch gradient descent.

    Update rule (one full-batch step per epoch):
        p     = sigmoid(X_b @ w)
        grad  = (1/n) * X_b.T @ (p − y)
        w    -= learning_rate * grad

    Print the cross-entropy loss every 50 epochs:
        print(f"  Epoch {epoch:>3d} | Cross-Entropy Loss: {loss:.4f}")

    Hint: use add_bias_column(X) to get X_b and initialise weights to zeros.

    Args:
        X:             Feature matrix, shape (n, d).
        y:             Binary targets in {0, 1}, shape (n,).
        learning_rate: Step size for gradient descent (default 0.1).
        epochs:        Number of full-dataset passes (default 200).

    Returns:
        Learned weight vector of shape (d+1,) where index 0 is the bias.
    """
    raise NotImplementedError("TODO: implement this")


# ── main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    # ── Task 1-3: sanity checks ────────────────────────────────────────────────
    print("=== SANITY CHECKS ===")

    sig_zero = sigmoid(np.array([0.0]))
    sig_point8 = sigmoid(np.array([0.8]))
    print(f"sigmoid(0.0)             : [{sig_zero[0]:.4f}]")
    print(f"sigmoid(0.8)             : [{sig_point8[0]:.4f}]")

    mse_ex = mse_loss(np.array([3.0, 5.0]), np.array([2.5, 4.5]))
    print(f"MSE worked example       : {mse_ex:.4f}")

    ce_ex = cross_entropy_loss(np.array([1.0]), np.array([0.6900]))
    print(f"Cross-Entropy worked ex  : {ce_ex:.4f}")

    # ── Task 4: linear regression ──────────────────────────────────────────────
    print("\n=== LINEAR REGRESSION ===")
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = load_regression_data()

    weights_ols = fit_linear_ols(X_train_reg, y_train_reg)

    # Predict: prepend bias column then dot with weights
    y_pred_scratch_reg = add_bias_column(X_test_reg) @ weights_ols

    mse_scratch = mse_loss(y_test_reg, y_pred_scratch_reg)

    # scikit-learn reference
    sk_reg = LinearRegression().fit(X_train_reg, y_train_reg)
    mse_sklearn = mean_squared_error(y_test_reg, sk_reg.predict(X_test_reg))

    print(f"Dataset        : n=300, 3 features, noise=20")
    print(f"Scratch OLS    | Test MSE : {mse_scratch:.4f}")
    print(f"sklearn        | Test MSE : {mse_sklearn:.4f}")
    print(f"Delta                     : {abs(mse_scratch - mse_sklearn):.6f}  (must be < 1.0)")

    # ── Task 5: logistic regression ────────────────────────────────────────────
    print("\n=== LOGISTIC REGRESSION — GD Training ===")
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = load_classification_data()

    weights_logistic = fit_logistic_gd(
        X_train_cls, y_train_cls.astype(float), learning_rate=0.1, epochs=200
    )

    # Predict probabilities and binary labels
    y_prob_scratch = sigmoid(add_bias_column(X_test_cls) @ weights_logistic)
    y_pred_scratch_cls = (y_prob_scratch >= 0.5).astype(int)

    # ── Task 6: comparison table ───────────────────────────────────────────────
    print("\n=== CLASSIFICATION RESULTS ===")

    # scikit-learn reference
    sk_cls = LogisticRegression(max_iter=500).fit(X_train_cls, y_train_cls)
    y_prob_sklearn = sk_cls.predict_proba(X_test_cls)[:, 1]
    y_pred_sklearn = sk_cls.predict(X_test_cls)

    acc_scratch = accuracy_score(y_test_cls, y_pred_scratch_cls)
    ll_scratch = log_loss(y_test_cls, y_prob_scratch)
    acc_sklearn = accuracy_score(y_test_cls, y_pred_sklearn)
    ll_sklearn = log_loss(y_test_cls, y_prob_sklearn)
    gap = abs(acc_scratch - acc_sklearn)

    print(f"{'':14s}  Accuracy   Log-Loss")
    print(f"Scratch LR  : {acc_scratch:.4f}     {ll_scratch:.4f}")
    print(f"sklearn      : {acc_sklearn:.4f}     {ll_sklearn:.4f}")
    print(f"Gap (acc)             : {gap:.4f}  (must be ≤ 0.05)")


if __name__ == "__main__":
    main()

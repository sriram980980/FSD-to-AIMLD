# Dependencies: numpy>=1.24, matplotlib>=3.7
# Node: 1.1.2 — Calculus: Derivatives & Gradients
# Run: python starter.py

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple


# ---------------------------------------------------------------------------
# Dataset — fixed seed for reproducibility
# True relationship: y = 3.0 * x + 0.5 + noise
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(42)
X_DATA: np.ndarray = RNG.uniform(0, 10, size=50)
Y_DATA: np.ndarray = 3.0 * X_DATA + 0.5 + RNG.normal(0, 1.0, size=50)


# ---------------------------------------------------------------------------
# Implemented helpers — provided to reduce boilerplate
# ---------------------------------------------------------------------------

def make_loss_curve_plot(
    epochs: np.ndarray,
    losses: list[float],
    title: str,
    filename: str,
) -> None:
    """Save a single loss-vs-epoch figure to disk."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, losses, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    print(f"Saved: {filename}")


def make_lr_sensitivity_plot(
    lr_losses: dict[float, list[float]],
    filename: str,
) -> None:
    """Save a multi-line LR sensitivity figure to disk."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for lr, losses in lr_losses.items():
        ax.plot(losses, linewidth=2, label=f"lr={lr}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Learning Rate Sensitivity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    print(f"Saved: {filename}")


def print_section(title: str) -> None:
    """Print a clearly labelled section header."""
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")


# ---------------------------------------------------------------------------
# Task 1 — Numerical Derivative via Finite Difference
# ---------------------------------------------------------------------------

def numerical_derivative(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """Estimate f'(x) using the central-difference formula: (f(x+h) - f(x-h)) / (2*h).

    Args:
        f: A scalar function of one variable.
        x: The point at which to evaluate the derivative.
        h: The step size for finite difference (default 1e-5).

    Returns:
        Approximate derivative of f at x.
    """
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Task 2 — Analytical Gradient of MSE Loss
# ---------------------------------------------------------------------------

def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute mean squared error: (1/n) * sum((y_pred - y_true)^2).

    Args:
        y_pred: Predicted values, shape (n,).
        y_true: Ground-truth values, shape (n,).

    Returns:
        Scalar MSE loss.
    """
    raise NotImplementedError("TODO: implement this")


def mse_gradient(
    weight: float, bias: float, X: np.ndarray, y: np.ndarray
) -> Tuple[float, float]:
    """Compute analytical partial derivatives of MSE w.r.t. weight and bias.

    Model: y_hat = weight * X + bias
    Loss:  L = (1/n) * sum((y_hat - y)^2)

    Partial derivatives:
        dL/dw = (2/n) * sum((y_hat - y) * X)
        dL/db = (2/n) * sum((y_hat - y))

    Args:
        weight: Current scalar weight.
        bias:   Current scalar bias.
        X:      Input features, shape (n,).
        y:      True targets, shape (n,).

    Returns:
        Tuple (dL/dw, dL/db).
    """
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Task 3 — Gradient Descent Loop
# ---------------------------------------------------------------------------

def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    epochs: int = 500,
) -> Tuple[float, float, list[float]]:
    """Run gradient descent on a single-weight linear model with MSE loss.

    Start from weight=0.0, bias=0.0. At each epoch:
        1. Compute predictions: y_hat = weight * X + bias
        2. Compute loss via mse_loss()
        3. Compute gradients via mse_gradient()
        4. Update: weight -= learning_rate * dL/dw
                   bias   -= learning_rate * dL/db

    Args:
        X:             Input features, shape (n,).
        y:             True targets, shape (n,).
        learning_rate: Step size alpha.
        epochs:        Number of full-dataset passes.

    Returns:
        Tuple (final_weight, final_bias, loss_history) where
        loss_history is a list of length `epochs`.
    """
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Task 4 — Gradient Check
# ---------------------------------------------------------------------------

def gradient_check(
    weight: float,
    bias: float,
    X: np.ndarray,
    y: np.ndarray,
    h: float = 1e-5,
) -> None:
    """Verify analytical gradient against numerical gradient for weight and bias.

    For each parameter p, define a scalar loss function L(p) by holding all
    other parameters fixed. Compute the numerical derivative with
    numerical_derivative() and compare to the analytical value from
    mse_gradient().

    Relative error formula:
        rel_error = |g_analytical - g_numerical| / (|g_analytical| + |g_numerical| + 1e-8)

    Print PASS if rel_error < 1e-5, else FAIL.

    Args:
        weight: Weight at which to evaluate gradients.
        bias:   Bias at which to evaluate gradients.
        X:      Input features, shape (n,).
        y:      True targets, shape (n,).
        h:      Finite-difference step size.
    """
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# main — calls all tasks in order
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # Task 1: Numerical Derivative
    # ------------------------------------------------------------------
    print_section("Task 1: Numerical Derivative")

    def f(x: float) -> float:
        return x**3 - 4 * x

    x0 = 2.0
    analytical_deriv = 3 * x0**2 - 4          # f'(x) = 3x^2 - 4
    numerical_deriv = numerical_derivative(f, x0, h=1e-5)
    abs_error = abs(analytical_deriv - numerical_deriv)

    print(f"Analytical f'({x0}) = {analytical_deriv:.6f}")
    print(f"Numerical  f'({x0}) = {numerical_deriv:.6f} (h=1e-05)")
    print(f"Absolute error     = {abs_error:.6e} (< 1e-06: {'PASS' if abs_error < 1e-6 else 'FAIL'})")

    # ------------------------------------------------------------------
    # Task 2: Analytical MSE Gradient
    # ------------------------------------------------------------------
    print_section("Task 2: Analytical MSE Gradient")

    w0, b0 = 0.0, 0.0
    dw, db = mse_gradient(w0, b0, X_DATA, Y_DATA)
    y_hat0 = w0 * X_DATA + b0
    initial_loss = mse_loss(y_hat0, Y_DATA)

    print(f"Initial weight={w0}, bias={b0}")
    print(f"  Initial loss = {initial_loss:.4f}")
    print(f"  dL/dw        = {dw:.4f}")
    print(f"  dL/db        = {db:.4f}")

    # ------------------------------------------------------------------
    # Task 3: Gradient Descent
    # ------------------------------------------------------------------
    print_section("Task 3: Gradient Descent")

    final_weight, final_bias, loss_history = gradient_descent(
        X_DATA, Y_DATA, learning_rate=0.01, epochs=500
    )

    log_epochs = [0, 100, 200, 300, 400, 499]
    for ep in log_epochs:
        print(f"Epoch {ep:>3d} | Loss: {loss_history[ep]:>8.4f}")

    print(f"\nFinal weight = {final_weight:.4f}  (expected ~3.0)")
    print(f"Final bias   = {final_bias:.4f}  (expected ~0.5)")

    # ------------------------------------------------------------------
    # Task 4: Gradient Check
    # ------------------------------------------------------------------
    print_section("Task 4: Gradient Check")
    gradient_check(w0, b0, X_DATA, Y_DATA)

    # ------------------------------------------------------------------
    # Task 5: Loss Curve
    # ------------------------------------------------------------------
    print_section("Task 5: Loss Curve")
    make_loss_curve_plot(
        epochs=np.arange(len(loss_history)),
        losses=loss_history,
        title="Gradient Descent Convergence",
        filename="loss_curve.png",
    )

    # ------------------------------------------------------------------
    # Task 6: Learning Rate Sensitivity
    # ------------------------------------------------------------------
    print_section("Task 6: LR Sensitivity")
    learning_rates = [0.001, 0.01, 0.1]
    lr_losses: dict[float, list[float]] = {}

    for lr in learning_rates:
        _, _, losses = gradient_descent(X_DATA, Y_DATA, learning_rate=lr, epochs=500)
        lr_losses[lr] = losses
        print(f"lr={lr:.3f} | final loss: {losses[-1]:.4f}")

    make_lr_sensitivity_plot(lr_losses, filename="lr_sensitivity.png")


if __name__ == "__main__":
    main()

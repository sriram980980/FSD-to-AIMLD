# Dependencies: torch>=2.1, numpy>=1.24, matplotlib>=3.7
# Node: 3.1.2 — Optimization — Gradient Descent Variants
# Run: python starter.py

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Any


# ── Experiment grid (do not modify) ──────────────────────────────────────────

OPTIMIZER_NAMES = ["SGD", "SGD+Momentum", "Adam"]
LEARNING_RATES  = [0.1, 0.01, 0.001, 0.0001, 0.00001]
EPOCHS          = 200
INPUT_DIM       = 2
OUTPUT_DIM      = 1

# XOR dataset — four samples, fully in-memory, no external files needed
X_DATA = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y_DATA = torch.tensor([[0.0], [1.0], [1.0], [0.0]])


# ── Fully implemented helpers — provided, do not modify ──────────────────────

def set_seed(seed: int = 42) -> None:
    """Fix PyTorch and NumPy random seeds for reproducible weight initialisation."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def format_lr(lr: float) -> str:
    """Return a short scientific-notation string for a learning rate."""
    return f"{lr:.0e}"


def find_best(results: dict[tuple[str, float], float]) -> tuple[str, float, float]:
    """Return (optimizer_name, learning_rate, final_loss) for the best-performing run."""
    best_key = min(results, key=lambda k: results[k])
    return best_key[0], best_key[1], results[best_key]


# ── Task 1 — Implement build_model ───────────────────────────────────────────

def build_model(input_dim: int = INPUT_DIM, output_dim: int = OUTPUT_DIM) -> nn.Module:
    """Return a fresh nn.Sequential MLP: input → 64 → ReLU → 32 → ReLU → output → Sigmoid.

    Each call must return an independent model with freshly initialised weights.
    """
    raise NotImplementedError("TODO: implement this")


# ── Task 2 — Implement make_optimizer ────────────────────────────────────────

def make_optimizer(
    optimizer_name: str,
    parameters: Any,
    learning_rate: float,
) -> torch.optim.Optimizer:
    """Construct and return the correct torch.optim object for optimizer_name.

    Supported names: "SGD" (momentum=0.0), "SGD+Momentum" (momentum=0.9), "Adam".
    Raise ValueError for any other name.
    """
    raise NotImplementedError("TODO: implement this")


# ── Task 3 — Implement train_one_run ─────────────────────────────────────────

def train_one_run(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
) -> list[float]:
    """Train model with BCELoss for the given number of epochs.

    Training loop order: zero_grad → forward → loss → backward → step.
    Return a list of per-epoch scalar loss values (length == epochs).
    """
    raise NotImplementedError("TODO: implement this")


# ── Task 4 — Implement run_grid_search ───────────────────────────────────────

def run_grid_search(
    optimizer_names: list[str],
    learning_rates: list[float],
    epochs: int,
    X: torch.Tensor,
    y: torch.Tensor,
) -> dict[tuple[str, float], float]:
    """Run one training experiment per (optimizer_name, learning_rate) combination.

    For each combination:
      1. Call set_seed(42) to ensure fair comparison.
      2. Call build_model() for a fresh model.
      3. Call make_optimizer() with the current name and learning rate.
      4. Call train_one_run() and record the final epoch loss.
      5. Print a progress line: "Running {name:<14s} lr={lr:.0e} ...  final loss: {loss:.4f}"

    Return a dict mapping (optimizer_name, learning_rate) → final_loss.
    The returned dict must have exactly len(optimizer_names) * len(learning_rates) entries.
    """
    raise NotImplementedError("TODO: implement this")


# ── Task 5 — Implement plot_heatmap ──────────────────────────────────────────

def plot_heatmap(
    results: dict[tuple[str, float], float],
    optimizer_names: list[str],
    learning_rates: list[float],
    output_path: str = "tuning_heatmap.png",
) -> None:
    """Save a heatmap of final losses to output_path.

    Layout:
      - Rows (Y-axis): optimizer_names  (label each row)
      - Columns (X-axis): learning_rates formatted with format_lr()  (label each column)
      - Each cell: colour-coded by loss value (lower = darker using "YlOrRd_r" colormap)
      - Annotate each cell with its loss value formatted to 4 decimal places.

    Save with: plt.savefig(output_path, dpi=120, bbox_inches="tight")
    """
    raise NotImplementedError("TODO: implement this")


# ── main — orchestrates all tasks ────────────────────────────────────────────

def main() -> None:
    print(
        f"=== Hyperparameter Grid Search — "
        f"{len(OPTIMIZER_NAMES)} optimizers × {len(LEARNING_RATES)} learning rates "
        f"× {EPOCHS} epochs ===\n"
    )

    # Task 4: run the full grid
    results = run_grid_search(
        optimizer_names=OPTIMIZER_NAMES,
        learning_rates=LEARNING_RATES,
        epochs=EPOCHS,
        X=X_DATA,
        y=Y_DATA,
    )

    # Task 6: report best combination
    best_name, best_lr, best_loss = find_best(results)
    print(
        f"\nBest combination: {best_name}  lr={best_lr}  final loss: {best_loss:.4f}"
    )

    # Task 5: save heatmap
    plot_heatmap(results, OPTIMIZER_NAMES, LEARNING_RATES)
    print("Heatmap saved → tuning_heatmap.png")


if __name__ == "__main__":
    main()

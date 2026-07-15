# Dependencies: torch>=2.1, numpy>=1.24, matplotlib>=3.7, scikit-learn>=1.3
# Node: 3.1.1 — Multi-Layer Perceptrons — Forward & Backward Pass
# Run: python starter.py

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from typing import List, Tuple


# ── Implemented Helpers ───────────────────────────────────────────────────────


def load_moons_data(
    n_samples: int = 1000,
    noise: float = 0.2,
    random_state: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate moons dataset and return train/test float32 tensors."""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    # Unsqueeze to shape (N, 1) — required by BCELoss
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    return X_train_t, X_test_t, y_train_t, y_test_t


def plot_loss_curve(train_losses: List[float], val_losses: List[float]) -> None:
    """Save training and validation loss curves to loss_curve.png."""
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Val Loss", linewidth=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("MLP Training Loss — 3.1.1")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=120)
    plt.close()
    print("Loss curve saved → loss_curve.png")


def plot_decision_boundary(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor
) -> None:
    """Save 2D decision boundary contour plot to decision_boundary.png."""
    model.eval()
    x_min = X[:, 0].min().item() - 0.5
    x_max = X[:, 0].max().item() + 0.5
    y_min = X[:, 1].min().item() - 0.5
    y_max = X[:, 1].max().item() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        probs = model(grid).numpy().reshape(xx.shape)
    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, probs, levels=50, cmap="RdBu", alpha=0.8)
    plt.colorbar(label="P(class = 1)")
    y_np = y.squeeze().numpy()
    plt.scatter(
        X[:, 0].numpy(),
        X[:, 1].numpy(),
        c=y_np,
        cmap="RdBu",
        edgecolors="k",
        s=20,
    )
    plt.title("MLP Decision Boundary — 3.1.1")
    plt.tight_layout()
    plt.savefig("decision_boundary.png", dpi=120)
    plt.close()
    print("Decision boundary saved → decision_boundary.png")


# ── Student Tasks ─────────────────────────────────────────────────────────────


def build_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Module:
    """
    Task 1: Build an MLP with two hidden layers and a sigmoid output.

    Return an nn.Sequential with EXACTLY this architecture:
        Linear(input_dim  → hidden_dim) → ReLU
        Linear(hidden_dim → hidden_dim) → ReLU
        Linear(hidden_dim → output_dim) → Sigmoid

    The model accepts batches of shape (N, input_dim) and outputs (N, output_dim).
    Do NOT add BatchNorm or Dropout — keep it minimal for this exercise.
    """
    raise NotImplementedError("TODO: implement this")


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 32,
) -> float:
    """
    Task 2: Run one full epoch of mini-batch gradient descent.

    Use TensorDataset + DataLoader (both already imported) to iterate over
    batches of size `batch_size`. For EACH batch:
        1. optimizer.zero_grad()           — clear stale gradients
        2. predictions = model(X_batch)    — forward pass
        3. loss = criterion(predictions, y_batch)  — compute BCE loss
        4. loss.backward()                 — backprop through the graph
        5. optimizer.step()                — update weights

    Set model.train() at the start of this function.
    Return the mean loss (float) across all batches in the epoch.
    """
    raise NotImplementedError("TODO: implement this")


def evaluate_accuracy(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor
) -> float:
    """
    Task 3: Compute binary classification accuracy (no gradient tracking).

    - Call model.eval() to disable dropout/batchnorm (good habit even without them).
    - Wrap the forward pass in torch.no_grad() to save memory.
    - Threshold: model output >= 0.5 → predict class 1, else class 0.
    - Return accuracy as a float in [0, 1].

    Acceptance threshold: >= 0.90 on the test set after 50 epochs.
    """
    raise NotImplementedError("TODO: implement this")


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 50,
) -> Tuple[List[float], List[float]]:
    """
    Task 4: Run the full training loop for `epochs` epochs.

    Each epoch:
        a. Call train_one_epoch → get train loss for this epoch.
        b. Set model.eval(), compute val loss inside torch.no_grad():
               val_preds = model(X_val)
               val_loss  = criterion(val_preds, y_val).item()
        c. Print exactly:
               f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"

    Return (train_losses, val_losses) — one float per epoch, two lists of length `epochs`.
    """
    raise NotImplementedError("TODO: implement this")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    print("=== 3.1.1 — Build basic PyTorch MLP ===\n")

    # Load synthetic moons dataset
    X_train, X_test, y_train, y_test = load_moons_data(n_samples=1000, noise=0.2)
    print(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}\n")

    # Task 1 — build model (input_dim=2 features, hidden_dim=64, output_dim=1 probability)
    model = build_mlp(input_dim=2, hidden_dim=64, output_dim=1)
    print(f"Model architecture:\n{model}\n")

    # Optimizer and loss — Adam + BCELoss are the standard pair for this setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    # Tasks 2, 3, 4 — full training loop
    train_losses, val_losses = run_training(
        model,
        optimizer,
        criterion,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=50,
    )

    # Task 3 — final accuracy evaluation
    accuracy = evaluate_accuracy(model, X_test, y_test)
    print(f"\nFinal Test Accuracy: {accuracy:.4f}")
    print(f"Final Train Loss:    {train_losses[-1]:.4f}")
    print(f"Final Val Loss:      {val_losses[-1]:.4f}")

    # Implemented helpers — save visualizations
    plot_loss_curve(train_losses, val_losses)
    plot_decision_boundary(model, X_test, y_test)


if __name__ == "__main__":
    main()

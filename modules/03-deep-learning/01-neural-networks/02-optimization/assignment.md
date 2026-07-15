# Assignment 3.1.2 — Execute Hyperparameter Tuning Experiment

## Objective

Build a systematic hyperparameter tuning experiment that compares SGD, SGD+Momentum, and Adam
optimizers across five learning rates on a multi-layer perceptron, then produce a convergence heatmap
that reveals which optimizer/learning-rate pair minimises training loss fastest.

## Background

The lesson showed that Adam adapts its step size per-parameter while SGD applies a fixed global step.
Both optimizers are sensitive to the initial learning rate: too large causes divergence, too small
causes stagnation. Refer to the **Theory → Adam** section of `lesson.md` for the bias-correction
equations and the **Learning Rate Scheduling** section for the cosine annealing formula used here.
This assignment replaces the lesson's illustrative Rosenbrock benchmark with a repeatable tabular
training experiment so you can observe the sensitivity empirically rather than analytically.

## Setup

```bash
pip install torch>=2.1 numpy>=1.24 matplotlib>=3.7
```

No GPU required. Every experiment in this assignment runs in under 90 seconds on a CPU laptop.

## Tasks

### Task 1 — Implement `build_model`

Complete the stub that returns a fresh `nn.Sequential` MLP with the architecture:
`input_dim → 64 → ReLU → 32 → ReLU → output_dim → Sigmoid`.

Each call to `build_model` must return an **independent** model (separate weights) so that
experiments do not share state across learning-rate trials.

### Task 2 — Implement `make_optimizer`

Complete the stub that receives `optimizer_name` (`"SGD"`, `"SGD+Momentum"`, `"Adam"`),
`model.parameters()`, and a `learning_rate` float, then constructs and returns the matching
`torch.optim` object.

- `"SGD"` → `torch.optim.SGD` with `momentum=0.0`
- `"SGD+Momentum"` → `torch.optim.SGD` with `momentum=0.9`
- `"Adam"` → `torch.optim.Adam` with default betas

Raise `ValueError` for any unrecognised optimizer name.

### Task 3 — Implement `train_one_run`

Complete the stub that trains a model for a fixed number of epochs using BCELoss and returns
the **per-epoch loss list**. The function signature is:

```python
def train_one_run(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
) -> list[float]:
```

The training loop must call `optimizer.zero_grad()`, compute `criterion(output, target)`,
call `loss.backward()`, then `optimizer.step()` — standard PyTorch training order.

### Task 4 — Implement `run_grid_search`

Complete the stub that:
1. Iterates over every `(optimizer_name, learning_rate)` combination in the provided grid.
2. Calls `build_model`, `make_optimizer`, and `train_one_run` for each combination.
3. Returns a `dict` keyed by `(optimizer_name, learning_rate)` mapping to the final loss value
   (last element of the epoch-loss list).

### Task 5 — Implement `plot_heatmap`

Complete the stub that converts the `results` dict into a 2-D grid (rows = optimizer names,
columns = learning rates) and saves a `matplotlib` heatmap to `tuning_heatmap.png`.
Cells should display the final loss as a float formatted to 4 decimal places.
Use `plt.savefig("tuning_heatmap.png", dpi=120, bbox_inches="tight")`.

### Task 6 — Run the Full Experiment and Analyse

Call `run_grid_search` with the pre-defined grid below, then call `plot_heatmap`:

```python
OPTIMIZER_NAMES = ["SGD", "SGD+Momentum", "Adam"]
LEARNING_RATES  = [0.1, 0.01, 0.001, 0.0001, 0.00001]
EPOCHS          = 200
```

After the heatmap is saved, print the single best `(optimizer_name, learning_rate)` pair —
the combination with the **lowest** final loss. Confirm your printed result matches the
visually darkest cell in the saved heatmap image.

## Expected Output

```
=== Hyperparameter Grid Search — 3 optimizers × 5 learning rates × 200 epochs ===

Running SGD            lr=1e-01 ...  final loss: 0.6931
Running SGD            lr=1e-02 ...  final loss: 0.5843
Running SGD            lr=1e-03 ...  final loss: 0.6893
Running SGD            lr=1e-04 ...  final loss: 0.6931
Running SGD            lr=1e-05 ...  final loss: 0.6931
Running SGD+Momentum   lr=1e-01 ...  final loss: 0.0541
Running SGD+Momentum   lr=1e-02 ...  final loss: 0.2174
Running SGD+Momentum   lr=1e-03 ...  final loss: 0.6730
Running SGD+Momentum   lr=1e-04 ...  final loss: 0.6930
Running SGD+Momentum   lr=1e-05 ...  final loss: 0.6931
Running Adam           lr=1e-01 ...  final loss: 0.0138
Running Adam           lr=1e-02 ...  final loss: 0.0201
Running Adam           lr=1e-03 ...  final loss: 0.0689
Running Adam           lr=1e-04 ...  final loss: 0.4112
Running Adam           lr=1e-05 ...  final loss: 0.6901

Best combination: Adam  lr=0.1  final loss: 0.0138
Heatmap saved → tuning_heatmap.png
```

> **Tolerance:** Final loss values may vary by ±0.05 due to random weight initialisation.
> The ranking pattern (Adam outperforming SGD at moderate–high learning rates) must be preserved.

## Evaluation Criteria

- [ ] `build_model` returns an `nn.Sequential` with exactly 5 layers (Linear, ReLU, Linear, ReLU, Linear/Sigmoid) and independent weights on each call
- [ ] `make_optimizer` raises `ValueError` for unrecognised optimizer names and correctly sets `momentum=0.0` for plain `"SGD"`
- [ ] `train_one_run` follows the correct PyTorch training order (`zero_grad → forward → loss → backward → step`) and returns a list of length `epochs`
- [ ] `run_grid_search` executes all 15 optimizer/learning-rate combinations and returns a dict with exactly 15 entries
- [ ] `plot_heatmap` saves `tuning_heatmap.png` with labelled axes (optimizer names on Y-axis, learning rates on X-axis) and per-cell loss annotations
- [ ] The printed best combination matches the minimum-loss cell in the saved heatmap

## Extension Challenge

Extend the experiment with **cosine annealing** learning rate scheduling on top of every optimizer:
wrap each optimizer with `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)`
and call `scheduler.step()` after each `optimizer.step()`.

Produce a **side-by-side** comparison: two heatmaps saved to `tuning_heatmap_no_schedule.png` and
`tuning_heatmap_with_schedule.png`. Quantify how many of the 15 cells improve (lower final loss)
when scheduling is enabled and print a summary:

```
Cells improved with cosine annealing: X / 15
Mean loss improvement: +0.XXXX
```

No starter code is provided for the extension. The cosine annealing formula and its `T_max`/`eta_min`
semantics are covered in the lesson's **Theory → Learning Rate Scheduling** section.

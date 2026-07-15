# Assignment 1.1.2 — Code Manual Gradient Descent Calculation

## Objective

Implement gradient descent from scratch — computing numerical derivatives, analytical gradients, and weight updates by hand — to prove you understand how a training loop works at the calculus level before any framework hides it from you.

## Background

The lesson showed that a gradient is a vector of partial derivatives: one per weight, each measuring how steeply the loss rises if that weight increases. Gradient descent moves each weight in the **negative** gradient direction by a small step controlled by the learning rate. The chain rule makes this tractable for composed functions (layers). This assignment strips away NumPy's `autograd` and PyTorch's `.backward()` to expose exactly what those tools compute for you.

Refer back to `lesson.md` — specifically the **Chain Rule** and **Gradient Descent** sections — before starting Task 3.

## Setup

```bash
pip install "numpy>=1.24" "matplotlib>=3.7"
```

## Tasks

### Task 1 — Numerical Derivative via Finite Difference

Implement `numerical_derivative(f, x, h)` that estimates $f'(x)$ using the central-difference formula:

$$f'(x) \approx \frac{f(x + h) - f(x - h)}{2h}$$

Verify your implementation against the known analytical derivative of $f(x) = x^3 - 4x$ at $x = 2.0$.
Expected: the numerical result is within `1e-6` of the analytical value `8.0`.

### Task 2 — Analytical Gradient of MSE Loss

Implement `mse_loss(y_pred, y_true)` and `mse_gradient(weight, bias, X, y)` that return the **exact** partial derivatives of the mean-squared-error loss with respect to a single `weight` and `bias`:

$$L = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2, \quad \hat{y}_i = w \cdot x_i + b$$

$$\frac{\partial L}{\partial w} = \frac{2}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i) \cdot x_i \qquad \frac{\partial L}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)$$

Print both partial derivatives for the provided dataset at initial `weight=0.0, bias=0.0`.

### Task 3 — Gradient Descent Loop

Implement `gradient_descent(X, y, learning_rate, epochs)` that runs the full update loop:

$$w \leftarrow w - \alpha \cdot \frac{\partial L}{\partial w}, \qquad b \leftarrow b - \alpha \cdot \frac{\partial L}{\partial b}$$

where $\alpha$ is the learning rate. Return the final `weight`, `bias`, and the list of `loss` values per epoch.
Run for **500 epochs** with `learning_rate=0.01`. Print final weight, bias, and loss.

### Task 4 — Gradient Check

Implement `gradient_check(weight, bias, X, y, h)` that verifies your analytical gradient (Task 2) against the numerical gradient (Task 1) for both `weight` and `bias`.

The **relative error** between analytical gradient $g_a$ and numerical gradient $g_n$ must satisfy:

$$\text{relative error} = \frac{|g_a - g_n|}{|g_a| + |g_n| + \varepsilon} < 10^{-5}$$

Print `PASS` or `FAIL` with the relative error for each parameter.

### Task 5 — Loss Curve Visualization

Using the loss history from Task 3, plot loss vs. epoch with `matplotlib`. Save the figure as `loss_curve.png` in the same directory. The plot must have:
- x-axis label: `"Epoch"`
- y-axis label: `"MSE Loss"`
- title: `"Gradient Descent Convergence"`

### Task 6 — Learning Rate Sensitivity

Call `gradient_descent` three times with `learning_rates = [0.001, 0.01, 0.1]`, all for 500 epochs. Plot all three loss curves on a single figure, one line per learning rate. Save as `lr_sensitivity.png`.

Observe: which learning rate converges fastest? Which diverges or oscillates?

## Expected Output

```
=== Task 1: Numerical Derivative ===
Analytical f'(2.0) = 8.000000
Numerical  f'(2.0) = 8.000000 (h=1e-05)
Absolute error     = 0.000000 (< 1e-06: PASS)

=== Task 2: Analytical MSE Gradient ===
Initial weight=0.0, bias=0.0
  dL/dw = -32.6800  (approx, depends on dataset)
  dL/db = -5.4600   (approx, depends on dataset)

=== Task 3: Gradient Descent ===
Epoch   0 | Loss: 89.1236
Epoch 100 | Loss: 12.3451
Epoch 200 | Loss:  4.5678
Epoch 300 | Loss:  2.3456
Epoch 400 | Loss:  1.8765
Epoch 499 | Loss:  1.7834
Final weight = 2.9500 (expected ~3.0)
Final bias   = 0.4800 (expected ~0.5)

=== Task 4: Gradient Check ===
weight: analytical=-32.6800, numerical=-32.6800, relative_error=0.0000000 → PASS
bias  : analytical= -5.4600, numerical= -5.4600, relative_error=0.0000000 → PASS

=== Task 5: Loss Curve ===
Saved: loss_curve.png

=== Task 6: LR Sensitivity ===
lr=0.001 | final loss: ~8.xx
lr=0.010 | final loss: ~1.xx  ← converges
lr=0.100 | final loss: may oscillate or diverge
Saved: lr_sensitivity.png
```

> **Tolerance:** Final weight within ±0.15 of 3.0 and final bias within ±0.15 of 0.5 for `learning_rate=0.01`. Gradient check relative errors must both be `< 1e-5`.

## Evaluation Criteria

- [ ] `numerical_derivative` returns a value within `1e-6` of the analytical derivative for $f(x) = x^3 - 4x$ at $x = 2.0$
- [ ] `mse_gradient` returns correct `dL/dw` and `dL/db` (verified by gradient check PASS for both)
- [ ] Gradient check prints `PASS` for both `weight` and `bias` (relative error `< 1e-5`)
- [ ] Gradient descent converges: final loss is lower than initial loss and final weight is within ±0.15 of 3.0
- [ ] `loss_curve.png` is written to disk with correct axis labels and title
- [ ] `lr_sensitivity.png` shows all three learning rates on one figure

## Extension Challenge

Extend gradient descent to a **multi-feature** linear model with $d = 5$ input features and a weight vector $\mathbf{w} \in \mathbb{R}^5$. Replace the scalar update with the vectorized form:

$$\mathbf{w} \leftarrow \mathbf{w} - \frac{\alpha}{n} \cdot \mathbf{X}^T (\mathbf{X}\mathbf{w} + b - \mathbf{y})$$

Generate a synthetic dataset (`n=200` samples, true weights `[1, -2, 3, 0.5, -1]`, `bias=2.0`, Gaussian noise `σ=0.5`). Run for 1000 epochs and confirm your recovered weights are within ±0.1 of the true values. No starter code is provided — implement from scratch.

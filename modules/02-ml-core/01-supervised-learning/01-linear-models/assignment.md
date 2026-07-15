# Assignment 2.1.1 — Build Regressor from Scratch, Evaluate Metrics

## Objective

Build linear regression and logistic regression entirely from NumPy primitives, then benchmark
your implementations against scikit-learn to confirm you understand exactly what the library wraps.

## Background

The lesson established MSE and Cross-Entropy as the two foundational loss functions — one for
continuous targets, one for probability outputs. It derived the closed-form OLS solution for
linear regression and the batch gradient descent update rule for logistic regression. This
assignment removes the library crutch: you implement each mathematical primitive in raw NumPy,
then use scikit-learn as a correctness oracle. Any significant divergence from the sklearn result
means your math or vectorization has a bug.

See `lesson.md` for all formulas, symbol definitions, and numeric worked examples.

## Setup

```bash
pip install "numpy>=1.24" "scikit-learn>=1.3" "matplotlib>=3.7"
```

## Tasks

1. **Implement `sigmoid(z)`** — apply the formula $\sigma(z) = \frac{1}{1+e^{-z}}$ element-wise
   over a NumPy array. Confirm: `sigmoid(np.array([0.0]))` returns `[0.5000]` and
   `sigmoid(np.array([0.8]))` returns `≈[0.6900]`.

2. **Implement `mse_loss(y_true, y_pred)`** — compute mean squared error. Confirm: passing
   `y_true=[3.0, 5.0]` and `y_pred=[2.5, 4.5]` returns exactly `0.2500` (matches the Theory
   worked example in the lesson).

3. **Implement `cross_entropy_loss(y_true, y_prob, eps=1e-12)`** — compute binary cross-entropy,
   clipping all probabilities into `[eps, 1-eps]` before taking the log. Confirm: passing
   `y_true=[1.0]` and `y_prob=[0.6900]` returns a value in the range `[0.370, 0.372]`.

4. **Implement `fit_linear_ols(X, y)`** — solve the normal equation
   $\mathbf{w}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ using `np.linalg.pinv`
   for numerical stability. Use `add_bias_column` before solving; return a single weight vector
   whose first element is the bias. The resulting test MSE must fall within `1.0` of
   scikit-learn's `LinearRegression` on the same train/test split.

5. **Implement `fit_logistic_gd(X, y, learning_rate, epochs)`** — run batch gradient descent
   using $\nabla_\mathbf{w} L_{\text{CE}} = \frac{1}{n}\mathbf{X}^T(\hat{\mathbf{p}} - \mathbf{y})$.
   Print the cross-entropy loss every 50 epochs using the format
   `Epoch {epoch:>3d} | Cross-Entropy Loss: {loss:.4f}`. Loss must strictly decrease across all
   printed checkpoints, and the final model must reach test accuracy ≥ `0.85`.

6. **Evaluate and compare** — `main()` calls both models and prints the comparison table shown
   below. Run the full script end-to-end and verify: (a) OLS MSE delta vs sklearn < `1.0`, and
   (b) logistic regression accuracy gap vs sklearn ≤ `0.05`.

## Expected Output

```
=== SANITY CHECKS ===
sigmoid(0.0)             : [0.5000]
sigmoid(0.8)             : [0.6900]
MSE worked example       : 0.2500
Cross-Entropy worked ex  : 0.3711

=== LINEAR REGRESSION ===
Dataset        : n=300, 3 features, noise=20
Scratch OLS    | Test MSE : <value>
sklearn        | Test MSE : <value>
Delta                     : <value>  (must be < 1.0)

=== LOGISTIC REGRESSION — GD Training ===
  Epoch   0 | Cross-Entropy Loss: <decreasing>
  Epoch  50 | Cross-Entropy Loss: <decreasing>
  Epoch 100 | Cross-Entropy Loss: <decreasing>
  Epoch 150 | Cross-Entropy Loss: <decreasing>

=== CLASSIFICATION RESULTS ===
              Accuracy   Log-Loss
Scratch LR  : 0.8X00    0.XXXX
sklearn      : 0.8X00    0.XXXX
Gap (acc)             : <value>  (must be ≤ 0.05)
```

Concrete bounds: scratch OLS MSE delta < `1.0`; scratch logistic accuracy ≥ `0.85`; accuracy gap
vs sklearn ≤ `0.05`; cross-entropy loss strictly decreasing across all four printed checkpoints.

## Evaluation Criteria

- [ ] `sigmoid(np.array([0.0]))` returns `array([0.5])` exactly
- [ ] `mse_loss(np.array([3.0, 5.0]), np.array([2.5, 4.5]))` returns `0.25`
- [ ] `cross_entropy_loss(np.array([1.0]), np.array([0.6900]))` returns a value in `[0.370, 0.372]`
- [ ] Scratch OLS test MSE is within `1.0` of scikit-learn's test MSE on the same split
- [ ] Cross-entropy loss decreases at every logged checkpoint (Epoch 0 → 50 → 100 → 150)
- [ ] Scratch logistic regression test accuracy ≥ `0.85`
- [ ] Accuracy gap between scratch and sklearn ≤ `0.05`
- [ ] Script runs end-to-end without errors via `python starter.py`

## Extension Challenge

Replace the constant `learning_rate` in `fit_logistic_gd` with **step-decay scheduling**: halve
the learning rate every 50 epochs. Then add **L2 regularisation** by appending the term
$\frac{\lambda}{n}\mathbf{w}$ (exclude the bias index `0`) to the gradient before each update.
Grid-search `learning_rate ∈ {0.05, 0.1, 0.2}` × `lambda ∈ {0.0, 0.01, 0.1}` using 5-fold
cross-validation on the training set. Report the best hyperparameter pair and plot validation
accuracy vs `lambda` (one curve per `learning_rate`) using matplotlib. No starter code is
provided for this extension.

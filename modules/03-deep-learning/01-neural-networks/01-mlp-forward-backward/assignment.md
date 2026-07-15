# Assignment 3.1.1 — Build Basic PyTorch MLP

## Objective

Build a Multi-Layer Perceptron from first principles in PyTorch — implementing the model
architecture, mini-batch training loop, and accuracy evaluation — then prove the network
learns a non-linear decision boundary that a logistic regression cannot express.

## Background

The lesson showed how stacking linear layers with ReLU activations lets a network approximate
any continuous function, and how backpropagation propagates the Binary Cross-Entropy loss
gradient through every weight via the chain rule. PyTorch's `autograd` engine performs the
backward pass automatically once you call `loss.backward()` — your job is to wire the correct
forward architecture, the training loop that calls it, and the evaluation that confirms it
worked. See `lesson.md` — *Theory → Forward Pass* and *Backward Pass* sections.

## Setup

```bash
pip install torch>=2.1 numpy>=1.24 matplotlib>=3.7 scikit-learn>=1.3
```

All four packages are required. `scikit-learn` is used only for generating the moons dataset;
no ML models from sklearn are used.

## Tasks

### Task 1 — Define the MLP architecture

Implement `build_mlp(input_dim, hidden_dim, output_dim)` in `starter/starter.py`.

Return an `nn.Sequential` with **exactly** this architecture:

```
Linear(input_dim → hidden_dim) → ReLU
Linear(hidden_dim → hidden_dim) → ReLU
Linear(hidden_dim → output_dim) → Sigmoid
```

Run `python starter/starter.py` after this task — you should see the model architecture
printed without any error.

### Task 2 — Implement one epoch of mini-batch training

Implement `train_one_epoch(model, optimizer, criterion, X, y, batch_size)`.

For each batch of `batch_size` samples:

1. Zero gradients: `optimizer.zero_grad()`
2. Forward pass: `predictions = model(X_batch)`
3. Compute BCE loss: `loss = criterion(predictions, y_batch)`
4. Backward pass: `loss.backward()`
5. Update weights: `optimizer.step()`

Return the **mean loss** across all batches in the epoch as a `float`.

Use `torch.utils.data.TensorDataset` and `DataLoader` to iterate over batches — both are
already imported in the starter file.

### Task 3 — Full training loop

Implement `run_training(model, optimizer, criterion, X_train, y_train, X_val, y_val, epochs)`.

For each epoch:

1. Set `model.train()`, call `train_one_epoch` → record the returned train loss.
2. Set `model.eval()`, compute validation loss inside `torch.no_grad()` — single forward
   pass on the full `X_val`, no batching needed.
3. Print: `Epoch {e}/{epochs} | Train Loss: {train:.4f} | Val Loss: {val:.4f}`

Return `(train_losses, val_losses)` — one float per epoch each.

### Task 4 — Evaluate test accuracy

Implement `evaluate_accuracy(model, X, y)`.

- Use `torch.no_grad()` to disable gradient tracking.
- Classify as **class 1** when `model(X) >= 0.5`, else **class 0**.
- Return accuracy as a `float` in `[0, 1]`.

**Acceptance threshold:** accuracy must be ≥ 0.90 on the held-out test set.

### Task 5 — Verify outputs

Run the completed script end-to-end:

```bash
python starter/starter.py
```

Confirm:
- Per-epoch loss lines print for all 50 epochs.
- `loss_curve.png` is saved and shows a monotonically decreasing trend.
- `decision_boundary.png` is saved and shows a clear curved boundary separating the two moons.

## Expected Output

```
=== 3.1.1 — Build basic PyTorch MLP ===

Train samples: 800 | Test samples: 200

Model architecture:
Sequential(
  (0): Linear(in_features=2, out_features=64, bias=True)
  (1): ReLU()
  (2): Linear(in_features=64, out_features=64, bias=True)
  (3): ReLU()
  (4): Linear(in_features=64, out_features=1, bias=True)
  (5): Sigmoid()
)

Epoch 1/50  | Train Loss: 0.6820 | Val Loss: 0.6754
Epoch 5/50  | Train Loss: 0.5103 | Val Loss: 0.5021
...
Epoch 50/50 | Train Loss: 0.0743 | Val Loss: 0.0891

Final Test Accuracy: 0.9700   ← must be ≥ 0.90
Final Train Loss:    0.0743   ← must be < 0.15
Final Val Loss:      0.0891   ← must be < 0.20

Loss curve saved → loss_curve.png
Decision boundary saved → decision_boundary.png
```

Exact loss values will vary. Accuracy must land in **[0.90, 1.00]**.

## Evaluation Criteria

- [ ] `build_mlp()` returns an `nn.Sequential` with the exact 6-layer structure shown (2× Linear+ReLU, 1× Linear+Sigmoid)
- [ ] `train_one_epoch()` calls `zero_grad()`, `loss.backward()`, and `optimizer.step()` in the correct order
- [ ] `run_training()` prints one log line per epoch and returns two equal-length lists
- [ ] `evaluate_accuracy()` uses `torch.no_grad()` and threshold at 0.5
- [ ] Final test accuracy is ≥ 0.90
- [ ] Final train loss is < 0.15 after 50 epochs
- [ ] `loss_curve.png` and `decision_boundary.png` are present in the working directory

## Extension Challenge

**Swap Adam for a hand-rolled SGD + cosine annealing schedule and match Adam's accuracy.**

Requirements (no starter code provided):

1. Replace `torch.optim.Adam` with `torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)`.
2. Add `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)` and call
   `scheduler.step()` after each epoch.
3. Run for 100 epochs (double the base task).
4. Achieve final test accuracy within **2 percentage points** of your Adam run.
5. Plot both runs' val loss curves on the same axes (label each line) and save as
   `optimizer_comparison.png`.

The challenge requires understanding *why* cosine annealing recovers accuracy that vanilla SGD
loses — connect it back to the lesson's gradient landscape discussion.

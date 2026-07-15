# Assignment 3.2.1 — Code Isolated Attention Block

## Objective

Build a self-attention block from raw tensor operations, validate every mathematical
guarantee the lesson proved (weight normalization, causal masking, shape invariance),
and confirm multi-head output can be wired into a Transformer stack without shape changes.

## Background

The lesson showed that scaled dot-product attention computes
$\text{Attention}(Q,K,V) = \text{softmax}(QK^{\top}/\sqrt{d_k})V$, where dividing by
$\sqrt{d_k}$ prevents softmax saturation as embedding size grows. Multi-head attention
runs $h$ independent attention sub-functions in parallel — one per head — and concatenates
their outputs, letting each head specialise on a different relationship type. Review
`lesson.md` sections **Theory → Numeric Worked Example** and
**Python Implementation → `MultiHeadAttention`** before starting.

## Setup

```bash
pip install torch>=2.1 numpy>=1.24
```

No GPU required — every task runs on CPU in under 30 seconds.

## Tasks

1. **Implement `scaled_dot_product_attention`** in `starter/starter.py`.
   Use raw PyTorch tensor operations: `torch.matmul`, `torch.softmax`, and `masked_fill`
   for the optional mask. Do not call `torch.nn.functional.scaled_dot_product_attention`.
   The function must return a tuple `(output, attention_weights)`.

2. **Validate the lesson's 3-token numeric example** inside `validate_numeric_example`.
   Using the hand-computed Q, K, V matrices from the lesson, call your implementation and
   assert that attention weights for token 0 are within `±0.005` of `[0.401, 0.198, 0.401]`.
   Print `PASS` or `FAIL` with the actual values.

3. **Implement `make_causal_mask(seq_len)`** that returns a lower-triangular
   `torch.Tensor` of shape `(seq_len, seq_len)` with `1.0` where `j <= i` and `0.0`
   where `j > i`. Token `i` may attend to token `j` only when `j <= i`.

4. **Implement `MultiHeadSelfAttention.forward`** following the five-step comment in the
   stub: project → split heads → attend → merge heads → output projection.
   The function must call your `scaled_dot_product_attention` (not any built-in equivalent).

5. **Run `check_shape_invariance`**: pass inputs of shape `(1, 6, 64)`, `(2, 10, 128)`,
   and `(4, 1, 64)` through your `MultiHeadSelfAttention`. Output shape must exactly equal
   input shape for every case. Print `PASS` or `FAIL` per test case.

6. **Run `check_causal_constraint`**: pass a `(1, 8, 64)` input through
   `MultiHeadSelfAttention` with the causal mask from Task 3. For every head `h` and query
   position `p`, assert `attention_weights[:, h, p, p+1:]` are all `0.0` (to within
   `1e-6`). Print `PASS` or `FAIL` plus the count of any violations.

## Expected Output

```
============================================================
Node 3.2.1 — Self-Attention Mechanisms Assignment
============================================================

--- Task 2: Numeric Worked Example Validation ---
Token 0 attention weights: [0.4009, 0.1982, 0.4009]
Expected:                   [0.401,  0.198,  0.401 ]
PASS — weights match within tolerance 0.005

--- Task 5: Shape Invariance Check ---
Input (1, 6, 64)   → Output (1, 6, 64)   PASS
Input (2, 10, 128) → Output (2, 10, 128) PASS
Input (4, 1, 64)   → Output (4, 1, 64)   PASS

--- Task 6: Causal Constraint Check ---
Sequence length: 8 | Heads: 4
Future-position violations: 0
PASS — all future attention weights are 0.0

All tasks complete.
```

Numeric values in Task 2 may vary by `±0.005`. Task 5 shapes must be exact.
Task 6 must report exactly 0 violations.

## Evaluation Criteria

- [ ] `scaled_dot_product_attention` uses only raw tensor ops — no `F.scaled_dot_product_attention` or `F.softmax` wrapped black-box calls
- [ ] Task 2 prints `PASS` — numeric weights match the lesson's hand-computed values within `0.005`
- [ ] `make_causal_mask` returns a lower-triangular float tensor; `mask[i, j] == 0.0` for all `j > i`
- [ ] `MultiHeadSelfAttention.forward` calls the student's own `scaled_dot_product_attention`, not PyTorch's built-in
- [ ] Task 5 prints `PASS` for all three shape test cases
- [ ] Task 6 prints `PASS` with 0 violations — causal masking is correctly applied across all heads

## Extension Challenge

Implement **relative position bias** on top of your attention block.

Instead of adding a separate positional encoding to embeddings before attention, add a
learned bias matrix $B \in \mathbb{R}^{n \times n}$ directly to the raw attention scores
before softmax:

$$\text{scores\_biased} = \frac{QK^{\top}}{\sqrt{d_k}} + B$$

Create a `RelativeBiasAttention` class that:
1. Stores a learnable `nn.Parameter` of shape `(max_seq_len, max_seq_len)` as the bias
2. Slices `B[:seq_len, :seq_len]` at forward time (handles variable-length inputs)
3. Adds `B` to scores before softmax — no other change to the attention formula
4. Verifies that after one gradient step the bias values change (i.e., gradients flow through)

No starter code provided. Compare the attention weight distributions of plain attention vs.
relative-bias attention on the same input to show they differ — proving the bias is active.

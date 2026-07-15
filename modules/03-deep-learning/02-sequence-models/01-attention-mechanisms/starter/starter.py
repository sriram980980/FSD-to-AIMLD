# Dependencies: torch>=2.1, numpy>=1.24
# Node: 3.2.1 — Attention Mechanisms: Self-Attention
# Run: python starter.py

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Utility helpers — fully implemented, do NOT modify
# ---------------------------------------------------------------------------


def softmax_rows(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable softmax over the last dimension (manual implementation)."""
    x_max = x.max(dim=-1, keepdim=True).values
    shifted = x - x_max
    exp_x = torch.exp(shifted)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Reshape (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k)."""
    batch, seq_len, d_model = x.size()
    d_k = d_model // num_heads
    x = x.view(batch, seq_len, num_heads, d_k)
    return x.transpose(1, 2)  # → (batch, num_heads, seq_len, d_k)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """Reshape (batch, num_heads, seq_len, d_k) → (batch, seq_len, d_model)."""
    batch, num_heads, seq_len, d_k = x.size()
    return x.transpose(1, 2).contiguous().view(batch, seq_len, num_heads * d_k)


# ---------------------------------------------------------------------------
# Task 1 — Implement scaled dot-product attention
# ---------------------------------------------------------------------------


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V.

    Args:
        query: shape (..., seq_q, d_k)
        key:   shape (..., seq_k, d_k)
        value: shape (..., seq_k, d_v)
        mask:  optional float tensor shape (..., seq_q, seq_k).
               Positions where mask == 0 receive score -inf before softmax,
               driving their attention weight to 0.

    Returns:
        output:            shape (..., seq_q, d_v)
        attention_weights: shape (..., seq_q, seq_k) — rows sum to 1.0

    Implementation steps:
        1. Read d_k from query.size(-1).
        2. scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        3. If mask is not None: scores = scores.masked_fill(mask == 0, float("-inf"))
        4. attention_weights = torch.softmax(scores, dim=-1)
        5. output = torch.matmul(attention_weights, value)
        6. return output, attention_weights
    """
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Task 2 — Validate the lesson's 3-token numeric example
# ---------------------------------------------------------------------------


def validate_numeric_example() -> None:
    """
    Re-create the 3-token, d_k=2 hand-computed example from the lesson.

    Q = [[1,0],[0,1],[1,1]]   K = [[1,0],[0,1],[1,1]]   V = [[0.5,0],[0,0.5],[1,1]]

    Expected attention weights for token 0: approximately [0.401, 0.198, 0.401].
    Tolerance: ±0.005 per element.

    Print PASS or FAIL with both actual and expected values.
    """
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Task 3 — Implement a causal (autoregressive) mask
# ---------------------------------------------------------------------------


def make_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Build a lower-triangular mask of shape (seq_len, seq_len).

    mask[i, j] = 1.0  if j <= i  (token i may attend to token j)
    mask[i, j] = 0.0  if j >  i  (token i must NOT attend to future token j)

    Hint: torch.tril(torch.ones(seq_len, seq_len)) produces this directly.

    Returns:
        torch.Tensor of dtype torch.float32, shape (seq_len, seq_len)
    """
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Task 4 — Implement MultiHeadSelfAttention.forward
# ---------------------------------------------------------------------------


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention block.

    Architecture:
        x → W_q / W_k / W_v → split_heads → scaled_dot_product_attention
          → merge_heads → W_o → output
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Learned projection matrices — no bias (standard in the original paper)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Self-attention forward pass (Q, K, V all projected from the same input x).

        Args:
            x:    shape (batch, seq_len, d_model) — token embeddings
            mask: optional causal mask shape (seq_len, seq_len).
                  Pass the output of make_causal_mask(seq_len) directly.

        Returns:
            output:            (batch, seq_len, d_model)
            attention_weights: (batch, num_heads, seq_len, seq_len)

        Implementation steps:
            Step 1 — Project:
                Q = self.W_q(x)   shape (batch, seq_len, d_model)
                K = self.W_k(x)   shape (batch, seq_len, d_model)
                V = self.W_v(x)   shape (batch, seq_len, d_model)

            Step 2 — Split heads:
                Q = split_heads(Q, self.num_heads)  → (batch, num_heads, seq_len, d_k)
                K = split_heads(K, self.num_heads)
                V = split_heads(V, self.num_heads)

            Step 3 — Attend (per head in parallel):
                attended, attention_weights = scaled_dot_product_attention(Q, K, V, mask)
                attended shape: (batch, num_heads, seq_len, d_k)
                weights  shape: (batch, num_heads, seq_len, seq_len)

            Step 4 — Merge heads:
                attended = merge_heads(attended)  → (batch, seq_len, d_model)

            Step 5 — Output projection:
                output = self.W_o(attended)        → (batch, seq_len, d_model)

            Return output, attention_weights
        """
        raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Task 5 — Shape invariance check
# ---------------------------------------------------------------------------


def check_shape_invariance(model: MultiHeadSelfAttention) -> None:
    """
    Verify that self-attention preserves the input tensor's shape exactly.

    For each test case below, run model.forward(x) and compare
    output.shape == x.shape. Print PASS or FAIL per case.

    Test cases:
        x shape (1,  6,  64)   — single item, short sequence
        x shape (2, 10, 128)   — batch of 2, longer sequence, larger d_model
        x shape (4,  1,  64)   — batch of 4, single-token sequence
    """
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Task 6 — Causal constraint check
# ---------------------------------------------------------------------------


def check_causal_constraint(model: MultiHeadSelfAttention) -> None:
    """
    Confirm that a causal mask correctly zeroes all future attention weights.

    Setup:
        batch=1, seq_len=8, d_model=64 (model must have d_model=64)
        mask = make_causal_mask(seq_len=8)

    Verification:
        For every head h in range(model.num_heads):
            For every query position p in range(8):
                attention_weights[0, h, p, p+1:]  must all be < 1e-6

    Print:
        "Sequence length: 8 | Heads: <num_heads>"
        "Future-position violations: <count>"
        "PASS — all future attention weights are 0.0"   (if count == 0)
        "FAIL — <count> future weights are non-zero"    (if count > 0)
    """
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Main — orchestrates all tasks in order
# ---------------------------------------------------------------------------


def main() -> None:
    torch.manual_seed(42)
    print("=" * 60)
    print("Node 3.2.1 — Self-Attention Mechanisms Assignment")
    print("=" * 60)
    print()

    # ── Task 2: Numeric validation (depends on Task 1) ──
    print("--- Task 2: Numeric Worked Example Validation ---")
    validate_numeric_example()
    print()

    # ── Tasks 4-6: Multi-head attention ──
    d_model = 64
    num_heads = 4
    model = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
    model.eval()

    print("--- Task 5: Shape Invariance Check ---")
    check_shape_invariance(model)
    print()

    print("--- Task 6: Causal Constraint Check ---")
    check_causal_constraint(model)
    print()

    print("All tasks complete.")


if __name__ == "__main__":
    main()

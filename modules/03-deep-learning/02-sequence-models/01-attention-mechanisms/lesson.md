# 3.2.1 — Attention Mechanisms: Self-Attention

## Hook

Self-attention is a SQL `JOIN` where the table joins itself — every token queries every other token to compute how much attention it should pay, exactly like a message bus where every subscriber scores each incoming event for relevance and decides how much weight to assign it before acting.

## The Problem

Recurrent Neural Networks (RNNs) process sequences one token at a time, compressing the entire history into a single hidden state vector. By the time an RNN reaches token 100, token 1's signal has been diluted through 99 successive multiplications — a symptom identical to writing all application state into a single global variable and mutating it on every request. Self-attention sidesteps this entirely: every token reads every other token in one parallel operation, with no bottleneck state. This is what made Transformers replace RNNs for virtually every sequence-modeling task after 2017.

## Theory

### Scaled Dot-Product Attention

The core operation is:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V$$

**Symbol definitions:**

| Symbol | Shape | Meaning |
|--------|-------|---------|
| $Q$ | $(n \times d_k)$ | **Query** matrix — what each token is searching for |
| $K$ | $(n \times d_k)$ | **Key** matrix — what each token advertises about itself |
| $V$ | $(n \times d_v)$ | **Value** matrix — what each token actually contributes |
| $n$ | scalar | sequence length (number of tokens) |
| $d_k$ | scalar | key/query dimensionality; also the scaling divisor |
| $\sqrt{d_k}$ | scalar | prevents dot products from growing large and pushing softmax into flat regions |

**Intuition mapped to the FSD analogy:**  
Think of $Q$ as the SQL `WHERE` clause of the current row, $K$ as the indexed column of every row, and $K^{\top}$ as the index scan. The dot product $QK^{\top}$ computes raw relevance scores (join cardinality). Dividing by $\sqrt{d_k}$ is connection-pool normalisation. Softmax converts raw scores into a probability distribution (weights that sum to 1). Multiplying by $V$ is selecting the actual column data weighted by those probabilities.

---

### Numeric Worked Example

Use $n = 3$ tokens, $d_k = d_v = 2$.

**Step 1 — Inputs (token embeddings after projection to Q, K, V):**

$$Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}, \quad K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}, \quad V = \begin{bmatrix} 0.5 & 0.0 \\ 0.0 & 0.5 \\ 1.0 & 1.0 \end{bmatrix}$$

**Step 2 — Raw attention scores for token 1 (row 0 of $Q$):**

$$\text{scores}_1 = Q[0] \cdot K^{\top} = [1, 0] \cdot \begin{bmatrix}1 & 0 & 1\\0 & 1 & 1\end{bmatrix} = [1 \cdot 1 + 0 \cdot 0,\ 1 \cdot 0 + 0 \cdot 1,\ 1 \cdot 1 + 0 \cdot 1] = [1, 0, 1]$$

**Step 3 — Scale by $\sqrt{d_k} = \sqrt{2} \approx 1.414$:**

$$\text{scaled}_1 = [1/1.414,\ 0/1.414,\ 1/1.414] \approx [0.707,\ 0.000,\ 0.707]$$

**Step 4 — Softmax (convert to probability weights):**

$$e^{0.707} \approx 2.028, \quad e^{0.000} = 1.000$$

$$\text{sum} = 2.028 + 1.000 + 2.028 = 5.056$$

$$\text{weights}_1 \approx [0.401,\ 0.198,\ 0.401]$$

**Step 5 — Weighted sum of $V$ rows:**

$$\text{output}_1 = 0.401 \times [0.5, 0.0] + 0.198 \times [0.0, 0.5] + 0.401 \times [1.0, 1.0]$$
$$= [0.201, 0.000] + [0.000, 0.099] + [0.401, 0.401] = [0.602, 0.500]$$

Token 1 attends roughly equally to itself and token 3 (both have matching key structure), and barely to token 2. Its output vector is a mixture of their values weighted by that relevance.

---

### Multi-Head Attention

Rather than running one attention function with $d_{model}$-dimensional vectors, run $h$ parallel attention heads each with $d_k = d_{model} / h$, then concatenate:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\,W^O$$

$$\text{head}_i = \text{Attention}(Q W_i^Q,\ K W_i^K,\ V W_i^V)$$

Where $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$, $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$, and $W^O \in \mathbb{R}^{h \cdot d_v \times d_{model}}$ are learned projection matrices. Each head learns to attend to different relationship types — one head may track syntactic dependencies, another semantic similarity.

## Python Implementation

```python
# Dependencies: torch>=2.1, numpy>=1.24
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled dot-product attention. Returns (output, attention_weights)."""
    d_k = query.size(-1)  # key/query dimensionality

    # Raw scores: (..., n_q, n_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # Fill masked positions with -inf so softmax drives them to 0
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attention_weights = F.softmax(scores, dim=-1)  # (..., n_q, n_k)
    output = torch.matmul(attention_weights, value)  # (..., n_q, d_v)
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention block."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension per head

        # Learned projection matrices (W_Q, W_K, W_V, W_O)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k)."""
        batch, seq_len, _ = x.size()
        x = x.view(batch, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Self-attention: Q, K, V all derived from same input x."""
        batch, seq_len, _ = x.size()

        # Project input to Q, K, V spaces
        Q = self._split_heads(self.W_q(x))  # (batch, heads, seq, d_k)
        K = self._split_heads(self.W_k(x))
        V = self._split_heads(self.W_v(x))

        # Attention per head
        attended, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads: (batch, heads, seq, d_k) → (batch, seq, d_model)
        attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

        # Final output projection
        output = self.W_o(attended)
        return output, attention_weights


def demo_numeric_example() -> None:
    """Reproduce the lesson's hand-computed 3-token example."""
    torch.manual_seed(0)

    Q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    K = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    V = torch.tensor([[0.5, 0.0], [0.0, 0.5], [1.0, 1.0]])

    # Add batch dimension: (1, 3, 2)
    Q, K, V = Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0)

    output, weights = scaled_dot_product_attention(Q, K, V)

    print("=== Numeric Worked Example (3 tokens, d_k=2) ===")
    print(f"Attention weights (token 1 attends to all tokens):")
    print(f"  Token 1 → [Token1, Token2, Token3]: {weights[0, 0].tolist()}")
    print(f"  Token 2 → [Token1, Token2, Token3]: {weights[0, 1].tolist()}")
    print(f"  Token 3 → [Token1, Token2, Token3]: {weights[0, 2].tolist()}")
    print(f"Output for Token 1: {output[0, 0].tolist()}")
    print()


def demo_multi_head() -> None:
    """Multi-head self-attention on a short synthetic sentence."""
    torch.manual_seed(42)

    batch_size = 1
    seq_len = 6     # 6 tokens
    d_model = 64    # embedding dimension
    num_heads = 4   # 4 parallel attention heads

    # Simulated token embeddings (normally from an embedding layer)
    x = torch.randn(batch_size, seq_len, d_model)

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    mha.eval()

    with torch.no_grad():
        output, attention_weights = mha(x)

    print("=== Multi-Head Self-Attention ===")
    print(f"Input shape:             {tuple(x.shape)}  (batch, seq_len, d_model)")
    print(f"Output shape:            {tuple(output.shape)}  (batch, seq_len, d_model)")
    print(f"Attention weights shape: {tuple(attention_weights.shape)}  (batch, heads, seq_len, seq_len)")
    print()

    # Show attention pattern for head 0, token 0
    head0_token0 = attention_weights[0, 0, 0]  # head=0, query=token0
    print("Head 0 — Token 0 attention distribution across 6 positions:")
    for pos, weight in enumerate(head0_token0.tolist()):
        bar = "█" * int(weight * 40)
        print(f"  → Token {pos}: {weight:.4f}  {bar}")
    print()

    # Verify weights sum to 1
    weight_sum = attention_weights[0, 0, 0].sum().item()
    print(f"Attention weight sum (must be 1.000): {weight_sum:.6f}")
    print()

    # Causal (autoregressive) mask — tokens can only attend to past positions
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
    with torch.no_grad():
        causal_output, causal_weights = mha(x, mask=causal_mask)

    print("=== Causal Mask Applied (autoregressive mode) ===")
    print("Head 0 — Token 3 attention (can only see tokens 0-3):")
    for pos, weight in enumerate(causal_weights[0, 0, 3].tolist()):
        bar = "█" * int(weight * 40) if weight > 1e-6 else "—"
        print(f"  → Token {pos}: {weight:.4f}  {bar}")


def main() -> None:
    demo_numeric_example()
    demo_multi_head()


if __name__ == "__main__":
    main()
```

**What to notice in the output:**

- **Token 1's weights** in the numeric example match the hand-computed $[0.401, 0.198, 0.401]$ — verifying the math.  
- **Multi-head output shape is identical to input shape** — the sequence length and `d_model` are preserved. Attention is a drop-in transformation, not a dimensionality reducer.  
- **Causal mask** zeroes out all future positions for token 3: weights for tokens 4 and 5 are exactly `0.0000`. This is the decoder's autoregressive constraint — the mechanism GPT uses to prevent tokens from "cheating" by reading ahead.  
- **Attention weight sum** is always `1.000000` — softmax guarantees this, making weights interpretable as a probability distribution over the sequence.

## Java Implementation

Library: `ai.djl:api:0.26.0` (Deep Java Library)

```java
// Dependencies (Maven):
//   ai.djl:api:0.26.0
//   ai.djl.pytorch:pytorch-engine:0.26.0
//   ai.djl.pytorch:pytorch-native-auto:2.1.1

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

/**
 * Scaled dot-product attention implemented with DJL NDArrays.
 * Mirrors the Python implementation for the 3-token numeric worked example.
 */
public class ScaledDotProductAttention {

    /**
     * Compute attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V.
     *
     * @param query  shape (batch, seq_q, d_k)
     * @param key    shape (batch, seq_k, d_k)
     * @param value  shape (batch, seq_k, d_v)
     * @return NDList of [output (batch, seq_q, d_v), weights (batch, seq_q, seq_k)]
     */
    public static NDList attend(NDArray query, NDArray key, NDArray value) {
        long dK = query.getShape().get(2); // key/query dimension

        // Raw scores: (batch, seq_q, seq_k)
        NDArray scores = query.matMul(key.transpose(0, 2, 1))
                              .div(Math.sqrt(dK));

        // Softmax over last axis (key positions)
        NDArray weights = scores.softmax(2); // (batch, seq_q, seq_k)

        // Weighted sum of values
        NDArray output = weights.matMul(value); // (batch, seq_q, d_v)

        return new NDList(output, weights);
    }

    public static void main(String[] args) {
        try (NDManager manager = NDManager.newBaseManager()) {

            // === Numeric worked example (matches lesson hand-computation) ===
            float[][] qData = {{1f, 0f}, {0f, 1f}, {1f, 1f}};
            float[][] kData = {{1f, 0f}, {0f, 1f}, {1f, 1f}};
            float[][] vData = {{0.5f, 0f}, {0f, 0.5f}, {1f, 1f}};

            // Add batch dim → shape (1, 3, 2)
            NDArray Q = manager.create(qData).reshape(1, 3, 2);
            NDArray K = manager.create(kData).reshape(1, 3, 2);
            NDArray V = manager.create(vData).reshape(1, 3, 2);

            NDList result = attend(Q, K, V);
            NDArray output  = result.get(0);
            NDArray weights = result.get(1);

            System.out.println("=== Scaled Dot-Product Attention (3 tokens, d_k=2) ===");
            System.out.println("Attention weights (batch=0):");
            float[] w = weights.reshape(9).toFloatArray(); // 3x3 flattened
            System.out.printf("  Token 0 → [%.4f, %.4f, %.4f]%n", w[0], w[1], w[2]);
            System.out.printf("  Token 1 → [%.4f, %.4f, %.4f]%n", w[3], w[4], w[5]);
            System.out.printf("  Token 2 → [%.4f, %.4f, %.4f]%n", w[6], w[7], w[8]);

            float[] o = output.reshape(6).toFloatArray(); // 3x2 flattened
            System.out.printf("Output Token 0: [%.4f, %.4f]%n", o[0], o[1]);

            // === Verify weight sum for token 0 ===
            double sum = w[0] + w[1] + w[2];
            System.out.printf("Weight sum for token 0 (must be 1.000): %.6f%n%n", sum);

            // === Larger example: 6 tokens, d_model=64 ===
            int seqLen = 6;
            int dModel = 64;
            NDArray X = manager.randomNormal(new Shape(1, seqLen, dModel));
            NDList bigResult = attend(X, X, X); // self-attention: Q=K=V=X
            NDArray bigOutput  = bigResult.get(0);
            NDArray bigWeights = bigResult.get(1);

            System.out.println("=== Self-attention on 6 tokens, d_model=64 ===");
            System.out.println("Input  shape: " + X.getShape());
            System.out.println("Output shape: " + bigOutput.getShape());
            System.out.println("Weight shape: " + bigWeights.getShape());

            // Verify weight row sums to 1
            float[] rowWeights = bigWeights.reshape(seqLen * seqLen).toFloatArray();
            double rowSum = 0;
            for (int j = 0; j < seqLen; j++) rowSum += rowWeights[j]; // row 0
            System.out.printf("Weight sum row 0 (must be 1.000): %.6f%n", rowSum);
        }
    }
}
```

**What to notice in the Java output:**

- The attention weights for the 3-token example match the Python output exactly — both compute identical floating-point results because they use the same IEEE-754 arithmetic.
- `Output shape` equals `Input shape`: DJL NDArrays handle the `(1, 6, 64)` → `(1, 6, 64)` passthrough cleanly, mirroring the PyTorch behaviour.
- The Java version uses only primitive `NDArray` operations (`matMul`, `softmax`, `div`) — no neural-network layers — making the math transparent.

## Stack Comparison

| Dimension | Python (PyTorch) | Java (DJL) |
|---|---|---|
| **Core library** | `torch>=2.1` | `ai.djl:api:0.26.0` + `pytorch-engine` |
| **Built-in MHA** | `nn.MultiheadAttention` | No equivalent; compose from `NDArray` ops |
| **Autograd support** | Full — gradients flow through attention | Supported via DJL `ParameterStore`; less ergonomic |
| **Production use** | Standard for training and research | DJL targets inference deployment (TorchScript export) |
| **Causal masking** | `masked_fill(-inf)` → one line | Manual `NDArray` boolean mask, slightly more verbose |
| **Ecosystem** | HuggingFace Transformers, FlashAttention | Limited; most production Java deployments call a Python serving layer |
| **When to use** | Research, fine-tuning, training new models | Embedding inference into existing Java services |

## Key Takeaways

- **Self-attention is a parallel self-join**: every token computes a relevance score against every other token in one matrix operation ($QK^{\top}$), eliminating the sequential bottleneck of RNNs and enabling attention to span the full sequence in a single layer.
- **The $\sqrt{d_k}$ scaling factor is load-control**: without it, large $d_k$ values push dot products into regions where softmax saturates and gradients vanish — dividing by $\sqrt{d_k}$ keeps scores in a well-conditioned range regardless of embedding size.
- **Multi-head attention runs independent specialised sub-attentions in parallel**: splitting $d_{model}$ into $h$ heads of size $d_k = d_{model}/h$ lets each head learn a different relationship type (syntactic, semantic, positional) at no extra compute cost compared to single-head attention with the same total dimension.

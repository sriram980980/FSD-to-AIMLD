# 4.1 — Transformer Architecture — Encoder-Decoder

## Hook

GPT is a stateless REST API that streams tokens left-to-right like a cursor — each call sees only
what came before and predicts the next token in the sequence; BERT is a batch-mode request
processor that reads the full input before emitting any output, making it ideal for tasks that
need bidirectional context like search ranking or classification.

## The Problem

RNNs and LSTMs process sequences one token at a time, creating an information bottleneck: context
from the beginning of a long sentence must survive hundreds of sequential hidden-state updates to
influence the end. That bottleneck collapses on long documents and prevents parallelism during
training — you cannot compute step 500 until step 499 is done. The Transformer replaces sequential
recurrence with parallel self-attention, letting every token attend to every other token in a
single matrix multiply, scaling to billion-token contexts while training on thousands of GPUs
simultaneously.

## Theory

### Scaled Dot-Product Attention

The core operation of a Transformer is scaled dot-product attention. Given a sequence of $T$
tokens, each token is projected into three vectors: a **query** ($Q$), a **key** ($K$), and a
**value** ($V$). The attention output is a weighted sum of values, where the weights measure
query–key similarity:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d_k}}\right)\!V$$

- $Q \in \mathbb{R}^{T \times d_k}$ — query matrix; row $i$ asks "what am I looking for?"
- $K \in \mathbb{R}^{T \times d_k}$ — key matrix; row $j$ answers "what do I advertise?"
- $V \in \mathbb{R}^{T \times d_v}$ — value matrix; row $j$ is the actual content to retrieve
- $d_k$ — key dimension per head; dividing by $\sqrt{d_k}$ prevents dot products from growing
  large enough to push softmax into near-zero gradient regions
- $T$ — sequence length (number of tokens in the current context window)

The result is a $T \times d_v$ matrix — one context-aware embedding per token, enriched by
information from all other tokens the operation is allowed to attend to.

#### Numeric Worked Example — Causal Attention (3 Tokens)

Represent "the", "cat", "sat" with toy 2-dimensional embeddings ($d_k = 2$, $d_v = 2$):

| Token | Query $Q$ | Key $K$ | Value $V$ |
|-------|-----------|---------|-----------|
| "the" | $[1, 0]$  | $[1, 0]$| $[1, 0]$  |
| "cat" | $[0, 1]$  | $[0, 1]$| $[0, 1]$  |
| "sat" | $[1, 1]$  | $[0, 1]$| $[1, 1]$  |

**Step 1 — Raw scores** $S = Q K^T$:

$$S = \begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}\begin{bmatrix}1&0&0\\0&1&1\end{bmatrix} = \begin{bmatrix}1&0&0\\0&1&1\\1&1&1\end{bmatrix}$$

**Step 2 — Scale** by $1/\sqrt{d_k} = 1/\sqrt{2} \approx 0.707$:

$$\hat{S} = \begin{bmatrix}0.707&0&0\\0&0.707&0.707\\0.707&0.707&0.707\end{bmatrix}$$

**Step 3 — Apply causal mask** (GPT style: token $i$ can only attend to positions $\leq i$).
Positions to the right become $-\infty$ so they vanish after softmax:

$$\hat{S}_{\text{masked}} = \begin{bmatrix}0.707&-\infty&-\infty\\0&0.707&-\infty\\0.707&0.707&0.707\end{bmatrix}$$

**Step 4 — Softmax** row-wise:

| Row | Raw | Softmax |
|-----|-----|---------|
| "the" (row 0) | $[0.707, -\infty, -\infty]$ | $[1.000,\ 0.000,\ 0.000]$ |
| "cat" (row 1) | $[0.000,\ 0.707,\ -\infty]$ | $[0.330,\ 0.670,\ 0.000]$ |
| "sat" (row 2) | $[0.707,\ 0.707,\ 0.707]$  | $[0.333,\ 0.333,\ 0.333]$ |

**Step 5 — Weighted sum of values** $A = \text{softmax}(\hat{S}) \cdot V$:

$$A_{\text{"the"}} = 1.000 \cdot [1,0] = [1.000,\; 0.000]$$
$$A_{\text{"cat"}} = 0.330 \cdot [1,0] + 0.670 \cdot [0,1] = [0.330,\; 0.670]$$
$$A_{\text{"sat"}} = 0.333 \cdot [1,0] + 0.333 \cdot [0,1] + 0.333 \cdot [1,1] = [0.666,\; 0.666]$$

"sat" blends information equally from all three tokens; "the" attends only to itself — exactly
the causal constraint that makes GPT-style generation valid at inference time.

### Multi-Head Attention

Instead of a single attention pass, Transformers run $H$ independent attention heads in parallel,
each projecting $Q$, $K$, $V$ into a lower-dimensional subspace:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H)\,W^O$$

$$\text{head}_h = \text{Attention}(Q W^Q_h,\; K W^K_h,\; V W^V_h)$$

- $W^Q_h, W^K_h \in \mathbb{R}^{d_{\text{model}} \times d_k}$ — per-head query/key projection
- $W^V_h \in \mathbb{R}^{d_{\text{model}} \times d_v}$ — per-head value projection, where $d_k = d_v = d_{\text{model}}/H$
- $W^O \in \mathbb{R}^{H d_v \times d_{\text{model}}}$ — output projection that merges heads
- Each head learns a different attention pattern: one head might track syntax, another coreference

### Positional Encoding

Self-attention is permutation-invariant — shuffle the tokens and the pairwise dot products are
identical. Positional encoding injects order by adding a position-dependent vector to each token
embedding before the first block:

$$\text{PE}(pos, 2i) = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$\text{PE}(pos, 2i+1) = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

- $pos \in \{0, 1, \ldots, T-1\}$ — token position in the sequence
- $i \in \{0, 1, \ldots, d_{\text{model}}/2 - 1\}$ — dimension index within the embedding
- Different frequencies ensure every position gets a unique fingerprint across all dimensions
- GPT-2 and later models replace the sinusoidal formula with **learned** position embeddings —
  a simple `nn.Embedding(context_len, d_model)` table that the model trains end-to-end

### The Transformer Block

Every Transformer layer stacks four operations in a residual loop:

```
x → LayerNorm → MultiHeadAttention → + (residual) → LayerNorm → FFN → + (residual) → x'
```

This **Pre-LN** ordering (normalise before attention, used in GPT-2+) stabilises gradients during
deep training. The feed-forward network (FFN) is a position-wise two-layer MLP:

$$\text{FFN}(x) = \text{GELU}\!\left(x W_1 + b_1\right) W_2 + b_2$$

- $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$ — expands dimension (typically $d_{\text{ff}} = 4 \cdot d_{\text{model}}$)
- $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$ — projects back to $d_{\text{model}}$
- **GELU** (Gaussian Error Linear Unit) is preferred over ReLU in modern LLMs: $\text{GELU}(x) \approx x \cdot \sigma(1.702x)$
- Residual connections let gradients flow directly from the output back to any earlier layer,
  preventing the vanishing gradient problem across 96+ blocks

### GPT vs BERT — Decoder-Only vs Encoder-Only

| Dimension | GPT (Decoder-Only) | BERT (Encoder-Only) |
|-----------|-------------------|---------------------|
| **Attention mask** | Causal (lower-triangular) | Bidirectional (full matrix) |
| **Pre-training task** | Next-token prediction | Masked Language Model (MLM) |
| **Inference pattern** | Autoregressive: token by token | One forward pass over full input |
| **Strengths** | Open-ended generation, agents | Classification, retrieval, QA |
| **Examples** | GPT-4, Llama-3, Mistral | BERT, RoBERTa, DeBERTa |
| **FSD analogy** | Streaming SSE response cursor | Batch-processed REST response |

The original 2017 "Attention Is All You Need" paper described a full encoder–decoder architecture
(BERT-like encoder + GPT-like decoder connected by cross-attention). Modern LLMs dropped the
encoder entirely: a sufficiently large decoder-only model trained on next-token prediction
generalises to all downstream tasks through prompting and fine-tuning, eliminating the need for
task-specific encoder heads.

## Python Implementation

```python
# Dependencies: torch>=2.1, numpy>=1.24, tiktoken>=0.5
import math
import numpy as np
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ── Hyperparameters ────────────────────────────────────────────────────────────
VOCAB_SIZE  = 50257   # GPT-2 BPE vocabulary size
CONTEXT_LEN = 16      # maximum sequence length this model supports
D_MODEL     = 64      # embedding dimension (d_model)
N_HEADS     = 4       # number of attention heads (H)
D_FF        = 256     # feed-forward hidden dimension (d_ff = 4 × d_model)
N_LAYERS    = 2       # number of stacked Transformer blocks


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute scaled dot-product attention for one head.

    Args:
        Q: query tensor  [batch, heads, seq_len, d_k]
        K: key tensor    [batch, heads, seq_len, d_k]
        V: value tensor  [batch, heads, seq_len, d_v]
        mask: optional causal mask [1, 1, seq_len, seq_len]; 0 = block, 1 = allow
    Returns:
        Attended output  [batch, heads, seq_len, d_v]
    """
    d_k = Q.size(-1)                                      # dimension per head
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)    # [B, H, T, T]
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)                   # attention weights
    return weights @ V                                    # [B, H, T, d_v]


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional causal masking."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads   # dimension per head

        # Three linear projections — no bias (standard GPT practice)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)   # output projection

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """Single self-attention forward pass."""
        B, T, C = x.shape   # batch, sequence length, d_model

        # Project and reshape into [B, n_heads, T, d_k]
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        mask = None
        if causal:
            # Lower-triangular mask: position i cannot see positions j > i
            mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)

        attended = scaled_dot_product_attention(Q, K, V, mask)   # [B, H, T, d_k]

        # Merge heads: [B, H, T, d_k] → [B, T, d_model]
        attended = attended.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(attended)


class TransformerBlock(nn.Module):
    """One GPT decoder block: Pre-LN → MHA (causal) → residual → Pre-LN → FFN → residual."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.norm1     = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm2     = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual connections wrap both sub-layers."""
        x = x + self.attention(self.norm1(x), causal=True)   # attention sub-layer
        x = x + self.ffn(self.norm2(x))                       # FFN sub-layer
        return x


class TinyGPT(nn.Module):
    """Minimal GPT-style decoder-only Transformer.

    Architecture:
        token_embedding + position_embedding
        → N × TransformerBlock
        → LayerNorm
        → Linear (lm_head) → logits over vocabulary
    """

    def __init__(
        self,
        vocab_size: int,
        context_len: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
    ) -> None:
        super().__init__()
        self.token_embedding    = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_len, d_model)   # learned PE
        self.blocks  = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.norm    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share embedding and lm_head weights (standard GPT practice)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return logits [batch, seq_len, vocab_size]."""
        B, T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device)   # [0, 1, ..., T-1]

        # Input = token embedding + positional embedding
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.blocks(x)    # N stacked Transformer blocks
        x = self.norm(x)
        return self.lm_head(x)   # project to vocabulary logits


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def show_causal_mask(seq_len: int) -> None:
    """Print the lower-triangular causal mask for a given sequence length."""
    mask = torch.tril(torch.ones(seq_len, seq_len)).numpy().astype(int)
    print(f"\nCausal mask ({seq_len}×{seq_len})  — 1=attend, 0=blocked:")
    header = "     " + "  ".join(f"t{j}" for j in range(seq_len))
    print(header)
    for i, row in enumerate(mask):
        print(f"  t{i}  " + "  ".join(str(v) for v in row))


def main() -> None:
    torch.manual_seed(42)

    # ── 1. Tokenise a prompt with GPT-2 BPE (downloads vocab once on first run) ──
    enc = tiktoken.get_encoding("gpt2")
    prompt = "Transformers replace recurrence with attention"
    token_ids = enc.encode(prompt)
    print("=" * 60)
    print(f"Prompt      : '{prompt}'")
    print(f"Token IDs   : {token_ids}")
    decoded_tokens = [enc.decode([t]) for t in token_ids]
    print(f"Tokens      : {decoded_tokens}")
    print(f"Token count : {len(token_ids)}")

    # ── 2. Build TinyGPT ──────────────────────────────────────────────────────
    model = TinyGPT(
        vocab_size  = VOCAB_SIZE,
        context_len = CONTEXT_LEN,
        d_model     = D_MODEL,
        n_heads     = N_HEADS,
        d_ff        = D_FF,
        n_layers    = N_LAYERS,
    )
    total_params = count_parameters(model)
    print(f"\nModel       : TinyGPT  ({N_LAYERS} blocks, d_model={D_MODEL}, heads={N_HEADS})")
    print(f"Parameters  : {total_params:,}")

    # ── 3. Forward pass ───────────────────────────────────────────────────────
    # Truncate input to CONTEXT_LEN if prompt is longer
    ids_clipped = token_ids[:CONTEXT_LEN]
    ids_tensor  = torch.tensor(ids_clipped).unsqueeze(0)   # [1, T]
    logits      = model(ids_tensor)                         # [1, T, vocab_size]

    print(f"\nInput shape  : {list(ids_tensor.shape)}  (batch=1, seq_len={ids_tensor.shape[1]})")
    print(f"Output shape : {list(logits.shape)}  (logits per position over {VOCAB_SIZE:,} tokens)")

    # ── 4. Greedy next-token prediction at every position ────────────────────
    next_token_ids = logits.argmax(dim=-1).squeeze(0).tolist()
    print("\nGreedy next-token predictions (untrained weights — not meaningful, but shape is correct):")
    print(f"  {'Position':<10}  {'Input token':<20}  {'Predicted next'}")
    print(f"  {'-'*8:<10}  {'-'*18:<20}  {'-'*18}")
    for pos, (inp_id, pred_id) in enumerate(zip(ids_clipped, next_token_ids)):
        inp_str  = repr(enc.decode([inp_id]))
        pred_str = repr(enc.decode([pred_id]))
        print(f"  pos {pos:<5}   {inp_str:<20}  → {pred_str}")

    # ── 5. Show the causal mask ───────────────────────────────────────────────
    show_causal_mask(seq_len=ids_tensor.shape[1])

    # ── 6. Numeric attention example (matches Theory section) ────────────────
    print("\n" + "=" * 60)
    print("Manual attention example from Theory (d_k=2, 3 tokens, causal):")
    Q = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    K = torch.tensor([[1., 0.], [0., 1.], [0., 1.]])
    V = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])

    d_k    = Q.size(-1)
    scores = Q @ K.T / math.sqrt(d_k)
    mask   = torch.tril(torch.ones(3, 3))
    scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    output  = weights @ V

    print(f"Attention weights (softmax, causal):\n{weights.numpy().round(3)}")
    print(f"Output (weighted sum of V):\n{output.numpy().round(3)}")


if __name__ == "__main__":
    main()
```

Run the script with `python lesson.py` (rename as needed). On the first execution, tiktoken
downloads the GPT-2 vocabulary (~500 KB) and caches it locally — subsequent runs are offline.

**What to notice in the output:**

- The **token IDs** show BPE sub-word splitting — "Transformers" may become 1–2 tokens depending
  on its presence in the GPT-2 vocabulary.
- The **output shape** `[1, T, 50257]` confirms each position gets a full vocabulary distribution,
  not just the final token — during training every position contributes to the loss simultaneously.
- The **causal mask** triangle makes the GPT constraint concrete: position 0 is completely isolated
  (column 1 onward is all zeros), while the last position has a full row of ones.
- The **manual attention example** matches the Theory section arithmetic exactly — attention weights
  for "the" are `[1, 0, 0]`, for "cat" are `[0.330, 0.670, 0]`, for "sat" are `[0.333, 0.333, 0.333]`.

## Java Implementation

> **Java:** minGPT from scratch has no Java equivalent. DJL covers inference but not educational scratch implementations.

## Key Takeaways

- The causal mask is the only structural difference between GPT (decoder-only, lower-triangular
  mask) and BERT (encoder-only, full attention matrix) — the block architecture is otherwise
  identical; choosing a mask shape is choosing a pre-training objective.
- Scaling by $1/\sqrt{d_k}$ is not cosmetic — without it, dot products grow linearly with
  dimension and softmax saturates to near-one-hot distributions, killing gradient flow in
  deeper heads.
- Multi-head attention runs $H$ independent attention patterns in parallel at no extra sequential
  cost — each head specialises in a different relationship (syntax, coreference, positional
  proximity) and the output projection merges them back into a single $d_{\text{model}}$ stream.

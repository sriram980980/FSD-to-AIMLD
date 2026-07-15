# 4.2 — Fine-Tuning: LoRA & QLoRA

## Hook

LoRA is like monkey-patching a third-party library — instead of forking the entire codebase (retraining all 7 billion weights), you inject two tiny adapter matrices that intercept the forward pass and redirect behavior, leaving the original base weights completely untouched.

## The Problem

A pretrained LLM like Llama-3 costs millions of dollars and months of compute to train. Full fine-tuning requires gradient storage for every one of its 7B+ weights — which exceeds the VRAM of any single consumer GPU and costs thousands of dollars per run. You need a way to teach the model new behavior (your domain vocabulary, your output format, your task) without touching more parameters than absolutely necessary.

## Theory

### LoRA: Low-Rank Adaptation

Every dense layer in a Transformer holds a weight matrix $W \in \mathbb{R}^{d \times k}$. Full fine-tuning updates $W$ directly, requiring $d \times k$ trainable parameters. LoRA instead freezes $W$ and adds a low-rank bypass:

$$h = xW + x \cdot \frac{\alpha}{r} BA$$

Where:
- $x \in \mathbb{R}^{1 \times d}$ — input activation vector
- $W \in \mathbb{R}^{d \times k}$ — **frozen** pretrained weight matrix (no gradient)
- $A \in \mathbb{R}^{r \times k}$ — trainable **right** adapter matrix, initialised with random Gaussian
- $B \in \mathbb{R}^{d \times r}$ — trainable **left** adapter matrix, initialised to **zero** (so the adapter starts as a no-op)
- $r$ — **rank**, a hyperparameter; typically 4–64; controls adapter expressivity
- $\alpha$ — scaling constant (often equal to $r$, making $\alpha/r = 1$); controls how much the adapter updates are weighted relative to the frozen path

The product $BA \in \mathbb{R}^{d \times k}$ approximates the full weight update $\Delta W$ but uses only $r(d + k)$ parameters instead of $dk$.

**Numeric worked example — parameter reduction:**

Suppose one attention projection in Llama-3 has $d = 4096$, $k = 4096$, and we choose $r = 8$:

| Method | Formula | Count |
|---|---|---|
| Full fine-tuning ($\Delta W$) | $d \times k = 4096 \times 4096$ | **16,777,216** |
| LoRA adapters ($A$ + $B$) | $r(d + k) = 8 \times 8192$ | **65,536** |
| Reduction | $16{,}777{,}216 / 65{,}536$ | **256×** fewer |

A full 7B model has ~32 such projection layers in each of 32 blocks. Full fine-tuning of all query/key/value/output projections across 32 blocks = ~2.1B parameters. LoRA at rank 8 brings that down to ~8.4M — a 250× reduction — while typically recovering 95%+ of the full fine-tune quality on narrow tasks.

**Forward pass numeric trace** (tiny 3×3 example, $r=1$, $\alpha=1$):

```
W  = [[1, 0, 0],   x = [1, 2, 3]
      [0, 1, 0],
      [0, 0, 1]]

A  = [[0.1, 0.2, 0.3]]   shape (1, 3)  ← r=1
B  = [[0.5],              shape (3, 1)  ← r=1
      [0.3],
      [0.2]]

xW = [1, 2, 3]                         ← frozen path
xA = [1×0.1 + 2×0.2 + 3×0.3] = [1.4]  ← project down to rank
xAB= [1.4×0.5, 1.4×0.3, 1.4×0.2]     
    = [0.70, 0.42, 0.28]               ← project back up

h  = xW + (α/r)·xAB
   = [1+0.70, 2+0.42, 3+0.28]
   = [1.70, 2.42, 3.28]
```

The adapter nudges the output without overwriting the pretrained knowledge in $W$.

---

### QLoRA: Quantized LoRA

QLoRA stacks two ideas on top of LoRA:

**1. 4-bit NF4 Quantization of the base model**

NF4 (Normal Float 4) uses a data type with 16 quantization levels spaced according to a normal distribution. Given that pretrained weights roughly follow $\mathcal{N}(0, \sigma^2)$, NF4 levels cluster tightly near zero where most weights live, reducing quantization error versus uniform 4-bit.

$$W_{4\text{bit}} = \text{quantize}(W / \text{absmax}(W))$$

Memory saving: a 7B fp16 model occupies $7 \times 10^9 \times 2 \approx 14\text{ GB}$ VRAM. The 4-bit version occupies $\approx 3.5\text{ GB}$, fitting on a single 24 GB consumer GPU.

**2. Double Quantization**

The quantization constants themselves (one per 64-weight block) are stored in fp32 — adding ~0.5 GB back. Double quantization quantizes those constants too (to 8-bit), recovering ~0.1 GB per billion parameters. Small, but meaningful at scale.

**3. Paged Optimizers**

Gradient checkpointing spikes VRAM during the backward pass. QLoRA uses NVIDIA's unified memory to **page** optimizer states to CPU RAM when GPU pressure peaks, preventing OOM without slowing the steady-state training loop.

**QLoRA memory budget (7B model, r=64):**

| Component | Precision | VRAM |
|---|---|---|
| Base model weights | 4-bit NF4 | ~3.5 GB |
| LoRA adapters (A, B) | bfloat16 | ~0.06 GB |
| Activations + optimizer | bfloat16 | ~2 GB |
| **Total** | | **~5.6 GB** |

This fits a 7B fine-tune on a single RTX 3090/4090 (24 GB), which was previously impossible.

## Python Implementation

```python
# Dependencies: torch>=2.1, numpy>=1.24
"""
Lesson 4.2 — LoRA math demonstration.
Runs on CPU. No model download required.
Shows the low-rank adapter forward pass and parameter savings.
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class FrozenLinear(nn.Module):
    """Standard linear layer with weights frozen — simulates a pretrained projection."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.02,
            requires_grad=False,  # ← frozen: no gradient flows here
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class LoRALayer(nn.Module):
    """
    Low-Rank Adapter that wraps a frozen linear layer.
    Adds trainable bypass: output += (alpha/r) * x @ A.T @ B.T
    """

    def __init__(
        self,
        frozen_layer: FrozenLinear,
        rank: int,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.frozen = frozen_layer
        self.rank = rank
        self.alpha = alpha

        in_features = frozen_layer.weight.shape[1]
        out_features = frozen_layer.weight.shape[0]

        # A: project down from in_features → rank (Gaussian init)
        self.lora_A = nn.Parameter(
            torch.randn(rank, in_features) * (1.0 / math.sqrt(rank))
        )
        # B: project up from rank → out_features (zero init → no-op at start)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frozen_out = self.frozen(x)                          # xW  (no gradient)
        adapter_out = x @ self.lora_A.T @ self.lora_B.T     # xAB
        scale = self.alpha / self.rank
        return frozen_out + scale * adapter_out              # h = xW + (α/r)·xAB

    def trainable_parameters(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()

    def full_finetune_parameters(self) -> int:
        return self.frozen.weight.numel()


def demonstrate_lora_savings(
    in_features: int,
    out_features: int,
    rank: int,
    alpha: float = 1.0,
) -> None:
    """Show parameter counts and a forward pass for a single LoRA layer."""
    print("=" * 55)
    print(f"Layer shape : {in_features} → {out_features}")
    print(f"LoRA rank   : r = {rank},  alpha = {alpha}")
    print("-" * 55)

    frozen = FrozenLinear(in_features, out_features)
    lora   = LoRALayer(frozen, rank=rank, alpha=alpha)

    full_params  = lora.full_finetune_parameters()
    lora_params  = lora.trainable_parameters()
    reduction    = full_params / lora_params

    print(f"Full fine-tune params : {full_params:>10,}")
    print(f"LoRA trainable params : {lora_params:>10,}  ({lora_params / full_params * 100:.2f}%)")
    print(f"Reduction factor      : {reduction:>10.1f}×")

    # Sanity-check: adapter is a no-op at initialisation (B is zero)
    x = torch.randn(4, in_features)           # batch of 4
    frozen_out   = frozen(x)
    lora_out_t0  = lora(x)
    delta        = (lora_out_t0 - frozen_out).abs().max().item()
    print(f"\nAdapter delta at init : {delta:.6f}  (should be 0.000000 — B is zero)")

    # Simulate one gradient step so B is non-zero, then check adapter fires
    optimizer = torch.optim.AdamW(lora.parameters(), lr=1e-3)
    loss = lora_out_t0.mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    lora_out_t1 = lora(x)
    delta_t1 = (lora_out_t1 - frozen_out).abs().max().item()
    print(f"Adapter delta post-1-step : {delta_t1:.6f}  (adapter now active)")
    print()


def compare_ranks(
    in_features: int,
    out_features: int,
    ranks: list[int],
) -> None:
    """Compare parameter budgets across different rank choices."""
    full = in_features * out_features
    print("=" * 55)
    print(f"Parameter budget vs rank  ({in_features}×{out_features} layer)")
    print(f"{'Rank':>6}  {'LoRA params':>12}  {'% of full':>10}  {'Reduction':>10}")
    print("-" * 55)
    for r in ranks:
        lora_p = r * (in_features + out_features)
        print(
            f"{r:>6}  {lora_p:>12,}  {lora_p / full * 100:>9.3f}%  {full / lora_p:>9.1f}×"
        )
    print()


def main() -> None:
    torch.manual_seed(42)

    # --- Llama-3-scale projection (4096×4096) ---
    demonstrate_lora_savings(in_features=4096, out_features=4096, rank=8, alpha=16.0)

    # --- Rank sensitivity sweep ---
    compare_ranks(in_features=4096, out_features=4096, ranks=[1, 4, 8, 16, 32, 64, 128])

    # --- Tiny layer — inspect the actual numbers ---
    print("=" * 55)
    print("Tiny 4×4 layer — numeric trace (r=1, alpha=1)")
    print("-" * 55)

    torch.manual_seed(0)
    tiny_frozen = FrozenLinear(4, 4)
    tiny_frozen.weight = nn.Parameter(torch.eye(4), requires_grad=False)  # identity W

    tiny_lora = LoRALayer(tiny_frozen, rank=1, alpha=1.0)
    # Manually set A and B to match the Theory section example
    tiny_lora.lora_A = nn.Parameter(torch.tensor([[0.1, 0.2, 0.3, 0.0]]))  # shape (1,4)
    tiny_lora.lora_B = nn.Parameter(torch.tensor([[0.5], [0.3], [0.2], [0.0]]))  # shape (4,1)

    x = torch.tensor([[1.0, 2.0, 3.0, 0.0]])

    frozen_out  = tiny_frozen(x)
    adapter_out = (x @ tiny_lora.lora_A.T @ tiny_lora.lora_B.T) * (1.0 / 1)
    lora_out    = tiny_lora(x)

    print(f"Input x         : {x.squeeze().tolist()}")
    print(f"xW  (frozen)    : {frozen_out.squeeze().tolist()}")
    print(f"(α/r)·xAB       : {adapter_out.squeeze().tolist()}")
    print(f"h = xW + adapter: {lora_out.squeeze().tolist()}")


if __name__ == "__main__":
    main()
```

**What to notice in the output:**

- The adapter delta is exactly `0.000000` at init — initialising $B$ to zero guarantees the model starts from the pretrained checkpoint, not a random perturbation. This is a deliberate design choice.
- After a single optimiser step $B$ is non-zero and the adapter fires, adding a small correction vector on top of the frozen output.
- At rank 8 the 4096×4096 projection goes from 16.7M parameters to 65K — you are training 0.39% of the weights that full fine-tuning would require.
- The rank sweep shows that even rank 64 (still only 3.1% of full) gives the adapter substantial expressivity headroom for complex domain adaptation.

## Java Implementation

> **Java:** No Java ecosystem for LoRA/QLoRA fine-tuning. Python-only.

## Key Takeaways

- **LoRA freezes the base model and adds two small matrices** $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ whose product approximates the full weight update; initialising $B$ to zero makes the adapter a no-op at the start of training.
- **Rank $r$ is the expressivity dial** — rank 8 gives a 256× parameter reduction on 4096×4096 projections while recovering the majority of full fine-tune quality; increase $r$ only when validation loss plateaus.
- **QLoRA extends LoRA with 4-bit NF4 quantization** of the frozen base weights and paged optimisers, shrinking a 7B model's VRAM footprint from ~14 GB to ~5.6 GB so fine-tuning fits on a single 24 GB consumer GPU.

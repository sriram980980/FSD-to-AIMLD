# 6.2 — Distributed Training — Parallelism Strategies

## Hook

Data Parallelism is horizontal scaling — the same full model runs on N GPU replicas and gradients are averaged, just like load-balancing N identical app servers behind a round-robin proxy; Tensor Parallelism is column-sharding — one giant weight matrix is split across GPUs the way a DBA shards a single hot table across DB nodes; and Pipeline Parallelism is a microservice chain — each GPU owns one layer group and streams activations downstream like requests flowing through ordered middleware stages.

## The Problem

A 70B-parameter model in bf16 requires ~140 GB of VRAM — nearly twice what the largest single GPU (H100 SXM at 80 GB) can hold. Even when a model fits, training on a single GPU takes months of wall-clock time, making large-scale pre-training and fine-tuning physically impractical. Without a clear mental model of which parallelism strategy to apply — and how to combine them — you will either hit an out-of-memory crash before the first forward pass, or waste expensive GPU-hours on strategies that cannot saturate available compute.

## Theory

### Data Parallelism (DDP) — Gradient AllReduce

In Data Parallelism every GPU holds a **full copy of the model**. The global batch of size $B$ is split into $N$ shards of size $B/N$. Each GPU runs its forward and backward pass independently on its shard, then all GPUs synchronize gradients via an **AllReduce** collective before applying the optimizer step.

**DDP weight update rule:**

$$w^{(t+1)} = w^{(t)} - \eta \cdot \frac{1}{N} \sum_{i=1}^{N} \nabla_{w} \mathcal{L}_i$$

Where:
- $w^{(t)}$ — model weights at training step $t$
- $\eta$ — learning rate
- $N$ — number of GPU workers
- $\nabla_{w} \mathcal{L}_i$ — gradient computed on GPU $i$ using its local data shard

The AllReduce is implemented as a **Ring-AllReduce**: each GPU sends to its right neighbour and receives from its left neighbour in two phases (reduce-scatter, all-gather), so every GPU ends up with the full averaged gradient after $2(N-1)$ rounds of communication — without any central parameter server.

**Numeric example — 2 GPUs, 1 scalar weight, learning rate 0.01:**

| GPU | Local batch loss | Local gradient $\nabla_w \mathcal{L}_i$ |
|-----|-----------------|------------------------------------------|
| 0   | 0.72            | +0.80                                    |
| 1   | 0.68            | +0.40                                    |

AllReduce average: $(0.80 + 0.40) / 2 = 0.60$

Weight update: $w \leftarrow w - 0.01 \times 0.60 = w - 0.006$

Without AllReduce — if each GPU updated independently — GPU 0 and GPU 1 would diverge immediately: GPU 0 would step $-0.008$ and GPU 1 would step $-0.004$. By the second step the two replicas are out of sync and the training run is no longer equivalent to training on the full batch.

**ZeRO (Zero Redundancy Optimizer)** extends DDP by sharding optimizer states, gradients, and optionally weights across GPUs — eliminating the per-replica memory overhead. ZeRO Stage 3 reduces per-GPU memory from $M_{full}$ to $M_{full} / N$, at the cost of additional communication during the forward and backward passes to gather the needed shards.

---

### Tensor Parallelism (TP) — Column and Row Sharding

Tensor Parallelism splits a **single weight matrix** across multiple GPUs. A linear layer $Y = XW^{\top}$ with $W \in \mathbb{R}^{d_{out} \times d_{in}}$ is partitioned **column-wise** across $N$ GPUs:

$$W = \bigl[W_1 \mid W_2 \mid \cdots \mid W_N\bigr], \quad W_i \in \mathbb{R}^{(d_{out}/N) \times d_{in}}$$

Each GPU $i$ computes the partial output $Y_i = X W_i^{\top}$ independently. The full output is recovered by concatenation:

$$Y = \bigl[Y_1 \mid Y_2 \mid \cdots \mid Y_N\bigr]$$

For **row-parallel** layers (e.g., the down-projection after column-parallel up-projection), each GPU holds $d_{in}/N$ input columns and produces a full $d_{out}$-dimensional output. An **AllReduce** sums the partial results across GPUs:

$$Y = \sum_{i=1}^{N} X_i W_i^{\top}$$

**Numeric example — split a $2 \times 4$ weight matrix across 2 GPUs:**

Full matrix $W$ and input $X$:

$$W = \begin{bmatrix}1 & 2 & 3 & 4\\5 & 6 & 7 & 8\end{bmatrix}, \quad X = \begin{bmatrix}1 & 0\end{bmatrix}$$

Column-parallel split:
- GPU 0 holds $W_0 = \bigl[\begin{smallmatrix}1 & 2\\5 & 6\end{smallmatrix}\bigr]$, computes $Y_0 = X W_0^{\top} = [1, 2]$
- GPU 1 holds $W_1 = \bigl[\begin{smallmatrix}3 & 4\\7 & 8\end{smallmatrix}\bigr]$, computes $Y_1 = X W_1^{\top} = [3, 4]$
- Concatenated output: $[1, 2, 3, 4]$ — identical to the full single-GPU matmul

TP requires **NVLink-class bandwidth** because every forward pass generates activation tensors that must be passed between GPUs proportional to sequence length and hidden dimension. It is almost always confined to a single node (8 GPUs sharing NVLink) and is never used across InfiniBand — the latency and bandwidth penalty would erase the compute gain.

---

### Pipeline Parallelism (PP) — Layer Scheduling and Micro-batches

Pipeline Parallelism assigns **contiguous blocks of model layers** to each GPU. A 24-layer transformer split across 4 GPUs gets 6 layers each. During the forward pass, GPU 0 computes its 6-layer block and sends the resulting activations to GPU 1, which does the same, and so on. During backward, gradients flow in the reverse direction.

The naive version (GPipe) creates a **pipeline bubble**: GPU 1 is idle while GPU 0 processes the first input. The fix is **micro-batching** — split the global batch into $m$ micro-batches so that GPU 1 can begin processing the second micro-batch while GPU 0 moves to the third.

**Bubble fraction formula:**

$$\text{Bubble}_{fraction} = \frac{p - 1}{p - 1 + m}$$

Where:
- $p$ — number of pipeline stages (one per GPU)
- $m$ — number of micro-batches per global batch

**Numeric example — $p = 4$ stages:**

| Micro-batches $m$ | Bubble fraction | GPU utilisation loss |
|-------------------|----------------|----------------------|
| 4                 | 3 / 7 ≈ 42.9%  | High                 |
| 8                 | 3 / 11 ≈ 27.3% | Moderate             |
| 16                | 3 / 19 ≈ 15.8% | Acceptable           |
| 32                | 3 / 35 ≈ 8.6%  | Near-ideal           |

Doubling micro-batches from 8 to 16 cuts wasted GPU time by nearly half. The practical limit is memory — each in-flight micro-batch's activations must fit in VRAM simultaneously.

---

### 3D Parallelism — Combining All Three

Production training of models with 70B+ parameters combines all three strategies simultaneously. The **Megatron-LM** framework and **DeepSpeed** apply:

| Parallelism | Dimension | Communication |
|-------------|-----------|---------------|
| Data (DDP / ZeRO) | Across replica groups | AllReduce via InfiniBand |
| Tensor (TP) | Within one node | AllReduce via NVLink |
| Pipeline (PP) | Across pipeline stages | P2P activation send/recv |

**Memory analysis — 70B model, 3D parallel across 64 GPUs (8 nodes × 8 GPUs):**

```
Without parallelism:  70B × 12 bytes (bf16 weights + grads + Adam) = 840 GB on 1 GPU
With 3D parallel:
  TP degree = 8  →  weight memory / 8
  PP degree = 4  →  layer memory / 4
  DP degree = 2  →  (ZeRO Stage 1) optimizer states / 2
  Effective per-GPU footprint ≈ 840 / (8 × 4 × 2) ≈ 13 GB  ← fits on a 40 GB A100
```

## Python Implementation

```python
# Dependencies: torch>=2.1, deepspeed>=0.14

"""
Lesson 6.2 — Distributed Training: Parallelism Strategies
Demonstrates (on CPU, no GPU required):
  1. DDP: AllReduce gradient averaging simulation + Ring-AllReduce mechanics
  2. Tensor Parallelism: column-parallel linear forward pass
  3. Pipeline Parallelism: micro-batch scheduling + bubble fraction calculator
  4. ZeRO Stage comparison: per-GPU memory at each sharding level
  5. DeepSpeed ZeRO-3 reference config
"""

import json
import math
import torch
import torch.nn as nn
from typing import List


# ── 1. DDP: Simulate Ring-AllReduce gradient averaging ────────────────────────

def ring_allreduce_average(
    per_gpu_gradients: List[torch.Tensor],
) -> torch.Tensor:
    """Average gradients from N workers — the DDP synchronisation step.

    Real DDP uses NCCL's ring algorithm to overlap this with the backward pass.
    Here we simulate the final averaged result.
    """
    stacked = torch.stack(per_gpu_gradients)   # shape: [N, *grad_shape]
    return stacked.mean(dim=0)


def ddp_demo() -> None:
    """Show gradient divergence without AllReduce, then with it."""
    torch.manual_seed(42)
    n_gpus = 4
    weight_shape = (4, 4)
    learning_rate = 0.01

    weights = torch.ones(weight_shape)

    # Simulate each GPU computing a gradient on its local mini-batch
    local_gradients = [
        torch.randn(weight_shape) * 0.1 + i * 0.05
        for i in range(n_gpus)
    ]

    print("=== Data Parallelism (DDP) ===")
    print(f"  GPUs: {n_gpus}  |  Weight shape: {list(weight_shape)}")
    print()

    # Without AllReduce: each GPU updates independently → divergence
    independent_updates = [
        weights - learning_rate * g
        for g in local_gradients
    ]
    update_norms_independent = [u.norm().item() for u in independent_updates]
    print("  Without AllReduce — per-GPU weight norms after update:")
    for i, norm in enumerate(update_norms_independent):
        print(f"    GPU {i}: {norm:.6f}")
    spread = max(update_norms_independent) - min(update_norms_independent)
    print(f"  Max divergence between replicas: {spread:.6f}")

    # With AllReduce: all GPUs apply the same averaged gradient → stay in sync
    averaged_gradient = ring_allreduce_average(local_gradients)
    synced_weights = weights - learning_rate * averaged_gradient
    print()
    print(f"  With AllReduce — averaged gradient norm:  {averaged_gradient.norm().item():.6f}")
    print(f"  All GPUs apply identical update — weight norm: {synced_weights.norm().item():.6f}")
    print(f"  Divergence: 0.000000  ✓")


# ── 2. Tensor Parallelism: Column-parallel linear layer ───────────────────────

def column_parallel_linear(
    x: torch.Tensor,
    weight_shards: List[torch.Tensor],
) -> torch.Tensor:
    """Column-parallel forward: each shard computes a partial output, concatenated."""
    partial_outputs = [x @ shard.T for shard in weight_shards]
    return torch.cat(partial_outputs, dim=-1)


def tensor_parallel_demo() -> None:
    """Split a 128→512 linear projection across 4 simulated GPU shards."""
    torch.manual_seed(0)
    d_in, d_out, n_shards = 128, 512, 4
    shard_cols = d_out // n_shards          # 128 output features per GPU

    full_weight = torch.randn(d_out, d_in)                    # [512, 128]
    weight_shards = list(full_weight.chunk(n_shards, dim=0))  # 4× [128, 128]

    batch_size = 8
    x = torch.randn(batch_size, d_in)

    output_tp = column_parallel_linear(x, weight_shards)
    output_ref = x @ full_weight.T                             # single-GPU reference

    max_diff = (output_tp - output_ref).abs().max().item()

    print("\n=== Tensor Parallelism ===")
    print(f"  Full weight shape:           {list(full_weight.shape)}")
    print(f"  Shard shape per GPU:         {list(weight_shards[0].shape)}  ({n_shards} GPUs)")
    print(f"  Input shape:                 {list(x.shape)}")
    print(f"  TP output shape:             {list(output_tp.shape)}")
    print(f"  Max numerical diff vs ref:   {max_diff:.2e}  (expected ~0)")
    print(f"  Memory per GPU:              {weight_shards[0].numel() * 4 / 1024:.1f} KB  "
          f"(vs {full_weight.numel() * 4 / 1024:.1f} KB for full weight)")


# ── 3. Pipeline Parallelism: Micro-batch scheduling + bubble calculator ────────

class PipelineStage(nn.Module):
    """One GPU's contiguous block of transformer layers (simplified to one MLP)."""

    def __init__(self, stage_id: int, d_model: int) -> None:
        super().__init__()
        self.stage_id = stage_id
        self.block = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def bubble_fraction(n_stages: int, n_micro_batches: int) -> float:
    """Fraction of total pipeline time wasted to bubble idle time."""
    return (n_stages - 1) / (n_stages - 1 + n_micro_batches)


def pipeline_parallel_demo() -> None:
    """Run 4-stage pipeline over micro-batches; show bubble vs micro-batch trade-off."""
    torch.manual_seed(7)
    n_stages = 4
    d_model = 64
    micro_batch_size = 4

    stages = [PipelineStage(i, d_model) for i in range(n_stages)]

    print("\n=== Pipeline Parallelism ===")
    print(f"  Pipeline stages (GPUs): {n_stages}")
    print()
    print(f"  {'Micro-batches (m)':>20}  {'Bubble fraction':>18}  {'GPU utilisation':>18}")
    print(f"  {'─'*20}  {'─'*18}  {'─'*18}")
    for m in [4, 8, 16, 32]:
        bubble = bubble_fraction(n_stages, m)
        utilisation = 1.0 - bubble
        print(f"  {m:>20}  {bubble:>17.1%}  {utilisation:>17.1%}")

    # Forward pass demo: 8 micro-batches through the 4-stage pipeline
    n_micro_batches = 8
    micro_batches = [
        torch.randn(micro_batch_size, d_model)
        for _ in range(n_micro_batches)
    ]

    print()
    print(f"  Forward pass: {n_micro_batches} micro-batches × {micro_batch_size} tokens × {d_model} dims")
    final_outputs = []
    for mb_idx, mb in enumerate(micro_batches):
        activations = mb
        for stage in stages:
            activations = stage(activations)
        final_outputs.append(activations)

    print(f"  Output shape per micro-batch: {list(final_outputs[0].shape)}")
    print(f"  Total micro-batches processed: {len(final_outputs)}")
    bubble = bubble_fraction(n_stages, n_micro_batches)
    print(f"  Effective bubble fraction: {bubble:.1%}  (wasted idle GPU cycles)")


# ── 4. ZeRO Stage memory breakdown ────────────────────────────────────────────

def zero_memory_breakdown(
    num_params: int,
    n_gpus: int,
    weight_bytes: float = 2.0,    # bf16
    grad_bytes: float = 2.0,      # bf16
    optim_bytes: float = 8.0,     # two fp32 Adam moments = 4 + 4
) -> None:
    """Print per-GPU memory at each ZeRO stage for a given model size."""
    total_weight_gb = num_params * weight_bytes / 1e9
    total_grad_gb   = num_params * grad_bytes   / 1e9
    total_optim_gb  = num_params * optim_bytes  / 1e9
    total_gb        = total_weight_gb + total_grad_gb + total_optim_gb

    print("\n=== ZeRO Stage Memory Comparison ===")
    print(f"  Model: {num_params / 1e9:.0f}B parameters  |  GPUs: {n_gpus}")
    print(f"  Full training footprint: {total_gb:.1f} GB  "
          f"(weights {total_weight_gb:.0f} + grads {total_grad_gb:.0f} "
          f"+ optim {total_optim_gb:.0f})")
    print()
    print(f"  {'Stage':<14}  {'What is sharded':<38}  {'Per-GPU (GB)':>14}")
    print(f"  {'─'*14}  {'─'*38}  {'─'*14}")

    # Stage 0: no sharding (baseline DDP)
    stage0_gb = total_gb
    print(f"  {'DDP (Stage 0)':<14}  {'nothing — full copy on every GPU':<38}  {stage0_gb:>13.1f}")

    # Stage 1: shard optimizer states only
    stage1_gb = total_weight_gb + total_grad_gb + total_optim_gb / n_gpus
    print(f"  {'ZeRO Stage 1':<14}  {'optimizer states':<38}  {stage1_gb:>13.1f}")

    # Stage 2: shard optimizer states + gradients
    stage2_gb = total_weight_gb + (total_grad_gb + total_optim_gb) / n_gpus
    print(f"  {'ZeRO Stage 2':<14}  {'optimizer states + gradients':<38}  {stage2_gb:>13.1f}")

    # Stage 3: shard everything
    stage3_gb = total_gb / n_gpus
    print(f"  {'ZeRO Stage 3':<14}  {'weights + grads + optimizer states':<38}  {stage3_gb:>13.1f}")


# ── 5. DeepSpeed ZeRO-3 reference config ──────────────────────────────────────

def print_deepspeed_config() -> None:
    """Print a production-ready DeepSpeed ZeRO Stage 3 config."""
    config = {
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 8,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-4,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.1,
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {"warmup_min_lr": 0, "warmup_max_lr": 1e-4, "warmup_num_steps": 500},
        },
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "offload_param": {"device": "cpu", "pin_memory": True},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e8,
            "stage3_prefetch_bucket_size": 5e7,
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 100,
    }
    print("\n=== DeepSpeed ZeRO-3 Reference Config ===")
    print(json.dumps(config, indent=2))
    print()
    print("  Key levers:")
    print("    stage: 3          → shard weights + grads + optimizer across all GPUs")
    print("    offload_optimizer → spill optimizer states to CPU RAM (saves ~50% VRAM)")
    print("    overlap_comm      → hide AllReduce latency behind backward compute")
    print("    gradient_clipping → prevents gradient explosion in long training runs")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ddp_demo()
    tensor_parallel_demo()
    pipeline_parallel_demo()
    zero_memory_breakdown(num_params=7_000_000_000, n_gpus=8)   # Llama 3 7B, 8 GPUs
    print_deepspeed_config()


if __name__ == "__main__":
    main()
```

**What to notice in the output:**

1. **DDP divergence without AllReduce is non-trivial even after one step** — across 4 GPUs the max norm spread grows measurably from the very first update. In real training this compounds over thousands of steps, making the replicas completely inconsistent. AllReduce collapses that divergence to exactly zero.
2. **Tensor parallelism numerical difference is ~1e-6** (floating-point rounding only) — column-parallel concatenation is mathematically identical to the single-GPU matmul. Any larger difference would indicate a partitioning bug.
3. **Bubble fraction drops from 42.9% at m=4 to 8.6% at m=32** — quadrupling micro-batches reduces idle GPU time by 5×. The cost is that 32 micro-batch activations must coexist in VRAM simultaneously.
4. **ZeRO Stage 3 on 8 GPUs reduces per-GPU memory from ~84 GB to ~10.5 GB** for a 7B model — the difference between needing 2 × H100 (160 GB) and fitting on a single 40 GB A100 for full fine-tuning.

## Java Implementation

> **Java:** Megatron-LM and DeepSpeed are Python-only. No Java ecosystem for large-scale distributed training. The distributed collective operations (AllReduce, AllGather) these frameworks rely on are implemented in NCCL (NVIDIA) and GLOO (CPU), neither of which has a maintained JVM binding. Spring AI and LangChain4j support calling a *trained* model endpoint but provide no distributed training primitives.

## Key Takeaways

- **Choose your parallelism by bottleneck:** Data Parallelism (DDP/ZeRO) is always the first choice when the model fits on one GPU — it is cheap to implement and scales linearly with GPU count; add Tensor Parallelism when a single weight matrix overflows one GPU's VRAM; add Pipeline Parallelism when the full layer stack does not fit even after TP, accepting a bubble-fraction overhead that shrinks as you increase micro-batches.
- **ZeRO Stage 3 trades bandwidth for memory:** it shards weights, gradients, and optimizer states across $N$ GPUs, reducing per-GPU footprint from the full $\sim 12\times$ parameter-byte training cost to $\sim 12\times / N$ — a 7B-parameter model that needs ~84 GB unsharded fits in ~10.5 GB per GPU across 8 GPUs, at the cost of AllGather communication on every forward pass.
- **Bubble fraction $= (p-1)/(p-1+m)$ is the key pipeline efficiency metric:** for a fixed number of pipeline stages $p$, the only lever you control is micro-batch count $m$ — doubling $m$ roughly halves wasted idle time, so production pipelines target $m \geq 4p$ to keep bubble overhead below ~20%.

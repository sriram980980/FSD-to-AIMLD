# Assignment 6.1 — Profile Memory/Network Bottlenecks During Distributed Run

## Objective

Build a GPU hardware profiler that queries (or simulates) a GPU spec, applies the Roofline model to classify ML kernels, budgets VRAM for three model sizes, computes all-reduce transfer times across interconnect tiers, and produces a written bottleneck report — proving you can diagnose whether a distributed run is constrained by VRAM, compute, or interconnect before spending a dollar on cloud GPUs.

## Background

Lesson 6.1 established three diagnostic tools: the **Roofline model** (arithmetic intensity $I = F/B$ vs the ridge point $I^* = P_{compute}/P_{bandwidth}$), the **VRAM footprint model** ($M_{train} = P \times (B_{param} + B_{grad} + 2 \times B_{moment})$), and the **interconnect hierarchy** (HBM → NVLink → PCIe → InfiniBand). Together these let you predict, without running a single training job, which hardware constraint will bite first. The starter code provides the GPU spec query and all pre-defined kernel/model constants; you implement the five analysis functions.

> **No GPU required.** All tasks run on CPU. When no NVIDIA GPU is detected, the profiler falls back to a simulated H100 SXM spec (80 GB VRAM, 3350 GB/s HBM bandwidth, 312 TFLOPS bf16). All expected output is based on this simulated spec.

## Setup

```bash
pip install "torch>=2.1" "nvidia-ml-py>=12.0"
```

## Tasks

**Task 1 — Implement `compute_ridge_point(spec)`**

Compute $I^* = P_{compute} / P_{bandwidth}$ for the queried GPU spec. `peak_tflops_bf16` is in TFLOPS (multiply by `1e12` for FLOP/s); `hbm_bandwidth_gbs` is in GB/s (multiply by `1e9` for bytes/s). Return the ridge point as a `float` in FLOPs/byte and print it.

```
Ridge point (I*): 93.1 FLOPs/byte
```

**Task 2 — Implement `classify_kernels(kernels, ridge)`**

For each kernel dict in `KERNELS` (keys: `"name"`, `"flops"`, `"bytes"`), compute arithmetic intensity $I = flops / bytes$, compare to `ridge`, and return a list of result dicts with keys `"name"`, `"intensity"`, and `"bound"` (either `"memory-bound"` or `"compute-bound"`). Print a formatted table. All four pre-defined kernels must appear; at least one must be `"compute-bound"`.

```
Kernel                                     FLOPs/byte  Bound
softmax_attention (2048² fp32)                   0.42  memory-bound
layer_norm (32×2048×4096 bf16)                   0.83  memory-bound
matmul (4096³ fp32)                            682.67  compute-bound
embedding_lookup (vocab=32K, d=4096 bf16)        0.25  memory-bound
```

**Task 3 — Implement `build_vram_budget(models, spec)`**

For each `(name, num_params)` tuple in `models`, compute:

- Inference VRAM in GB for dtypes `fp32`, `bf16`, `int8`, `int4` using `BYTES_PER_DTYPE`
- Training VRAM in GB using bf16 weights + bf16 gradients + two fp32 Adam moments: `P × (2 + 2 + 8)` bytes
- Whether the model fits on the queried GPU for **bf16 inference** and **bf16 training**

Return a list of result dicts and print a formatted table. Values must match the expected output within ±0.5 GB.

```
Model          fp32(GB)  bf16(GB)  int8(GB)  int4(GB)  Train-bf16(GB)  Fit-80GB(inf-bf16)?
Llama3-7B          26.1      13.0       6.5       3.3            78.2  ✓ fits (1.8 GB headroom)
Llama3-13B         48.4      24.2      12.1       6.1           145.3  ✓ fits (55.8 GB headroom)
Llama3-70B        260.8     130.4      65.2      32.6           782.3  ✗ does not fit (50.4 GB short)
```

**Task 4 — Implement `compute_allreduce_times(models, links)`**

For each model, treat its bf16 inference VRAM as the gradient blob size (worst-case all-reduce payload). For each `(link_name, bandwidth_gbs)` tuple in `INTERCONNECTS`, compute transfer time in milliseconds: `time_ms = (grad_gb / bandwidth_gbs) × 1000`. Return a list of dicts and print a formatted table. Values must match within ±1 ms.

```
Model       Grad(GB)  NVLink 4.0(ms)  PCIe 5.0 ×16(ms)  InfiniBand NDR(ms)
Llama3-7B      13.0            14.5             203.7               260.8
Llama3-13B     24.2            26.9             378.3               484.3
Llama3-70B    130.4           144.9            2037.5              2608.0
```

**Task 5 — Implement `generate_bottleneck_report(spec, vram_results, allreduce_results)`**

Synthesize the VRAM and interconnect results into a structured report. For each model print:

- Whether bf16 inference fits on one GPU and, if not, which dtype does fit
- Whether bf16 training fits on one GPU; if not, how many GPUs are required (ceiling of `train_gb / vram_80gb`)
- The slowest interconnect tier (InfiniBand) all-reduce latency in ms
- A one-line **binding constraint** statement: `"compute utilisation"` if training fits on one GPU, `"training VRAM"` if training overflows but inference fits, or `"inter-node InfiniBand bandwidth"` if inference also overflows

```
── Bottleneck Report ────────────────────────────────────────────────────
[Llama3-7B]
  Inference (bf16): fits on 1× H100 (13.0 GB / 80 GB). No quantisation needed.
  Training  (bf16): fits on 1× H100 (78.2 GB / 80 GB) — tight, 1.8 GB headroom.
  Interconnect:     InfiniBand all-reduce = 260.8 ms.
  Binding constraint: compute utilisation.

[Llama3-13B]
  Inference (bf16): fits on 1× H100 (24.2 GB / 80 GB). No quantisation needed.
  Training  (bf16): needs 2× H100s (145.3 GB required).
  Interconnect:     InfiniBand all-reduce = 484.3 ms.
  Binding constraint: training VRAM.

[Llama3-70B]
  Inference (bf16): does NOT fit — use int8 (65.2 GB fits) or int4 (32.6 GB fits).
  Training  (bf16): needs 10× H100s (782.3 GB required).
  Interconnect:     InfiniBand all-reduce = 2608.0 ms (2.6 s per gradient sync).
  Binding constraint: inter-node InfiniBand bandwidth.
```

## Expected Output

Running `python starter.py` after implementing all five stubs produces output matching this template (simulated H100 spec; your live GPU spec will differ in VRAM/bandwidth/TFLOPS but the classification logic must remain correct):

```
GPU: H100 SXM (simulated)
VRAM: 80 GB | Bandwidth: 3350 GB/s | bf16 TFLOPS: 312

── Task 1: Ridge Point ──────────────────────────────────────────────────
Ridge point (I*): 93.1 FLOPs/byte

── Task 2: Kernel Classification ────────────────────────────────────────
Kernel                                     FLOPs/byte  Bound
softmax_attention (2048² fp32)                   0.42  memory-bound
layer_norm (32×2048×4096 bf16)                   0.83  memory-bound
matmul (4096³ fp32)                            682.67  compute-bound
embedding_lookup (vocab=32K, d=4096 bf16)        0.25  memory-bound

── Task 3: VRAM Budget (GB) ─────────────────────────────────────────────
Model          fp32(GB)  bf16(GB)  int8(GB)  int4(GB)  Train-bf16(GB)  Fit-80GB(inf-bf16)?
Llama3-7B          26.1      13.0       6.5       3.3            78.2  ✓ fits (1.8 GB headroom)
Llama3-13B         48.4      24.2      12.1       6.1           145.3  ✓ fits (55.8 GB headroom)
Llama3-70B        260.8     130.4      65.2      32.6           782.3  ✗ does not fit (50.4 GB short)

── Task 4: All-Reduce Transfer Times ────────────────────────────────────
Model       Grad(GB)  NVLink 4.0(ms)  PCIe 5.0 ×16(ms)  InfiniBand NDR(ms)
Llama3-7B      13.0            14.5             203.7               260.8
Llama3-13B     24.2            26.9             378.3               484.3
Llama3-70B    130.4           144.9            2037.5              2608.0

── Task 5: Bottleneck Report ────────────────────────────────────────────
[Llama3-7B]
  Inference (bf16): fits on 1× H100 (13.0 GB / 80 GB). No quantisation needed.
  Training  (bf16): fits on 1× H100 (78.2 GB / 80 GB) — tight, 1.8 GB headroom.
  Interconnect:     InfiniBand all-reduce = 260.8 ms.
  Binding constraint: compute utilisation.

[Llama3-13B]
  Inference (bf16): fits on 1× H100 (24.2 GB / 80 GB). No quantisation needed.
  Training  (bf16): needs 2× H100s (145.3 GB required).
  Interconnect:     InfiniBand all-reduce = 484.3 ms.
  Binding constraint: training VRAM.

[Llama3-70B]
  Inference (bf16): does NOT fit — use int8 (65.2 GB fits) or int4 (32.6 GB fits).
  Training  (bf16): needs 10× H100s (782.3 GB required).
  Interconnect:     InfiniBand all-reduce = 2608.0 ms (2.6 s per gradient sync).
  Binding constraint: inter-node InfiniBand bandwidth.
```

Tolerance: VRAM values ±0.5 GB, time values ±1 ms. Classification labels must be exact strings.

## Evaluation Criteria

- [ ] `compute_ridge_point()` returns a value between 90–100 FLOPs/byte for the simulated H100 spec
- [ ] `classify_kernels()` correctly labels `matmul (4096³)` as `"compute-bound"` and all three other kernels as `"memory-bound"`
- [ ] `build_vram_budget()` reports Llama3-7B bf16 training as fitting (≤80 GB) and Llama3-70B bf16 inference as not fitting (>80 GB)
- [ ] `compute_allreduce_times()` shows NVLink at least 10× faster than PCIe for all models
- [ ] `generate_bottleneck_report()` assigns `"inter-node InfiniBand bandwidth"` as the binding constraint for Llama3-70B
- [ ] `python starter.py` runs to completion without errors on a CPU-only machine

## Extension Challenge

**Multi-GPU VRAM partitioning with communication overhead.**

For each model and each number of GPUs $G \in \{1, 2, 4, 8\}$ (all on the same NVLink domain), compute:

1. VRAM per GPU required for bf16 training with data parallelism (assume weights + optimizer states replicated, gradients reduced): $M_{per\_gpu} = M_{train} + M_{inference\_bf16}$
2. All-reduce time over NVLink for the gradient blob at batch size $G$ (gradient blob stays the same size; time scales with number of hops in a ring-all-reduce: $T_{ring} = \frac{2(G-1)}{G} \times \frac{grad\_gb}{nvlink\_bw\_gbs} \times 1000$ ms)
3. **GPU efficiency ratio**: what fraction of a theoretical 312 TFLOPS peak does the all-reduce dead time consume, assuming 1 training step takes 500 ms of compute? $\eta = 1 - T_{ring} / (500 + T_{ring})$

Print a table and identify the $G$ at which communication overhead exceeds 5% of total step time. No starter code provided.

# 6.1 — GPU/TPU Hardware — Interconnect & Memory

## Hook

NVLink between GPUs is like a high-speed internal LAN — orders of magnitude faster than PCIe (the "WAN") — so keep hot tensors local to the same node; HBM (High Bandwidth Memory) is your L2 cache where the live model weights reside during inference, and system DRAM is the slow "disk" you never want to spill onto.

## The Problem

Running large models on GPU hardware is not just a "get more GPUs" problem — it is a data-movement problem. A GPU performing a matrix multiply at 312 TFLOPS stalls the moment it cannot feed operands fast enough from memory. Understanding which bottleneck you are hitting — compute, memory bandwidth, or interconnect — determines whether you buy a faster GPU, more NVLink bandwidth, or simply change your batch size. Without this mental model you will spend $50k on hardware and still hit the same wall.

## Theory

### Arithmetic Intensity and the Roofline Model

The **Roofline model** classifies every kernel as either *memory-bound* or *compute-bound* using a single number: **arithmetic intensity**.

$$I = \frac{F}{B}$$

Where:
- $I$ — arithmetic intensity (FLOPs per byte transferred), the ratio that decides your fate
- $F$ — number of floating-point operations performed by the kernel (FLOPs)
- $B$ — number of bytes read from / written to memory (DRAM or HBM) during that kernel

The GPU's **ridge point** $I^*$ separates the two regimes:

$$I^* = \frac{P_{compute}}{P_{bandwidth}}$$

Where:
- $P_{compute}$ — peak throughput in FLOP/s (e.g., 312 × 10¹² FLOP/s for an H100 SXM)
- $P_{bandwidth}$ — peak HBM bandwidth in bytes/s (e.g., 3.35 × 10¹² bytes/s for an H100 SXM)

**Rule:** if $I < I^*$ the kernel is *memory-bound* (faster memory solves it); if $I > I^*$ it is *compute-bound* (more CUDA cores help).

**Numeric worked example — H100 SXM ridge point:**

```
P_compute   = 312 × 10¹² FLOP/s   (bf16 tensor core throughput)
P_bandwidth =   3.35 × 10¹² bytes/s   (HBM3 bandwidth)

I* = 312 × 10¹² / 3.35 × 10¹² ≈ 93 FLOPs/byte
```

Now consider a **vector addition** kernel over 1 M float32 elements:

```
F = 1 × 10⁶ FLOPs   (one add per element)
B = 3 × 4 × 10⁶ bytes   (read A, read B, write C — 3 arrays × 4 bytes each)
  = 12 × 10⁶ bytes

I = 10⁶ / (12 × 10⁶) ≈ 0.08 FLOPs/byte
```

`0.08 << 93` → this kernel is **heavily memory-bound**. Buying a GPU with more TFLOPS does nothing; you need higher HBM bandwidth.

---

### VRAM Footprint Model

Knowing whether a model fits on a GPU requires estimating its memory footprint before you try to load it.

$$M_{model} = P \times B_{param}$$

Where:
- $M_{model}$ — memory required for weights alone (bytes)
- $P$ — total number of trainable parameters
- $B_{param}$ — bytes per parameter: 4 for fp32, 2 for fp16/bf16, 1 for int8, 0.5 for int4

At training time you also need gradients and optimizer states. For the **Adam optimizer** (the default):

$$M_{train} = P \times (B_{param} + B_{grad} + 2 \times B_{moment})$$

Where:
- $B_{grad}$ — bytes per parameter for gradient storage (typically equals $B_{param}$)
- $B_{moment}$ — bytes per parameter per Adam moment (fp32 even in mixed-precision training = 4 bytes each, two moments)

**Numeric worked example — Llama 3 8B in bf16 training:**

```
P              = 8 × 10⁹ parameters
B_param        = 2 bytes   (bf16 weights)
B_grad         = 2 bytes   (bf16 gradients)
B_moment × 2   = 8 bytes   (two fp32 Adam moments)

M_train = 8 × 10⁹ × (2 + 2 + 8)
        = 8 × 10⁹ × 12
        = 96 × 10⁹ bytes
        ≈ 96 GB
```

A single H100 SXM has 80 GB VRAM — this model **does not fit on one GPU at training time**. You need either model parallelism, gradient checkpointing, or a quantised optimizer.

---

### Interconnect Hierarchy

| Link | Typical Bandwidth (per direction) | Analogy |
|---|---|---|
| HBM ↔ SM (on-chip) | 3.35 TB/s (H100) | L2 cache |
| NVLink 4.0 (GPU ↔ GPU, same node) | 900 GB/s total | Internal LAN (10 Gbit) |
| PCIe 5.0 ×16 (GPU ↔ CPU) | 64 GB/s | WAN uplink |
| InfiniBand NDR 400 (node ↔ node) | 50 GB/s per port | Data-centre backbone |

**NVLink vs PCIe in practice:** An `all-reduce` collective across 8 GPUs on the same NVLink domain moves gradients at ~900 GB/s. If those GPUs were connected only via PCIe through the host CPU, the same all-reduce degrades to ~64 GB/s — a **14× slowdown** for the same operation.

**InfiniBand** sits between nodes. Multi-node training stacks multiple InfiniBand ports (e.g., 8 × NDR = 400 GB/s per server). Even so, inter-node bandwidth is still ~2× slower than NVLink within a node, which is why data-parallel splits prefer intra-node and model-parallel splits tolerate inter-node.

## Python Implementation

```python
# Dependencies: torch>=2.1, nvidia-ml-py>=12.0

"""
Lesson 6.1 — GPU/TPU Hardware: Interconnect & Memory
Demonstrates:
  1. Live VRAM query via pynvml (falls back to simulated spec if no GPU present)
  2. VRAM footprint estimation for a parameterised model
  3. Roofline model: classify kernels as memory-bound or compute-bound
"""

import sys
from typing import NamedTuple


# ── GPU spec dataclass ──────────────────────────────────────────────────────

class GPUSpec(NamedTuple):
    name: str
    vram_bytes: int            # total VRAM in bytes
    hbm_bandwidth_gbs: float   # HBM bandwidth in GB/s
    peak_tflops_bf16: float    # peak bf16 tensor-core throughput in TFLOPS


# ── VRAM query (live if GPU present, simulated otherwise) ────────────────────

def query_gpu_spec() -> GPUSpec:
    """Return live GPU spec via pynvml, or fall back to an H100 simulation."""
    try:
        import pynvml  # nvidia-ml-py
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_bytes = mem_info.total
        # Bandwidth and TFLOPS are not exposed by nvml — use known lookup table
        SPECS: dict[str, tuple[float, float]] = {
            "H100": (3350.0, 312.0),
            "A100": (2000.0, 312.0),
            "A10G": ( 600.0,  31.2),
            "T4":   ( 300.0,   8.1),
            "V100": ( 900.0, 112.0),
        }
        bw, tflops = next(
            (v for k, v in SPECS.items() if k in name),
            (900.0, 312.0),  # default: assume A100-class
        )
        pynvml.nvmlShutdown()
        return GPUSpec(name=name, vram_bytes=vram_bytes,
                       hbm_bandwidth_gbs=bw, peak_tflops_bf16=tflops)
    except Exception:
        print("[INFO] No NVIDIA GPU detected — using simulated H100 SXM spec.\n")
        return GPUSpec(
            name="H100 SXM (simulated)",
            vram_bytes=80 * (1024 ** 3),   # 80 GB
            hbm_bandwidth_gbs=3350.0,
            peak_tflops_bf16=312.0,
        )


# ── Roofline model ───────────────────────────────────────────────────────────

def ridge_point(spec: GPUSpec) -> float:
    """Compute the ridge point I* = peak_flops / peak_bandwidth (FLOPs/byte)."""
    peak_flop_s = spec.peak_tflops_bf16 * 1e12          # FLOP/s
    peak_bw_byte_s = spec.hbm_bandwidth_gbs * 1e9       # bytes/s
    return peak_flop_s / peak_bw_byte_s


def classify_kernel(flops: float, bytes_transferred: float, ridge: float) -> str:
    """Return 'memory-bound' or 'compute-bound' for a kernel."""
    intensity = flops / bytes_transferred
    bound = "compute-bound" if intensity > ridge else "memory-bound"
    return f"I = {intensity:.2f} FLOPs/byte  →  {bound}  (ridge = {ridge:.1f})"


# ── VRAM footprint estimator ─────────────────────────────────────────────────

BYTES_PER_DTYPE: dict[str, float] = {
    "fp32":  4.0,
    "bf16":  2.0,
    "fp16":  2.0,
    "int8":  1.0,
    "int4":  0.5,
}


def estimate_inference_vram(num_params: int, dtype: str) -> float:
    """Return VRAM required for weights only (inference), in GB."""
    bpp = BYTES_PER_DTYPE[dtype]
    return num_params * bpp / (1024 ** 3)


def estimate_training_vram(num_params: int, weight_dtype: str) -> float:
    """Return approximate VRAM for AdamW training (weights + grads + 2× fp32 moments)."""
    bpp = BYTES_PER_DTYPE[weight_dtype]
    bytes_weights    = num_params * bpp
    bytes_grads      = num_params * bpp
    bytes_moments    = num_params * 4.0 * 2   # two fp32 Adam moments
    total_bytes      = bytes_weights + bytes_grads + bytes_moments
    return total_bytes / (1024 ** 3)


def fits_on_gpu(required_gb: float, spec: GPUSpec) -> str:
    available_gb = spec.vram_bytes / (1024 ** 3)
    margin_gb = available_gb - required_gb
    if margin_gb >= 0:
        return f"✓ Fits  ({available_gb:.0f} GB available, {margin_gb:.1f} GB headroom)"
    return f"✗ Does NOT fit  ({available_gb:.0f} GB available, {abs(margin_gb):.1f} GB short)"


# ── Interconnect bandwidth table ─────────────────────────────────────────────

INTERCONNECTS: list[tuple[str, float, str]] = [
    ("HBM ↔ SM (H100 on-chip)",    3_350.0, "L2 cache equivalent"),
    ("NVLink 4.0  (same node)",       900.0, "Internal LAN"),
    ("PCIe 5.0 ×16 (GPU ↔ CPU)",      64.0, "WAN uplink"),
    ("InfiniBand NDR 400 (node-node)", 50.0, "DC backbone"),
]


def print_interconnect_table() -> None:
    print("─" * 72)
    print(f"{'Link':<38} {'BW (GB/s)':>10}  {'FSD Analogy'}")
    print("─" * 72)
    for name, bw, analogy in INTERCONNECTS:
        print(f"{name:<38} {bw:>10,.0f}  {analogy}")
    print("─" * 72)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    spec = query_gpu_spec()
    print(f"GPU: {spec.name}")
    print(f"VRAM: {spec.vram_bytes / (1024**3):.0f} GB")
    print(f"HBM Bandwidth: {spec.hbm_bandwidth_gbs:,.0f} GB/s")
    print(f"Peak bf16 Throughput: {spec.peak_tflops_bf16:.0f} TFLOPS\n")

    # ── 1. Ridge point ──────────────────────────────────────────────────────
    ridge = ridge_point(spec)
    print(f"Ridge point (I*): {ridge:.1f} FLOPs/byte\n")

    # ── 2. Classify example kernels ─────────────────────────────────────────
    print("── Kernel Classification ──────────────────────────────────────────")
    elements = 1_000_000
    # Vector addition: 1 FLOP per element, 3 arrays × 4 bytes
    print(f"  vector_add (1M fp32):   "
          f"{classify_kernel(elements, 3 * 4 * elements, ridge)}")
    # GEMM (4096×4096×4096): ~2×N³ FLOPs, reads A+B writes C = 3N² × 4 bytes
    N = 4096
    gemm_flops = 2 * N ** 3
    gemm_bytes = 3 * N ** 2 * 4
    print(f"  matmul (4096³ fp32):    "
          f"{classify_kernel(gemm_flops, gemm_bytes, ridge)}")
    print()

    # ── 3. VRAM footprint for popular models ────────────────────────────────
    models: list[tuple[str, int]] = [
        ("Llama 3 8B",   8_000_000_000),
        ("Llama 3 70B", 70_000_000_000),
        ("GPT-4 (est.)", 1_800_000_000_000),
    ]
    print("── Inference VRAM Footprint ────────────────────────────────────────")
    print(f"  {'Model':<18} {'bf16 (GB)':>10} {'int8 (GB)':>10} {'int4 (GB)':>10}  Fit (bf16)?")
    print(f"  {'─'*18} {'─'*10} {'─'*10} {'─'*10}  {'─'*26}")
    for name, params in models:
        gb_bf16 = estimate_inference_vram(params, "bf16")
        gb_int8 = estimate_inference_vram(params, "int8")
        gb_int4 = estimate_inference_vram(params, "int4")
        fit = fits_on_gpu(gb_bf16, spec)
        print(f"  {name:<18} {gb_bf16:>10.1f} {gb_int8:>10.1f} {gb_int4:>10.1f}  {fit}")
    print()

    # ── 4. Training footprint (8B Llama) ────────────────────────────────────
    print("── Training VRAM Footprint — Llama 3 8B (AdamW, bf16) ─────────────")
    train_gb = estimate_training_vram(8_000_000_000, "bf16")
    print(f"  Weights + Grads + Adam moments ≈ {train_gb:.1f} GB")
    print(f"  {fits_on_gpu(train_gb, spec)}\n")

    # ── 5. Interconnect table ───────────────────────────────────────────────
    print("── Interconnect Bandwidth Hierarchy ────────────────────────────────")
    print_interconnect_table()

    # ── 6. NVLink vs PCIe all-reduce contrast ───────────────────────────────
    model_grad_gb = estimate_inference_vram(8_000_000_000, "bf16")  # same bytes as weights
    nvlink_bw = 900.0   # GB/s, NVLink 4.0
    pcie_bw   =  64.0   # GB/s, PCIe 5.0 ×16
    time_nvlink_ms = (model_grad_gb / nvlink_bw) * 1000
    time_pcie_ms   = (model_grad_gb / pcie_bw)   * 1000
    print(f"\n  all-reduce for 8B bf16 grads ({model_grad_gb:.0f} GB):")
    print(f"    via NVLink 4.0 : {time_nvlink_ms:.0f} ms")
    print(f"    via PCIe 5.0   : {time_pcie_ms:.0f} ms")
    print(f"    Speedup        : {time_pcie_ms / time_nvlink_ms:.1f}×")


if __name__ == "__main__":
    main()
```

**What to notice in the output:**

1. **Ridge point ~93 FLOPs/byte** for the H100 confirms that most LLM token-generation kernels (attention, layer-norm) fall *below* this threshold — they are memory-bound, not compute-bound. Doubling CUDA cores would not help; doubling HBM bandwidth would.
2. **Vector addition scores ~0.08 FLOPs/byte** vs **matmul at ~5,590 FLOPs/byte** — a 70,000× difference. Matmul is why GPUs exist; elementwise ops are an afterthought.
3. **The 8B model at 16 GB (bf16 inference)** fits comfortably on a single H100, but **96 GB for training** does not — demonstrating why parameter-efficient techniques like LoRA (node 4.2) and distributed training (node 6.2) are not optional at this scale.
4. **NVLink delivers ~14× faster all-reduce** than PCIe for the same gradient blob — the direct reason multi-GPU servers use NVLink domains rather than CPU-mediated transfers.

## Java Implementation

> **Java:** GPU profiling tooling is Python/CUDA only. No Java equivalent. The NVIDIA Management Library (NVML) exposes a C API; there is no maintained, production-grade JVM binding. Python's `pynvml` package is the standard interface used by NVIDIA's own tooling (nvidia-smi, DCGM).

## Key Takeaways

- **Arithmetic intensity ($I = F/B$) determines your bottleneck:** if $I$ is below the ridge point (~93 FLOPs/byte on H100), adding compute does nothing — you need more HBM bandwidth or you need to restructure data access patterns to reuse data already in the SM registers.
- **VRAM footprint for training is ~6× the raw weight size** in bf16 with AdamW (weights + gradients + two fp32 moments), which is why a 8B-parameter model that fits easily for inference requires more than a single 80 GB GPU for a full fine-tune.
- **NVLink is ~14× faster than PCIe for GPU-to-GPU transfers**, making intra-node all-reduce essentially "free" relative to compute time; InfiniBand bridges nodes but at 10–20× lower bandwidth than NVLink, so inter-node communication is always the distributed training bottleneck to minimise.

# Dependencies: torch>=2.1, nvidia-ml-py>=12.0
# Node: 6.1 — GPU/TPU Hardware — Interconnect & Memory
# Run: python starter.py
#
# No GPU required — falls back to a simulated H100 SXM spec automatically.

import math
from typing import NamedTuple


# ── GPU spec container ───────────────────────────────────────────────────────

class GPUSpec(NamedTuple):
    name: str
    vram_bytes: int            # total VRAM in bytes
    hbm_bandwidth_gbs: float   # HBM bandwidth (GB/s)
    peak_tflops_bf16: float    # peak bf16 tensor-core throughput (TFLOPS)


# ── Pre-defined constants (fully implemented — do not modify) ────────────────

BYTES_PER_DTYPE: dict[str, float] = {
    "fp32": 4.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}

# Each kernel dict: name (str), flops (int), bytes (int)
KERNELS: list[dict] = [
    {
        "name": "softmax_attention (2048² fp32)",
        # 5 ops/element: subtract-max, exp, row-sum, divide — over 2048×2048 matrix
        "flops": 5 * 2048 * 2048,
        # 3 fp32 arrays: input, intermediate exp buffer, output
        "bytes": 3 * 2048 * 2048 * 4,
    },
    {
        "name": "layer_norm (32×2048×4096 bf16)",
        # 5 ops/element: mean, variance, normalise, scale, shift
        "flops": 5 * 32 * 2048 * 4096,
        # 3 passes of bf16 data: input read, intermediate, output write
        "bytes": 3 * 32 * 2048 * 4096 * 2,
    },
    {
        "name": "matmul (4096³ fp32)",
        # Standard GEMM: 2 × M × N × K FLOPs
        "flops": 2 * 4096 ** 3,
        # Read A (M×K) + read B (K×N) + write C (M×N), all fp32
        "bytes": 3 * 4096 ** 2 * 4,
    },
    {
        "name": "embedding_lookup (vocab=32K, d=4096 bf16)",
        # Gather: one copy per element over (batch=32, seq=1024) tokens
        "flops": 32 * 1024 * 4096,
        # Read selected rows from table + write output, both bf16
        "bytes": 2 * 32 * 1024 * 4096 * 2,
    },
]

# (model_name, num_params) tuples
MODELS: list[tuple[str, int]] = [
    ("Llama3-7B",  7_000_000_000),
    ("Llama3-13B", 13_000_000_000),
    ("Llama3-70B", 70_000_000_000),
]

# (link_name, bandwidth_gbs) tuples — GPU-to-GPU interconnect tiers
INTERCONNECTS: list[tuple[str, float]] = [
    ("NVLink 4.0",      900.0),   # same NVLink domain, same physical server
    ("PCIe 5.0 ×16",    64.0),    # GPU-to-CPU-to-GPU through host memory
    ("InfiniBand NDR",  50.0),    # cross-node via network fabric
]


# ── Helper: query live GPU spec with pynvml fallback ────────────────────────
# Fully implemented — do not modify.

def query_gpu_spec() -> GPUSpec:
    """Return live GPU spec via pynvml, or fall back to a simulated H100 SXM spec."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name: str = pynvml.nvmlDeviceGetName(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_bytes: int = mem_info.total
        # nvml does not expose bandwidth/TFLOPS — use a lookup table for known GPUs
        _KNOWN: dict[str, tuple[float, float]] = {
            "H100": (3350.0, 312.0),
            "A100": (2000.0, 312.0),
            "A10G": ( 600.0,  31.2),
            "T4":   ( 300.0,   8.1),
            "V100": ( 900.0, 112.0),
        }
        bw, tflops = next(
            (v for k, v in _KNOWN.items() if k in name),
            (900.0, 312.0),  # default: A100-class fallback
        )
        pynvml.nvmlShutdown()
        return GPUSpec(
            name=name,
            vram_bytes=vram_bytes,
            hbm_bandwidth_gbs=bw,
            peak_tflops_bf16=tflops,
        )
    except Exception:
        print("[INFO] No NVIDIA GPU detected — using simulated H100 SXM spec.\n")
        return GPUSpec(
            name="H100 SXM (simulated)",
            vram_bytes=80 * (1024 ** 3),   # 80 GB
            hbm_bandwidth_gbs=3350.0,
            peak_tflops_bf16=312.0,
        )


# ── Helper: bytes → gigabytes ────────────────────────────────────────────────
# Fully implemented — do not modify.

def _to_gb(byte_count: int | float) -> float:
    """Convert a byte count to gigabytes (base-2)."""
    return byte_count / (1024 ** 3)


# ── Helper: GPU VRAM in GB ───────────────────────────────────────────────────
# Fully implemented — do not modify.

def _vram_gb(spec: GPUSpec) -> float:
    """Return the queried GPU's total VRAM in GB."""
    return _to_gb(spec.vram_bytes)


# ════════════════════════════════════════════════════════════════════════════
# STUDENT TODO — implement the five functions below
# Each raises NotImplementedError until you fill it in.
# The main() function calls them in order and prints labeled results.
# ════════════════════════════════════════════════════════════════════════════


def compute_ridge_point(spec: GPUSpec) -> float:
    """Return the Roofline ridge point I* = peak_flops / peak_bandwidth (FLOPs/byte).

    Args:
        spec: GPU specification from query_gpu_spec().

    Returns:
        Ridge point as a float in FLOPs/byte.

    Hints:
        - spec.peak_tflops_bf16 is in TFLOPS; multiply by 1e12 for FLOP/s.
        - spec.hbm_bandwidth_gbs is in GB/s; multiply by 1e9 for bytes/s.
        - Divide peak FLOP/s by peak bytes/s to get FLOPs/byte.
    """
    raise NotImplementedError("TODO: implement this")


def classify_kernels(kernels: list[dict], ridge: float) -> list[dict]:
    """Classify each kernel as 'memory-bound' or 'compute-bound' via the Roofline model.

    Args:
        kernels: List of kernel dicts, each with keys 'name', 'flops', 'bytes'.
        ridge:   Ridge point I* in FLOPs/byte from compute_ridge_point().

    Returns:
        List of result dicts, each with keys:
            'name'      (str)   — kernel name, unchanged from input
            'intensity' (float) — arithmetic intensity I = flops / bytes
            'bound'     (str)   — 'memory-bound' if I < ridge, else 'compute-bound'

    Hints:
        - Iterate over kernels; for each compute intensity = flops / bytes.
        - Compare intensity to ridge to choose the bound label.
        - Return a new list of dicts; do not modify the input list.
    """
    raise NotImplementedError("TODO: implement this")


def build_vram_budget(models: list[tuple[str, int]], spec: GPUSpec) -> list[dict]:
    """Compute VRAM requirements for each model across inference dtypes and training.

    Args:
        models: List of (model_name, num_params) tuples (use the global MODELS).
        spec:   GPU specification from query_gpu_spec().

    Returns:
        List of result dicts, one per model, each with keys:
            'model'          (str)   — model name
            'params'         (int)   — total parameter count
            'fp32_gb'        (float) — inference VRAM in GB for fp32
            'bf16_gb'        (float) — inference VRAM in GB for bf16
            'int8_gb'        (float) — inference VRAM in GB for int8
            'int4_gb'        (float) — inference VRAM in GB for int4
            'train_bf16_gb'  (float) — training VRAM in GB (bf16 weights + bf16 grads
                                        + two fp32 Adam moments = P × 12 bytes total)
            'fits_inf_bf16'  (bool)  — True if bf16_gb <= GPU VRAM
            'fits_train_bf16'(bool)  — True if train_bf16_gb <= GPU VRAM

    Hints:
        - Use BYTES_PER_DTYPE for inference: inference_bytes = params * bytes_per_dtype.
        - Training formula: train_bytes = params * (2 + 2 + 8)  [weights + grads + 2×fp32 moments].
        - Use _to_gb() to convert bytes to GB.
        - Use _vram_gb(spec) for the GPU capacity comparison.
    """
    raise NotImplementedError("TODO: implement this")


def compute_allreduce_times(
    models: list[tuple[str, int]],
    links: list[tuple[str, float]],
) -> list[dict]:
    """Compute all-reduce transfer time (ms) per model per interconnect link.

    Treat each model's bf16 inference VRAM as the gradient blob size (worst-case payload).

    Args:
        models: List of (model_name, num_params) tuples (use the global MODELS).
        links:  List of (link_name, bandwidth_gbs) tuples (use INTERCONNECTS).

    Returns:
        List of result dicts, one per model, each with keys:
            'model'    (str)            — model name
            'grad_gb'  (float)          — gradient blob size in GB (= bf16 inference VRAM)
            'times_ms' (dict[str,float])— mapping link_name → transfer time in ms

    Formula:
        time_ms = (grad_gb / bandwidth_gbs) * 1000

    Hints:
        - Compute grad_gb from num_params using BYTES_PER_DTYPE['bf16'] and _to_gb().
        - Iterate over links to fill the times_ms dict for each model.
    """
    raise NotImplementedError("TODO: implement this")


def generate_bottleneck_report(
    spec: GPUSpec,
    vram_results: list[dict],
    allreduce_results: list[dict],
) -> None:
    """Print a structured bottleneck report synthesising VRAM and interconnect data.

    For each model print:
      - Inference (bf16): fits / does not fit; if not, which dtype does fit (int8 or int4).
      - Training (bf16):  fits / needs N× GPUs (ceil(train_gb / vram_80gb)).
      - Interconnect:     InfiniBand NDR all-reduce latency in ms.
      - Binding constraint (one of three labels):
          'compute utilisation'              — training fits on 1 GPU
          'training VRAM'                    — training overflows but inference fits in bf16
          'inter-node InfiniBand bandwidth'  — bf16 inference also overflows (needs quantisation)

    Args:
        spec:             GPU specification from query_gpu_spec().
        vram_results:     Output of build_vram_budget().
        allreduce_results:Output of compute_allreduce_times().

    Hints:
        - Pair vram_results[i] with allreduce_results[i] (same order as MODELS).
        - Use _vram_gb(spec) as the per-GPU capacity for the "N× GPUs" calculation.
        - math.ceil(train_gb / vram_gb) gives the minimum GPU count.
        - Look up InfiniBand NDR time from result['times_ms']['InfiniBand NDR'].
        - Determine binding constraint:
            if vr['fits_train_bf16']  → 'compute utilisation'
            elif vr['fits_inf_bf16']  → 'training VRAM'
            else                      → 'inter-node InfiniBand bandwidth'
    """
    raise NotImplementedError("TODO: implement this")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── GPU spec ─────────────────────────────────────────────────────────────
    spec = query_gpu_spec()
    vram_gb = _vram_gb(spec)
    print(f"GPU: {spec.name}")
    print(f"VRAM: {vram_gb:.0f} GB | Bandwidth: {spec.hbm_bandwidth_gbs:,.0f} GB/s "
          f"| bf16 TFLOPS: {spec.peak_tflops_bf16:.0f}")

    # ── Task 1: Ridge point ───────────────────────────────────────────────────
    print("\n── Task 1: Ridge Point " + "─" * 52)
    ridge = compute_ridge_point(spec)
    print(f"Ridge point (I*): {ridge:.1f} FLOPs/byte")

    # ── Task 2: Kernel classification ────────────────────────────────────────
    print("\n── Task 2: Kernel Classification " + "─" * 41)
    kernel_results = classify_kernels(KERNELS, ridge)
    print(f"{'Kernel':<42} {'FLOPs/byte':>10}  {'Bound'}")
    print(f"{'─' * 42} {'─' * 10}  {'─' * 14}")
    for kr in kernel_results:
        print(f"{kr['name']:<42} {kr['intensity']:>10.2f}  {kr['bound']}")

    # ── Task 3: VRAM budget ───────────────────────────────────────────────────
    print("\n── Task 3: VRAM Budget (GB) " + "─" * 47)
    vram_results = build_vram_budget(MODELS, spec)
    header = (
        f"{'Model':<14} {'fp32(GB)':>8} {'bf16(GB)':>8} "
        f"{'int8(GB)':>8} {'int4(GB)':>8} {'Train-bf16(GB)':>14}  Fit-{vram_gb:.0f}GB(inf-bf16)?"
    )
    print(header)
    print("─" * len(header))
    for vr in vram_results:
        capacity_gb = _vram_gb(spec)
        headroom = capacity_gb - vr["bf16_gb"]
        if vr["fits_inf_bf16"]:
            fit_str = f"✓ fits ({headroom:.1f} GB headroom)"
        else:
            fit_str = f"✗ does not fit ({abs(headroom):.1f} GB short)"
        print(
            f"{vr['model']:<14} {vr['fp32_gb']:>8.1f} {vr['bf16_gb']:>8.1f} "
            f"{vr['int8_gb']:>8.1f} {vr['int4_gb']:>8.1f} {vr['train_bf16_gb']:>14.1f}  {fit_str}"
        )

    # ── Task 4: All-reduce transfer times ────────────────────────────────────
    print("\n── Task 4: All-Reduce Transfer Times " + "─" * 38)
    allreduce_results = compute_allreduce_times(MODELS, INTERCONNECTS)
    link_names = [name for name, _ in INTERCONNECTS]
    header_ar = (
        f"{'Model':<14} {'Grad(GB)':>8}"
        + "".join(f"  {n + '(ms)':<18}" for n in link_names)
    )
    print(header_ar)
    print("─" * len(header_ar))
    for ar in allreduce_results:
        times_str = "".join(
            f"  {ar['times_ms'][n]:>16.1f}  " for n in link_names
        )
        print(f"{ar['model']:<14} {ar['grad_gb']:>8.1f}{times_str}")

    # ── Task 5: Bottleneck report ─────────────────────────────────────────────
    print("\n── Task 5: Bottleneck Report " + "─" * 46)
    generate_bottleneck_report(spec, vram_results, allreduce_results)


if __name__ == "__main__":
    main()

# Dependencies: fastapi>=0.111, uvicorn>=0.29, httpx2>=2.0, numpy>=1.24
# Node: 5.1 — Inference Serving — High-Throughput Engines
# Run: python starter.py

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import fastapi
import numpy as np
from fastapi.testclient import TestClient


# ── Implemented helpers ───────────────────────────────────────────────────────
# These are complete. Read and use them inside your stubs.


def mock_generate(
    prompt: str,
    max_tokens: int,
    delay_per_token_ms: float = 2.0,
) -> dict[str, Any]:
    """Simulate token-by-token generation with a fixed per-token delay (CPU-safe)."""
    t0 = time.perf_counter()
    time.sleep(delay_per_token_ms * max_tokens / 1000.0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "text": f"[MOCK] {prompt[:40].strip()}... ({max_tokens} tokens)",
        "tokens_generated": max_tokens,
        "latency_ms": round(elapsed_ms, 2),
    }


def compute_throughput(total_tokens: int, elapsed_seconds: float) -> float:
    """Return tokens-per-second given total tokens and wall-clock elapsed time."""
    if elapsed_seconds <= 0:
        return 0.0
    return total_tokens / elapsed_seconds


def percentile(values: list[float], p: int) -> float:
    """Return the p-th percentile of a list of floats (nearest-rank method)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = max(0, min(int(len(sorted_vals) * p / 100), len(sorted_vals) - 1))
    return sorted_vals[idx]


# ── Student stubs ─────────────────────────────────────────────────────────────


def quantize_weights_int8(weights_fp32: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Apply symmetric per-tensor INT8 quantisation to a float32 weight array.

    Steps:
      1. Compute scale = max(abs(weights_fp32)) / 127.0
      2. Divide weights_fp32 by scale, round to the nearest integer.
      3. Clip values to [-127, 127] and cast the result to np.int8.
      4. Return (int8_weights, scale).

    Args:
        weights_fp32: float32 NumPy array of any shape.

    Returns:
        Tuple of (int8_weights: np.ndarray[int8], scale: float).
    """
    raise NotImplementedError("TODO: implement this")


def dequantize_weights(int8_weights: np.ndarray, scale: float) -> np.ndarray:
    """
    Recover a float32 weight array from its INT8 representation.

    Steps:
      1. Cast int8_weights to float32.
      2. Multiply element-wise by scale.
      3. Return the recovered float32 array.

    Args:
        int8_weights: INT8 NumPy array produced by quantize_weights_int8().
        scale: The scale factor returned alongside int8_weights.

    Returns:
        Recovered float32 NumPy array of the same shape as int8_weights.
    """
    raise NotImplementedError("TODO: implement this")


def compute_max_reconstruction_error(
    original: np.ndarray,
    recovered: np.ndarray,
) -> float:
    """
    Compute the max absolute reconstruction error as a percentage of the original range.

    Formula:
        error_pct = max(abs(original - recovered)) / max(abs(original)) * 100

    Args:
        original: The original float32 weight array before quantisation.
        recovered: The float32 array after quantise → dequantise.

    Returns:
        Error percentage (float). A correct INT8 implementation yields < 1.0%.
    """
    raise NotImplementedError("TODO: implement this")


def simulate_continuous_batching(
    batch_sizes: list[int],
    step_time_ms: float,
) -> dict[int, float]:
    """
    Simulate inference throughput (tok/s) for each batch size.

    Each decode step produces exactly 1 token per request in the batch.
    Throughput formula (from lesson 5.1 Theory):
        throughput = batch_size / (step_time_ms / 1000.0)

    Args:
        batch_sizes: List of batch sizes to evaluate, e.g. [1, 4, 16, 32].
        step_time_ms: Wall-clock time per decode step in milliseconds.

    Returns:
        Dict mapping each batch_size to its throughput in tokens/second.
        Example: {1: 50.0, 4: 200.0, 16: 800.0, 32: 1600.0}
    """
    raise NotImplementedError("TODO: implement this")


def build_app() -> fastapi.FastAPI:
    """
    Create and return a FastAPI application with two routes:

    GET /health
        Returns JSON: {"status": "ok", "backend": "mock"}

    POST /generate
        Request body (JSON):
            {"prompt": str, "max_tokens": int}   # max_tokens defaults to 32
        Response (JSON):
            {"text": str, "tokens_generated": int, "latency_ms": float}
        Internally call mock_generate(prompt, max_tokens) to produce the response.

    Hint: define Pydantic models for request and response bodies so FastAPI
    validates types automatically.

    Returns:
        A configured fastapi.FastAPI instance (do NOT call uvicorn.run here).
    """
    raise NotImplementedError("TODO: implement this")


def run_load_test(
    app: fastapi.FastAPI,
    num_requests: int = 20,
    max_tokens: int = 32,
) -> dict[str, float | bool]:
    """
    Fire num_requests concurrent POST /generate requests against the app and
    collect latency statistics.

    Steps:
      1. Use ThreadPoolExecutor to dispatch all requests concurrently.
         Each worker thread must create its own TestClient(app) instance to
         avoid sharing state across threads.
      2. Each worker sends:
             POST /generate  body={"prompt": "Load test prompt", "max_tokens": max_tokens}
         and records the "latency_ms" value from the JSON response body.
      3. Collect all HTTP status codes and latencies.
      4. Compute P50 and P99 with the percentile() helper.
      5. Set all_200 = True only if every response returned HTTP 200.

    Args:
        app: The FastAPI app instance returned by build_app().
        num_requests: Total number of concurrent requests to fire.
        max_tokens: max_tokens to request in each /generate call.

    Returns:
        {"p50_ms": float, "p99_ms": float, "all_200": bool}
    """
    raise NotImplementedError("TODO: implement this")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    # ── Tasks 1 & 2: INT8 Quantisation ───────────────────────────────────────
    print("── INT8 Quantisation ──")
    rng = np.random.default_rng(42)
    weights_fp32 = rng.uniform(-1.0, 1.0, size=(64, 64)).astype(np.float32)

    int8_weights, scale = quantize_weights_int8(weights_fp32)
    recovered = dequantize_weights(int8_weights, scale)
    error_pct = compute_max_reconstruction_error(weights_fp32, recovered)

    print(f"Scale factor     : {scale:.6f}")
    print(f"INT8 dtype check : {int8_weights.dtype}")
    print(f"Max recon. error : {error_pct:.4f}%  (must be < 1.0%)")

    assert int8_weights.dtype == np.int8, "int8_weights must have dtype np.int8"
    assert error_pct < 1.0, (
        f"Reconstruction error {error_pct:.4f}% exceeds 1.0% — check your scale formula"
    )

    # Memory footprint reference table
    param_count = 7_000_000_000
    print(f"FP16 VRAM (7B)   : {param_count * 2 / 1e9:.1f} GB")
    print(f"INT8 VRAM (7B)   : {param_count * 1 / 1e9:.1f} GB  (2× compression)")
    print(f"INT4 VRAM (7B)   : {param_count * 0.5 / 1e9:.1f} GB  (4× compression)")

    # ── Task 3: Throughput Simulation ────────────────────────────────────────
    print("\n── Continuous Batching Simulation ──")
    throughput_map = simulate_continuous_batching(
        batch_sizes=[1, 4, 16, 32],
        step_time_ms=20.0,
    )
    for batch_size, throughput in sorted(throughput_map.items()):
        print(f"Batch={batch_size:>2}  → {throughput:>8.1f} tok/s")

    expected = {1: 50.0, 4: 200.0, 16: 800.0, 32: 1600.0}
    assert throughput_map == expected, (
        f"Throughput map mismatch.\nExpected: {expected}\nGot: {throughput_map}"
    )

    # ── Tasks 4 & 5: FastAPI App + Load Test ─────────────────────────────────
    print("\n── FastAPI App ──")
    app = build_app()
    client = TestClient(app)

    health_resp = client.get("/health")
    assert health_resp.status_code == 200, f"GET /health returned {health_resp.status_code}"
    print(f"GET /health  : {health_resp.json()}")

    gen_resp = client.post("/generate", json={"prompt": "Hello world", "max_tokens": 16})
    assert gen_resp.status_code == 200, f"POST /generate returned {gen_resp.status_code}"
    gen_data = gen_resp.json()
    assert gen_data["tokens_generated"] == 16, (
        f"Expected tokens_generated=16, got {gen_data['tokens_generated']}"
    )
    print(
        f"POST /generate : tokens={gen_data['tokens_generated']}, "
        f"latency={gen_data['latency_ms']:.1f} ms"
    )

    print("\n── Load Test (20 concurrent requests) ──")
    load_results = run_load_test(app, num_requests=20, max_tokens=32)
    print(f"P50 latency      : {load_results['p50_ms']:.1f} ms")
    print(f"P99 latency      : {load_results['p99_ms']:.1f} ms")
    print(f"All requests 200 : {load_results['all_200']}")

    assert load_results["all_200"], "Some /generate requests did not return HTTP 200"
    assert load_results["p99_ms"] < 5000, (
        f"P99 latency {load_results['p99_ms']:.1f} ms exceeded 5000 ms ceiling"
    )

    print("\nAll assertions passed ✓")


if __name__ == "__main__":
    main()

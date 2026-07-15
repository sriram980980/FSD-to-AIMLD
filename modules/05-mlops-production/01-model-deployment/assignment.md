# Assignment 5.1 — Deploy Quantized Model via API Endpoint

## Objective

Build a FastAPI inference server that applies INT8 quantization to model weights, models
continuous-batching throughput scaling, and measures P50/P99 latency under concurrent load —
proving you can reason about and instrument a production inference stack without a GPU.

## Background

Lesson 5.1 showed that naive sequential inference leaves ~95 % of GPU compute idle, and that
two techniques close that gap: **PagedAttention / continuous batching** (maximising effective
batch size by recycling freed KV-cache memory pages) and **INT8/INT4 quantisation** (cutting
VRAM footprint by 2–4×). This assignment makes both techniques concrete — you will implement
the quantisation arithmetic, simulate the throughput formula, and wire up a real HTTP server
so you can observe latency under concurrent load. Refer to the Theory section of `lesson.md`
for the formulas you will implement here.

## Setup

```bash
pip install "fastapi>=0.111" "uvicorn>=0.29" "httpx2>=2.0" "numpy>=1.24"
```

> **GPU not required.** The entire assignment runs on CPU using mock inference.
> If you have a GPU with ≥ 8 GB VRAM, you can optionally swap the mock with a real vLLM
> engine (`pip install "vllm>=0.4"`) — the API contract is identical.

## Tasks

### Task 1 — Implement INT8 Quantisation

Open `starter/starter.py` and implement `quantize_weights_int8()`:

1. Compute `scale = max(abs(weights_fp32)) / 127.0` — this maps the full FP32 range to
   the INT8 range `[-127, 127]`.
2. Divide every weight by `scale`, round to the nearest integer, clip to `[-127, 127]`,
   and cast to `np.int8`.
3. Return `(int8_weights, scale)`.

### Task 2 — Implement Dequantisation and Error Measurement

Implement `dequantize_weights()` and `compute_max_reconstruction_error()`:

1. `dequantize_weights`: cast `int8_weights` to `float32`, multiply element-wise by
   `scale`, return the recovered array.
2. `compute_max_reconstruction_error`: compute
   `max(abs(original - recovered)) / max(abs(original)) * 100` and return it as a
   percentage. The assertion in `main()` will fail if this exceeds **1.0 %** — confirming
   quantisation fidelity.

### Task 3 — Model Throughput Under Continuous Batching

Implement `simulate_continuous_batching()`:

1. For each `batch_size` in the input list, apply the throughput formula from the lesson:
   `throughput = batch_size / (step_time_ms / 1000.0)` — one token per request per step.
2. Return a dict mapping each `batch_size` to its `throughput` in tokens/second.

### Task 4 — Build the FastAPI Inference App

Implement `build_app()` so it returns a `fastapi.FastAPI` instance with two routes:

| Route | Method | Behaviour |
|-------|--------|-----------|
| `/health` | `GET` | Return `{"status": "ok", "backend": "mock"}` |
| `/generate` | `POST` | Accept JSON body `{"prompt": str, "max_tokens": int}`, call `mock_generate()`, return `{"text": str, "tokens_generated": int, "latency_ms": float}` |

Use the pre-built `mock_generate()` helper — do **not** add any external model calls.

### Task 5 — Concurrent Load Test

Implement `run_load_test()` to stress-test your app:

1. Create a `TestClient` wrapping the app.
2. Use `ThreadPoolExecutor` to fire `num_requests` concurrent `POST /generate` requests
   (each thread creates its own `TestClient` instance to avoid shared-state issues).
3. Collect the `latency_ms` field from each successful response body.
4. Use the `percentile()` helper to compute P50 and P99.
5. Return `{"p50_ms": float, "p99_ms": float, "all_200": bool}` where `all_200` is `True`
   only if every response had HTTP status 200.

### Task 6 — Run and Verify

Run the completed starter:

```bash
cd modules/05-mlops-production/01-model-deployment/starter
python starter.py
```

Confirm all five assertions pass (no `AssertionError`). Record your P50/P99 values.

## Expected Output

```
── INT8 Quantisation ──
Scale factor     : 0.007937
INT8 dtype check : int8
Max recon. error : 0.XXXX%  (must be < 1.0%)
FP16 VRAM (7B)   : 14.0 GB
INT8 VRAM (7B)   : 7.0 GB  (2× compression)
INT4 VRAM (7B)   : 3.5 GB  (4× compression)

── Continuous Batching Simulation ──
Batch= 1  →    50.0 tok/s
Batch= 4  →   200.0 tok/s
Batch=16  →   800.0 tok/s
Batch=32  → 1600.0 tok/s

── FastAPI App ──
GET /health  : {'status': 'ok', 'backend': 'mock'}
POST /generate : tokens=16, latency=XX.X ms

── Load Test (20 concurrent requests) ──
P50 latency      : XX.X ms
P99 latency      : XX.X ms
All requests 200 : True
```

**Exact values:**

| Output | Acceptable range |
|--------|-----------------|
| `Max recon. error` | < 1.0 % |
| Batch=1 throughput | exactly 50.0 tok/s |
| Batch=4 throughput | exactly 200.0 tok/s |
| Batch=16 throughput | exactly 800.0 tok/s |
| Batch=32 throughput | exactly 1600.0 tok/s |
| `all_200` | `True` |
| P99 latency | < 5000 ms |

## Evaluation Criteria

- [ ] `quantize_weights_int8()` returns an `np.int8` array and a positive float scale
- [ ] `compute_max_reconstruction_error()` returns a value strictly less than 1.0 %
- [ ] `simulate_continuous_batching()` returns exactly `{1: 50.0, 4: 200.0, 16: 800.0, 32: 1600.0}`
- [ ] `GET /health` returns HTTP 200 with `{"status": "ok", "backend": "mock"}`
- [ ] `POST /generate` returns HTTP 200 with `tokens_generated` equal to the requested `max_tokens`
- [ ] `run_load_test()` returns `all_200 = True` for 20 concurrent requests
- [ ] All five assertions in `main()` pass without modification

## Extension Challenge

Replace the mock inference engine with a real quantised model served via vLLM:

1. Install vLLM on a GPU machine: `pip install "vllm>=0.4"`.
2. Update `build_app()` to initialise a `vllm.engine.async_llm_engine.AsyncLLMEngine` on
   startup with `model="facebook/opt-125m"` and `dtype="float16"`.
3. Update the `/generate` route to await token generation from the real engine instead of
   calling `mock_generate()`.
4. Re-run your `run_load_test()` (increase `num_requests` to 100) and compare real P50/P99
   values against the mock baseline. Quantify the throughput improvement when you increase
   `max_tokens` from 32 to 128 — does P99 scale linearly? Explain why or why not.

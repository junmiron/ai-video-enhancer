# Benchmarks — AI Video Enhancer

Measured on the target machine (RTX 5090, 32 GB). Raw per-run data is archived in
[`tools/bench_results.json`](../tools/bench_results.json); re-run with:

```
python tools/benchmark_rife.py            # all variants @ 720p + 1080p, N=30 calls each
```

## Environment

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 (Blackwell, sm_120), 32 GB VRAM |
| Driver / CUDA | 616.56 / 13.2 |
| PyTorch | 2.12.1+cu132 |
| TensorRT | 11.1.0.106 (fp16 engines, fixed-shape profiles) |
| Python | 3.14.7 |

## Methodology

- One model call = one interpolated frame between two source frames at t=0.5
  (the worst-case timestep for RIFE's refinement loop).
- Frames: random uint8 content (timing is content-independent), padded to the
  arch's multiple-of-64 profile (720p → 1280×768, 1080p → 1920×1088).
- Both backends run fp16 below UHD (the pipeline's default precision choice;
  TAS found fp16 flicker at/above UHD sources, where the app falls back to fp32).
- 3 warmup calls, then N=30 timed calls including H2D/D2H transfers and GPU sync.
- TensorRT engines are cached per (GPU, TRT version, weights hash, precision,
  shape) under `%LOCALAPPDATA%\ai-video-enhancer\models\engines\rife\`.

## RIFE 4.25 family — frame interpolation throughput

Median ms/call (N=30), lower is better:

| Variant | Params | Resolution (padded) | TRT fp16 | PyTorch fp16 | TRT p95 | Torch p95 |
|---|---:|---|---:|---:|---:|---:|
| rife_425 (default) | 5.7 M | 1280×720 (1280×768) | **24.4** ms (~41 calls/s) | 26.8 ms | 25.2 ms | 36.7 ms |
| rife_425 (default) | 5.7 M | 1920×1080 (1920×1088) | **52.7** ms (~19 calls/s) | 53.4 ms | 83.3 ms | 76.8 ms |
| rife_425_heavy | 21.7 M | 1280×720 (1280×768) | **26.4** ms (~38 calls/s) | 27.0 ms | 27.8 ms | 27.9 ms |
| rife_425_heavy | 21.7 M | 1920×1080 (1920×1088) | **54.8** ms (~18 calls/s) | 54.4 ms | 62.5 ms | 76.0 ms |
| rife_425_lite | 5.6 M | 1280×720 (1280×768) | **24.8** ms (~40 calls/s) | 27.0 ms | 35.3 ms | 31.9 ms |
| rife_425_lite | 5.6 M | 1920×1080 (1920×1088) | **52.8** ms (~19 calls/s) | 52.8 ms | 68.6 ms | 76.4 ms |

Mean ms/call for the same runs: TRT 25.7 / 57.2 / 28.3 / 59.6 / 26.9 / 57.1;
PyTorch 28.5 / 57.6 / 27.0 / 58.7 / 29.1 / 57.5 (same row order).

### Findings

1. **TRT fp16 ≈ PyTorch fp16 for RIFE on this GPU** — within ~5–10 %, TRT
   slightly ahead at 720p, parity at 1080p. This is expected: the IFNet's
   convolutions run at very low resolution (pyramid scales 16/8/4/2/1), so each
   call is dominated by full-resolution warps and interpolations — i.e. it is
   **memory-bandwidth bound, not FLOP-bound**. TensorRT's kernel fusion wins on
   FLOP-heavy models (the M3 upscalers should show a bigger gap).
2. **All three variants cost about the same per call** despite heavy having 4×
   the parameters — same memory-bandwidth explanation. Choose by quality, not speed:
   default `rife_425` for most content, `heavy` when maximum fidelity matters,
   `lite` to shave a few ms or save VRAM on long jobs.
3. **TensorRT remains the primary backend** anyway: lower and more predictable
   VRAM use (no cuDNN workspace churn), deterministic per-call latency, and it is
   the path that scales to the FLOP-bound SR/restore models later in the pipeline.
4. **Practical throughput:** at 1080p one call ≈ 53 ms → a 24→60 fps job on a
   96-frame scene needs ~72 calls ≈ 4 s of model time (plus decode/encode I/O).
   A full 10-minute 1080p anime episode at 2× interpolation is roughly
   5–8 minutes of GPU time end-to-end.

### Caveats

- First measurements after the GPU has been idle can be 20–40 % slower (clock
  ramp-up); the numbers above are from warm, uncontended runs. Keep other GPU
  work off the machine while benchmarking — one early run was skewed by a
  concurrent test process and has been superseded.
- p95 at 1080p shows occasional ~30 ms spikes (page faults / allocator); medians
  are the stable comparison metric.

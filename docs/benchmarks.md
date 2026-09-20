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

## x265 10-bit anime preset — encode defaults review (M7 A4)

Reviewed the `x265-10bit-anime` final-encode preset on flat-color anime content:
`samples/m5-demo/anime_enhanced.mp4` (1280×720@60, 179 frames — the M5.x FI+SR
demo output), CRF 19 throughout. ffmpeg 9.0.1 full build (libx265 v4), quality
measured against the encode input with VMAF (default model), SSIM and PSNR.

| Config | x265-params (beyond `preset slow`) | Size KB | Encode s | VMAF | SSIM All | PSNR avg |
|---|---|---:|---:|---:|---:|---:|
| A current | `bframes=8:rc-lookahead=60` | 529.0 | 1.9 | 96.597 | 0.998094 (27.20 dB) | 53.19 dB |
| B deep decision | A + `rd=6:rdoq-level=2:ref=4:limit-modes=0` | 533.9 (+0.9%) | **4.2 (+121%)** | 96.513 (−0.08) | 0.997936 (−0.35 dB) | 52.85 dB (−0.33) |
| C anime AQ | A + `aq-mode=3:aq-strength=1.0:rdoq-level=2` | 589.9 (+11.5%) | 1.9 | **96.779** | **0.998597 (+1.33 dB)** | **54.42 dB (+1.24)** |
| D isolation: rdoq only | A + `rdoq-level=2` | 529.0 (=A) | 1.9 | =A (bit-identical metrics) | =A | =A |
| E isolation: AQ only | A + `aq-mode=3:aq-strength=1.0` | 589.9 (=C) | 1.9 | =C (bit-identical metrics) | =C | =C |

### Findings

1. **Adopted: `bframes=8:rc-lookahead=60:aq-mode=3`** (now the preset default in
   [`aive/stages/encode.py`](../aive/stages/encode.py)). aq-mode is x265's adaptive
   quantization selector (x265 manual: 0=off, 1=uniform cost, 2=variance [default],
   3=bias variance); mode 3 redistributes bits toward flat regions and line art —
   exactly the anime content profile. Measured +1.3 dB SSIM / +1.24 dB PSNR /
   VMAF 96.78 vs 96.60 at identical encode speed, for ~+11% file size (still tiny:
   this clip is 590 KB). The UI size estimate was bumped to match
   (`_BPP_BASE` 0.10 → 0.11 in `aive/ui/estimates.py`).
2. **Deeper mode decision (rd=6, limit-modes=0, ref=4) is a loss on this content**:
   2.2× the encode time for slightly *worse* metrics — at CRF 19 preset `slow`
   already spends enough decision effort; the extra search just re-allocates bits
   differently without perceptual gain. Rejected.
3. **rdoq-level=2 is a no-op here** (D bit-matches A): at CRF 19 quantization is
   gentle enough that RDOQ's coefficient decisions don't change. Dropped from the
   preset to keep it minimal; revisit if the default CRF ever moves up.
4. Caveats: single short synthetic clip near the quality ceiling (VMAF ~96+), and
   VMAF's default model is live-action-trained — but all three metrics agree on the
   ordering C > A > B, so the conclusion is robust. Re-benchmark on real episodes
   if encode-time or size regressions are ever reported.

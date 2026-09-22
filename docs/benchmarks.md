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

## GMFSS — frame interpolation throughput (M2.x)

GMFSS is **CUDA-only** (the softsplat splatting is an `index_add_` scatter with
data-dependent boundary masks that ONNX cannot represent), so there is no TensorRT
column. Measured PyTorch fp16 below UHD; above-1080p sources additionally run the
flow at half resolution (`scale=0.5`) and, at/above UHD, in fp32 — both handled by
`GMFSSRunner`. Re-run with `python tools/benchmark_gmfss.py`.

ms/call (N=15), lower is better:

| Model | Resolution (padded) | Backend | Precision | Flow scale | Mean ms/call | Median | p95 | Calls/s |
|---|---|---|---|---:|---:|---:|---:|---:|
| gmfss | 1280×720 (1280×768) | pytorch | fp16 | 1.0 | **100.9** ms (~9.9 calls/s) | 101.2 ms | 107.2 ms |
| gmfss | 1920×1080 (1920×1088) | pytorch | fp16 | 1.0 | **198.7** ms (~5.0 calls/s) | 184.5 ms | 238.9 ms |

### Findings

1. **GMFSS is ~4× slower per call than RIFE at the same resolution** (≈101 vs ≈24
   ms at 720p, ≈199 vs ≈53 ms at 1080p). It carries a GMFlow transformer plus
   multi-scale feature extraction and three softsplat warps per frame — FLOP- and
   memory-heavy in ways RIFE's lightweight IFNet is not. Choose RIFE for speed,
   GMFSS when its anime-oriented quality is wanted.
2. **The per-pair reuse cache matters**: flow + metrics + features are computed once
   per frame pair and reused across every intermediate slot of that pair, so a 2×
   job costs one `reuse()` plus N `inference()` calls rather than N full forwards.
3. **Practical throughput:** at 1080p one call ≈ 199 ms → a 24→60 fps job on a
   96-frame scene needs ~72 calls ≈ 14 s of model time (vs ~4 s for RIFE).

### Caveats

- These runs had a concurrent GPU workload present (a local LLM server holding most
  of the VRAM), so absolute latency is likely somewhat higher than fully idle,
  uncontended measurements — treat them as relative data points. N=15 (vs the RIFE
  table's N=30) for the same reason; medians are the stable metric.

## M2M — frame interpolation throughput (M2.x)

M2M is **CUDA-only** (the many-to-many splatting is an `index_add_` scatter with
data-dependent boundary masks that ONNX cannot represent), so there is no TensorRT
column. Measured PyTorch fp16 below UHD / fp32 at/above it (`M2MRunner`). The model
self-pads (replicate) to a multiple of 64 internally; the "padded" column shows that
internal pad. Re-run with `python tools/benchmark_m2m.py`.

ms/call (N=30), lower is better:

| Model | Resolution (padded) | Backend | Precision | Mean ms/call | Median | p95 | Calls/s |
|---|---|---|---:|---:|---:|---:|---:|
| m2m | 1280×720 (1280×768) | pytorch | fp16 | **191.8** ms (~5.2 calls/s) | 189.0 ms | 203.3 ms |
| m2m | 1920×1080 (1920×1088) | pytorch | fp16 | **226.4** ms (~4.4 calls/s) | 223.0 ms | 241.9 ms |

### Findings

1. **M2M is the slowest FI model per call** (≈192 vs ≈101 GMFSS / ≈24 RIFE at 720p;
   ≈226 vs ≈199 / ≈53 at 1080p). The cost is the many-to-many splat: each slot runs
   occlusion photo-metrics plus **eight full-resolution warps** (4 branches × 2
   directions), whereas GMFSS warps at half resolution. fp16 is safe without extra
   guards — the checkpoint's `paramAlpha ≈ 2.9` bounds the splat exponent by e^2.9,
   far below fp16 max; the cost volume additionally accumulates in fp32 as insurance.
2. **The per-pair reuse cache is M2M's design advantage at high multipliers.** The
   benchmark cycles *distinct* pairs so every timed call pays for a fresh flow
   estimate (PWCNet bidir + MotionRefineNet) plus one splat — the realistic worst
   case per slot. In a real 4×–8× job, consecutive intermediate slots of the same
   pair hit the cache and pay only for photo-metrics + splat, so the amortized
   cost-per-output-frame drops as the multiplier grows. That is why M2M is targeted
   at high multipliers despite its higher per-call cost.
3. **Practical throughput:** at 1080p one fresh-pair call ≈ 226 ms → a 2× job on a
   96-frame scene needs ~95 calls ≈ 21 s of model time (vs ~4 s RIFE / ~14 s GMFSS);
   a 4× job's later slots are cheaper than this number implies.

### Caveats

- These runs had a concurrent GPU workload present (a local LLM server holding most
  of the VRAM), so absolute latency is likely somewhat higher than fully idle,
  uncontended measurements — treat them as relative data points. N=30; medians are
  the stable metric.

## FILM — frame interpolation throughput & backend deviation (M2.x)

FILM is **TensorRT-only** (TF2 model, no torch port), so there is no PyTorch
column. The runner builds one cached fixed-shape engine per resolution from the
bundled dynamic ONNX (min=opt=max profile — memory scales with video size like
RIFE); each timed call is a full stateless inference (no reuse cache to amortize).
fp16 below UHD / fp32 at/above 1080p sources (project flicker rule, same as RIFE).
Re-run with `python tools/benchmark_film.py`.

ms/call (N=30), lower is better:

| Model | Resolution (padded) | Backend | Precision | Mean ms/call | Median | p95 | Calls/s |
|---|---|---|---:|---:|---:|---:|---:|
| film | 640×360 (640×384) | tensorrt | fp16 | **27.9** ms (~35.8 calls/s) | 27.6 ms | 28.9 ms |

### Findings

1. **Speed class:** ~28 ms/call @640×384 — already in the same class as RIFE's
   ≈24 ms at *twice* the resolution (720p); M2M/GMFSS were measured at 720p+
   (≈192 / ≈101 ms). A same-resolution comparison follows once the 720p engine can
   be built under current VRAM constraints (see caveats).
2. **Real-content backend deviation is sub-perceptual** (m5-demo frames, TRT-fp16
   vs ORT-CPU fp32, both sides clamped+rounded to uint8 — what a viewer actually
   sees):
   - *Pure fp16 precision* (identical graph): ≤1 gray level at **every** pixel;
     mean ≈0.008/255; 0% of pixels >1. Imperceptible.
   - *Total*, including the TRT-mandated `round_prefer_floor` nearest sampling in
     four FusionNet-decoder resizes: mean 0.09–0.13/255 across four frame pairs;
     worst-case isolated-pixel spikes up to ~65–81/255 on high-frequency edges
     (hair lines); ~1.5–1.7% of pixels >1 gray level, ~0.7% >2. The deviation is
     deterministic per (frame pair, t) — it adds no temporal variance relative to
     the TF original, so M4's flicker detector sees nothing new.
   - *Measurement gotcha:* FILM's raw output legitimately exceeds [0,1] by up to
     ~11% at edge pixels on **both** backends; comparing clamped uint8 against
     unclamped float inflates the "fp16-only" max diff to ~29/255. The true
     user-visible number is ≤1 — clamp both sides before comparing.
3. **E2E:** m5-demo ×2 through the real `InterpolateStage`: 90→179 frames,
   endpoints pixel-exact, ~35 ms/call incl. pad/H2D/D2H at 640×384.

### Caveats

- Concurrent GPU workload present (a local LLM server holding most of the VRAM):
  the 720p engine build OOMs with ~2.6 GiB free, so only the 360p-class row is
  measured; the 1280×720 / 1920×1080 rows are deferred until more VRAM frees (the
  runner surfaces a clear error naming the requirement). N=30; medians are the
  stable metric.

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

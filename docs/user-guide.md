# User Guide — AI Video Enhancer

A desktop app that upscales video with state-of-the-art AI models, adds frame
interpolation, and then **automatically detects artifacts introduced by those
AI stages and repairs them** — handing back a clean file plus a transparent
artifact report. Personal-use Windows app; requires an NVIDIA GPU (CUDA +
TensorRT).

```
input video ──► scene detection ──► frame interpolation ──► upscale ──► artifact QA ──► encode ──► output
                    (cuts never crossed)      (RIFE/GMFSS/M2M)        (CUGAN/ESRGAN/…)  (detect+repair)  (H.265 10-bit)
                                                                                                    │
                                                                                          optional depth-map pass
```

Stage order is configurable (interpolate-first vs upscale-first); artifact QA
always runs last, right before encoding.

---

## First launch

1. Start the app — the top banner shows your GPU and runtimes, e.g.
   `RTX 5090 — CUDA ready · TensorRT 11.x · VRAM 32 GiB`. If it says *No NVIDIA
   GPU detected*, AI models cannot run on this machine (the app will not queue jobs).
2. On the **Home** page, drop a video onto the window (or use *Browse*). The first
   job for a preset triggers a one-time download of that preset's default models
   (progress shown in the first-run card; weights are cached under
   `%LOCALAPPDATA%\ai-video-enhancer\models\` and never re-downloaded).
3. No video handy? Use **Generate sample clip** on the Home page to create a short
   anime-style test clip, then run *Fast Preview* on it to see the whole pipeline
   in under a minute.

## Basic workflow

1. **Home**: add one or more videos (they form a queue; each row shows its status).
2. Pick a **preset** in Settings (or just use the default — Anime).
3. Click **Enhance all**. The Processing page shows per-stage progress, overall
   ETA, VRAM/RAM meters and a live log. *Pause / Resume / Cancel* work at any
time; cancelling is safe — finished jobs are never touched.
4. When done: the **Report** page shows what artifacts were found and fixed, and
   the **Preview** page lets you compare before/after frame-by-frame with artifact
   markers on the timeline.

## Presets

Presets set sensible defaults for a content type; every field stays overridable in
the settings panel, and your tweaks survive switching presets for unrelated groups.

| Preset | Upscale | Interpolation | QA | Encode | Notes |
|---|---|---|---|---|---|
| **Anime** (default) | ShuffleCUGAN x2 | RIFE 4.25 → 60 fps, light de-duplication of held frames | auto + AI repairs | H.265 10-bit anime profile, CRF 19 | Interpolate-first (fast, TAS-proven order). Depth model when enabled: Limbo v2 |
| **Live Action** | Real-ESRGAN x4plus (x2) | RIFE 4.25 → 60 fps, no de-duplication | auto + AI repairs | H.265 10-bit anime profile, CRF 19 | Upscale-first: RIFE runs at target resolution for best motion fidelity (slower). Depth model when enabled: Depth-Anything V2 Small |
| **Mixed / Unknown** | Real-ESRGAN x4plus (x2) | RIFE 4.25 ×2 multiplier | auto + AI repairs | H.265 10-bit anime profile, CRF 19 | Safe defaults when content type is unclear |
| **Fast Preview** | ShuffleCUGAN x2 | off | detect-only (report, no repairs) | NVENC H.265 10-bit, CRF 23 | Quick test run before committing to a long job |

## Settings reference

### Upscale
- **Enable upscaling** — off = pass-through at source resolution.
- **Model** — see [Models](#models-and-licenses). Hover any entry for its license and notes; `✓` means the weights are already cached.
- **Scale** — target output scale. When it is an exact power of the model's native scale (e.g. ×4 or ×8 on an x2 model), the app chains that many AI passes so every step gets real super-resolution detail (true Nx) rather than one pass + resample. Any other target runs a single pass at the native scale, then a high-quality LANCZOS resize to the requested size (e.g. the Live Action preset runs the x4 Real-ESRGAN model at a 2× target).

### Frame interpolation
- **Mode: Target FPS** — pick a target frame rate (24/30/50/60/120); the app computes the needed multiplier per scene.
- **Mode: Multiplier** — fixed ×2 / ×3 / ×4 on top of the source rate.
- **Model** — RIFE 4.25 family (with a *Quality* selector: Lite / Balanced / Heavy), or the experimental GMFSS and M2M. Any multiplier works with all models (continuous timestep).
- **De-duplicate held frames (threshold)** — anime often holds identical frames; values >0 skip interpolating pairs that differ by less than the threshold (Anime preset: 0.01).

### Artifact detection & repair
The differentiator stage. Runs after upscale+interpolation, before encoding.
- **Mode:**
  - *Auto* — detect everything and apply repairs automatically.
  - *Detect only* — measure and report, never modify frames (Fast Preview uses this).
  - *Manual* — you choose which artifact types are active via the checkboxes.
- **Sensitivity sliders** (per artifact, %): higher = more aggressive detection.
  Defaults (50%) were calibrated so that on clean content false positives stay
  under ~10% while injected artifacts are caught at ≥90%.
- **AI repairs** — enables learned repair models (SCUNet for flicker, deepDeband-f
  for banding) with automatic CV fallback when VRAM or weights are unavailable.
  Line-art repair (FastLineDarken) is pure OpenCV and always available.
- **Show report on completion** — jump straight to the Report page when a job finishes.

### Depth map (optional)
Generates an extra depth output *after* the enhanced file is written (a failure
here never loses your enhanced video).
- **Model**: Limbo v1/v2 (anime-tuned, temporally consistent streaming),
  Depth-Anything V2 Small (per-frame, general content), or experimental
  Depth-Anything-3 small/base.
- **Output**: 10-bit grayscale MP4 (`<name>_depth.mp4`) or lossless 16-bit PNG
  sequence (`<name>_depth/`).
- **Quality** (per-frame models): high / medium / low inference resolution.
- **Window** (streaming models): 4/8/16/32 frames — larger windows are smoother
  over time but slower and more VRAM-hungry.

### Output
- **Container**: MP4 or MKV.
- **Codec**: *H.265 10-bit (x265, anime profile)* — default; tuned for flat-color
  content (`aq-mode=3`, measured +1.3 dB SSIM vs the stock preset). *NVENC* is much
  faster with slightly lower quality. *H.264* for maximum compatibility (8-bit).
- **Quality (CRF)**: 0–51, lower = better; the label next to it estimates output size.
  CRF 19 is the default — visually lossless for anime at reasonable sizes.
- **Audio passthrough**: keeps your original audio track untouched (no re-encode).

### Advanced
- **Scene-cut detection** — never interpolate across a cut (always on in practice;
  interpolation runs per scene segment).
- **Keep lossless workspace chunks after success** — debugging aid; off by default.
- **Stage order**: *Interpolate → Upscale* (fast, TAS-proven for anime) or
  *Upscale → Interpolate* (max motion detail on live action; slower and more VRAM).

### About
Version info and a manual *Check for updates* button (update channel not yet
configured — it will tell you so).

## Reading the artifact report

The Report page shows one row per artifact type found in the job:

| Column | Meaning |
|---|---|
| **Type** | One of the seven detectors below. |
| **Count** | How many frames had a finding for this artifact. |
| **Severity** | Worst detector confidence on any frame: `minor` < 0.3 · `low` ≥ 0.3 · `medium` ≥ 0.5 · `high` ≥ 0.7. |
| **Repair applied** | The dominant repair method used (most frequent across frames). |
| **Before → After** | The detector's own metric re-measured after the repair — a drop means the fix worked; this is the app verifying its own work, not just applying it blindly. |

The seven artifacts and how they're repaired:

| Artifact | What it is (and why AI stages cause it) | Repair |
|---|---|---|
| **Flicker** | Temporal noise in regions that should be static — super-resolution models re-hallucinate fine detail independently per frame. | SCUNet denoise on flagged static regions (AI), else temporal smoothing. Original (non-interpolated) frames are never judged for ghosting/flicker caused by their own motion. |
| **Ghosting** | Double-exposure smearing where interpolation blended two very different frames — typical of RIFE-class models at extreme fast motion or occlusion boundaries. | Neighbor-blend toward the better-matching source frame (flow-corroborated). |
| **Banding** | Visible steps in smooth gradients after 8-bit-style quantization / aggressive denoising. | deepDeband-f on flagged plateau regions (AI), else dither-based deband. |
| **Halo / ringing** | Overshoot rings around high-contrast edges from strong SR models. | Edge-soften on the halo mask. |
| **Line-art** | Anime line art that got blurred, broken or shifted by interpolation/SR. | FastLineDarken (exact port of TAS's CV filter) — re-darkens and thins lines against the source as ground truth. |
| **Color drift** | Per-scene color shift relative to a scene reference frame. | Color-match back toward the reference. |
| **Regression frame** | A single frame whose overall image quality (IQA z-score vs its neighbors) dropped — e.g. an encode hiccup or bad interpolation slot. | Temporal stabilization / re-blend with neighbors. |

Export buttons write a machine-readable **JSON** report and a summary **PNG**
next to your output (`<name>_report.json` / `.png`).

## Models and licenses

All models are open weights; the app is personal-use, so non-commercial (NC)
weights are acceptable here per project decision D3 — they carry a ⚠️ badge in
the UI tooltips. Full audit with verification dates: [docs/licenses.md](licenses.md).

**Upscale (x2 unless noted)**

| Model | License | Backend | Notes |
|---|---|---|---|
| ShuffleCUGAN x2 | arch MIT; weights personal use | TRT + CUDA | Anime default. Tiny, fast, anime-tuned. |
| Fallin soft / strong | arch MIT; weights personal use | TRT + CUDA | Same CUGAN arch, different weights — softer vs stronger detail. |
| SPAN V2 (Modern Spanimation) | arch Apache-2.0; weights personal use | TRT + CUDA | Higher quality than CUGAN-class; slightly more edge overshoot on extreme contrast. |
| Cyte V1 SuperUltraCompact | arch Apache-2.0; **weights CC BY-NC 4.0 ⚠️** | TRT + CUDA | Ultra-light (fastest of the lineup). |
| OpenProteus Compact i2 | arch Apache-2.0; weights personal use | TRT + CUDA | Good speed/quality balance. |
| Real-ESRGAN x4plus | code BSD-3; **weights non-commercial ⚠️** | TRT + CUDA | Live-action default (x4 model, run at 2×). Battle-tested SR+denoise. |

**Frame interpolation**

| Model | License | Backend | Notes |
|---|---|---|---|
| RIFE 4.25 (Lite / Balanced / Heavy) | MIT | TRT + CUDA | Default for all content. Fast (~24 ms/call @720p on a 5090). Any multiplier. |
| GMFSS | code MIT; **weights unlicensed ⚠️** | CUDA only | Experimental, anime-oriented quality; ~4× slower than RIFE per call. |
| M2M | **Adobe Research License (non-commercial) ⚠️** | CUDA only | Experimental; slowest per call but its per-pair flow reuse makes it efficient at high multipliers (4×–8×). |
| AMT-S / AMT-L (CVPR'23) | code CC BY-NC 4.0; weights CC BY-NC-SA 4.0 ⚠️ | CUDA only | Experimental anime/large-motion interpolator (all-pairs multi-field transforms). S = fast default, L = higher quality/slower. No TensorRT path (correlation volume + warp aren't exportable); fp16 below UHD / fp32 above. |

**Artifact repair (used internally by the QA stage)**: SCUNet (Apache-2.0 arch,
personal-use weights) and deepDeband-f (BSD-3 + MIT archs, personal-use weights);
both auto-download on first use when AI repairs are enabled.

**Depth maps**: Limbo v1/v2 (**CC BY-NC 4.0 ⚠️**, anime finetunes), Depth-Anything
V2 Small and Depth-Anything-3 small/base (Apache-2.0).

## Output files

For an input `myvideo.mp4` you get, next to the original:

| File | What it is |
|---|---|
| `myvideo_enhanced.mp4` | The enhanced video (H.265 10-bit by default) with audio passthrough. A numeric suffix (`_enhanced_1`) avoids overwriting existing files. |
| `myvideo_report.json` / `.png` | Artifact report (machine-readable + image). |
| `myvideo_depth.mp4` or `myvideo_depth/` | Depth map output, only when depth generation is enabled. |

## Performance notes

- **First run per model/resolution builds a TensorRT engine** (tens of seconds to
  minutes); it's cached under `%LOCALAPPDATA%\ai-video-enhancer\models\engines\`
  and reused forever after (invalidated automatically on TRT/weights change).
- **VRAM is shared with everything else on your GPU.** If another process holds
  most of the VRAM, the app degrades gracefully: it skips doomed engine builds and
  falls back to PyTorch-CUDA, or runs upscalers in tiled mode. You'll see notices
  in the log — output quality is unaffected, speed may be.
- **Throughput ballpark** (RTX 5090, warm): RIFE ~24 ms/call @720p (~41 fps of
  interpolated frames), SR models ~10–13 ms/frame @360p input for the x2 anime
  lineup. A full 10-minute 1080p episode at 2× interpolation is roughly 5–8 minutes
  of GPU time end-to-end. See [docs/benchmarks.md](benchmarks.md) for measured tables.

## Troubleshooting

| Symptom | What it means / what to do |
|---|---|
| *No NVIDIA GPU detected* banner | The app needs a CUDA-capable NVIDIA GPU; nothing else runs without one. |
| *TensorRT engine build failed / skipped — using PyTorch-CUDA* in the log | Not an error: another process (e.g. a local LLM server) is holding most of the VRAM, so the app used its fallback path. Close the other GPU workload for full speed; no action needed otherwise. |
| *Out of memory* on a job | Lower the resolution target, use a smaller model (Cyte V1 / RIFE Lite), or free VRAM. The app fails the job with a clear message rather than corrupting output. |
| Model download failure | The downloader retries 3× with backoff and falls back to mirrors; persistent failures are usually network-side. Delete the partial file under `%LOCALAPPDATA%\ai-video-enhancer\models\` and retry. |
| *Disk full* mid-run | The job fails cleanly; nothing is left half-written in your output location (lossless workspace chunks may remain if you enabled "keep workspace"). |
| Job cancelled | Safe at any point — finished jobs are untouched. Re-queue the video to run it again. |
| Report shows many findings on a flat/texture-less clip | Known calibration note: very flat synthetic content (solid colors, in-place motion) can over-trigger temporal detectors. On real anime episodes the defaults are well calibrated; lower the sensitivity sliders if you want fewer flags. |

## Feedback loop (beta)

This is a beta build. If something misbehaves: note the **job's log lines**
(Processing page → live log, or `%LOCALAPPDATA%\ai-video-enhancer\logs\aive.log`, rotating 5 MB ×4), the
preset/model/resolution involved, and what you expected to happen — that's all
the information needed to reproduce it.

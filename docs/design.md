# Design — AI Video Enhancer

> Status: **Draft v2 for review** (incorporates your feedback) · 2026-09-17
> Companion doc: [plan.md](plan.md) (implementation phases & milestones)

## Confirmed decisions (from your review of v1)

| # | Decision | Consequence in this design |
|---|---|---|
| D1 | Reference tool "TAS" = **[TheAnimeScripter](https://github.com/NevermindNilas/TheAnimeScripter)** (actively maintained, last push 2026-09-14) | Primary reference: pipeline order, anime model lineup, TensorRT backend pattern, restoration models (§2–§5) |
| D2 | GPU = **RTX 5090, 32 GB VRAM** (Blackwell) | Full-frame inference by default (no tiling needed up to ~4K output); heavy model variants viable; toolchain pinned to CUDA 12.8 / TensorRT 10.x (§6) |
| D3 | **Personal use only** | Non-commercial weights (Real-ESRGAN, SwinIR/HAT NC sets) may be bundled freely (§8) |
| D4 | Output = **H.265, anime profile, 10-bit** | Default encode: libx265 main10, preset slow, CRF 19 + anime-oriented `x265-params` (the old `anime`/`stillimage` tunes were removed in x265 v4); NVENC HEVC 10-bit as "fast" option (§7) |
| D5 | **GUI only** (no CLI product surface) | Headless core kept internally for testing/worker process, but no CLI UX is built or shipped (§6, plan M5/M6) |
| D6 | **TensorRT + CUDA acceleration** for models | TensorRT is the primary inference backend with PyTorch-CUDA fallback; engines cached per GPU/model/precision/shape (§6.2) |

---

## 1. Goals & scope

### In scope (v1)
1. **AI upscaling** of video with current open anime-first + general models, selectable scale factor (default 2× like TAS; up to 4× where models exist).
2. **Frame interpolation** to a target FPS or multiplier (e.g., 24→60), with scene-cut protection and smart deduplication for anime.
3. **Artifact detection & repair**: after upscale + interpolation, automatically detect artifacts the AI stages introduced and apply targeted fixes; produce a human-readable report. *(This is what TAS does not have — its restoration options are manual. This is our core differentiator.)*
4. **User-friendly GUI** (Windows): presets, one-click enhance, live progress, before/after preview, artifact report. Single-file installer, no Python required by the user.
5. **Optional depth map generation**: per-frame monocular depth video output (Depth-Anything V2 / DA3 streaming / anime-tuned Limbo), standalone or alongside enhancement — TAS-style (§3.4).

### Out of scope for v1 (candidates for later)
- Real-time playback enhancement / streaming; After Effects plugin (TAS has one — possible v2).
- Face restoration, depth maps, object detection, stabilization (TAS features we deliberately skip in v1).
- macOS/Linux builds (architecture won't preclude them; packaging is Windows-first).
- Training/fine-tuning models inside the app.

---

## 2. What existing tools do (and don't) do

| Capability | **TheAnimeScripter** | Video2X v6 | Flowframes | VapourSynth ecosystem | **This app** |
|---|---|---|---|---|---|
| AI upscale (anime-first models) | ✅ CUGAN/Adore/Fallin/SPAN/Cyte/OpenProteus/AniScale/RTMOSR… | ✅ Anime4K/Real-ESRGAN/Real-CUGAN | ❌ | ✅ (plugins) | ✅ same class of models + registry |
| Frame interpolation | ✅ RIFE 4.6→4.25-heavy (+Elexor mod) | ✅ RIFE | ✅ RIFE/DAIN/FLAVR | ✅ vs-rife etc. | ✅ RIFE 4.25(-heavy)/AMT |
| Multi-backend (CUDA/TensorRT/DirectML/OpenVINO) | ✅ | ❌ (ncnn/Vulkan only) | partial (NCNN/CUDA) | n/a | ✅ **TensorRT primary + CUDA fallback** (D6) |
| Pipeline order | dedup → interpolate → upscale → restore | separate modes | interpolation only | manual scripts | **same as TAS by default** (§3), advanced option to swap |
| Scene-cut protection / smart dedup | ✅ | partial | ✅ | manual | ✅ automatic |
| Restoration (denoise/dejpeg/line enhance) | ✅ manual options (SCUNet, DeJpeg, Anime1080fixer…) | ❌ | ❌ | manual filters | ✅ **automatic, driven by detection** (§4) |
| Depth maps | ✅ DA v2 / DA3 streaming / Limbo (anime), TRT-validated | ❌ | ❌ | manual scripts | ✅ optional stage, TensorRT-accelerated (§5.4) |
| **Automatic artifact detection + targeted repair + report** | ❌ | ❌ | ❌ | ❌ | ✅ **core feature** |
| Before/after preview + timeline markers | live preview (no markers) | ❌ | ❌ | n/a | ✅ split-slider + color-coded markers |

---

## 3. Processing pipeline

### 3.1 Stage order (default = TAS-proven, speed-first)

```
 input ─▶ Ingest ─▶ Scene split ─▶ Dedup (anime) ─▶ Interpolate ─▶ Upscale ─▶ Artifact QA/Restore ─▶ Encode ─▶ output + report
          (probe,      (cuts)        (2D anim only)    (RIFE at       (AI SR on     (detect → repair →         (x265 10-bit
           decode)                                                        source res)  all frames)   verify)            anime profile)
```

**Why interpolate first (TAS order):** RIFE cost scales ~quadratically with pixel count — interpolating at 720p/1080p source resolution is dramatically faster than at the upscaled target, and on a 32 GB card everything still fits in memory for one pass. Anime SR models are stable enough that running them over interpolated frames produces clean results (TAS's demos confirm this). Our QA stage then cleans whatever artifacts remain.

**Advanced option — "Upscale first" mode:** upscale → interpolate, for maximum motion detail on live-action content (RIFE runs in full resolution; slower but best motion fidelity). Exposed as a quality toggle in the Live Action preset.

### 3.2 Stage details

| # | Stage | What happens | Notes |
|---|---|---|---|
| 1 | **Ingest** | FFmpeg probe (codec, res, fps, duration, audio); decode via NVDEC; validate container | Reject/flag: >8K, VFR without CFR conversion, odd pixel formats |
| 2 | **Scene split** | PySceneDetect content-aware cut detection → segments | Never interpolate across a cut. Segments = unit of work & checkpointing |
| 3 | **Dedup** *(anime preset)* | Remove consecutive duplicate frames before FI, restore after (TAS/Flowframes-style sensitivity) | Off for live action |
| 4 | **Interpolate** | RIFE-family model generates intermediate frames to target FPS; arbitrary-timestep recursion for N× | TensorRT engine per model version; UHD mode auto above 1080p source |
| 5 | **Upscale** | AI SR on every frame (real + interpolated), full-frame inference at 32 GB VRAM | Models in §5. Tiled inference kept as safety valve for >4K targets or future smaller GPUs |
| 6 | **Artifact QA / Restore** | Detect → repair → verify (§4). Runs on a 720p proxy first; only flagged segments get full-res repair | The differentiator. Overhead target: <~20% of total runtime |
| 7 | **Encode** | libx265 **10-bit, anime profile** (default) / NVENC HEVC 10-bit (fast) / H.264; audio passthrough or re-encode; MP4/MKV | Estimated output size shown before start (§7.2) |

### 3.3 Checkpointing & resume
Work is per scene-segment. Completed segments are written to a temp workspace (lossless FFV1-in-MKV chunks). On crash/cancel, the job resumes from the last completed segment boundary — no full re-run. Disk usage estimate shown up front; workspace auto-cleans on success.

### 3.4 Depth map generation (optional, TAS-style)
Depth is an **independent output**, not part of the enhance chain: a toggle in settings generates a depth video alongside (or instead of) the enhanced file. It runs on source frames by default — depth models are trained at fixed input resolutions (e.g., Limbo's 504×280-class inputs), so upscaling first buys little and costs a lot; an advanced option runs it post-upscale for higher-resolution depth. Output: **10-bit grayscale MP4** or **16-bit PNG sequence** (for editing/3D workflows). Video-streaming models (DA3, Limbo) process overlapping chunk windows with temporal alignment — TAS has validated exactly this pattern on TensorRT at ~2–3× over CUDA with negligible flicker regression.

---

## 4. Artifact detection & repair system (core feature)

### 4.1 Design principle
Each detector emits **findings**: `(segment, frame range, region mask, artifact type, severity score)`. Repair policies consume findings and apply **targeted** fixes only where needed — no global filters that degrade good areas. A final verify pass re-measures the same metrics and records before/after deltas for the report.

### 4.2 Detectors (v1 set)

| Artifact | Cause in our pipeline | Detection method (proxy-res, cheap CV) | Severity signal |
|---|---|---|---|
| **Temporal flicker / detail instability** | Per-frame SR hallucinations differ between frames in static regions | In low-flow regions (optical flow magnitude ≈ 0), compute temporal σ of pixel values over a 5–9 frame window; flag regions above adaptive threshold | σ vs. scene noise floor ratio |
| **Ghosting / double-exposure** | FI blending at fast motion, occlusions, or across imperfect cuts | For each interpolated frame: similarity to both neighbors (SSIM/abs-diff); ghost mask = high-diff-to-both + flow-inconsistency regions | Mask area × contrast of ghost |
| **Banding / posterization** | 8-bit source + SR smoothing in gradients (mitigated by our 10-bit output, but sources vary) | Histogram plateau detection on smooth-gradient patches (Laplacian ≈ 0) | Plateau width & count |
| **Halo / ringing / over-sharpening** | SR edge overshoot; line-art halos in anime | Edge-response analysis: measure high-freq overshoot around strong edges vs. the source's own edge profile (we have the original!) | Overshoot amplitude |
| **Line-art degradation** *(anime)* | Blurred/thickened/darkened lines after upscale | Line-mask extraction (high-contrast thin structures) compared to source line mask: width/position drift | Drift magnitude on line pixels |
| **Color shift / exposure drift** | Per-frame model output variance, encode | Per-scene color histogram comparison against scene reference frame; per-channel drift tracking | ΔE-style distance |
| **Global quality regression frames** | Rare bad frames from any stage | No-reference IQA score (NIQE/BRISQUE-class) per frame; flag statistical outliers vs. segment median | Z-score of IQA |

> Because we keep the original source, several detectors compare *against ground truth* (source edge profile, line mask, color), which is far more reliable than blind no-reference checks — a structural advantage over tools that only see the output.

### 4.3 Repairs (mapped per artifact) — hybrid CV + AI

| Artifact | Repair policy (in priority order) |
|---|---|
| Flicker | 1) **SCUNet** blind-denoise pass on flagged regions (TensorRT): independent per-frame denoising collapses temporal variance in static regions, where each frame differs only by SR hallucination noise. 2) Fallback: masked temporal smoothing — weighted median blend across time **only inside flagged regions** (motion edges excluded by flow mask). |
| Ghosting | 1) Region-wise blend toward the temporally closer source frame (per-pixel weight from similarity). 2) Re-interpolate flagged sub-segments with the alternate RIFE variant (e.g., 4.25 ↔ 4.25-heavy / lite) or reduced multiplier locally. |
| Banding | 1) **deepDeband-f** AI debanding model on flagged regions (TAS-proven). 2) Fallback: classical Deband-style filter + fine dither on smooth regions only. |
| Halo/ringing | Local edge softening: unsharp rollback proportional to measured overshoot, confined to flagged edges; never global blur. |
| Line-art degradation *(anime)* | **FastLineDarken**-style line enhancement pass (TAS-proven) + optional Anime1080fixer for 1080p-class outputs; applied only where line drift was detected. |
| Color drift | Per-scene color match back to source palette (per-channel affine fit), applied with a temporal ramp at scene boundaries to avoid pops. |
| Regression frames | Re-process the frame/short window with the fallback SR model and blend in if verify confirms improvement. |

### 4.4 User controls for this stage
- **Mode:** `Auto` (detect + fix, default) · `Detect only` (report, no changes — useful for tuning) · `Manual` (per-artifact on/off).
- **Sensitivity slider** per artifact type (low = fewer fixes / high = aggressive).
- **Report always generated**, even in Auto mode: table of findings + applied repairs + before/after metric deltas; exportable as JSON and PNG.

### 4.5 Why proxy-res first
Detectors run on a 720p proxy of the processed stream (cheap, ~seconds per minute of video). Only segments with findings get full-resolution repair. This keeps the QA stage from becoming another full model pass.

---

## 5. Model selection ("latest available", anime-first)

Research done against upstream repos on 2026-09-17 (links in §9). Models are **pinned by version + checksum** at build time; a JSON **model registry** lets us add newer drops without code changes (new manifest entry + loader config). Lineup mirrors what TAS has validated, plus live-action options.

### 5.1 Upscaling models

| Model | Best for | Why included | Backend support | License |
|---|---|---|---|---|
| **ShuffleCugan** (x2) | Anime — default | Current CUGAN evolution used by TAS; ONNX/TRT-ready (styler00dollar) | CUDA + TRT ✅ | verify at build |
| **Fallin Soft / Fallin Strong** (x2) | Anime, soft vs. strong detail | renarchi/Re-SISR; popular in anime community; TRT-ready per TAS | CUDA + TRT ✅ | arch MIT; weights personal-use (TAS mirror) ✅ |
| **SPAN** (x2) | Anime, high quality | Research SOTA for anime SR; TRT-ready per TAS | CUDA + TRT ✅ | arch Apache-2.0 (traiNNer-redux); Conv3XC MIT; weights personal-use (TAS mirror) ✅ |
| **Cyte V1** (x2) | Anime | In TAS lineup, all backends | CUDA + TRT ✅ | **CC BY-NC 4.0** ⚠️ (NevermindNilas/Ai-models) — NC, marked in UI per D3 |
| **OpenProteus / AniScale 2** | Anime (advanced) | Sirosky/Upscale-Hub; newer community favorites | CUDA + TRT ✅ | OpenProteus: arch Apache-2.0 + weights personal-use ✅ · AniScale 2 pending (not added) |
| **RTMOSR / Saryn / Gauss** | Experimental max-quality anime | umzi2's RT/DIS-based models; behind "experimental" flag | CUDA (+TRT for most) | verify at build ⚠️ |
| **Real-ESRGAN** (x4plus, animevideov3) | Live action & general | Battle-tested default outside anime | CUDA + TRT ✅ | BSD-3 / NC weights (OK per D3) |
| **SAFA** (WACV 2024, RIFE team) | Live action with motion — experimental | *Temporal* video SR; built for temporal consistency → less flicker at the source | CUDA (TRT if exportable) | research — verify ⚠️ |

**Defaults:** Anime preset → ShuffleCugan x2 · Live Action preset → Real-ESRGAN (SAFA offered as experimental) · Fast preset → Anime4K v4 (real-time filter, any scale).
**Custom models:** import of community weights via a Spandrel-compatible loader (see §8 licensing note on GPL isolation).

### 5.2 Frame interpolation models

| Model | Best for | Why included | Backend support | License |
|---|---|---|---|---|
| **RIFE 4.25 / 4.25-heavy** — default | Everything | Upstream's recommended default; TAS ships the full 4.x line incl. heavy variant (great on 32 GB) | CUDA + TRT ✅ | MIT ✅ |
| **RIFE 4.25-lite** | Fast mode / ghost-repair fallback | Lower-cost variant for speed preset & targeted re-interpolation | CUDA + TRT ✅ | MIT ✅ |
| **AMT-S / AMT-L** ✅ implemented | Anime, 2D animation (experimental) | All-pairs multi-field transforms (CVPR'23); strong on anime/large-motion; alternate model for ghost repair | CUDA only (no TRT) | CC BY-NC(-SA) ⚠️ personal use |
| **M2M** (CVPR'22) | High multipliers (4×–8×), general | Many-to-many splatting: motion estimated once per frame pair → very efficient when generating many intermediate frames; ideal for 120 fps+ targets | CUDA (TRT if exportable) | "Other" license ⚠️ personal use |
| **GMFSS** | Anime, experimental | GMFlow-based anime interpolation (SVFI-team lineage); community successors (Fortuna/union) improve it further — we track the best maintained variant at build time | CUDA | no license file ⚠️ personal use only |

> **FILM** (Google, Apache-2.0) was evaluated during M2.x but **removed from the product on 2026-10-15**: its TensorRT engine build fails at upscaled resolutions on the target GPU and it has no PyTorch fallback, so shipping it would be broken. Not in the shipped lineup; kept here for provenance only.

### 5.3 Restoration models (used by the QA stage & optional manual restore)

| Model | Used for | Backend support |
|---|---|---|
| **deepDeband-f** | AI debanding (repair: banding) | CUDA ✅ |
| **SCUNet** | Denoise / flicker stabilizing pass | CUDA + TRT ✅ |
| **FastLineDarken** | Anime line enhancement (repair: line-art) | CUDA + TRT ✅ |
| **Anime1080fixer** | 1080p-class anime cleanup | CUDA + TRT ✅ |
| **DeJpeg / DeH264** (Real-Plksr) | Optional *pre-stage* source cleanup for heavily compressed inputs (user toggle, not part of auto-QA) | CUDA (+TRT for DeH264) |

### 5.4 Depth map models

| Model | Best for | Why included | Backend support | License |
|---|---|---|---|---|
| **Depth-Anything V2** (Small/Base, "faster" variants) | General depth — default | Current standard monocular depth; per-frame or streaming | CUDA + TRT ✅ | Apache-2.0 ✅ |
| **DA3 / Depth Anything 3** (video streaming) | Temporally consistent video depth — experimental | ByteDance's newer model; chunk-windowed streaming with overlap alignment (the TAS pattern); metric checkpoints available | CUDA (+TRT via validated export path) | Apache-2.0 ✅ |
| **Limbo v1 / v2** | Anime depth — recommended for the Anime preset | Anime fine-tune of DA3-small at fixed input resolutions; TAS has validated TensorRT streaming (~2–3× over CUDA, ≤~1% flicker change) | CUDA + TRT ✅ (streaming export) | verify at build ⚠️ |

### 5.5 Model manager
- Manifest-driven: name, task (sr/fi/restore/depth), version, source URL(s), SHA-256, min VRAM, backend support matrix, recommended content type, license.
- First-use download with progress UI; cached in `%LOCALAPPDATA%\ai-video-enhancer\models`; checksum verification on load.
- TensorRT engines built from ONNX on first use and cached (see §6.2).

---

## 6. Architecture

```
┌──────────────────────────── UI process (PySide6 / Qt 6) ────────────────────────────┐
│ Home/drop zone · Settings+presets · Preview compare · Progress · Artifact report    │
└──────────────────────────────▲──────────────────────────────────────────────────────┘
                               │ IPC: job queue + event stream (multiprocessing.Queue / local socket)
┌──────────────────────────────▼──────────────────────────────────────────────────────┐
│ Worker process(es) — pipeline orchestrator                                          │
│  ingest → scene split → dedup → interpolate → upscale → artifact QA → encode        │
│                                                                                     │
│  Inference backend layer:                                                           │
│    ┌─ TensorRT (primary): ONNX export → .engine cache per GPU/model/precision/shape │
│    └─ PyTorch CUDA (fallback + for non-exportable models)                           │
│  · FFmpeg (NVDEC/NVENC, libx265 10-bit)   · model manager (download/verify/cache)   │
│  · checkpoint store (FFV1 segment chunks)                                           │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 6.1 Key decisions & rationale
| Decision | Choice | Rationale |
|---|---|---|
| Language/runtime | **Python 3.11 + PyTorch (cu128)** | Every target model is natively PyTorch; fastest path to "latest models". Blackwell (RTX 5090) requires CUDA 12.8+ toolchain — pinned in the installer. |
| UI toolkit | **PySide6 (Qt 6)** | Native Windows look, mature frame display, excellent packaging with PyInstaller + Inno Setup; one language across the app. GUI-only product surface per D5 (headless core exists internally for tests/worker). |
| Process model | UI process + separate worker(s) | GPU work must never block the UI; clean cancel/pause via job control events; crash isolation. |
| Video I/O | **FFmpeg** (bundled static build), NVDEC/NVENC when present | Industry-standard decode/encode; hardware accel on both ends. |
| Scene detection | PySceneDetect | Content-aware, well-maintained, Python-native (same choice TAS makes). |
| VRAM strategy (32 GB) | **Full-frame inference by default**; fp16/bf16; tiled inference only as safety valve (>4K targets or future smaller GPUs); RIFE UHD mode auto above 1080p source | On a 5090 the tiling complexity is unnecessary for normal use — simpler, faster, no seams. |

### 6.2 Inference backend layer (D6: TensorRT + CUDA)
- **Primary path:** models ship as ONNX (or are exported from PyTorch on first use); a TensorRT engine builder compiles `.engine` files at FP16 and caches them in the model cache, keyed by `(GPU arch, model name+version, precision, input shape profile)`.
- **Fallback path:** if a model has no clean ONNX export (e.g., SAFA, some experimental models), it runs via PyTorch CUDA — same adapter interface, so the pipeline never knows the difference.
- **Shape profiles:** fixed-shape engines per common tile/frame size; dynamic shapes used where TRT 10.x supports them cleanly for RIFE's variable timesteps.
- **Expected gains** (per TAS/Video2X community benchmarks): ~1.5–3× over raw PyTorch CUDA for SR models, similar for RIFE — meaningful on long jobs even with a fast card.
- Engine build time (~minutes per model) happens once at first use, shown in the UI ("Preparing model…").

---

## 7. UI / UX design

Design language: dark theme, large drop zone, minimal jargon; every advanced option collapsed behind "Advanced". Presets do the thinking for the user; power users can override anything. (TAS-Standalone's chain-builder + live log is a good UX reference.)

### 7.1 Screen map
```
┌───────────────────────────────────────────────────────────────────────┐
│  AI Video Enhancer                                    [⚙ Settings] [?] │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│              ┌─────────────────────────────────────┐                  │
│              │                                     │                  │
│              │        Drop video here (or Browse)  │   ← Home screen  │
│              │                                     │                  │
│              └─────────────────────────────────────┘                  │
│                                                                       │
│  queue:  episode01.mp4  720p·23.976fps·24min      [Enhance all]       │
│         episode02.mp4  1080p·25fps·24min                                 │
└───────────────────────────────────────────────────────────────────────┘
```

### 7.2 Settings panel (right side, grouped)
```
 Preset:  [ Anime ▾ ]   (Anime · Live Action · Mixed/Unknown · Fast Preview)

 ─ Upscale ─────────────────────────────────────────────
   Model:      [ ShuffleCugan x2 ▾ ]     Scale: [2×] [3×] [4×]
   Source cleanup (optional): dejpeg / deh264  [off]

 ─ Frame interpolation ─────────────────────────────────
   Enable:     [✓]   Target FPS: [60 ▾]  (or multiplier 2×/3×/4×)
   Model:      [ RIFE 4.25 ▾ ]           Quality: [Balanced ▾]

 ─ Artifact handling ───────────────────────────────────
   Mode:       (●) Auto   ( ) Detect only   ( ) Manual
   Sensitivity: flicker ●─  ghost ●─  banding ●─  halo ●─  lines ●─  color ●─
   [✓] Show artifact report when done

 ─ Depth map (optional) ────────────────────────────────
   Generate: [ ]   Model: [ Limbo v2 ▾ ]  Output: [10-bit MP4 ▾] / PNG seq

 ─ Output ──────────────────────────────────────────────
   Container: [MP4 ▾]   Codec: [H.265 10-bit (x265, anime profile) ▾]
                 · fast option: H.265 10-bit (NVENC)
   Quality: CRF [19]    Est. size: ~3.4 GB    Audio: passthrough
```

**Anime-profile encode defaults (D4):** libx265 main10, preset slow, CRF 19, `-x265-params bframes=8:rc-lookahead=60` (flat colors/line art; x265 v4 removed the `anime`/`stillimage` tunes, so their intent is emulated with explicit params); NVENC HEVC 10-bit (`-preset p7 -rc vbr`) as the fast option. 10-bit output also reduces banding in flat anime skies — synergistic with the deband repair stage.

### 7.3 Preview & artifact report
- **Preview:** pick any frame → split-slider before/after; timeline strip below with color-coded markers (🟠 flicker, 🟣 ghost, 🔵 banding, 🟡 halo, ⚫ line-art, ⚪ color). Click a marker → jump to frame + tooltip "ghosting @ 00:12:34.5 — repaired by neighbor-blend".
- **Report view** after run:

```
 Artifact report — episode01.mp4 (24 min)          [Export JSON] [Export PNG]
 ┌────────────┬────────┬───────────┬──────────────────────────┬───────────────┐
 │ Type       │ Count  │ Severity  │ Repair applied           │ Before → After│
 ├────────────┼────────┼───────────┼──────────────────────────┼───────────────┤
 │ Flicker    │ 14 seg │ medium    │ masked temporal smooth   │ σ 2.3 → 0.6   │
 │ Ghosting   │ 7 seg  │ low       │ neighbor-frame blend     │ area 8%→1%    │
 │ Banding    │ 3 seg  │ low       │ deepDeband-f             │ bands 5→0     │
 │ Halo       │ 2 seg  │ minor     │ edge softening           │ overshoot ↓40%│
 │ Line-art   │ 5 seg  │ minor     │ FastLineDarken (masked)  │ drift ↓60%    │
 │ Color drift│ 1 seg  │ minor     │ scene color match        │ ΔE 3.1→0.4    │
 └────────────┴────────┴───────────┴──────────────────────────┴───────────────┘
```

### 7.4 Processing view
Per-stage progress (Interpolate ▓▓░ · Upscale ▓▓▓░ 62% · QA ░░░), ETA, live log (collapsible), VRAM/RAM meters, **Pause / Cancel** buttons. Queue continues to next file automatically.

### 7.5 First-run experience
1. Launch → GPU detected ("NVIDIA RTX 5090 — CUDA + TensorRT ready") or clear warning.
2. "Get started" downloads default models (ShuffleCugan, RIFE 4.25) with progress; other models download on demand; TRT engines build at first use of each model.
3. Sample clip offered for a 1-minute test run before the user commits to a long job.

---

## 8. Licensing & compliance notes (personal use per D3)

| Item | License | Action |
|---|---|---|
| App code (ours) | **MIT** (proposed — your call) | — |
| RIFE / Practical-RIFE weights | MIT ✅ | bundle freely |
| Anime4K v4, Real-CUGAN ncnn-vulkan | MIT ✅ | bundle freely |
| Real-ESRGAN code / weights | BSD-3 / **weights non-commercial** ⚠️ | OK per D3; document in app's license screen |
| SwinIR/HAT (if added later) | some weights CC-BY-NC ⚠️ | same treatment |
| ShuffleCugan, Fallin, SPAN, Cyte, OpenProteus, AniScale 2, RTMOSR/Saryn/Gauss, deepDeband-f, SCUNet, FastLineDarken, Anime1080fixer, DeJpeg/DeH264 | mixed — **per-model license check at integration** (most are MIT/Apache; SPAN & some research models need verification) ⚠️ | ship only after check passes; NC/research ones marked in UI model list |
| SAFA (pending); AMT-S/L ✅ integrated | research repos — AMT verified at integration: code CC BY-NC 4.0, weights CC BY-NC-SA 4.0 ⚠️ | AMT shipped with "experimental" flag; SAFA still pending |
| M2M (CVPR'22) | **"Other" / non-standard license** ⚠️ | personal use per D3; no redistribution claims; marked in UI model list |
| GMFSS | **no license file** ⚠️ | personal use per D3 only; we track the best-maintained community variant (Fortuna/union) at build time and re-check its terms then |
| ~~FILM~~ (google-research — removed 2026-10-15) | Apache-2.0 ✅ | no longer bundled; model removed from product, so its license is no longer relevant to shipping |
| Depth-Anything V2 / DA3 checkpoints | Apache-2.0 ✅ | bundle freely |
| Limbo v1/v2 (anime fine-tune of DA3-small) | **verify at integration** ⚠️ | "experimental" flag until cleared; TAS's TRT streaming validation gives us a proven export path either way |
| **Spandrel** (chaiNNer's model loader) for custom-model import | **GPL-3.0** ⚠️ | Two clean options: (a) implement the feature in a separate subprocess that talks to our MIT core over IPC (mere aggregation — keeps app MIT), or (b) license the whole app AGPL like TAS. Default plan: **(a)**; custom-model import is an advanced/optional feature, not v1-critical |
| Video2X / TheAnimeScripter (reference only) | AGPLv3 | we do **not** use their code; feature/model reference only → no contamination |

---

## 9. Sources checked (2026-09-17)

- **TheAnimeScripter**: https://github.com/NevermindNilas/TheAnimeScripter — pipeline order, model lineup + backend matrix, restoration models, TAS-Standalone UX
- RIFE main repo: https://github.com/hzwer/ECCV2022-RIFE · Practical-RIFE (4.25/4.26 recommended default; SAFA v0.5): https://github.com/hzwer/Practical-RIFE
- Video2X v6.x README (C++/Qt6, Anime4K v4 + Real-CUGAN + Real-ESRGAN + RIFE via ncnn/Vulkan, AGPLv3, standard test clip): https://github.com/k4yt3x/video2x
- Flowframes README (RIFE PyTorch/NCNN, DAIN, FLAVR; scene-cut fix, de-duplication, UHD mode): https://github.com/n00mkrad/flowframes
- VapourSynth ecosystem: vs-rife (https://github.com/HolyWu/vs-rife), RIFE-ncnn-Vulkan plugin, Deband filter — feature references.
- M2M (Many-to-many Splatting for Efficient Video Frame Interpolation, CVPR'22): https://github.com/feinanshan/M2M_VFI
- GMFSS (GMFlow-based anime FI; community successors Fortuna/union): https://github.com/routineLife1/GMFSS · https://github.com/98mxr/GMFSS_Fortuna
- FILM (google-research, Apache-2.0): https://github.com/google-research/frame-interpolation
- Depth Anything 3 (ByteDance-Seed, Apache-2.0 checkpoints): https://github.com/ByteDance-Seed/Depth-Anything-3 · Limbo = anime fine-tune of DA3-small per TAS model registry; TAS TRT streaming validation: `docs/LIMBO_TENSORRT_VALIDATION.md` in the TAS repo (2026-09-12).

---

## 10. Remaining open questions (minor)

1. **Default SR model for the Anime preset:** ShuffleCugan x2 (TAS's workhorse) or Fallin Soft? I'll default to ShuffleCugan and make both one click away — OK?
2. **App license:** MIT with subprocess-isolated Spandrel feature (§8) — OK, or do you prefer AGPL like TAS for simplicity?
3. Anything else from v1 you'd change? Otherwise I start M0+M1 on approval.

# AI Video Enhancer

A desktop app that upscales video with state-of-the-art AI models, adds frame
interpolation, and then **automatically detects artifacts introduced by the
upscale/interpolation and repairs them** — with a friendly UI and a transparent
artifact report. Personal-use Windows app for NVIDIA GPUs (CUDA + TensorRT).

## Status: v1 beta shipped (M0–M8 complete)

| Document | Purpose |
|---|---|
| [docs/user-guide.md](<docs/user-guide.md>) | **Start here** — presets, settings, reading the artifact report, models & licenses, troubleshooting |
| [docs/design.md](<docs/design.md>) | Product design, architecture, model selection, artifact system, UI/UX |
| [docs/plan.md](<docs/plan.md>) | Phased implementation plan with milestones and acceptance criteria (progress log at the bottom) |
| [docs/licenses.md](<docs/licenses.md>) | License audit: per-model weight/arch tables + verification dates |
| [docs/benchmarks.md](<docs/benchmarks.md>) | Measured throughput tables (auto-updated snapshot + per-milestone findings); re-run with `python tools/bench_all.py` |
| [packaging/BUILD.md](<packaging/BUILD.md>) | How to build the Windows package (`powershell -File packaging\build.ps1`) |

## One-paragraph pitch

Drop in a video → pick a preset (Anime / Live Action / Mixed / Fast Preview) →
the app runs **AI upscale → frame interpolation → artifact detection & repair**
using current open models, then hands back a clean H.265 10-bit file plus a
report showing what artifacts were found and fixed — with an optional depth-map
output on top.

## What's in the box (v1)

- **Upscale**: ShuffleCUGAN x2, Fallin soft/strong, SPAN V2, Cyte V1, OpenProteus
  Compact i2 (anime), Real-ESRGAN x4plus (live action) — TensorRT fp16 with
  PyTorch-CUDA fallback, automatic tiled mode as a VRAM safety valve.
- **Frame interpolation**: RIFE 4.25 family (Lite/Balanced/Heavy) + experimental
  GMFSS and M2M; scene-cut protection, de-duplication of held frames, any
  multiplier (target-FPS or fixed ×).
- **Artifact QA** (the differentiator): detects flicker, ghosting, banding,
  halo/ringing, line-art drift, color drift and regression frames after the AI
  stages; applies targeted repairs (incl. SCUNet + deepDeband-f AI models),
  re-measures to verify each fix, and writes a JSON/PNG report.
- **Depth maps** (optional post-pass): Limbo v1/v2 (anime streaming),
  Depth-Anything V2 Small, experimental Depth-Anything-3 small/base → 10-bit
  grayscale MP4 or lossless PNG sequence.
- **UI**: PySide6 dark theme — queue + presets, per-stage progress with VRAM/RAM
  meters and live log, before/after preview with artifact markers, report view.

## Quick start (dev machine)

```powershell
# run from source (Python 3.14 with the project deps installed)
C:\Python314\python.exe run_gui.py

# or build the portable bundle (~8 min; multi-GB output under dist\)
powershell -ExecutionPolicy Bypass -File packaging\build.ps1
dist\ai-video-enhancer\ai-video-enhancer.exe
```

Model weights download on first use (cached under
`%LOCALAPPDATA%\ai-video-enhancer\models\`). No video handy? Use *Generate
sample clip* on the Home page.

## Reference tools studied

| Tool | What we take from it |
|---|---|
| [TheAnimeScripter](https://github.com/NevermindNilas/TheAnimeScripter) (TAS) | Model lineup + hosts, depth-map streaming pattern, x265 anime encode tuning, QA mindset |
| [Video2X](https://github.com/k4yt3x/video2x) v6.x | Model lineup cross-check (Real-CUGAN, Real-ESRGAN, RIFE), standard test clip for benchmarks |
| [Flowframes](https://github.com/n00mkrad/flowframes) | Interpolation UX: scene-cut protection, frame de-duplication (anime), UHD mode, audio/metadata preservation |
| [VapourSynth ecosystem](https://www.vapoursynth.com/) | Feature ideas: scriptable chains, Deband-style filters; possible future "export as .py script" for power users |

## Key differentiator

No existing tool automatically measures what went wrong after AI upscale +
interpolation (flicker, ghosting, banding, halos, color drift) and applies
**targeted** fixes — then verifies the fix worked. That QA-and-repair stage is
the core of this project.

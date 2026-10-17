# AI Video Enhancer

A desktop app that upscales video with state-of-the-art AI models, adds frame
interpolation, and then **automatically detects artifacts introduced by the
upscale/interpolation and repairs them** — with a friendly UI and a transparent
artifact report. Personal-use Windows app for NVIDIA GPUs (CUDA + TensorRT).

## Status: v1 beta shipped (M0–M10 complete)

| Document | Purpose |
|---|---|
| [docs/user-guide.md](<docs/user-guide.md>) | **Start here** — presets, settings, reading the artifact report, models & licenses, troubleshooting |
| [docs/design.md](<docs/design.md>) | Product design, architecture, model selection, artifact system, UI/UX |
| [docs/plan.md](<docs/plan.md>) | Phased implementation plan with milestones and acceptance criteria (progress log at the bottom) |
| [docs/licenses.md](<docs/licenses.md>) | License audit: per-model weight/arch tables + verification dates |
| [docs/benchmarks.md](<docs/benchmarks.md>) | Measured throughput tables (auto-updated snapshot + per-milestone findings); re-run with `python tools/bench_all.py` |
| [packaging/BUILD.md](<packaging/BUILD.md>) | How to build the Windows package, installer and release ZIP (`powershell -File packaging\build.ps1`) |

## One-paragraph pitch

Drop in a video → pick a preset (Anime / Live Action / Mixed / Fast Preview) →
the app runs **AI upscale → frame interpolation → artifact detection & repair**
using current open models, then hands back a clean H.265 10-bit file plus a
report showing what artifacts were found and fixed — with an optional depth-map
output on top.

## What's in the box (v1)

- **Upscale**: ShuffleCUGAN x2, Fallin soft/strong, SPAN V2, Cyte V1, OpenProteus
  Compact i2 (anime), Real-ESRGAN x4plus (live action) — TensorRT fp16 with
  PyTorch-CUDA fallback, automatic tiled mode as a VRAM safety valve. Integer-power
  targets beyond a model's native scale (e.g. ×4 on an ×2 model) are reached by
  **cascading AI passes** — no plain resampling in between.
- **Frame interpolation**: RIFE 4.25 family (Lite/Balanced/Heavy), GMFSS, M2M and
  AMT-S / AMT-L (CVPR'23) — scene-cut protection, de-duplication of held frames,
  any multiplier (target-FPS or fixed ×).
- **Artifact QA** (the differentiator): detects flicker, ghosting, banding,
  halo/ringing, line-art drift, color drift and regression frames after the AI
  stages; applies targeted repairs (incl. SCUNet + deepDeband-f AI models),
  re-measures to verify each fix, and writes a JSON/PNG report.
- **Depth maps** (optional post-pass): Limbo v1/v2 (anime streaming),
  Depth-Anything V2 Small, experimental Depth-Anything-3 small/base → 10-bit
  grayscale MP4 or lossless PNG sequence.
- **UI**: PySide6 dark theme — queue + presets, per-stage progress with VRAM/RAM
  meters and live log, before/after preview with artifact markers, report view.

## Get it

### Option 1 — release ZIP (no build needed)

Download the latest `ai-video-enhancer-<ver>-win-x64.zip` from
[Releases](https://github.com/junmiron/ai-video-enhancer/releases). GitHub caps
each release asset at 2 GB, so the archive ships as **split parts** — reassemble
once, then extract:

```bat
:: Windows cmd (adjust part count to what you downloaded)
copy /b ai-video-enhancer-0.1.0-win-x64.zip.part01 + ^
        ai-video-enhancer-0.1.0-win-x64.zip.part02 + ^
        ai-video-enhancer-0.1.0-win-x64.zip.part03 = ai-video-enhancer-0.1.0-win-x64.zip

:: then extract (built-in tar, or right-click -> Extract All)
tar -xf ai-video-enhancer-0.1.0-win-x64.zip
```

Run `ai-video-enhancer\ai-video-enhancer.exe`. Requirements: Windows 10/11 x64 +
NVIDIA GPU with a current driver. FFmpeg is bundled inside the folder (no PATH
setup); model weights download on first use and are cached under
`%LOCALAPPDATA%\ai-video-enhancer\models\`.

### Option 2 — build from source

Prerequisites:

- Windows 10/11 x64, NVIDIA GPU with a current driver (CUDA)
- Python 3.11–3.14 (3.14 verified): `winget install Python.Python.3.14` or python.org
- FFmpeg full build: `winget install Gyan.FFmpeg` — or drop `ffmpeg.exe` +
  `ffprobe.exe` into `tools\ffmpeg\` in the repo

```powershell
git clone https://github.com/junmiron/ai-video-enhancer.git
cd ai-video-enhancer

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .                 # numpy, opencv, torch (CUDA), tensorrt, PySide6, ...
# optional: pip install -e ".[speed]"   # xformers (depth speed-up) + onnxruntime (LIMBO re-export)

python run_gui.py                # run from source
```

Build the distributable bundle (~8–15 min; multi-GB output under `dist\`):

```powershell
pip install pyinstaller
powershell -ExecutionPolicy Bypass -File packaging\build.ps1
.\dist\ai-video-enhancer\ai-video-enhancer.exe
```

See [packaging/BUILD.md](<packaging/BUILD.md>) for the full build matrix
(onedir bundle, Inno Setup installer, GitHub release ZIP).

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

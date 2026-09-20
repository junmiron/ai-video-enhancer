# Building the Windows package (M7 B)

Produces two artifacts from a clean checkout on the dev machine:

| Artifact | Path | What it is |
|---|---|---|
| Portable bundle | `dist/ai-video-enhancer/` | PyInstaller **onedir**: `ai-video-enhancer.exe` + `_internal/` (Python, torch+TensorRT, Qt). Runnable as-is if ffmpeg is on PATH. |
| Installer | `packaging/output/ai-video-enhancer-setup-0.1.0.exe` | Inno Setup installer: bundle **plus** FFmpeg under `<install>\ffmpeg\`, which the app finds automatically in frozen builds (`aive/stages/ingest.py`). |

## Prerequisites (dev machine)

- Python 3.14 with the project deps installed — `C:\Python314\python.exe` on this machine
  (torch 2.12+cu132, tensorrt 11.1 cu13, PySide6 6.11, opencv-python, scenedetect, …).
- **PyInstaller ≥ 6** (`pip install pyinstaller`) — 6.22.3 verified.
- **Inno Setup 6** for the installer: `winget install JRSoftware.InnoSetup`.
  > Not installed on this machine as of M7 B — the portable bundle still builds fine;
  > only step 3 is skipped with a warning.
- FFmpeg full build (Gyan) for bundling — default source path in `build.ps1` points at the
  WinGet install (`ffmpeg-9.0.1-full_build\bin`). Override with `-FfmpegDir`.

## One-shot build

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build.ps1
```

Useful switches: `-SkipPyInstaller` (reuse existing `dist/`), `-SkipInstaller`,
`-Python <path>`, `-FfmpegDir <bin-folder>`.

## Manual steps

```powershell
# 1. PyInstaller onedir bundle (~5–15 min; multi-GB output)
C:\Python314\python.exe -m PyInstaller packaging\aive.spec --noconfirm

# 2. Stage FFmpeg for the installer (build.ps1 does this automatically)
Copy-Item <ffmpeg-bin>\ffmpeg.exe, <ffmpeg-bin>\ffprobe.exe packaging\ffmpeg-bin\

# 3. Installer
& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\ai-video-enhancer.iss
```

## How the frozen app finds its pieces

- **Model manifest** — `assets/models/manifest.json` ships inside `_internal/assets/…`;
  `aive.models.registry.default_manifest_path()` checks `<exe>\assets\…` then
  `_MEIPASS\assets\…` (onedir: `_MEIPASS == _internal`).
- **FFmpeg** — PATH first, then `<exe>\ffmpeg\` when frozen (installer layout).
- **Qt plugins** — bundled under `_internal/PySide6/Qt/plugins`; `run_gui.py` sets
  `QT_PLUGIN_PATH` before `QApplication` starts.
- **Model weights** — *not* bundled; downloaded on first use to
  `%LOCALAPPDATA%\ai-video-enhancer\models\` (see the model cards in Settings).

## Notes & known trade-offs

- Bundle size is dominated by torch (~2.8 GB, CUDA DLLs inside `torch/lib`) and TensorRT
  libs (~2.2 GB) — expected for a self-contained AI app; D: has plenty of room.
- The Gyan *full* FFmpeg build is ~450 MB for the two exes (all codecs/filters). A leaner
  GPL build would shrink the installer if size ever matters.
- No application icon yet — add an `.ico` and set `icon=` in `packaging/aive.spec` +
  `SetupIconFile`/`UninstallDisplayIcon` in the `.iss`.
- Inno compresses with LZMA (default level). If installer build time becomes painful,
  switch to `Compression=lzma2/fast` or `no` in the `.iss`.

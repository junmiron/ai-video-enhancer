# Building the Windows package (M7 B)

Produces two artifacts from a clean checkout on the dev machine:

| Artifact | Path | What it is |
|---|---|---|
| Portable bundle | `dist/ai-video-enhancer/` | PyInstaller **onedir**: `ai-video-enhancer.exe` + `_internal/` (Python, torch+TensorRT, Qt). Runnable as-is if ffmpeg is on PATH. |
| Installer | `packaging/output/ai-video-enhancer-setup-0.1.0.exe` | Inno Setup installer: bundle **plus** FFmpeg under `<install>\ffmpeg\`, which the app finds automatically in frozen builds (`aive/stages/ingest.py`). |
| Release ZIP | `dist/ai-video-enhancer-<ver>-win-x64.zip[.partNN]` | The whole bundle (exe + `_internal` + `ffmpeg`) as one logical zip for GitHub Releases; split into <2 GB parts when needed (see below). |

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

## Release ZIP for GitHub

GitHub release assets are capped at **2 GB each**, and this bundle is ~6 GB, so a
release ships as one logical ZIP split into `.partNN` files. A true single-`.exe`
portable build is **not possible**: PyInstaller's onefile archive format uses
32-bit offsets/sizes (~4 GB hard cap) — an onefile spec was attempted in Oct 2026
and failed with `struct.error: 'I' format requires 0 <= number <= 4294967295`.
Do not retry it; the split-ZIP release is the shipping format.

```powershell
# after a fresh build.ps1 run (which stages FFmpeg into packaging\ffmpeg-bin\):
powershell -ExecutionPolicy Bypass -File packaging\make_release_zip.ps1 [-Version 0.1.0]
```

What it does: 1) copies `ffmpeg.exe`/`ffprobe.exe` into the bundle's `ffmpeg\`
folder so the release is self-contained (frozen builds find `<exe>\ffmpeg\`),
2) zips the whole `ai-video-enhancer\` folder with 7-Zip (`-tzip -mx=1`, fast;
ZIP64 handled automatically for >4 GB), 3) if the zip exceeds ~1.9 GB, splits it
into ~1.85 GB `.partNN` files and prints the reassemble command.

Users reassemble before extracting (Windows cmd):

```bat
copy /b ai-video-enhancer-0.1.0-win-x64.zip.part01 + ^
        ai-video-enhancer-0.1.0-win-x64.zip.part02 + ^
        ai-video-enhancer-0.1.0-win-x64.zip.part03 = ai-video-enhancer-0.1.0-win-x64.zip
tar -xf ai-video-enhancer-0.1.0-win-x64.zip
```

Upload the parts as release assets via `gh release create` (see README "Get it").

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

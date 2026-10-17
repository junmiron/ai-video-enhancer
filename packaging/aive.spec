# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller onedir spec — AI Video Enhancer (M7 B).

Build from the repo root::

    C:\\Python314\\python.exe -m PyInstaller packaging/aive.spec --noconfirm

Output layout (onedir)::

    dist/ai-video-enhancer/
        ai-video-enhancer.exe      <- windowed GUI entry (run_gui.py)
        _internal/                 <- sys._MEIPASS: python + all binaries/data
            assets/models/manifest.json   (model registry, see aive.models.registry)
            PySide6/Qt/plugins/...        (Qt plugins; QT_PLUGIN_PATH set in run_gui.py)

Notes on collection strategy:
  * torch / torchvision / tensorrt(_libs) / onnxruntime / cv2 / timm are handled by
    community hooks (pyinstaller-hooks-contrib): CUDA DLLs live inside torch/lib,
    TensorRT DLLs inside the tensorrt_libs package (preloaded via ctypes at import).
  * PySide6 has NO hook in this environment, so collect_all() bundles the whole
    package (Qt .pyd modules + Qt DLLs + plugin tree); run_gui.py points Qt at the
    bundled plugins dir before QApplication is created.
  * multiprocessing spawn works frozen: CPython launches children with
    ``exe --multiprocessing-fork`` and PyInstaller's bootloader handles it; the
    worker target (aive.ui.worker.worker_main) is statically importable in-bundle.
  * tools.make_sample_clip is bundled explicitly (hiddenimports): the worker
    generates sample clips IN-PROCESS (frozen exes cannot re-run arbitrary .py),
    which also pulls Pillow along via its hook.
"""
import os

from PyInstaller.utils.hooks import collect_all

REPO = os.path.dirname(os.path.dirname(os.path.abspath(SPEC)))  # noqa: F821 - SPEC injected by PyInstaller

# --- PySide6 (no hook available -> collect the whole package) -----------------
pyside_datas, pyside_binaries, pyside_hiddenimports = collect_all("PySide6")

a = Analysis(
    [os.path.join(REPO, "run_gui.py")],
    pathex=[REPO],
    binaries=pyside_binaries,
    datas=pyside_datas + [(os.path.join(REPO, "assets"), "assets")],
    hiddenimports=[
        *pyside_hiddenimports,
        "torchgen",               # imported by torch at runtime
        "tools.make_sample_clip",  # in-process sample-clip generation (aive.ui.worker)
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib"],  # never imported; keep the bundle lean
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ai-video-enhancer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # GUI-only app (user requirement); logs go to %APPDATA%\ai-video-enhancer\logs\aive.log
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="ai-video-enhancer",
)

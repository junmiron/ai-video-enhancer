"""SeedVR2-3B video super-resolution (experimental).

Vendors ByteDance's SeedVR architecture (Apache-2.0) under ``vendor/`` and
provides a clip-based :class:`aive.stages.upscale._seedvr.runner.SeedVrRunner`
that restores whole temporal clips via one-step diffusion denoising — unlike the
per-frame CNN SR models in the rest of the upscale stage (see
:mod:`aive.stages.upscale.stage`).

The vendored code keeps its original top-level package names (``common``,
``models``) so the upstream omegaconf ``__object__`` string paths resolve
unchanged; :func:`ensure_vendor_on_path` puts the vendor directory on
``sys.path`` before those modules are imported.

License: Apache-2.0 (c) 2025 Bytedance Ltd. — see ``vendor/NOTICE.md``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_VENDOR_DIR = str(Path(__file__).resolve().parent / "vendor")

#: Absolute path of the vendored SeedVR tree (configs, checkpoints' config files).
VENDOR_DIR = Path(_VENDOR_DIR)


def ensure_vendor_on_path() -> None:
    """Idempotently add the vendored SeedVR tree to ``sys.path``.

    Must be called before importing any ``common.*`` / ``models.*`` module so
    that upstream absolute imports and omegaconf string paths resolve to our
    vendored copies rather than (nonexistent) top-level packages.
    """
    if _VENDOR_DIR not in sys.path:
        sys.path.insert(0, _VENDOR_DIR)

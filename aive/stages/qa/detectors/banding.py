"""Banding / posterization detector (design.md §4.2).

In a true smooth gradient the value changes by ~1 level per pixel, so there are no
long runs of identical values. Banding shows up as *plateaus*: stretches where the
value is stuck at one quantized level before jumping to the next. We measure the
per-pixel plateau length (horizontal and vertical) in smooth regions — i.e. away
from edges/textures, found via a low-Laplacian mask — and flag long plateaus.
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import FrameWindow, luma

_SMOOTH_LAP = 0.01  # |Laplacian| below this => flat/gradient (not an edge)


def _plateau_len(arr: np.ndarray, tol: float) -> np.ndarray:
    """Per-pixel length of the run of near-constant values it belongs to (rows)."""
    h, w = arr.shape
    d = np.abs(np.diff(arr, axis=1)) > tol  # [h, w-1] change between col i-1 and i
    pos = np.arange(w)[None, :].astype(np.int32)
    change = np.concatenate([np.ones((h, 1), bool), d], axis=1)  # [h, w]
    last_change = np.maximum.accumulate(np.where(change, pos, -1).astype(np.int32), axis=1)
    return (pos - last_change + 1).astype(np.float32)


def detect(win: FrameWindow, sens: float) -> tuple[np.ndarray, float]:
    lc = luma(win.center).astype(np.float32)
    lap = cv2.Laplacian(lc, cv2.CV_32F)
    smooth = (np.abs(lap) < _SMOOTH_LAP).astype(np.float32)

    # A plateau is a run of *near-identical* values (true quantization). We use a
    # sub-level tolerance so a smooth gradient that changes by >=1 level per pixel
    # does NOT read as banded, while genuinely-stuck levels do.
    tol = 0.75 / 255.0
    ph = _plateau_len(lc, tol)
    pv = _plateau_len(lc.T.copy(), tol).T
    plateaus = np.maximum(ph, pv)

    # Higher sensitivity => shorter plateaus count as banding.
    l_ref = 16.0 - 8.0 * sens
    raw = (np.clip(plateaus / max(4.0, l_ref), 0.0, 1.0) * smooth).astype(np.float32)

    sev = float(np.clip((raw > 0.5).mean(), 0.0, 1.0))
    return raw, sev

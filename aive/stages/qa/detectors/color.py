"""Color-shift / exposure-drift detector (design.md §4.2).

Within a scene the overall color should be stable; per-frame model output variance
or encode drift shows up as a shift of the whole per-channel distribution relative
to a scene reference frame. We track per-channel mean drift normalized by the
scene's own per-channel spread (a ΔE-like magnitude) and report when it exceeds a
sensitivity-dependent gate. The spatial map highlights where the deviation from the
scene palette is largest.
"""
from __future__ import annotations

import numpy as np

from .base import FrameWindow


def detect(win: FrameWindow, sens: float) -> tuple[np.ndarray, float]:
    h, w = win.center.shape[:2]
    if win.scene_ref is None:
        return np.zeros((h, w), dtype=np.float32), 0.0

    c = win.center.astype(np.float32)
    ref = win.scene_ref.astype(np.float32)
    dm = np.abs(c - ref)  # [H,W,3] per-pixel per-channel deviation

    ch_drift = dm.mean(axis=(0, 1))  # [3] mean drift per channel
    scene_std = ref.std(axis=(0, 1)) + 1e-3  # [3]
    rel = ch_drift / scene_std  # in units of the scene's own variation
    total = float(np.sqrt(float((rel ** 2).sum())))  # ΔE-like global drift

    gate = max(0.2, 0.9 - 0.6 * sens)  # higher sensitivity => lower bar
    if total < gate:
        return np.zeros((h, w), dtype=np.float32), 0.0

    pix_dev = dm / (scene_std[None, None, :] + 1e-3)  # [H,W,3]
    spatial = np.clip(pix_dev.max(axis=2), 0.0, 1.0).astype(np.float32)
    weight = float(np.clip((total - gate) / gate + 0.5, 0.5, 1.0))
    mask = (spatial * weight).astype(np.float32)

    sev = float(np.clip(total / 3.0, 0.0, 1.0))
    return mask, sev

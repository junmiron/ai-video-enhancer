"""Temporal-flicker detector (design.md §4.2).

Per-frame SR hallucinations differ between frames even where nothing moves, so a
static region that should be temporally constant shows pixel-value variance. We
measure the temporal standard deviation of luma over a short window and flag
regions whose variance exceeds an adaptive threshold — but only inside *low-flow*
regions (real motion is excluded via the optical-flow mask), which is what keeps
this from lighting up on legitimate movement.
"""
from __future__ import annotations

import numpy as np

from ..motion import static_mask
from .base import FrameWindow, luma

_FLOW_THRESH = 0.5  # px of displacement considered "static"


def detect(win: FrameWindow, sens: float) -> tuple[np.ndarray, float]:
    """Return (region mask [H,W] in [0,1], severity in [0,1]) for the center frame."""
    frames = win.window if len(win.window) >= 3 else ([win.center] * 3)
    lumas = np.stack([luma(f) for f in frames[:9]], axis=0)  # [T,H,W]
    std_t = lumas.std(axis=0).astype(np.float32)

    static = static_mask(win.mags, thresh=_FLOW_THRESH) if win.mags else np.ones_like(std_t)

    static_sel = std_t[static > 0.5]
    floor = float(static_sel.min()) if static_sel.size else 0.0
    # Higher sensitivity -> smaller margin above the noise floor -> more flags.
    margin = (1.0 - sens) * 0.03 + 0.004
    thresh = floor + margin

    raw = np.clip((std_t - thresh) / max(1e-3, 0.05), 0.0, 1.0).astype(np.float32)
    mask = (raw * static).astype(np.float32)

    flagged = mask > 0.3
    if flagged.any():
        sev = float(np.clip((std_t[flagged].mean() - thresh) / 0.05, 0.0, 1.0))
    else:
        sev = 0.0
    return mask, sev

"""Temporal-flicker detector (design.md §4.2).

Per-frame SR hallucinations differ between frames even where nothing moves, so a
static region that should be temporally constant shows pixel-value variance. We
measure the temporal standard deviation of luma over a short window and flag
regions whose variance exceeds an adaptive threshold — but only inside *low-flow*
regions (real motion is excluded via the optical-flow mask), which is what keeps
this from lighting up on legitimate movement.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..motion import static_mask
from .base import FrameWindow, luma

_FLOW_THRESH = 0.5  # px of displacement considered "static"
# Spatial-coherence gate (M7 A3 calibration on samples/m5-demo): SR re-hallucinates
# sub-pixel edge jitter per frame that flow cannot track on flat content; it shows up
# as scattered single pixels / tiny components (observed: <=19 px, 43-92 flags/frame)
# and floods the report. Real hallucination flicker is patch-like (injected acceptance
# regions are ~5k-70k px at proxy res). Components below this size are dropped before
# severity so isolated shimmer never records a finding.
MIN_COMPONENT_PX = 64


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

    # Spatial coherence: drop connected components below MIN_COMPONENT_PX so isolated
    # edge shimmer cannot record a finding on its own.
    flagged = mask > 0.3
    if flagged.any():
        n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(
            flagged.astype(np.uint8), connectivity=8
        )
        keep_ids = np.where(stats[1:, cv2.CC_STAT_AREA] >= MIN_COMPONENT_PX)[0] + 1
        if len(keep_ids) < n_cc - 1:
            drop_ids = np.setdiff1d(np.arange(1, n_cc), keep_ids)
            mask = np.where(np.isin(labels, drop_ids), 0.0, mask).astype(np.float32)
            flagged = mask > 0.3

    if flagged.any():
        sev = float(np.clip((std_t[flagged].mean() - thresh) / 0.05, 0.0, 1.0))
    else:
        sev = 0.0
    return mask, sev

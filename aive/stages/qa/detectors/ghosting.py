"""Ghosting / double-exposure detector (design.md §4.2).

Frame interpolation blends two source frames; at fast motion, occlusions or
imperfect cuts this can leave a semi-transparent "ghost" — content that matches
*neither* neighbor. The primary signal is therefore **dual-diff**: per-pixel
distance to both the previous and next frame (high only when far from *both*).
Flow inconsistency between the two sides of the interpolated frame corroborates
severity, but it must not suppress a strong dual-diff below reportable level —
in static scenes flow is ~0 on both sides, so inconsistency there is just noise.
The percentile gate keeps precision (only the most anomalous pixels are flagged).
Only runs on frames the stage marks as interpolated.
"""
from __future__ import annotations

import numpy as np

from ..motion import flow_inconsistency, flow_magnitude, motion_diff
from .base import FrameWindow, luma, robust_scale


def detect(win: FrameWindow, sens: float) -> tuple[np.ndarray, float]:
    h, w = win.center.shape[:2]
    if not win.is_interpolated or win.prev is None or win.next is None:
        return np.zeros((h, w), dtype=np.float32), 0.0

    d_prev = motion_diff(win.center, win.prev).astype(np.float32)
    d_next = motion_diff(win.center, win.next).astype(np.float32)
    both = np.minimum(d_prev, d_next)  # high only when far from BOTH neighbors

    mag_pc = flow_magnitude(luma(win.prev), luma(win.center))
    mag_cn = flow_magnitude(luma(win.center), luma(win.next))
    inc = flow_inconsistency(mag_pc, mag_cn)  # [H,W] in [0,1]

    scale = robust_scale(both, lo=50.0, hi=98.0) + 1e-3
    both_n = np.clip(both / scale, 0.0, 1.0).astype(np.float32)
    # Dual-diff is the primary signal; flow inconsistency only modulates severity.
    raw = np.clip(both_n * (0.7 + 0.3 * inc), 0.0, 1.0)

    gate = float(np.percentile(both, 98)) if both.size else 0.0
    thresh = (1.0 - sens) * gate + 0.02
    mask = np.where(both > thresh, raw, 0.0).astype(np.float32)

    flagged = mask > 0.3
    sev = float(mask[flagged].mean()) if flagged.any() else 0.0
    return mask, sev

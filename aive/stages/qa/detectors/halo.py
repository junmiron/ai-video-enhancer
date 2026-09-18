"""Halo / ringing / over-sharpening detector (design.md §4.2).

We keep the original source, so this compares against *ground truth*: SR ringing
shows up as deviation from the source in a thin ring around strong source edges
(the overshoot lobes sit just outside an edge that was flat in the original). We
measure |processed - source| luma inside that ring and normalize by its robust
scale. A faithful upscale stays close to the source there; ringing does not.
"""
from __future__ import annotations

import numpy as np

from .base import FrameWindow, dilate_mask, erode_mask, luma, robust_scale, sobel_mag


def detect(win: FrameWindow, sens: float) -> tuple[np.ndarray, float]:
    h, w = win.center.shape[:2]
    if win.src_center is None:
        return np.zeros((h, w), dtype=np.float32), 0.0

    lc = luma(win.center).astype(np.float32)
    ls = luma(win.src_center).astype(np.float32)

    edge_mag = sobel_mag(ls)
    edge_thresh = float(np.percentile(edge_mag, 90)) if edge_mag.size else 0.0
    edges = (edge_mag > max(1e-3, edge_thresh)).astype(np.float32)
    if not edges.any():
        return np.zeros((h, w), dtype=np.float32), 0.0

    # A ring around the strong source edges where overshoot lobes appear.
    band = np.clip(dilate_mask(edges, k=9) - erode_mask(edges, k=1), 0.0, 1.0).astype(np.float32)

    resid = np.abs(lc - ls).astype(np.float32)
    scale = robust_scale(resid[edges > 0], lo=50.0, hi=98.0) + 1e-3
    divisor = max(1e-3, scale * (1.6 - sens))  # higher sensitivity => lower bar
    raw = np.clip(resid / divisor, 0.0, 1.0).astype(np.float32) * band

    sev = float(raw[band > 0].mean()) if (band > 0).any() else 0.0
    return raw, sev

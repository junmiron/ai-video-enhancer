"""Global quality-regression frame detector (design.md §4.2).

A rare bad frame from any stage — an SR hallucination blow-up, a dropped or
over-smoothed frame — shows up as an outlier in the per-frame no-reference IQA
time series rather than as a localized region. We compute a lightweight no-ref
sharpness/texture score for the center frame and flag it when its *robust*
z-score against the recent rolling window of scores (``win.recent_iqa``) exceeds a
sensitivity-dependent gate. The spatial mask localizes where the frame deviates
from the temporal median of recent frames so repair can target it, falling back to
a full-frame map when no localization is available.

The core pieces (:func:`iqa_feature`, :func:`robust_z`) are pure functions over a
scalar / list so they unit-test without any pipeline or GPU.
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import FrameWindow, luma


def iqa_feature(frame: np.ndarray) -> float:
    """Lightweight no-reference sharpness/texture score (higher = more detail).

    Laplacian variance of the luma plane — a classic no-ref sharpness proxy. It is
    content-dependent in absolute value but *stable* across temporally adjacent
    frames, which is exactly what the rolling z-score exploits: a regression frame
    jumps away from its neighbors regardless of scene content (a blurry or noisy
    bad frame both move this score off the local baseline).
    """
    g = luma(frame).astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    return float(lap.var())


def robust_z(x: float, ref: list[float]) -> float:
    """Robust z-score of ``x`` against a reference window (median / MAD).

    Uses the median and median-absolute-deviation instead of mean/std so a single
    extreme value in the reference cannot inflate the scale. Returns 0 when there
    are too few references to establish a baseline.
    """
    if len(ref) < 3:
        return 0.0
    arr = np.asarray(ref, dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    # 1.4826 * MAD ~= std for gaussian data; floor the scale so a perfectly flat
    # reference (zero spread) still yields a finite, meaningful z.
    scale = 1.4826 * mad if mad > 1e-9 else max(1e-9, abs(med) * 0.05 + 1e-9)
    return float((x - med) / scale)


def detect(win: FrameWindow, sens: float) -> tuple[np.ndarray, float]:
    """Return (region mask [H,W] in [0,1], severity in [0,1]) for the center frame."""
    h, w = win.center.shape[:2]
    if len(win.recent_iqa) < 3:
        return np.zeros((h, w), dtype=np.float32), 0.0

    iqa_now = iqa_feature(win.center)
    z = robust_z(iqa_now, win.recent_iqa)
    # Higher sensitivity => lower bar (more frames flagged as regressions).
    gate = max(1.5, 4.0 - 2.5 * sens)
    if abs(z) < gate:
        return np.zeros((h, w), dtype=np.float32), 0.0

    sev = float(np.clip(abs(z) / (gate + 1.5), 0.0, 1.0))

    # Localize the anomaly: per-pixel deviation from the temporal median of recent
    # frames. The median is robust to the single bad frame being in the window.
    if len(win.window) >= 3:
        stack = np.stack([f.astype(np.float32) for f in win.window], axis=0)
        med = np.median(stack, axis=0)  # [H,W,3]
        dev = np.abs(win.center.astype(np.float32) - med).mean(axis=2)
        peak = float(dev.max())
        mask = (np.clip(dev / max(peak, 1e-6), 0.0, 1.0) * sev).astype(np.float32)
    else:
        mask = np.full((h, w), sev, dtype=np.float32)

    return mask, sev

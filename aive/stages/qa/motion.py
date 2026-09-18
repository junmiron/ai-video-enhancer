"""Cheap motion / optical-flow helpers for the QA detectors (design.md §4.2).

Two things drive several detectors:
* **low-flow regions** — where optical-flow magnitude is ~0, so temporal pixel
  variance there indicates flicker rather than real motion (flicker detector);
* **flow inconsistency** — incoherent motion between the two halves around an
  interpolated frame, a signature of ghosting / double-exposure.

Flow is computed with OpenCV's Farneback estimator at proxy resolution (cheap,
CPU-friendly). All functions are pure numpy/cv2 so they unit-test without a GPU.
"""
from __future__ import annotations

import cv2
import numpy as np


def _to_u8(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32) if gray.dtype != np.float32 else gray
    if g.max() > 1.5:  # already uint8 range
        return g.astype(np.uint8)
    return (np.clip(g, 0.0, 1.0) * 255.0).astype(np.uint8)


def flow_magnitude(prev_gray: np.ndarray, next_gray: np.ndarray) -> np.ndarray:
    """Per-pixel displacement magnitude (in pixels) between two grayscale frames.

    Inputs may be float [0,1] or uint8; they are normalized internally. Returns a
    single-channel float32 array of the same spatial size as the inputs.
    """
    p = _to_u8(prev_gray)
    n = _to_u8(next_gray)
    flow = cv2.calcOpticalFlowFarneback(
        p, n, None, 0.5, 3, 21, 3, 5, 1.2, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    )
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return mag.astype(np.float32)


def static_mask(mags: list[np.ndarray], thresh: float = 0.5) -> np.ndarray:
    """Soft "staticness" mask in [0,1]: high where motion stays below `thresh`.

    ``mags`` are per-consecutive-pair flow-magnitude fields over a window. A pixel
    is static only if it is low-motion in (nearly) every pair — we use the minimum
    of the per-pair low-motion indicators so one moving frame disqualifies it.
    """
    if not mags:
        return np.zeros((0, 0), dtype=np.float32)
    shape = mags[0].shape
    acc = None
    for m in mags:
        ind = (m < thresh).astype(np.float32)
        acc = ind if acc is None else np.minimum(acc, ind)
    return acc


def flow_inconsistency(mag_a: np.ndarray, mag_b: np.ndarray) -> np.ndarray:
    """Soft mask in [0,1] where two adjacent flow fields disagree.

    ``mag_a``/``mag_b`` are the magnitudes on either side of an interpolated frame
    (prev->center and center->next). Large local disagreement is a ghosting cue.
    The difference is normalized by the local max so it is scale-free, then
    spatially smoothed to suppress single-pixel noise.
    """
    diff = np.abs(mag_a - mag_b)
    denom = np.maximum(np.maximum(mag_a, mag_b), 1e-3)
    score = np.clip(diff / denom, 0.0, 1.0).astype(np.float32)
    k = max(3, (min(score.shape) // 48) | 1)
    return cv2.GaussianBlur(score, (k, k), 0)


def motion_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cheap per-pixel change magnitude between two [0,1] HWC frames (no flow).

    Used as a fast fallback / for the ghosting dual-similarity term. Returns a
    single-channel float32 in roughly [0,1].
    """
    d = np.abs(a.astype(np.float32) - b.astype(np.float32))
    return d.mean(axis=2)

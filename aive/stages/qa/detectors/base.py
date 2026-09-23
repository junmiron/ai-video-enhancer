"""Shared context + small CV helpers for the QA detectors (design.md §4.2).

Every detector is a pure function of a :class:`FrameWindow` plus a sensitivity,
returning ``(mask, severity)``: a proxy-resolution region mask in ``[0, 1]`` and
an overall ``[0, 1]`` severity for the center frame. Keeping detectors as pure
numpy/cv2 functions over explicit arrays is what makes them unit-testable with
synthetic tensors (no GPU, no pipeline).
"""
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field


@dataclass
class FrameWindow:
    """Everything a detector needs to judge one center frame.

    All spatial arrays are at PROXY resolution and in ``[0, 1]`` (HWC for frames).
    Optional fields may be ``None`` when the stage cannot provide them (e.g. no
    source reference); detectors must degrade gracefully.
    """

    center: np.ndarray  # proxy [0,1] HWC — the frame being judged
    prev: np.ndarray | None = None  # previous processed frame (proxy)
    next: np.ndarray | None = None  # next processed frame (proxy)
    window: list[np.ndarray] = field(default_factory=list)  # temporal frames incl. center
    src_center: np.ndarray | None = None  # source reference for this time (ground truth)
    scene_ref: np.ndarray | None = None  # scene reference frame (color drift)
    mags: list[np.ndarray] = field(default_factory=list)  # flow mag between consecutive window frames
    flows: list[np.ndarray] = field(default_factory=list)  # [H,W,2] flow vectors, same order as mags
    center_idx: int | None = None  # index of `center` within `window`; None -> middle (symmetric windows)
    is_interpolated: bool = False  # True if the center frame came from FI
    recent_iqa: list[float] = field(default_factory=list)  # prior per-frame IQA scores (regression)


def luma(frame: np.ndarray) -> np.ndarray:
    """HWC [0,1] RGB -> single-channel float32 luma in [0,1]."""
    f = frame.astype(np.float32) if frame.dtype != np.float32 else frame
    return cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)


def sobel_mag(gray: np.ndarray) -> np.ndarray:
    """Edge-response magnitude (Sobel) of a grayscale [0,1] array."""
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def dilate_mask(mask: np.ndarray, k: int = 5) -> np.ndarray:
    """Binary/soft mask dilation with a square kernel (returns float [0,1])."""
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.dilate(mask.astype(np.float32), kern).astype(np.float32)


def erode_mask(mask: np.ndarray, k: int = 5) -> np.ndarray:
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.erode(mask.astype(np.float32), kern).astype(np.float32)


def smooth(x: np.ndarray, k: int | None = None) -> np.ndarray:
    """Gaussian blur; kernel auto-sized from the smaller dim when ``k`` is None."""
    if k is None:
        k = max(3, (min(x.shape) // 48) | 1)
    return cv2.GaussianBlur(x.astype(np.float32), (k, k), 0)


def robust_scale(x: np.ndarray, lo: float = 5.0, hi: float = 95.0) -> float:
    """A noise-floor-ish scale from percentiles (avoids outlier blow-up)."""
    x = np.asarray(x, dtype=np.float32).ravel()
    if x.size == 0:
        return 1e-3
    lo_v = np.percentile(x, lo)
    hi_v = np.percentile(x, hi)
    return float(max(1e-3, hi_v - lo_v))

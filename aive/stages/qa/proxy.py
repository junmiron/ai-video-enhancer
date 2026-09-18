"""Proxy-resolution helpers for the QA stage (design.md §4.5).

Detectors run on a ~720p proxy of the processed stream so they stay cheap; only
flagged regions get full-resolution repair. These helpers convert between the
full-resolution HWC uint8 frames that flow through the pipeline and the small
float [0,1] proxy arrays the detectors operate on, plus mask up/down-scaling.

Conventions:
* Pipeline frames are ``H x W x 3`` uint8 (RGB).
* Proxy arrays are ``h x w x 3`` float32 in ``[0, 1]`` (RGB).
* Region masks are single-channel float in ``[0, 1]``.
"""
from __future__ import annotations

import cv2
import numpy as np


def proxy_dims(width: int, height: int, target_h: int = 720) -> tuple[int, int]:
    """Target proxy (w, h): scale down to ~target_h tall, keep aspect, even dims.

    Never upscales — a source already at or below the target keeps its native
    size (detection on tiny inputs is cheap anyway). Dims are rounded to even so
    downstream 4:2:0 / flow ops stay happy.
    """
    if height <= 0 or width <= 0:
        return (max(2, width), max(2, height))
    scale = min(1.0, target_h / float(height))
    pw = int(round(width * scale / 2.0) * 2)
    ph = int(round(height * scale / 2.0) * 2)
    return (max(2, pw), max(2, ph))


def to_proxy(frame: np.ndarray, pw: int, ph: int) -> np.ndarray:
    """HWC uint8 RGB frame -> proxy float32 [0,1] HWC at (pw, ph)."""
    if frame.shape[1] == pw and frame.shape[0] == ph:
        return frame.astype(np.float32) / 255.0
    small = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32) / 255.0


def to_full(proxy: np.ndarray, fw: int, fh: int) -> np.ndarray:
    """Proxy float32 [0,1] HWC -> full-res HWC uint8 (for thumbnails/preview)."""
    arr = np.clip(proxy * 255.0, 0, 255)
    if proxy.shape[1] != fw or proxy.shape[0] != fh:
        arr = cv2.resize(arr.astype(np.uint8), (fw, fh), interpolation=cv2.INTER_LINEAR)
    return arr.astype(np.uint8)


def upsample_mask(mask: np.ndarray, fw: int, fh: int) -> np.ndarray:
    """Single-channel proxy mask [0,1] -> full-res bilinear mask [0,1]."""
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    out = cv2.resize(m, (fw, fh), interpolation=cv2.INTER_LINEAR)
    return np.clip(out, 0.0, 1.0)


def downsample_mask(mask: np.ndarray, pw: int, ph: int) -> np.ndarray:
    """Full-res single-channel mask [0,1] -> proxy mask (inverse of upsample)."""
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    out = cv2.resize(m, (pw, ph), interpolation=cv2.INTER_AREA)
    return np.clip(out, 0.0, 1.0)


def gray(frame: np.ndarray) -> np.ndarray:
    """HWC RGB (uint8 or float [0,1]) -> single-channel grayscale float32 [0,1]."""
    f = frame.astype(np.float32) if frame.dtype != np.float32 else frame
    if f.max() > 1.5:  # uint8 range
        f = f / 255.0
    g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
    return g

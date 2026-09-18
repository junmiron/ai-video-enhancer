"""Shared context + CV helpers for the QA repairs (design.md §4.3).

Each repair is a pure function of a :class:`RepairContext` returning
``(repaired_frame, method)``: an HWC uint8 full-resolution frame and the name of
the fix applied (or ``"none"`` when it could not act). Repairs are *targeted* —
they only touch pixels inside the detector's region mask (upsampled to full res),
never a global filter, so good areas are left alone.

The CV path is always available and works without any model; AI-model repairs
(deepDeband-f / SCUNet / FastLineDarken) plug in behind the same interface later
and fall back to these when their weights/engine are unavailable (mirrors how M3.x
handled experimental additions).
"""
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field


@dataclass
class RepairContext:
    """Everything a repair needs to fix one full-res center frame.

    ``frame`` is the HWC uint8 processed frame to be repaired; ``mask`` is the
    detector's region map upsampled to full resolution (single channel in [0,1]).
    Temporal repairs use :attr:`window` (full-res processed frames including the
    center at :attr:`center_idx`). Color drift uses the proxy references for a
    cheap per-channel fit. Optional fields may be ``None``; repairs degrade to
    ``"none"`` when their required context is missing.
    """

    frame: np.ndarray  # HWC uint8 full-res center frame (to repair)
    mask: np.ndarray  # [H,W] float32 in [0,1], full res
    window: list[np.ndarray] = field(default_factory=list)  # temporal full-res frames incl. center
    center_idx: int = 0  # index of `frame` within `window`
    scene_ref_proxy: np.ndarray | None = None  # [h,w,3] float [0,1] proxy scene reference (color drift)
    center_proxy: np.ndarray | None = None  # [h,w,3] float [0,1] proxy of the center frame


def _bbox(mask: np.ndarray, thresh: float = 0.05) -> tuple[int, int, int, int] | None:
    """Tight bounding box (y0,y1,x0,x1) of mask > thresh, or None if empty."""
    ys, xs = np.where(mask > thresh)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _blend_region(frame: np.ndarray, replacement: np.ndarray, mask: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Blend ``replacement`` into ``frame`` where ``mask`` is high (full-res HWC)."""
    out = frame.copy()
    w = (np.clip(mask, 0.0, 1.0)[..., None] * strength)
    blended = frame.astype(np.float32) * (1.0 - w) + replacement.astype(np.float32) * w
    return np.clip(np.round(blended), 0, 255).astype(np.uint8)


def _temporal_stat(window: list[np.ndarray], center_idx: int, mask: np.ndarray, stat: str = "mean") -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    """Per-pixel temporal mean/median of ``window`` frames inside the mask bbox.

    Returns ``(stat_crop, box)`` where ``stat_crop`` is the [h,w,3] float result
    restricted to the mask's bounding box (memory-bounded — only flagged regions
    are accumulated), or ``(None, None)`` when there is nothing to do.
    """
    if len(window) < 3:
        return None, None
    box = _bbox(mask)
    if box is None:
        return None, None
    y0, y1, x0, x1 = box
    crops = [f[y0:y1, x0:x1].astype(np.float64) for f in window]
    stack = np.stack(crops, axis=0)  # [T,h,w,3]
    if stat == "median":
        out = np.median(stack, axis=0).astype(np.float32)
    else:
        out = (stack.mean(axis=0)).astype(np.float32)
    return out, box


def _neighbor_blend(window: list[np.ndarray], center_idx: int) -> np.ndarray | None:
    """Similarity-weighted blend of the frame's immediate temporal neighbors.

    Used by deghosting: a ghost is content that matches *neither* neighbor, so
    pulling it toward a weighted mix of prev/next (weight ~ 1/distance) removes
    the double-exposure. Returns an HWC float32 array or None if no neighbors.
    """
    if center_idx <= 0 or center_idx >= len(window) - 1:
        return None
    c = window[center_idx].astype(np.float32)
    prev = window[center_idx - 1].astype(np.float32)
    nxt = window[center_idx + 1].astype(np.float32)
    d_prev = np.abs(c - prev).mean(axis=2, keepdims=True)
    d_next = np.abs(c - nxt).mean(axis=2, keepdims=True)
    w_prev = 1.0 / (d_prev + 1e-3)
    w_next = 1.0 / (d_next + 1e-3)
    denom = w_prev + w_next + 1e-6
    return (prev * w_prev + nxt * w_next) / denom

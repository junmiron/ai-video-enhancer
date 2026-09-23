"""Ghosting / double-exposure detector (design.md §4.2).

Frame interpolation blends two source frames; at fast motion, occlusions or
imperfect cuts this can leave a semi-transparent "ghost" — content that matches
*neither* neighbor. The primary signal is therefore **dual-diff**: per-pixel
distance to both the previous and next frame (high only when far from *both*).

Fast uniform motion also raises dual-diff, but only in thin bands (~2x the flow
magnitude wide) along moving edges: every pixel there stays close to one of its
motion-compensated neighbors. A true ghost is a solid region elevated above BOTH
neighbors across an area wider than the local motion can explain. So after the
percentile gate selects candidate pixels, we keep only those whose distance to
the boundary of the high-dual-diff region exceeds a robust global estimate of
the frame's motion scale (p75 of the median-filtered flow magnitude) plus a 1px
margin — thin motion bands are shaved away, solid ghost patches survive.

Flow inconsistency between the two sides of the interpolated frame corroborates
severity, but it must not suppress a strong dual-diff below reportable level —
in static scenes flow is ~0 on both sides, so inconsistency there is just noise.
Only runs on frames the stage marks as interpolated.
"""
from __future__ import annotations

import cv2
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
    hi = both > thresh
    # Motion-width gate: uniform fast motion leaves dual-diff confined to thin
    # bands along moving edges (~2x the flow magnitude wide), while a ghost is a
    # solid region elevated above both neighbors across an area wider than any
    # plausible displacement. Keep only high-dual-diff pixels deeper into their
    # region than the robust global motion scale (p75 of median-filtered max
    # per-pair magnitude — median washes Farneback's spurious edge spikes) plus a
    # 1px margin. In static scenes m_glob ~ 0 and this is a no-op.
    m_glob = float(np.percentile(cv2.medianBlur(np.maximum(mag_pc, mag_cn), 5), 75))
    dist = cv2.distanceTransform(hi.astype(np.uint8), cv2.DIST_L2, 5)
    keep = (dist > m_glob + 1.0) & hi

    mask = np.where(keep, raw, 0.0).astype(np.float32)

    flagged = mask > 0.3
    sev = float(mask[flagged].mean()) if flagged.any() else 0.0
    return mask, sev

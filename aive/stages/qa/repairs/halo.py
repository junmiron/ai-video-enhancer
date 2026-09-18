"""Halo / ringing repair (design.md §4.3): local edge softening.

The detector measures high-frequency overshoot around strong source edges (SR
ringing). The CV fix blends the frame toward a light Gaussian blur *only* inside the
flagged edge ring, which flattens the overshoot lobes without blurring the rest of
the image — never a global blur. (A measured unsharp-rollback is the higher-quality
variant; this masked softening is the always-available CV path.)
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import RepairContext, _blend_region


def repair(ctx: RepairContext) -> tuple[np.ndarray, str]:
    if not (ctx.mask > 0.05).any():
        return ctx.frame, "none"
    blurred = cv2.GaussianBlur(ctx.frame, (5, 5), 1.2)
    out = _blend_region(ctx.frame, blurred, ctx.mask, strength=0.7)
    return out, "edge-soften"

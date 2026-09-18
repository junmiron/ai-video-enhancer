"""Line-art degradation repair, anime (design.md §4.3): masked line enhancement.

The detector flags where thin high-contrast lines drifted from the source (blurred,
thickened or eroded after upscale). The CV fix applies an unsharp mask *only* inside
the flagged line region to restore edge crispness and re-thin the strokes. (Priority-1
FastLineDarken / Anime1080fixer AI passes plug in behind this interface later.)
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import RepairContext, _blend_region


def repair(ctx: RepairContext) -> tuple[np.ndarray, str]:
    if not (ctx.mask > 0.05).any():
        return ctx.frame, "none"
    blurred = cv2.GaussianBlur(ctx.frame, (3, 3), 0.8)
    # Unsharp mask: push the frame away from its blur to sharpen thin structures.
    f = ctx.frame.astype(np.float32)
    sharpened = f + 1.4 * (f - blurred.astype(np.float32))
    out = _blend_region(ctx.frame, np.clip(sharpened, 0, 255), ctx.mask, strength=0.8)
    return out, "line-enhance"

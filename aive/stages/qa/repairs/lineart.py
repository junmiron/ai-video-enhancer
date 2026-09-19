"""Line-art degradation repair, anime (design.md §4.3): FastLineDarken.

The detector flags where thin high-contrast lines drifted from the source
(blurred, thickened or eroded after upscale). The fix is TAS's FastLineDarken —
an exact OpenCV/numpy port of its fixed conv pipeline (:mod:`fastlinedarken`,
no model involved): luma -> Sobel magnitude -> global normalize -> max-pool
thinning -> Gaussian soften, then ``image -= strength * softened_edges``. That
re-darkens and re-thins the strokes along detected edges, restoring line
crispness/density that upscaling washed out.

Applied *masked*: the edge map is computed on the full frame (the global
min-max normalization needs it) but the subtraction only lands inside the
detector's region, so good areas are left alone — consistent with the targeted-
repair principle. Replaces the earlier unsharp-mask pass ("line-enhance"),
which sharpened without restoring line darkness.
"""
from __future__ import annotations

import numpy as np

from .base import RepairContext, _blend_region


def repair(ctx: RepairContext) -> tuple[np.ndarray, str]:
    if not (ctx.mask > 0.05).any():
        return ctx.frame, "none"

    from .fastlinedarken import fast_line_darken

    darkened = fast_line_darken(ctx.frame)  # full-frame HWC uint8 (TAS default strength 0.7)
    if np.array_equal(darkened, ctx.frame):
        return ctx.frame, "none"  # no edges in the frame to darken — nothing to do

    out = _blend_region(ctx.frame, darkened, ctx.mask, strength=1.0)
    return out, "fastlinedarken"

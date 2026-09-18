"""Flicker repair (design.md §4.3): masked temporal smoothing.

The detector flags static regions whose luma varies frame-to-frame (per-frame SR
hallucinations). The fix replaces flagged pixels with the *temporal mean* of the
short window — which is exactly what minimizes that temporal variance — confined to
the mask so moving content is untouched. (Priority-2 SCUNet stabilizing pass plugs
in behind this interface later; CV smoothing always works.)
"""
from __future__ import annotations

import numpy as np

from .base import RepairContext, _blend_region, _temporal_stat


def repair(ctx: RepairContext) -> tuple[np.ndarray, str]:
    stat_crop, box = _temporal_stat(ctx.window, ctx.center_idx, ctx.mask, stat="mean")
    if stat_crop is None or box is None:
        return ctx.frame, "none"
    y0, y1, x0, x1 = box
    out = ctx.frame.copy()
    region_mask = np.clip(ctx.mask[y0:y1, x0:x1], 0.0, 1.0) * 0.9
    sub = ctx.frame[y0:y1, x0:x1]
    blended = sub.astype(np.float32) * (1.0 - region_mask[..., None]) + stat_crop * region_mask[..., None]
    out[y0:y1, x0:x1] = np.clip(np.round(blended), 0, 255).astype(np.uint8)
    return out, "temporal-smooth"

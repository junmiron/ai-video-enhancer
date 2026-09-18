"""Regression-frame repair (design.md §4.3): temporal stabilization.

The detector flags a frame whose no-ref IQA is an outlier vs its recent neighbors
(a rare bad frame from any stage). The CV fix replaces the flagged region with the
*temporal median* of the short window — robust to the single bad frame being in that
window, so it pulls the outlier back toward its neighbors' consensus. (Priority-1
fallback-SR reprocess + verify-blend plugs in behind this interface later.)
"""
from __future__ import annotations

import numpy as np

from .base import RepairContext, _temporal_stat


def repair(ctx: RepairContext) -> tuple[np.ndarray, str]:
    stat_crop, box = _temporal_stat(ctx.window, ctx.center_idx, ctx.mask, stat="median")
    if stat_crop is None or box is None:
        return ctx.frame, "none"
    y0, y1, x0, x1 = box
    out = ctx.frame.copy()
    region_mask = np.clip(ctx.mask[y0:y1, x0:x1], 0.0, 1.0) * 0.9
    sub = ctx.frame[y0:y1, x0:x1]
    blended = sub.astype(np.float32) * (1.0 - region_mask[..., None]) + stat_crop * region_mask[..., None]
    out[y0:y1, x0:x1] = np.clip(np.round(blended), 0, 255).astype(np.uint8)
    return out, "temporal-stabilize"

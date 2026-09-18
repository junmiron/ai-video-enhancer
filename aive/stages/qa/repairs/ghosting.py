"""Ghosting repair (design.md §4.3): neighbor-blend deghost.

A ghost is content that matches *neither* temporal neighbor (double-exposure from
FI blending at fast motion / occlusion). The fix pulls the flagged region toward a
similarity-weighted mix of the previous and next processed frames, which directly
reduces its distance to both neighbors — the very quantity the detector measures.
(Priority-2 alternate-RIFE re-interpolation plugs in behind this interface later.)
"""
from __future__ import annotations

import numpy as np

from .base import RepairContext, _blend_region, _neighbor_blend


def repair(ctx: RepairContext) -> tuple[np.ndarray, str]:
    blend = _neighbor_blend(ctx.window, ctx.center_idx)
    if blend is None or not (ctx.mask > 0.05).any():
        return ctx.frame, "none"
    out = _blend_region(ctx.frame, blend, ctx.mask, strength=0.85)
    return out, "neighbor-blend"

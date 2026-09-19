"""Flicker repair (design.md §4.3): temporal stabilization of static regions.

The detector flags static regions whose luma varies frame-to-frame (per-frame SR
hallucinations). Two fixes, tried in order:

1. **SCUNet** (``"scunet-denoise"``, when ``ctx.ai_repairs`` and the runner is
   available): a per-frame blind denoiser on the full frame, blended into the
   original via the detector mask. In a static region every frame's input differs
   only by hallucination noise, so independent denoising maps them all to ~the
   same clean estimate — collapsing exactly the temporal variance the detector
   measures.
2. **CV temporal smoothing** (``"temporal-smooth"`): replace flagged pixels with
   the *temporal mean* of the short window — which is what minimizes that
   temporal variance — confined to the mask so moving content is untouched.

Both are targeted: only masked pixels change, never a global filter.
"""
from __future__ import annotations

import numpy as np

from .base import RepairContext, _blend_region, _temporal_stat


def repair(ctx: RepairContext) -> tuple[np.ndarray, str]:
    if ctx.ai_repairs:
        from .ai import get_restore_runner

        h, w = ctx.frame.shape[:2]
        runner = get_restore_runner("scunet", w, h)
        if runner is not None and (ctx.mask > 0.05).any():
            denoised_u8 = np.clip(np.round(runner(ctx.frame) * 255.0), 0, 255).astype(np.uint8)
            out = _blend_region(ctx.frame, denoised_u8, ctx.mask, strength=1.0)
            return out, "scunet-denoise"

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

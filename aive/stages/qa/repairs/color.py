"""Color-drift repair (design.md §4.3): per-scene color match.

The detector flags a shift of the whole per-channel distribution relative to the
scene reference. Color drift is inherently a *frame-level* artifact, so — unlike the
other repairs — this one applies a global per-channel affine fit (gain + offset) that
maps the frame's palette back onto the scene reference's. The coefficients are computed
cheaply on the proxy references and applied at full resolution with a moderate strength
so it corrects drift without over-correcting legitimate in-scene lighting change.
"""
from __future__ import annotations

import numpy as np

from .base import RepairContext


def repair(ctx: RepairContext) -> tuple[np.ndarray, str]:
    if ctx.scene_ref_proxy is None or ctx.center_proxy is None:
        return ctx.frame, "none"
    ref = ctx.scene_ref_proxy.astype(np.float32)
    cur = ctx.center_proxy.astype(np.float32)

    gain = np.ones(3, dtype=np.float32)
    offset = np.zeros(3, dtype=np.float32)
    for c in range(3):
        ref_mean = float(ref[..., c].mean())
        ref_std = float(ref[..., c].std()) + 1e-4
        cur_mean = float(cur[..., c].mean())
        cur_std = float(cur[..., c].std()) + 1e-4
        gain[c] = ref_std / cur_std
        offset[c] = ref_mean - gain[c] * cur_mean

    strength = 0.7
    f = ctx.frame.astype(np.float32) / 255.0
    corrected = (f * (1.0 - strength) + (f * gain[None, None, :] + offset[None, None, :]) * strength)
    return np.clip(np.round(corrected * 255.0), 0, 255).astype(np.uint8), "color-match"

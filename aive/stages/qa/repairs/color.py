"""Color-drift repair (design.md §4.3): per-scene color match.

The detector flags a shift of the whole per-channel distribution relative to the
scene reference. Color drift is inherently a *frame-level* artifact, so — unlike the
other repairs — this one applies a global per-channel affine fit (gain + offset) that
maps the frame's palette back onto the scene reference's. The coefficients are computed
cheaply on the proxy references and applied at full resolution with a moderate strength
so it corrects drift without over-correcting legitimate in-scene lighting change.

Q4: the coefficients are the MEDIAN of per-frame fits across the temporal window, not
the center frame's own fit. A per-frame fit chases legitimate in-scene lighting change
— every flagged frame gets its own slightly different gain/offset, which reads as
pumping/blinking on real runs (thousands of consecutive findings). The median is
temporally stable: it changes smoothly as the window slides over a slow ramp, and for a
persistent shift most of the window shares the shifted palette so correction strength
is unchanged. All window frames belong to the same segment (the stage never spans scene
cuts inside one buffer), so no cross-scene contamination is possible.
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import RepairContext


def _fit(ref: np.ndarray, cur: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel affine (gain, offset) mapping ``cur``'s palette onto ``ref``'s."""
    gain = np.ones(3, dtype=np.float64)
    offset = np.zeros(3, dtype=np.float64)
    for c in range(3):
        ref_mean = float(ref[..., c].mean())
        ref_std = float(ref[..., c].std()) + 1e-4
        cur_mean = float(cur[..., c].mean())
        cur_std = float(cur[..., c].std()) + 1e-4
        gain[c] = ref_std / cur_std
        offset[c] = ref_mean - gain[c] * cur_mean
    return gain, offset


def repair(ctx: RepairContext) -> tuple[np.ndarray, str]:
    if ctx.scene_ref_proxy is None or ctx.center_proxy is None:
        return ctx.frame, "none"
    ref = ctx.scene_ref_proxy.astype(np.float32)

    # Window-median coefficients (Q4): fit every temporal neighbor too and take the
    # median per channel — temporally smooth across a sliding window.
    if len(ctx.window) >= 2:
        rh, rw = ref.shape[:2]
        fits = []
        for f in ctx.window:
            p = cv2.resize(f.astype(np.float32), (rw, rh), interpolation=cv2.INTER_AREA) / 255.0
            fits.append(_fit(ref, p))
    else:
        # No temporal context (e.g. unit-test single frame): legacy per-frame fit.
        fits = [_fit(ref, ctx.center_proxy.astype(np.float32))]

    gain = np.median([g for g, _ in fits], axis=0).astype(np.float32)
    offset = np.median([o for _, o in fits], axis=0).astype(np.float32)

    strength = 0.7
    f = ctx.frame.astype(np.float32) / 255.0
    corrected = (f * (1.0 - strength) + (f * gain[None, None, :] + offset[None, None, :]) * strength)
    return np.clip(np.round(corrected * 255.0), 0, 255).astype(np.uint8), "color-match"

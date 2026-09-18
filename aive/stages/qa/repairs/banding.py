"""Banding repair (design.md §4.3): deband + fine dither in smooth regions.

The detector flags long plateaus of identical values inside smooth gradients. The
CV fix adds a small amount of fine, spatially-varying dither *only* where the mask
says banding exists and the region is genuinely flat (low Laplacian), which breaks
the constant runs without adding visible grain to textured areas. (Priority-1
deepDeband-f AI debanding plugs in behind this interface later.)
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import RepairContext


def repair(ctx: RepairContext) -> tuple[np.ndarray, str]:
    frame = ctx.frame
    h, w = frame.shape[:2]
    g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
    smooth = (lap < 0.01).astype(np.float32)  # flat/gradient, not an edge

    region = np.clip(ctx.mask * smooth, 0.0, 1.0)
    if not (region > 0.05).any():
        return frame, "none"

    # Fine dither: ~±1 luma level, scaled by the region mask so it fades out at the
    # edges of the flagged banding and never touches textured pixels.
    rng = np.random.default_rng(1234)  # deterministic per-frame is fine (not visible)
    noise = (rng.uniform(-1.0, 1.0, size=(h, w)) * region).astype(np.float32)[..., None]
    out = frame.astype(np.float32) + noise
    return np.clip(np.round(out), 0, 255).astype(np.uint8), "deband-dither"

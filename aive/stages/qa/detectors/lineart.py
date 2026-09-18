"""Line-art degradation detector, anime (design.md §4.2).

Anime line art is high-contrast thin structure. We extract a line mask from both
the processed frame and the source (pixels markedly darker than their immediate
surrounds), then measure how much the local *line density* changed between them on
the union of the two masks. Thickened, blurred or eroded lines all shift that
density, so this catches width/position drift against ground truth.
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import FrameWindow, luma

_LINE_THRESH = 0.12  # luma difference from local surround to count as a line


def _line_mask(lum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lum = lum.astype(np.float32)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    surround = cv2.dilate(lum, kern)  # local max => brighter surroundings
    dark = np.clip(surround - lum, 0.0, None).astype(np.float32)
    mask = (dark > _LINE_THRESH).astype(np.float32)
    return mask, dark


def detect(win: FrameWindow, sens: float) -> tuple[np.ndarray, float]:
    h, w = win.center.shape[:2]
    if win.src_center is None:
        return np.zeros((h, w), dtype=np.float32), 0.0

    line_c, _ = _line_mask(luma(win.center))
    line_s, _ = _line_mask(luma(win.src_center))
    union = np.maximum(line_c, line_s)
    if not union.any():
        return np.zeros((h, w), dtype=np.float32), 0.0

    thick_c = cv2.boxFilter(line_c.astype(np.float32), -1, (5, 5))
    thick_s = cv2.boxFilter(line_s.astype(np.float32), -1, (5, 5))
    # Higher sensitivity => smaller density change counts as drift.
    ref = max(2.0, 6.0 - 4.0 * sens)
    drift = np.clip(np.abs(thick_c - thick_s) / ref, 0.0, 1.0).astype(np.float32) * union

    sev = float(drift[union > 0].mean()) if (union > 0).any() else 0.0
    return drift, sev

"""Detector registry (design.md §4.2): maps each artifact type to its detector.

Every detector is a pure function ``(FrameWindow, sens) -> (mask [H,W] in [0,1],
severity in [0,1])`` at proxy resolution. The registry keeps the import of the
individual detector modules lazy so that importing :mod:`aive.stages.qa` stays
cheap (no cv2/numpy pulled until a detector actually runs).
"""
from __future__ import annotations

import numpy as np

from ..findings import ArtifactType
from .base import FrameWindow


def detect(artifact: ArtifactType, win: FrameWindow, sens: float) -> tuple[np.ndarray, float]:
    """Run the detector for ``artifact`` on a proxy-res window.

    Returns ``(mask, severity)`` where ``mask`` is a single-channel region map in
    ``[0, 1]`` at proxy resolution and ``severity`` is an overall ``[0, 1]`` score
    for the center frame. Detectors degrade gracefully (zero mask / zero severity)
    when their required context is missing — see each module's docstring.
    """
    from . import banding, color, flicker, ghosting, halo, lineart, regression

    registry = {
        ArtifactType.FLICKER: flicker.detect,
        ArtifactType.GHOSTING: ghosting.detect,
        ArtifactType.BANDING: banding.detect,
        ArtifactType.HALO: halo.detect,
        ArtifactType.LINEART: lineart.detect,
        ArtifactType.COLOR_DRIFT: color.detect,
        ArtifactType.REGRESSION: regression.detect,
    }
    return registry[artifact](win, sens)


__all__ = ["detect", "FrameWindow"]

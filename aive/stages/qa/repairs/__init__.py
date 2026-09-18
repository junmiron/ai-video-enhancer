"""Repair registry (design.md §4.3): maps each artifact type to its repair policy.

Every repair is a pure function ``(RepairContext) -> (repaired_frame, method)`` at
full resolution. The import of the individual repair modules is lazy so importing
:mod:`aive.stages.qa` stays cheap until a repair actually runs.
"""
from __future__ import annotations

import numpy as np

from ..findings import ArtifactType
from .base import RepairContext


def repair(artifact: ArtifactType, ctx: RepairContext) -> tuple[np.ndarray, str]:
    """Apply the artifact-specific fix to one full-res frame.

    Returns ``(repaired_frame HWC uint8, method_name)`` where ``method_name`` is a
    short label for the report (e.g. ``"temporal-smooth"``) or ``"none"`` when the
    repair could not act on this frame.
    """
    from . import banding, color, flicker, ghosting, halo, lineart, regression

    registry = {
        ArtifactType.FLICKER: flicker.repair,
        ArtifactType.GHOSTING: ghosting.repair,
        ArtifactType.BANDING: banding.repair,
        ArtifactType.HALO: halo.repair,
        ArtifactType.LINEART: lineart.repair,
        ArtifactType.COLOR_DRIFT: color.repair,
        ArtifactType.REGRESSION: regression.repair,
    }
    return registry[artifact](ctx)


__all__ = ["repair", "RepairContext"]

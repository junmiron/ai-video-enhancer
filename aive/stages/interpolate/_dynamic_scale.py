"""Pyramid-scale selection for RIFE.

TAS's dynamic_scale picks the pyramid scale per frame pair from SSIM (requires
the external `frame_analytics` package). v1 keeps a fixed scale of 1.0 — TAS's
own default (`--scale=1.0`) — so this stub preserves the arch's import surface
without pulling in that dependency. A dynamic implementation can replace it
later without touching the architecture file.
"""
from __future__ import annotations

import torch


def pickScale(model, img0: torch.Tensor, img1: torch.Tensor) -> float:
    """Fixed pyramid scale (TAS default). See module docstring."""
    return 1.0

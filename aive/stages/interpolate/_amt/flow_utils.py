"""Minimal flow utilities vendored from AMT's ``utils/flow_utils.py``.

Only :func:`warp` is used by the AMT blocks; the upstream module's color-wheel /
visualization helpers are omitted to keep this dependency-light (no PIL/numpy).
The implementation is copied verbatim so results match upstream exactly.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def warp(img, flow):
    B, _, H, W = flow.shape
    xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
    yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([xx, yy], 1).to(img)
    flow_ = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)
    grid_ = (grid + flow_).permute(0, 2, 3, 1)
    output = F.grid_sample(input=img, grid=grid_, mode="bilinear", padding_mode="border", align_corners=True)
    return output

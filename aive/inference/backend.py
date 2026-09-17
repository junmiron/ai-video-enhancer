"""PyTorch CUDA runner (fallback backend + reference for TRT validation).

Contract: all runners take NCHW float32 numpy input and return NCHW float32
numpy output. Model adapters handle normalization/denormalization around this.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class TorchRunner:
    def __init__(self, model, device: str = "cuda", dtype=None):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.dtype = dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)
        self.model = model.to(self.device).eval()
        if self.dtype == torch.float16:
            self.model = self.model.half()

    def infer(self, x_nchw_f32: np.ndarray) -> np.ndarray:
        import torch

        t = torch.from_numpy(np.ascontiguousarray(x_nchw_f32)).to(self.device, non_blocking=True).to(self.dtype)
        with torch.no_grad():
            out = self.model(t)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.float().cpu().numpy()

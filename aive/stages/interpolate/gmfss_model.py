"""GMFSS frame-interpolation runner (PyTorch-CUDA).

Fixed-resolution runner mirroring :class:`aive.stages.interpolate.stage.RIFERunner`'s
contract — ``runner(img0, img1, t) -> HWC uint8`` — with two GMFSS-specific
behaviors:

- **per-pair reuse cache**: the model protocol is ``reuse(img0, img1)`` once per
  frame pair (bidirectional flow + metrics + features), then ``inference(t)``
  per intermediate slot. The stage calls the runner per slot, so we key a cheap
  content fingerprint of the pair and re-run ``reuse`` only when the pair
  changes. Cached GPU tensors also mean each source frame is uploaded once per
  pair instead of once per slot.
- **no TensorRT path**: the softsplat splatting is an ``index_add_`` scatter
  with data-dependent boundary masks, which the ONNX exporter cannot represent
  (and GMFlow's transformer stack has never been export-verified). CUDA fp16
  below UHD / fp32 at/above it (same TAS flicker rule as RIFE) is therefore
  the primary and only backend; see the manifest entry.

Flow resolution follows TAS: computed on a further half-downscaled copy above
1080p sources (``scale=0.5``), full otherwise.
"""
from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path

import numpy as np

from .stage import check_frame_dims  # lazy-safe: stage.py never imports this module at top level

log = logging.getLogger(__name__)


class GMFSSRunner:
    """Runs the GMFSS model at a fixed (padded) resolution on PyTorch-CUDA."""

    def __init__(self, weights_dir: str | Path, width: int, height: int):
        import torch

        from ._gmfss.model import GMFSSModel

        self.torch = torch
        self.width, self.height = width, height
        # TAS rule (OR condition): flow at half res above 1080p sources.
        self.scale = 0.5 if (width > 1920 or height > 1080) else 1.0
        # GMFlow's /4 feature uses attn_splits=8, so the flownet input must be a
        # multiple of 32 in both dims. The flownet runs at padded/2 (scale 1.0)
        # or padded/4 (scale 0.5), hence pad to 64 or 128 respectively — otherwise
        # split_feature asserts on aspect ratios whose ceil(dim/64) is odd.
        self.pad_mod = 64 if self.scale == 1.0 else 128
        self.pw = math.ceil(width / self.pad_mod) * self.pad_mod
        self.ph = math.ceil(height / self.pad_mod) * self.pad_mod
        # Same UHD precision rule as RIFERunner: fp32 at/above UHD, fp16 below.
        self.uhd = width > 1920 and height > 1080
        dtype = torch.float32 if self.uhd else torch.float16

        model = GMFSSModel()
        model.load_weights(weights_dir)
        model.to(device="cuda", dtype=dtype, memory_format=torch.channels_last).eval()
        self.model = model

        # Per-pair reuse cache (see module docstring).
        self._cache_key: str | None = None
        self._reuse = None
        self._pair_tensors = None  # (img0, img1) NCHW on GPU — kept for re-halving in inference()

        log.info(
            "GMFSS runner ready (%dx%d padded to %dx%d, flow scale %.2f, %s)",
            width, height, self.pw, self.ph, self.scale, dtype,
        )

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _fingerprint(frame_hwc_u8: np.ndarray) -> str:
        """Cheap content hash of a frame (stride-4 subsample of raw bytes)."""
        sub = np.ascontiguousarray(frame_hwc_u8[::4, ::4])
        return hashlib.sha256(sub.tobytes()).hexdigest()

    def _pad(self, frame_hwc_u8: np.ndarray) -> "np.ndarray":
        """HWC uint8 [h,w] -> NCHW float32 [0,1] padded to (ph,pw), zero border."""
        check_frame_dims(frame_hwc_u8, self.width, self.height, type(self).__name__)
        f = frame_hwc_u8.astype(np.float32) / 255.0
        nchw = np.ascontiguousarray(f.transpose(2, 0, 1)[None])
        if self.pw != self.width or self.ph != self.height:
            padded = np.zeros((1, 3, self.ph, self.pw), dtype=np.float32)
            padded[:, :, : self.height, : self.width] = nchw
            return padded
        return nchw

    def _to_gpu(self, frame_hwc_u8: np.ndarray):
        torch = self.torch
        dev = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        return (
            torch.from_numpy(self._pad(frame_hwc_u8))
            .to(dev, non_blocking=True)
            .to(dtype=dtype, memory_format=torch.channels_last)
        )

    # -- inference -------------------------------------------------------------

    def __call__(self, img0: np.ndarray, img1: np.ndarray, t: float) -> np.ndarray:
        """Interpolate one frame between HWC uint8 frames at timestep t in (0,1)."""
        torch = self.torch
        key = self._fingerprint(img0) + "|" + self._fingerprint(img1)
        if key != self._cache_key or self._reuse is None:
            a = self._to_gpu(img0)
            b = self._to_gpu(img1)
            with torch.inference_mode():
                self._reuse = self.model.reuse(a, b, scale=self.scale)
            self._pair_tensors = (a, b)
            self._cache_key = key
        else:
            a, b = self._pair_tensors

        with torch.inference_mode():
            out = self.model.inference(a, b, self._reuse, float(t))
        # Model output is clamped [0,1] NCHW at padded resolution — crop and scale.
        arr = out[0, :, : self.height, : self.width].float().cpu().numpy()
        return np.clip(np.round(arr.transpose(1, 2, 0) * 255.0), 0, 255).astype(np.uint8)

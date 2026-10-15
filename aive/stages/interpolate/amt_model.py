"""AMT frame-interpolation runner (PyTorch-CUDA).

Fixed-resolution runner mirroring :class:`aive.stages.interpolate.stage.RIFERunner`'s
contract — ``runner(img0, img1, t) -> HWC uint8`` — for the AMT architecture
(MCG-NKU/AMT, CVPR 2023), vendored under :mod:`._amt`.

Notes specific to AMT:

- **arbitrary timestep via ``embt``**: AMT takes a scalar time embedding shaped
  ``[1,1,1,1]``; the flow-completion stage scales the bidirectional flows by
  ``t`` / ``(1-t)`` internally. We pass ``embt = t`` clamped into ``(0,1)`` (the
  model divides by ``t`` and ``1-t``, so exact endpoints would blow up — but the
  stage never asks for them; original frames are emitted directly).
- **no per-pair reuse**: unlike GMFSS/M2M, AMT computes its bidirectional flow
  inside :meth:`forward` with no separable ``reuse`` step, so each inserted slot
  costs one full forward. That is the honest cost of this architecture behind our
  per-timestep interface; it stays comfortably real-time-ish on a 5090 for anime
  resolutions and matches how upstream's own multi-interval demo works.
- **no TensorRT path**: the correlation-volume + ``grid_sample`` warp stack has
  never been export-verified (and AMT is transformer-free but still dynamic), so
  CUDA fp16 below UHD / fp32 at/above it (the project's standard flicker rule) is
  the only backend; see the manifest entry.

Flow resolution follows upstream: a VRAM-aware ``scale_factor`` downscales the
flow computation for very large frames and leaves small ones at full resolution.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from .stage import check_frame_dims  # lazy-safe: stage.py never imports this module at top level

log = logging.getLogger(__name__)


def _flow_scale(width: int, height: int) -> float:
    """Upstream's VRAM-aware flow downscale factor (clamped to [0.25, 1.0]).

    Copied from AMT's demo heuristic: full resolution when the frame is small or
    VRAM is plentiful, otherwise step the flow map down in units of 1/16 so the
    correlation volume fits. Falls back to 1.0 if CUDA info is unavailable.
    """
    try:
        import torch

        vram_avail = torch.cuda.get_device_properties("cuda").total_memory
    except Exception:  # pragma: no cover - non-CUDA dev box
        return 1.0
    anchor_resolution = 1024 * 512
    anchor_memory = 1500 * 1024**2
    anchor_memory_bias = 2500 * 1024**2

    scale = anchor_resolution / (width * height) * math.sqrt((vram_avail - anchor_memory_bias) / anchor_memory)
    if not math.isfinite(scale) or scale <= 0:
        return 1.0
    scale = min(1.0, scale)
    denom = math.floor(1.0 / math.sqrt(scale) * 16.0)
    if denom < 1:
        return 1.0
    scale = 1.0 / denom * 16.0
    # Guard against pathologically small flow maps (upstream leaves this open).
    return max(0.25, min(scale, 1.0))


class AMTRunner:
    """Runs the AMT model at a fixed (padded) resolution on PyTorch-CUDA."""

    def __init__(self, weights_path: str | Path, width: int, height: int, variant: str = "s"):
        import torch

        from ._amt.model import build_amt, load_amt_weights

        self.torch = torch
        self.variant = variant.lower()
        self.width, self.height = width, height
        # AMT downsamples to H//16 internally; upstream pads to a multiple of 16.
        self.pad_mod = 16
        self.pw = math.ceil(width / self.pad_mod) * self.pad_mod
        self.ph = math.ceil(height / self.pad_mod) * self.pad_mod
        # Same UHD precision rule as RIFERunner/GMFSSRunner: fp32 at/above UHD,
        # fp16 below.
        self.uhd = width > 1920 and height > 1080
        dtype = torch.float32 if self.uhd else torch.float16

        model = build_amt(self.variant)
        load_amt_weights(model, weights_path)
        model.to(device="cuda", dtype=dtype).eval()
        self.model = model

        # Flow downscale computed on the padded working size.
        self.scale_factor = _flow_scale(self.pw, self.ph)

        log.info(
            "AMT-%s runner ready (%dx%d padded to %dx%d, flow scale %.2f, %s)",
            self.variant.upper(), width, height, self.pw, self.ph, self.scale_factor, dtype,
        )

    # -- helpers ---------------------------------------------------------------

    def _pad(self, frame_hwc_u8: np.ndarray) -> "np.ndarray":
        """HWC uint8 [h,w] -> NCHW float32 [0,1] padded to (ph,pw), edge border."""
        check_frame_dims(frame_hwc_u8, self.width, self.height, type(self).__name__)
        f = frame_hwc_u8.astype(np.float32) / 255.0
        nchw = np.ascontiguousarray(f.transpose(2, 0, 1)[None])
        if self.pw != self.width or self.ph != self.height:
            # Replicate-pad bottom/right only (content stays at [0,0]); upstream
            # uses replicate padding too. The padded margin is cropped away later.
            pad_h = self.ph - self.height
            pad_w = self.pw - self.width
            nchw = np.pad(nchw, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="edge")
        return nchw

    def _to_gpu(self, frame_hwc_u8: np.ndarray):
        torch = self.torch
        dev = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        return (
            torch.from_numpy(self._pad(frame_hwc_u8))
            .to(dev, non_blocking=True)
            .to(dtype=dtype)
        )

    # -- inference -------------------------------------------------------------

    def __call__(self, img0: np.ndarray, img1: np.ndarray, t: float) -> np.ndarray:
        """Interpolate one frame between HWC uint8 frames at timestep t in (0,1)."""
        torch = self.torch
        # AMT divides by embt and (1-embt); keep strictly inside (0,1). The stage
        # never requests the exact endpoints, but clamp defensively.
        tt = min(max(float(t), 1e-3), 1.0 - 1e-3)

        a = self._to_gpu(img0)
        b = self._to_gpu(img1)
        dev = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        embt = torch.tensor(tt, device=dev, dtype=dtype).view(1, 1, 1, 1)

        with torch.inference_mode():
            out = self.model(a, b, embt, scale_factor=self.scale_factor, eval=True)["imgt_pred"]
        # Model output is clamped [0,1] NCHW at padded resolution — crop and scale.
        arr = out[0, :, : self.height, : self.width].float().cpu().numpy()
        return np.clip(np.round(arr.transpose(1, 2, 0) * 255.0), 0, 255).astype(np.uint8)

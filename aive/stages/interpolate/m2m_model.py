"""M2M frame-interpolation runner (PyTorch-CUDA).

Fixed-resolution runner mirroring :class:`aive.stages.interpolate.stage.RIFERunner`'s
contract — ``runner(img0, img1, t) -> HWC uint8`` — with two M2M-specific
behaviors:

- **no external padding**: the model self-pads (replicate) to a multiple of
  ratio*16 = 64 inside :meth:`reuse` and crops back in :meth:`inference`, so
  exact-resolution frames go in and come out. ``pw``/``ph`` below are reported
  for benchmarking only — they describe the model's internal padding.
- **per-pair reuse cache**: flow estimation (PWCNet bidir at half resolution +
  MotionRefineNet -> 4 branch flows) is the expensive part and depends only on
  the frame pair, so it runs once per pair; each intermediate slot then costs
  just occlusion metrics + the many-to-many splat. The stage calls the runner
  per slot, so we key a cheap content fingerprint of the pair (same scheme as
  :class:`GMFSSRunner`).

No TensorRT path: the softsplat splatting is an ``index_add_`` scatter with
data-dependent boundary masks that ONNX cannot represent (same blocker as
GMFSS). CUDA fp16 below UHD / fp32 at/above it (the project's standard flicker
rule) is therefore the primary and only backend; see the manifest entry.
"""
from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class M2MRunner:
    """Runs the M2M model at a fixed resolution on PyTorch-CUDA."""

    def __init__(self, weights_path: str | Path, width: int, height: int):
        import torch

        from ._m2m.model import M2MModel

        self.torch = torch
        self.width, self.height = width, height
        # Informational only — the model pads internally (see module docstring).
        self.pw = math.ceil(width / 64) * 64
        self.ph = math.ceil(height / 64) * 64
        # Same UHD precision rule as RIFERunner/GMFSSRunner: fp32 at/above UHD,
        # fp16 below. The splat exponent is bounded by exp(paramAlpha) with the
        # checkpoint's paramAlpha ~= 2.9, so fp16 cannot overflow it.
        self.uhd = width > 1920 and height > 1080
        dtype = torch.float32 if self.uhd else torch.float16

        model = M2MModel()
        model.load_weights(weights_path)
        model.to(device="cuda", dtype=dtype, memory_format=torch.channels_last).eval()
        self.model = model

        # Per-pair reuse cache (see module docstring).
        self._cache_key: str | None = None
        self._state = None

        log.info(
            "M2M runner ready (%dx%d, internal pad %dx%d, %s)",
            width, height, self.pw, self.ph, dtype,
        )

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _fingerprint(frame_hwc_u8: np.ndarray) -> str:
        """Cheap content hash of a frame (stride-4 subsample of raw bytes)."""
        sub = np.ascontiguousarray(frame_hwc_u8[::4, ::4])
        return hashlib.sha256(sub.tobytes()).hexdigest()

    def _to_gpu(self, frame_hwc_u8: np.ndarray):
        """HWC uint8 [h,w] -> NCHW in model dtype on GPU (no padding)."""
        torch = self.torch
        dev = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        f = frame_hwc_u8.astype(np.float32) / 255.0
        nchw = np.ascontiguousarray(f.transpose(2, 0, 1)[None])
        return (
            torch.from_numpy(nchw)
            .to(dev, non_blocking=True)
            .to(dtype=dtype, memory_format=torch.channels_last)
        )

    # -- inference -------------------------------------------------------------

    def __call__(self, img0: np.ndarray, img1: np.ndarray, t: float) -> np.ndarray:
        """Interpolate one frame between HWC uint8 frames at timestep t in (0,1)."""
        torch = self.torch
        key = self._fingerprint(img0) + "|" + self._fingerprint(img1)
        if key != self._cache_key or self._state is None:
            a = self._to_gpu(img0)
            b = self._to_gpu(img1)
            with torch.inference_mode():
                self._state = self.model.reuse(a, b)
            self._cache_key = key

        with torch.inference_mode():
            out = self.model.inference(self._state, float(t))
        # Model output is [0,1]-ish NCHW at exact resolution — clamp and scale.
        arr = out[0].float().cpu().numpy()
        return np.clip(np.round(arr.transpose(1, 2, 0) * 255.0), 0, 255).astype(np.uint8)

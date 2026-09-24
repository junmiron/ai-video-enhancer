"""FILM frame-interpolation runner (TensorRT, per-resolution fixed-shape).

FILM (google-research/frame-interpolation, ECCV'22) is a TF2 model, so — unlike
RIFE/GMFSS/M2M which run on PyTorch-CUDA — it runs through the project's ONNX ->
TensorRT path. The shipped artifact is a **dynamic-spatial** ONNX (one file for
all resolutions), converted offline from the official Style SavedModel and baked
to be TRT-parseable; :class:`FILMRunner` builds one cached **fixed-shape** fp16
engine per resolution from it (min=opt=max profile on the dynamic dims — no
re-export needed, there is no TF in the main env). Memory therefore scales with
the actual video size instead of a 4K ceiling, matching RIFE's engine strategy.

I/O layout differs from RIFE: inputs are **NHWC** ``x0[1,H,W,3]`` /
``x1[1,H,W,3]`` plus a scalar ``time[1,1]`` in (0,1); the single output is the
full-res NHWC image (unclamped — it can exceed [0,1], so we clamp then scale).

TensorRT 11 constraint: for the four FusionNet-decoder nearest resizes that use
``coordinate_transformation_mode="tf_half_pixel_for_nn"``, TRT only accepts
``nearest_mode="round_prefer_floor"`` (it hard-rejects the tf2onnx default
``floor``). The offline bake therefore sets ``round_prefer_floor`` and bakes a
constant ``scales=[1,1,2,2]`` into those resizes. This is the only TRT-supported
sampling for that coordinate mode; on real (smooth) content it deviates from the
exact TF nearest sampling by well under a gray level on average — see the
manifest notes and docs/benchmarks.md.

Precision follows the project flicker rule (same as RIFE): fp16 below UHD,
fp32 at/above 1080p sources. No PyTorch fallback exists (there is no torch port
of FILM), so a failed engine build surfaces as a clear error rather than a
silent backend switch.
"""
from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path

import numpy as np

from .stage import check_frame_dims  # lazy-safe: stage.py never imports this module at top level

log = logging.getLogger(__name__)


class FILMRunner:
    """Runs the FILM model at a fixed (padded) resolution on TensorRT."""

    PAD_MOD = 64  # FILM's decoder pyramid divides by 2 repeatedly; keep dims integral

    def __init__(self, weights_path: str | Path, width: int, height: int):
        import torch  # noqa: F401 - ensure the CUDA context is up before engine use

        self.width, self.height = width, height
        self.pw = math.ceil(width / self.PAD_MOD) * self.PAD_MOD
        self.ph = math.ceil(height / self.PAD_MOD) * self.PAD_MOD
        # TAS: UHD + fp16 flickers -> fp32 above 1080p sources (same rule as RIFE).
        self.uhd = width > 1920 and height > 1080
        self.precision = "default" if self.uhd else "fp16"

        self._runner = None
        self._out_name = "Identity:0"
        self._build_trt(weights_path)

    # -- backend setup -------------------------------------------------------

    def _build_trt(self, weights_path: str | Path) -> None:
        from ...inference.trt import TrtRunner, build_engine, engine_cache_dir

        weights_path = Path(weights_path)
        sha = hashlib.sha256(weights_path.read_bytes()).hexdigest()[:16]
        shape = (1, self.ph, self.pw, 3)
        profile = f"fixed{self.width}x{self.height}|pad{self.ph}x{self.pw}"
        key = hashlib.sha256(f"{sha}|{self.precision}|{profile}".encode()).hexdigest()[:32]

        cache_dir = engine_cache_dir() / "film"
        engine_path = cache_dir / f"film_{key}_{self.precision}.engine"

        if not engine_path.exists():
            # Fixed-shape profile (min=opt=max) on the dynamic-spatial ONNX:
            # TRT specializes every layer for this exact size and sizes its
            # buffers to it — no 4K ceiling, memory scales with the video.
            profiles = {"x0": (shape, shape, shape), "x1": (shape, shape, shape)}
            try:
                build_engine(weights_path, engine_path, precision=self.precision, profiles=profiles)
            except Exception as exc:  # no torch fallback for FILM -> surface clearly
                raise RuntimeError(
                    "Failed to build the FILM TensorRT engine. This is usually a VRAM "
                    f"shortage (a {self.ph}x{self.pw} fp16 engine needs several GB free). "
                    "Close other GPU apps (e.g. an LLM server) and retry, or process at a "
                    f"lower resolution. Underlying error: {str(exc)[:300]}"
                ) from exc
        self._runner = TrtRunner(engine_path)
        # Robust to the output node name; FILM has exactly one image output.
        if len(self._runner.out_names) == 1:
            self._out_name = self._runner.out_names[0]
        log.info(
            "FILM TRT runner ready (%dx%d padded to %dx%d, %s)",
            self.width, self.height, self.pw, self.ph, self.precision,
        )

    # -- inference -----------------------------------------------------------

    def _pad(self, frame_hwc_u8: np.ndarray) -> np.ndarray:
        """HWC uint8 [h,w] -> NHWC float32 [0,1] padded to (ph,pw), zero border."""
        check_frame_dims(frame_hwc_u8, self.width, self.height, type(self).__name__)
        f = frame_hwc_u8.astype(np.float32) / 255.0
        if self.pw != self.width or self.ph != self.height:
            padded = np.zeros((self.ph, self.pw, 3), dtype=np.float32)
            padded[: self.height, : self.width, :] = f
            return np.ascontiguousarray(padded[None])
        return np.ascontiguousarray(f[None])

    def __call__(self, img0: np.ndarray, img1: np.ndarray, t: float) -> np.ndarray:
        """Interpolate one frame between HWC uint8 frames at timestep t in (0,1)."""
        out = self._runner.infer_multi(
            {
                "x0": self._pad(img0),
                "x1": self._pad(img1),
                "time": np.array([[t]], dtype=np.float32),
            }
        )[self._out_name]
        # Output is NHWC [1,ph,pw,3], unclamped (may exceed [0,1]) -> crop + clamp.
        arr = out[0, : self.height, : self.width, :]
        return np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)

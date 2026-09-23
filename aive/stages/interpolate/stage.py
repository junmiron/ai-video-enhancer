"""Frame-interpolation pipeline stage.

Model lineup (manifest ``task: fi``): the RIFE 4.25 family (TensorRT primary,
PyTorch-CUDA fallback), GMFSS and M2M (M2.x; PyTorch-CUDA — their softsplat
splatting is not ONNX-exportable, see :mod:`gmfss_model` / :mod:`m2m_model`),
and FILM (M2.x; TensorRT fixed-shape engines per resolution, built from a
baked TF->ONNX export with min=opt=max profiles — see :mod:`film_model`).

Scene-cut protection is structural: the orchestrator runs this stage once per
scene segment, so interpolation never crosses a cut. De-duplication skips model
calls for near-identical consecutive source frames (common in anime) and fills
their slots with copies of the previous frame — timing stays exact because slot
counting comes from :mod:`gapplan`.

Backends: TensorRT engine where available (fp16 below UHD, fp32 at/above it per
TAS's flicker finding), PyTorch-CUDA otherwise.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from ...core.jobqueue import check_control  # M6: per-frame pause/cancel polling
from .gapplan import fps_ratio, gap_plan, interpolate_timestep
from .rife_model import (
    RIFE_VARIANTS,
    RIFEExportWrapper,
    build_rife,
    load_rife,
    reset_for_call,
)

log = logging.getLogger(__name__)


@dataclass
class InterpolateConfig:
    """User-facing interpolation settings (M6 UI maps onto this)."""

    enabled: bool = False
    model: str = "rife_425"  # manifest registry name (task: fi)
    target_fps: float | None = None  # e.g. 60.0 — wins over multiplier when set
    multiplier: int | None = None  # e.g. 2, 3, 4
    dedup_threshold: float = 0.0  # mean abs diff (0..1); 0 disables de-duplication

    def ratio(self, source_fps: float) -> tuple[int, int]:
        """(factor_num, factor_den) for gap_plan given the source fps."""
        if self.target_fps is not None and self.target_fps > 0:
            return fps_ratio(source_fps, self.target_fps)
        m = int(self.multiplier or 2)
        return m, 1

    def output_fps(self, source_fps: float) -> float:
        num, den = self.ratio(source_fps)
        return source_fps * num / den


class RIFERunner:
    """Runs one RIFE 4.25-family model at a fixed (padded) resolution on TRT or CUDA."""

    PAD_MOD = 64  # TAS's mod for the 4.25 family; keeps every pyramid level integral

    def __init__(
        self,
        weights_path: str | Path,
        width: int,
        height: int,
        use_trt: bool = True,
        variant: str = "rife_425",
    ):
        import torch

        if variant not in RIFE_VARIANTS:
            raise ValueError(f"Unknown RIFE variant '{variant}'. Known: {list(RIFE_VARIANTS)}")
        self.variant = variant
        self.torch = torch
        self.width, self.height = width, height
        self.pw = math.ceil(width / self.PAD_MOD) * self.PAD_MOD
        self.ph = math.ceil(height / self.PAD_MOD) * self.PAD_MOD
        # TAS: UHD + fp16 flickers -> fp32 above 1080p sources.
        self.uhd = width > 1920 and height > 1080
        self.precision = "default" if self.uhd else "fp16"

        self._trt_runner = None
        self._torch_model = None
        if use_trt:
            try:
                self._build_trt(weights_path)
            except Exception as exc:  # pragma: no cover - TRT build edge cases
                log.warning("TRT engine unavailable for RIFE (%s); falling back to PyTorch-CUDA.", exc)
        if self._trt_runner is None:
            dtype = torch.float32 if self.uhd else torch.float16
            self._torch_model = load_rife(self.variant, weights_path, device="cuda", dtype=dtype).eval()

    # -- backend setup -------------------------------------------------------

    def _build_trt(self, weights_path: str | Path) -> None:
        import hashlib

        from ...inference.trt import TrtRunner, build_engine, engine_cache_dir

        weights_path = Path(weights_path)
        sha = hashlib.sha256(weights_path.read_bytes()).hexdigest()[:16]
        profile = f"{self.variant}|{self.width}x{self.height}|pad{self.pw}x{self.ph}"
        key = hashlib.sha256(f"{sha}|{self.precision}|{profile}".encode()).hexdigest()[:32]

        cache_dir = engine_cache_dir() / "rife"
        onnx_path = cache_dir / f"{self.variant}_{key}.onnx"
        engine_path = cache_dir / f"{self.variant}_{key}_{self.precision}.engine"

        if not engine_path.exists():
            model = build_rife(self.variant, device="cuda", dtype=None)  # fp32 export; TRT picks layer precision
            load_state(weights_path, model)
            RIFEExportWrapper.export(model, onnx_path, self.ph, self.pw)
            del model
            import torch

            torch.cuda.empty_cache()
            build_engine(onnx_path, engine_path, precision=self.precision)
        self._trt_runner = TrtRunner(engine_path)
        log.info(
            "RIFE TRT runner ready (%s, %s, %dx%d padded to %dx%d)",
            self.variant, self.precision, self.width, self.height, self.pw, self.ph,
        )

    # -- inference -----------------------------------------------------------

    def __call__(self, img0: np.ndarray, img1: np.ndarray, t: float) -> np.ndarray:
        """Interpolate one frame between HWC uint8 frames at timestep t in (0,1)."""
        if self._trt_runner is not None:
            out = self._run_trt(img0, img1, t)
        else:
            out = self._run_torch(img0, img1, t)
        # Model output is [0,1] (convex blend of the normalized inputs) — scale
        # to uint8 before rounding; rounding without *255 binarizes the frame.
        return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)

    def _pad(self, frame_hwc_u8: np.ndarray) -> "np.ndarray | object":
        """HWC uint8 [h,w] -> NCHW float32 [0,1] padded to (ph,pw), zero border."""
        f = frame_hwc_u8.astype(np.float32) / 255.0
        nchw = np.ascontiguousarray(f.transpose(2, 0, 1)[None])
        if self.pw != self.width or self.ph != self.height:
            padded = np.zeros((1, 3, self.ph, self.pw), dtype=np.float32)
            padded[:, :, : self.height, : self.width] = nchw
            return padded
        return nchw

    def _run_trt(self, img0: np.ndarray, img1: np.ndarray, t: float) -> np.ndarray:
        ts = np.full((1, 1, self.ph, self.pw), t, dtype=np.float32)
        out = self._trt_runner.infer_multi(
            {"img0": self._pad(img0), "img1": self._pad(img1), "timestep": ts}
        )["output"]
        return out[0, :, : self.height, : self.width].transpose(1, 2, 0)

    def _run_torch(self, img0: np.ndarray, img1: np.ndarray, t: float) -> np.ndarray:
        torch = self.torch
        model = self._torch_model
        dev = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        with torch.no_grad():
            a = torch.from_numpy(self._pad(img0)).to(dev, non_blocking=True).to(dtype)
            b = torch.from_numpy(self._pad(img1)).to(dev, non_blocking=True).to(dtype)
            ts = torch.full((1, 1, self.ph, self.pw), t, device=dev, dtype=dtype)
            reset_for_call(model)
            out = model(a, b, ts)
        return out[0, :, : self.height, : self.width].float().cpu().numpy().transpose(1, 2, 0)


def load_state(weights_path: Path, model) -> None:
    """Load weights into a pre-built arch (inference keys must all be present)."""
    from .rife_model import apply_rife_state, load_rife_state_dict

    apply_rife_state(model, load_rife_state_dict(weights_path))


def build_fi_runner(model_name: str, weights_path, width: int, height: int):
    """Runner-builder registry keyed by manifest model name (M2.x).

    Each builder returns an object with the ``runner(img0, img1, t) -> HWC
    uint8`` contract. ``weights_path`` is whatever :func:`download_model`
    returned for that model — a file for single-checkpoint models, a directory
    for multi-file bundles (GMFSS).
    """
    if model_name in RIFE_VARIANTS:
        return RIFERunner(weights_path, width, height, variant=model_name)
    if model_name == "gmfss":
        from .gmfss_model import GMFSSRunner

        return GMFSSRunner(weights_path, width, height)
    if model_name == "m2m":
        from .m2m_model import M2MRunner

        return M2MRunner(weights_path, width, height)
    if model_name == "film":
        from .film_model import FILMRunner

        return FILMRunner(weights_path, width, height)
    known = ", ".join(sorted(RIFE_VARIANTS + ("gmfss", "m2m", "film")))
    raise ValueError(
        f"Model '{model_name}' has no frame-interpolation adapter yet. "
        f"Available: {known}."
    )


class InterpolateStage:
    """Pipeline stage: source frames -> interpolated frames at the target rate."""

    name = "interpolate"

    def __init__(self, cfg: InterpolateConfig):
        self.cfg = cfg
        self._runner: RIFERunner | None = None
        self._runner_dims: tuple[int, int] | None = None  # (w, h) the runner was built for; None = injected externally
        self.model_calls = 0
        self.dedup_skips = 0

    # -- stage interface ------------------------------------------------------

    def output_dims(self, ctx) -> tuple[int, int]:
        """Interpolation changes frame rate, never resolution."""
        return (ctx.width, ctx.height)

    def process(self, frames: Iterator[np.ndarray], ctx) -> Iterator[np.ndarray]:
        import torch  # noqa: F401 - ensure CUDA context is up before first model use

        num, den = self.cfg.ratio(_fps_from_str(ctx.fps))
        prev: np.ndarray | None = None
        prev_pos = 0
        ctl = getattr(ctx, "control", None)
        for cur_pos, frame in enumerate(frames):
            check_control(ctl)  # M6: pause/cancel takes effect within one source frame
            self._ensure_runner(frame, ctx)
            if prev is None:
                yield frame
                prev = frame
                continue

            n_insert, timesteps = gap_plan(prev_pos, cur_pos, num, den)
            if (
                self.cfg.dedup_threshold > 0
                and n_insert > 0
                and _mean_abs_diff(prev, frame) < self.cfg.dedup_threshold
            ):
                # Near-duplicate pair: fill slots with copies, no model calls.
                for _ in range(n_insert):
                    yield prev.copy()
                self.dedup_skips += n_insert
            else:
                for i in range(n_insert):
                    t = interpolate_timestep(i, n_insert, timesteps)
                    yield self._runner(prev, frame, t)
                    self.model_calls += 1

            yield frame
            prev = frame
            prev_pos = cur_pos


    def _ensure_runner(self, frame: np.ndarray, ctx) -> None:
        """Lazily build the FI runner from the first real input frame.

        The working resolution must match the frames actually flowing into this
        stage — source resolution in fi_first order, but the UPSCALED target when
        this stage runs after upscale (sr_first). ctx always carries source dims,
        so it cannot be used here. A runner injected externally (tests) with
        ``_runner_dims`` left as None is trusted as-is; a production-built runner
        is rebuilt if the input resolution changes mid-stream.
        """
        from ...models.downloader import download_model
        from ...models.registry import get_model

        h, w = frame.shape[:2]
        if self._runner is not None and (self._runner_dims is None or self._runner_dims == (w, h)):
            return
        info = get_model(self.cfg.model)
        weights = download_model(info)
        log.info("Interpolation: %s @ %dx%d (target %.3ffps, dedup=%s)",
                 self.cfg.model, w, h,
                 self.cfg.output_fps(_fps_from_str(ctx.fps)),
                 f"{self.cfg.dedup_threshold:g}" if self.cfg.dedup_threshold > 0 else "off")
        if self._runner is not None:
            ow, oh = self._runner_dims
            log.warning("InterpolateStage: input resolution changed %dx%d -> %dx%d; rebuilding runner",
                        ow, oh, w, h)
        self._runner = build_fi_runner(self.cfg.model, weights, w, h)
        self._runner_dims = (w, h)


def _fps_from_str(fps: str) -> float:
    """Parse ffmpeg's rational fps string ('24/1', '30000/1001') to float."""
    if "/" in fps:
        num, den = fps.split("/")
        return float(num) / max(1.0, float(den))
    return float(fps)


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference of two HWC uint8 frames in [0, 1]."""
    if a.shape != b.shape:
        return 1.0
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()) / 255.0

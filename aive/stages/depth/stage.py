"""Depth-map model adapters (M5): per-frame DAv2 + streaming DA3/Limbo.

Two inference modes, mirroring TheAnimeScripter's validated backends:

* **Per-frame** — Depth-Anything V2 Small on one frame at a time. Process
  resolution follows TAS's ``calculateAspectRatio`` semantics (quality
  high = each axis rounded up to a multiple of 14; medium/low = fixed
  700^2 / 518^2 squares). Output: per-frame min-max stretch.
* **Streaming** — DA3-family models (Limbo v1/v2 anime finetunes + official
  Depth-Anything-3 small/base) over a sliding window of N ordered views with
  50% overlap alignment (:mod:`aive.stages.depth.chunk_stream`). Input size is
  fixed to the two validated export shapes (504x280 widescreen / 504x378
  4:3-squarer, picked by log-aspect like TAS's ``limboResolution``); output:
  1/depth disparity + per-frame 2/98-percentile stretch (TAS's ``_writeDepth``).

Backend chain mirrors :class:`~aive.stages.qa.repairs.ai.RestoreRunner`:
TensorRT fp16 engine cached under ``engines/depth/`` keyed by weights sha,
with a VRAM-headroom guard that skips doomed builds and falls back to
PyTorch-CUDA (fp32 weights; bf16 autocast for the DA3 family — TAS's CUDA
baseline). Normalization is ImageNet mean/std in fp32 for every model:
TAS found fp16 normalization "costs range for nothing".

Runners are cached per (model, frame size, quality/window) for the process
lifetime; weights auto-download from the manifest (``task: depth``) on first
use. Any failure degrades to ``None`` (cached) so the depth pass reports a
clear error instead of retrying per frame.
"""
from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# ImageNet normalization (fp32 — see module docstring).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# The two spatial shapes the streaming ONNX export supports (baked into the
# graph; multiples of 14). Stored as (height, width), like TAS's LIMBO_SHAPES.
STREAMING_SHAPES: tuple[tuple[int, int], ...] = ((280, 504), (378, 504))

VALID_WINDOWS = (4, 8, 16, 32)


@dataclass(frozen=True)
class DepthModelSpec:
    """Static properties of one depth model family."""

    name: str
    mode: str  # "per_frame" | "streaming"
    arch: str  # vendored-arch preset: "dav2-small" | "da3-small" | "da3-base"
    experimental: bool = False


DEPTH_SPECS: dict[str, DepthModelSpec] = {
    # Per-frame monocular depth (official DAv2 Small, Apache-2.0).
    "depth_anything_v2_small": DepthModelSpec("depth_anything_v2_small", "per_frame", "dav2-small"),
    # Anime finetunes of DA3-small (CC BY-NC 4.0; personal use per decision D3).
    "limbo_v1": DepthModelSpec("limbo_v1", "streaming", "da3-small"),
    "limbo_v2": DepthModelSpec("limbo_v2", "streaming", "da3-small"),
    # Official DA3 streaming (Apache-2.0) — experimental: same validated TRT
    # path as Limbo, fixed 504x280/504x378 inputs instead of TAS's dynamic
    # CUDA process resolution.
    "depth_anything_3_small": DepthModelSpec("depth_anything_3_small", "streaming", "da3-small", experimental=True),
    "depth_anything_3_base": DepthModelSpec("depth_anything_3_base", "streaming", "da3-base", experimental=True),
}


def spec_for_depth(name: str) -> DepthModelSpec:
    if name not in DEPTH_SPECS:
        raise ValueError(f"Unknown depth model '{name}'. Known: {list(DEPTH_SPECS)}")
    return DEPTH_SPECS[name]


# -- process-resolution selection (TAS parity) ---------------------------------


def dav2_process_size(width: int, height: int, quality: str = "high") -> tuple[int, int]:
    """DAv2 input size for one source frame — TAS ``calculateAspectRatio`` v2.

    high   : each axis rounded UP to a multiple of 14 (aspect preserved);
    medium : fixed 700x700 square; low: fixed 518x518 square.
    Returns (process_w, process_h).
    """
    if quality == "high":
        return ((width + 13) // 14 * 14, (height + 13) // 14 * 14)
    size = 700 if quality == "medium" else 518
    if quality not in ("medium", "low"):
        raise ValueError(f"Unknown depth quality '{quality}'. Known: high/medium/low")
    return (size, size)


def streaming_shape(width: int, height: int) -> tuple[int, int]:
    """Pick the baked streaming input shape closest to the source aspect ratio.

    Compared in log space (TAS ``limboResolution``): 16:9/16:10 land on
    504x280, anything squarer than ~1.55:1 (incl. portrait) on 504x378.
    Returns (height, width).
    """
    aspect = max(width, 1) / max(height, 1)
    return min(STREAMING_SHAPES, key=lambda hw: abs(math.log(aspect) - math.log(hw[1] / hw[0])))


def _resize_frame(frame_hwc_u8: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize one HWC uint8 frame to (width, height); AREA down, LINEAR up."""
    import cv2

    if frame_hwc_u8.shape[1] == width and frame_hwc_u8.shape[0] == height:
        return frame_hwc_u8
    interp = cv2.INTER_AREA if (frame_hwc_u8.shape[1] > width or frame_hwc_u8.shape[0] > height) else cv2.INTER_LINEAR
    return cv2.resize(frame_hwc_u8, (width, height), interpolation=interp)


def normalize_nchw(frames_hwc_u8: np.ndarray) -> np.ndarray:
    """HWC uint8 [N,H,W,3] (or single [H,W,3]) -> NCHW float32 ImageNet-normalized."""
    x = frames_hwc_u8.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN[None, None, None, :]) / IMAGENET_STD[None, None, None, :]
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


# -- output mapping (TAS parity) -------------------------------------------------


def disparity_stretch(depth: np.ndarray) -> np.ndarray:
    """Positive depth [H,W] -> [0,1] gray via 1/depth + per-frame 2/98 stretch.

    Exact port of TAS's ``_writeDepth`` math (DA3/Limbo path): invalid pixels
    (non-finite or <= 0) map to black; a frame with <= 10 valid pixels is all
    black rather than a percentile over noise.
    """
    depth = np.ascontiguousarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if int(valid.sum()) <= 10:
        return np.zeros_like(depth)
    disparity = np.full_like(depth, np.nan)
    disparity[valid] = 1.0 / depth[valid]
    low, high = np.percentile(disparity[valid], (2, 98))
    gray = ((disparity - low) / max(float(high - low), 1e-6)).clip(0.0, 1.0)
    gray[~valid] = 0.0
    return gray.astype(np.float32)


def minmax_stretch(depth: np.ndarray) -> np.ndarray:
    """Positive depth [H,W] -> [0,1] gray via per-frame min-max (DAv2 path)."""
    d = np.nan_to_num(np.ascontiguousarray(depth, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(d.min())
    return ((d - lo) / max(float(d.max() - lo), 1e-6)).clip(0.0, 1.0).astype(np.float32)


# -- arch construction / weight loading ------------------------------------------


def build_dav2(device: str = "cuda", dtype=None):
    """Construct the vendored DAv2 Small (weights NOT loaded).

    Pinned to the official ``depth_anything_v2_small`` checkpoint: vits encoder,
    features=64, out_channels=[48,96,192,384] — verified as a strict 239-key
    state-dict match (the arch auto-folds LayerScale on first load).
    """
    from ._dav2.dpt_v2 import DepthAnythingV2

    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model.to(device)


def load_dav2(weights_path: str | Path, device: str = "cuda", dtype=None):
    """Load the DAv2 Small checkpoint; fatal on ANY state-dict mismatch."""
    import torch

    model = build_dav2(device=device, dtype=dtype)
    if weights_path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file as _load_st

        state = _load_st(str(weights_path), device="cpu")
    else:
        state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    result = model.load_state_dict(state, strict=True)  # folds LayerScale internally
    del state
    if result.missing_keys or result.unexpected_keys:
        sample = ", ".join((result.missing_keys + result.unexpected_keys)[:6])
        raise RuntimeError(f"DAv2 weights do not match the arch ({weights_path}): {sample}")
    log.info("Loaded DAv2 Small weights from %s", Path(weights_path).name)
    return model.eval()


# -- per-frame runner -------------------------------------------------------------


class PerFrameDepthRunner:
    """Runs DAv2 at a fixed process resolution on TRT or CUDA; one frame in,
    one [H,W] float32 gray map in [0,1] at the INPUT size out."""

    def __init__(
        self,
        weights_path: str | Path,
        width: int,
        height: int,
        model_name: str = "depth_anything_v2_small",
        quality: str = "high",
        use_trt: bool = True,
    ):
        import torch

        spec = spec_for_depth(model_name)
        if spec.mode != "per_frame":
            raise ValueError(f"{model_name} is a streaming model; use StreamingDepthRunner")
        self.spec = spec
        self.torch = torch
        self.width, self.height = width, height  # frame (input/output) size
        self.pw, self.ph = dav2_process_size(width, height, quality)

        log.info("Per-frame depth %s @ %dx%d: process %dx%d (%s)",
                 model_name, width, height, self.pw, self.ph, quality)

        self._trt_runner = None
        self._torch_model = None
        if use_trt:
            try:
                self._build_trt(weights_path)
            except Exception as exc:  # pragma: no cover - TRT build edge cases
                log.warning("TRT engine unavailable for %s (%s); falling back to PyTorch-CUDA.",
                            model_name, str(exc)[:300])
        if self._trt_runner is None:
            self._torch_model = load_dav2(weights_path, device="cuda", dtype=None).eval()

    def _build_trt(self, weights_path: str | Path) -> None:
        from ...inference.onnx_export import _ensure_self_contained
        from ...inference.trt import TrtRunner, build_engine, engine_cache_dir, free_vram_bytes

        weights_path = Path(weights_path)
        sha = hashlib.sha256(weights_path.read_bytes()).hexdigest()[:16]
        key = hashlib.sha256(f"{sha}|fp16|{self.pw}x{self.ph}".encode()).hexdigest()[:32]

        cache_dir = engine_cache_dir() / "depth"
        onnx_path = cache_dir / f"dav2_{key}.onnx"
        engine_path = cache_dir / f"dav2_{key}_fp16.engine"

        if not engine_path.exists():
            # Engine builds need VRAM headroom beyond inference (fp32 trace
            # forward + largest intermediate tensor + workspace pool). When
            # another process holds most of the GPU, skip the doomed build and
            # let the PyTorch-CUDA fallback cover it.
            min_free = 6 * 2**30
            free = free_vram_bytes()
            if free < min_free:
                log.info("Skipping TRT build for %s: %.1f GiB VRAM free (< 6 GiB headroom); using PyTorch-CUDA.",
                         self.spec.name, free / 2**30)
                return
            model = load_dav2(weights_path, device="cuda", dtype=None)  # fp32 export; TRT picks layer precision
            x = torch.zeros(1, 3, self.ph, self.pw, device="cuda")
            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                import contextlib
                import io

                with contextlib.redirect_stdout(io.StringIO()):
                    torch.onnx.export(
                        model, (x,), str(onnx_path),
                        input_names=["input"], output_names=["depth"],
                        opset_version=18, dynamo=False,  # legacy exporter: TAS-proven for this family
                    )
            _ensure_self_contained(onnx_path)
            del model
            torch.cuda.empty_cache()
            build_engine(onnx_path, engine_path, precision="fp16")
        self._trt_runner = TrtRunner(engine_path)
        log.info("Per-frame depth TRT runner ready (%s fp16, process %dx%d)",
                 self.spec.name, self.pw, self.ph)

    def __call__(self, frame_hwc_u8: np.ndarray) -> np.ndarray:
        """One HWC uint8 frame -> [H,W] float32 gray in [0,1] at the same size."""
        proc = _resize_frame(frame_hwc_u8, self.pw, self.ph)
        nchw = normalize_nchw(proc[None])
        if self._trt_runner is not None:
            out = self._trt_runner.infer(nchw)  # [1,1,ph,pw] float32
        else:
            model = self._torch_model
            dev = next(model.parameters()).device
            with self.torch.no_grad(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
                x = self.torch.from_numpy(nchw).to(dev, non_blocking=True)
                out = model(x)  # [1,1,ph,pw] positive depth at process size
            out = out.float().cpu().numpy()
        d = np.ascontiguousarray(out[0, 0])
        if (d.shape[0], d.shape[1]) != (self.height, self.width):
            import cv2

            d = cv2.resize(d, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        return minmax_stretch(d)


# -- streaming runner ---------------------------------------------------------------


class StreamingDepthRunner:
    """Runs one DA3-family model over sliding windows of ordered views.

    ``infer_chunk(frames)`` takes N HWC uint8 frames (N <= window, all from ONE
    scene segment) and returns [N,H,W] float32 positive depth at the baked
    model resolution — the contract :func:`stream_depth_chunks` expects.
    """

    def __init__(
        self,
        weights_path: str | Path,
        width: int,
        height: int,
        model_name: str = "limbo_v2",
        window: int = 8,
        use_trt: bool = True,
    ):
        import torch

        spec = spec_for_depth(model_name)
        if spec.mode != "streaming":
            raise ValueError(f"{model_name} is a per-frame model; use PerFrameDepthRunner")
        if window not in VALID_WINDOWS:
            raise ValueError(f"Streaming depth window must be one of {VALID_WINDOWS}")
        self.spec = spec
        self.torch = torch
        self.width, self.height = width, height  # frame (input/output) size
        self.window = window
        self.mh, self.mw = streaming_shape(width, height)  # baked model resolution

        log.info("Streaming depth %s @ %dx%d: input %dx%d, window=%d",
                 model_name, width, height, self.mw, self.mh, window)

        self._trt_runner = None
        self._torch_mv = None  # MultiViewDepth over the loaded net (fp32 weights)
        if use_trt:
            try:
                self._build_trt(weights_path)
            except Exception as exc:  # pragma: no cover - TRT build edge cases
                log.warning("TRT engine unavailable for %s (%s); falling back to PyTorch-CUDA.",
                            model_name, str(exc)[:300])
        if self._trt_runner is None:
            from ._da3_loader import load_da3
            from ._limbo_export import MultiViewDepth

            wrapper = load_da3(spec.arch, weights_path, device="cuda", dtype=None)  # fp32; bf16 autocast at runtime
            self._torch_mv = MultiViewDepth(wrapper.model, self.mh, self.mw)

    def _build_trt(self, weights_path: str | Path) -> None:
        from ...inference.trt import TrtRunner, build_engine, engine_cache_dir, free_vram_bytes
        from ._limbo_export import export_streaming_onnx

        cache_dir = engine_cache_dir() / "depth"
        onnx_path = export_streaming_onnx(
            weights_path, self.spec.arch, cache_dir, self.mh, self.mw, half=True
        )
        with open(onnx_path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()[:16]
        engine_path = (
            cache_dir / f"{self.spec.name}_{sha}_w{self.window}_{self.mw}x{self.mh}_fp16.engine"
        )

        if not engine_path.exists():
            min_free = 6 * 2**30
            free = free_vram_bytes()
            if free < min_free:
                log.info("Skipping TRT build for %s: %.1f GiB VRAM free (< 6 GiB headroom); using PyTorch-CUDA.",
                         self.spec.name, free / 2**30)
                return
            # Dynamic view axis: full window is opt AND max (the common case),
            # tail chunks run at their real length down to 1.
            build_engine(
                onnx_path, engine_path, precision="fp16",
                profiles={"image": ((1, 3, self.mh, self.mw),
                                    (self.window, 3, self.mh, self.mw),
                                    (self.window, 3, self.mh, self.mw))},
            )
        self._trt_runner = TrtRunner(engine_path)
        log.info("Streaming depth TRT runner ready (%s fp16, window=%d, input %dx%d)",
                 self.spec.name, self.window, self.mw, self.mh)

    def infer_chunk(self, frames: list[np.ndarray]) -> np.ndarray:
        """N HWC uint8 frames (one scene) -> [N,mh,mw] float32 positive depth."""
        count = len(frames)
        if not 1 <= count <= self.window:
            raise ValueError(f"Chunk of {count} frames outside window 1..{self.window}")
        imgs = np.stack([_resize_frame(f, self.mw, self.mh) for f in frames])
        nchw = normalize_nchw(imgs)  # [N,3,mh,mw] fp32 ImageNet-normalized

        if self._trt_runner is not None:
            out = self._trt_runner.infer_dynamic({"image": nchw})["depth"]
        else:
            mv = self._torch_mv
            with self.torch.no_grad(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
                x = self.torch.from_numpy(nchw).to("cuda", non_blocking=True)
                out_t = mv(x)  # [N,mh,mw] positive depth (head runs fp32 inside the arch)
            out = out_t.float().cpu().numpy()
        return np.ascontiguousarray(out, dtype=np.float32)


# -- process-level runner cache ------------------------------------------------------

_runner_cache: dict[tuple, object | None] = {}


def get_depth_runner(
    model_name: str,
    width: int,
    height: int,
    quality: str = "high",
    window: int = 8,
) -> PerFrameDepthRunner | StreamingDepthRunner | None:
    """Get (building on first use) the depth runner for one frame size.

    Returns ``None`` when depth is unavailable — no CUDA device, download
    failure, or build/load error. Failures are cached so the pass reports once
    instead of retrying per frame. Weights auto-download from the manifest
    (``task: depth``) on first use; see module docstring.
    """
    spec = spec_for_depth(model_name)  # raises on unknown model (caller handles)
    key = (model_name, width, height, quality if spec.mode == "per_frame" else None, window)
    if key in _runner_cache:
        return _runner_cache[key]

    import torch

    if not torch.cuda.is_available():
        log.warning("Depth model '%s' unavailable: no CUDA device.", model_name)
        _runner_cache[key] = None
        return None

    try:
        from ...models.downloader import download_model
        from ...models.registry import get_model

        info = get_model(model_name, task="depth")
        weights = download_model(info)
        if spec.mode == "per_frame":
            runner = PerFrameDepthRunner(weights, width, height, model_name=model_name, quality=quality)
        else:
            runner = StreamingDepthRunner(weights, width, height, model_name=model_name, window=window)
    except Exception as exc:
        log.warning("Depth model '%s' unavailable (%s).", model_name, str(exc)[:300])
        _runner_cache[key] = None
        return None

    _runner_cache[key] = runner
    return runner


def clear_depth_runner_cache() -> None:
    """Drop cached runners (tests / VRAM reclamation)."""
    import torch

    _runner_cache.clear()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - non-CUDA environments
        pass


__all__ = [
    "DEPTH_SPECS",
    "DepthModelSpec",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "PerFrameDepthRunner",
    "STREAMING_SHAPES",
    "StreamingDepthRunner",
    "VALID_WINDOWS",
    "build_dav2",
    "clear_depth_runner_cache",
    "dav2_process_size",
    "disparity_stretch",
    "get_depth_runner",
    "load_dav2",
    "minmax_stretch",
    "normalize_nchw",
    "spec_for_depth",
    "streaming_shape",
]

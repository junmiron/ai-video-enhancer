"""Super-resolution pipeline stage (M3): AI upscaling after interpolation.

Pipeline order follows design.md (TAS-proven): interpolate at source resolution,
then upscale — so the SR model sees fewer frames and runs on smaller inputs.

Backends: TensorRT engine (primary; fp16) with a PyTorch-CUDA fallback. On a
32 GB card full-frame inference is the default; when the estimated activation
memory for one frame exceeds 75% of free VRAM, the runner switches to tiled
inference with overlap + feathered blending (see :mod:`tiling`) so no seam
survives at any resolution.

When the requested output scale is an exact integer power of the model's
native scale (e.g. x4 or x8 on an x2 model), the stage cascades that many AI
passes so every step gets real super-resolution detail (true Nx). Any other
target falls back to a single pass at the native scale followed by a LANCZOS
resize.
"""
from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from ...core.jobqueue import check_control  # M6: per-frame pause/cancel polling
from .sr_model import SPECS, SRExportWrapper, build_sr, load_sr, spec_for
from .tiling import (
    OVERLAP,
    apply_edge_floor,
    blend_into,
    choose_tile_size,
    feather_mask,
    plan_tiles,
    upscale_mask,
)

log = logging.getLogger(__name__)


@dataclass
class UpscaleConfig:
    """User-facing upscaling settings (M6 UI maps onto this)."""

    enabled: bool = False
    model: str = "shufflecugan_x2"  # registry name
    scale: int | None = None  # target output scale; None -> the model's native scale


class SRRunner:
    """Runs one SR model at a fixed (padded) resolution on TRT or CUDA.

    Either full-frame (default when it fits in VRAM) or tiled with feathered
    blending — decided once at construction from current free VRAM.
    """

    def __init__(
        self,
        weights_path: str | Path,
        width: int,
        height: int,
        model_name: str,
        use_trt: bool = True,
    ):
        import torch

        spec = spec_for(model_name)
        self.spec = spec
        self.torch = torch
        self.width, self.height = width, height
        self.scale = spec.scale

        # Padded full-frame dims (model size constraints).
        self.pw = max(math.ceil(width / spec.pad_mod) * spec.pad_mod, spec.min_dim)
        self.ph = max(math.ceil(height / spec.pad_mod) * spec.pad_mod, spec.min_dim)

        # Full frame vs tiling: estimate activation VRAM for one full frame.
        # Conservative (nvidia-smi-aware) estimate: cudaMemGetInfo is blind to
        # VMM-mapped usage by other processes and would over-commit the GPU.
        from ...inference.trt import free_vram_bytes

        tile_cover = choose_tile_size(
            height, width, free_vram_bytes(), spec.bytes_per_pixel, pad_mod=spec.pad_mod
        )
        self.tile_cover = tile_cover  # None -> full-frame mode

        if tile_cover is None:
            self.engine_h, self.engine_w = self.ph, self.pw
        else:
            side = tile_cover + 2 * OVERLAP
            self.engine_h = max(math.ceil(side / spec.pad_mod) * spec.pad_mod, spec.min_dim)
            self.engine_w = max(math.ceil(side / spec.pad_mod) * spec.pad_mod, spec.min_dim)

        log.info(
            "SR %s @ %dx%d -> x%d: %s (engine %dx%d)",
            model_name, width, height, self.scale,
            f"tiled C={tile_cover} ov={OVERLAP}" if tile_cover else "full-frame",
            self.engine_w, self.engine_h,
        )

        self._trt_runner = None
        self._torch_model = None
        if use_trt:
            try:
                self._build_trt(weights_path)
            except Exception as exc:  # pragma: no cover - TRT build edge cases
                log.warning("TRT engine unavailable for %s (%s); falling back to PyTorch-CUDA.", model_name, exc)
        if self._trt_runner is None:
            self._torch_model = load_sr(model_name, weights_path, device="cuda", dtype=torch.float16).eval()

    # -- backend setup -------------------------------------------------------

    def _build_trt(self, weights_path: str | Path) -> None:
        from ...inference.trt import TrtRunner, build_engine, engine_cache_dir, free_vram_bytes

        weights_path = Path(weights_path)
        sha = hashlib.sha256(weights_path.read_bytes()).hexdigest()[:16]
        profile = f"{self.spec.name}|{self.width}x{self.height}|eng{self.engine_w}x{self.engine_h}"
        key = hashlib.sha256(f"{sha}|fp16|{profile}".encode()).hexdigest()[:32]

        cache_dir = engine_cache_dir() / "sr"
        onnx_path = cache_dir / f"{self.spec.name}_{key}.onnx"
        engine_path = cache_dir / f"{self.spec.name}_{key}_fp16.engine"

        if not engine_path.exists():
            # Engine builds need VRAM headroom beyond inference (largest
            # intermediate tensor + workspace pool). When another process holds
            # most of the GPU, skip the doomed multi-minute build and let the
            # PyTorch-CUDA fallback cover it.
            min_free = (6 if self.spec.scale >= 4 else 3) * 2**30
            free = free_vram_bytes()
            if free < min_free:
                log.info(
                    "Skipping TRT build for %s: %.1f GiB VRAM free (< %.0f GiB headroom); using PyTorch-CUDA.",
                    self.spec.name, free / 2**30, min_free / 2**30,
                )
                return
            model = build_sr(self.spec.name, device="cuda", dtype=None)  # fp32 export; TRT picks layer precision
            from .sr_model import apply_sr_state, finalize_sr, load_sr_state_dict

            apply_sr_state(model, load_sr_state_dict(weights_path))
            finalize_sr(model)  # e.g. SPAN Conv3XC kernel fusion before export
            SRExportWrapper.export(model, self.spec, onnx_path, self.engine_h, self.engine_w)
            del model
            import torch

            torch.cuda.empty_cache()
            build_engine(onnx_path, engine_path, precision="fp16")
        self._trt_runner = TrtRunner(engine_path)
        log.info("SR TRT runner ready (%s fp16, engine %dx%d)", self.spec.name, self.engine_w, self.engine_h)

    # -- inference -----------------------------------------------------------

    def __call__(self, frame_hwc_u8: np.ndarray) -> np.ndarray:
        """Upscale one HWC uint8 frame; returns HWC uint8 at native model scale."""
        if self.tile_cover is None:
            out = self._run_full_frame(frame_hwc_u8)
        else:
            out = self._run_tiled(frame_hwc_u8)
        # Model output is [0,1] (sigmoid/tanh-normalized) — scale to uint8
        # before rounding; rounding without *255 binarizes the frame.
        return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)

    def _to_nchw(self, frame_hwc_u8: np.ndarray) -> np.ndarray:
        f = frame_hwc_u8.astype(np.float32) / 255.0
        return np.ascontiguousarray(f.transpose(2, 0, 1)[None])

    def _pad_to_engine(self, region_nchw: np.ndarray) -> np.ndarray:
        """Edge-pad (or pass through) a NCHW region to the fixed engine size.

        Content is kept at [0,0]; the deficit is padded on bottom/right only
        with edge replication. Both consumers crop from the top-left corner
        (full-frame ``[:h, :w]`` and tile cover offsets), so centered padding
        would silently shift every padded region — see test_sr_pad.py.
        The padded area is never used in the output; replicated edges just keep
        it plausible context for models with global ops (SE blocks).
        """
        _, _, rh, rw = region_nchw.shape
        if rh == self.engine_h and rw == self.engine_w:
            return region_nchw
        take_h = min(rh, self.engine_h)
        take_w = min(rw, self.engine_w)
        core = region_nchw[:, :, :take_h, :take_w]
        dh = self.engine_h - take_h
        if dh > 0:
            core = np.concatenate(
                [core, np.repeat(core[:, :, -1:], dh, axis=2)],
                axis=2,
            )
        dw = self.engine_w - take_w
        if dw > 0:
            core = np.concatenate(
                [core, np.repeat(core[:, :, :, -1:], dw, axis=3)],
                axis=3,
            )
        return core

    def _infer(self, nchw_f32: np.ndarray) -> np.ndarray:
        """Run the model on a NCHW float32 [0,1] array at engine size; returns NCHW float32."""
        if self._trt_runner is not None:
            out = self._trt_runner.infer_multi({"input": nchw_f32})["output"]
            return np.ascontiguousarray(out)
        model = self._torch_model
        dev = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        with self.torch.no_grad():
            x = self.torch.from_numpy(nchw_f32).to(dev, non_blocking=True).to(dtype)
            out = model(x)
        return out.float().cpu().numpy()

    def _run_full_frame(self, frame_hwc_u8: np.ndarray) -> np.ndarray:
        nchw = self._pad_to_engine(self._to_nchw(frame_hwc_u8))
        out = self._infer(nchw)
        s = self.scale
        return out[0, :, : self.height * s, : self.width * s].transpose(1, 2, 0)

    def _run_tiled(self, frame_hwc_u8: np.ndarray) -> np.ndarray:
        h, w = self.height, self.width
        s = self.scale
        ov = OVERLAP
        acc = np.zeros((1, 3, h * s, w * s), dtype=np.float32)  # NCHW accumulator
        wsum = np.zeros((h * s, w * s), dtype=np.float32)

        for tile in plan_tiles(h, w, self.tile_cover, ov):
            # Model input region: cover expanded by the overlap margin.
            in_t = max(0, tile.top - ov)
            in_b = min(h, tile.top + tile.cover_h + ov)
            in_l = max(0, tile.left - ov)
            in_r = min(w, tile.left + tile.cover_w + ov)
            region = self._to_nchw(frame_hwc_u8[in_t:in_b, in_l:in_r])
            nchw = self._pad_to_engine(region)
            out = self._infer(nchw)

            # Crop the cover part of the model output (offsets account for clamping).
            ct = (tile.top - in_t) * s
            cb = ct + tile.cover_h * s
            cl = (tile.left - in_l) * s
            cr = cl + tile.cover_w * s
            out_crop = np.ascontiguousarray(out[0, :, ct:cb, cl:cr][None])

            mask = feather_mask(
                tile.cover_h, tile.cover_w,
                nominal_h=self.tile_cover, nominal_w=self.tile_cover, overlap=ov
            )
            # Frame-edge rows/cols would otherwise sum to zero weight (the ramp
            # reaches 0 at the tile origin and no neighbor exists beyond the
            # frame) -> normalization would divide by zero. Flooring only those
            # mask rows/cols keeps every other pixel exact.
            apply_edge_floor(mask, tile.top, tile.left, tile.cover_h, tile.cover_w, h, w)
            # Blend at OUTPUT resolution: nearest-upscale the feather mask so it
            # matches out_crop's dims (exact under nearest sampling).
            blend_into(acc, wsum, tile.top * s, tile.left * s, out_crop, upscale_mask(mask, s))

        with np.errstate(divide="ignore", invalid="ignore"):
            out = acc[0] / wsum[None]
        return np.nan_to_num(out).transpose(1, 2, 0)


def _cascade_passes(native_scale: int, target_scale: int) -> int | None:
    """Number of chained passes so that ``native_scale ** n == target_scale``.

    Returns an integer >= 1 when the target is an exact power of the native
    scale (so a cascade reaches it with no resampling), else ``None`` (the caller
    falls back to one pass + LANCZOS). A target equal to the native scale yields
    1 (a single pass, no resize).
    """
    if native_scale <= 1 or target_scale < native_scale:
        return None
    n = 0
    t = target_scale
    while t % native_scale == 0:
        t //= native_scale
        n += 1
    return n if t == 1 else None


class UpscaleStage:
    """Pipeline stage: frames at source resolution -> AI-upscaled frames."""

    name = "upscale"

    #: Clip (temporal diffusion) SR models have no fixed native scale; this is
    #: the default target when the user leaves the scale unset.
    CLIP_DEFAULT_SCALE = 4

    def __init__(self, cfg: UpscaleConfig):
        self.cfg = cfg
        # One runner per input resolution. Cascading (target a multiple of the
        # model's native scale) needs a chain of runners at increasing sizes;
        # they're built once and reused across all segments/frames of a job.
        self._runners: dict[tuple[int, int], SRRunner] = {}
        # Clip models (SeedVR2): one runner per (input size, scale) — it is
        # resolution-bound like the per-frame runners.
        self._clip_runners: dict[tuple[int, int, int], object] = {}
        self.frames_processed = 0

    def _info(self):
        """Manifest entry for the configured model (cached; raises if unknown)."""
        from ...models.registry import get_model

        if not hasattr(self, "_model_info"):
            self._model_info = get_model(self.cfg.model, task="sr")
        return self._model_info

    def _get_runner(self, weights: str | Path, width: int, height: int) -> SRRunner:
        """Build (or fetch from cache) a runner sized for one pass's input dims."""
        key = (width, height)
        runner = self._runners.get(key)
        if runner is None:
            runner = SRRunner(weights, width, height, model_name=self.cfg.model)
            self._runners[key] = runner
        return runner

    def _process_clip_sr(self, frames: Iterator[np.ndarray], ctx, info) -> Iterator[np.ndarray]:
        """Temporal diffusion SR (SeedVR2): buffer the segment's frames and run
        windowed clips through :class:`aive.stages.upscale._seedvr.runner.SeedVrRunner`.

        Clip models see whole temporal windows, so — unlike the per-frame path —
        the segment is buffered first (source-resolution frames: small), then
        processed in one go; output frames are yielded lazily afterwards.
        """
        from ...models.downloader import download_model
        from ._seedvr.runner import SeedVrRunner

        scale = self.cfg.scale or self.CLIP_DEFAULT_SCALE
        out_w, out_h = ctx.width * scale, ctx.height * scale

        key = (ctx.width, ctx.height, scale)
        runner = self._clip_runners.get(key)
        if runner is None:
            weights_dir = download_model(info)  # multi-file models return the cache dir
            log.info(
                "Upscaling: %s @ %dx%d (target x%s) - temporal clip SR",
                info.name, ctx.width, ctx.height, scale,
            )
            runner = SeedVrRunner(weights_dir, ctx.width, ctx.height, scale=scale)
            runner.load()  # VRAM-heavy; once per resolution/scale
            self._clip_runners[key] = runner

        ctl = getattr(ctx, "control", None)
        buf: list[np.ndarray] = []
        for frame in frames:
            check_control(ctl)  # M6: pause/cancel takes effect within one frame
            buf.append(frame)

        out_frames = runner(buf)
        need_resize = (runner.out_h, runner.out_w) != (out_h, out_w)
        if need_resize:
            import cv2

            log.info("SeedVR output %dx%d -> LANCZOS to exact target %dx%d", runner.out_w, runner.out_h, out_w, out_h)
        for f in out_frames:
            if need_resize:
                f = cv2.resize(f, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
            self.frames_processed += 1
            check_control(ctl)
            yield f

    # -- stage interface ------------------------------------------------------

    def output_dims(self, ctx) -> tuple[int, int]:
        """Output (width, height) for a context with the given input dims."""
        if not self.cfg.enabled:
            return (ctx.width, ctx.height)
        info = self._info()
        default_scale = self.CLIP_DEFAULT_SCALE if info.kind == "clip" else spec_for(info.name).scale
        scale = self.cfg.scale or default_scale
        return (ctx.width * scale, ctx.height * scale)

    def process(self, frames: Iterator[np.ndarray], ctx) -> Iterator[np.ndarray]:
        import torch  # noqa: F401 - ensure CUDA context is up before first model use

        info = self._info()
        if info.kind == "clip":
            yield from self._process_clip_sr(frames, ctx, info)
            return

        native_scale = spec_for(info.name).scale
        target_scale = self.cfg.scale or native_scale

        # Cascade (Option A): when the requested scale is an exact integer power
        # of the model's native scale, chain that many AI passes so every step
        # gets real SR detail (true Nx) instead of one pass + Lanczos resample.
        # Any other target keeps a single native-scale pass followed by LANCZOS.
        passes = _cascade_passes(native_scale, target_scale)  # int >=1 or None

        out_w, out_h = ctx.width * target_scale, ctx.height * target_scale

        if not self._runners:
            from ...models.downloader import download_model

            weights = download_model(info)
            n = passes or 1
            if passes is not None and passes > 1:
                log.info(
                    "Upscaling: %s @ %dx%d (target x%s) - cascading x%s over %d passes for true x%s",
                    self.cfg.model, ctx.width, ctx.height, target_scale,
                    native_scale, passes, target_scale,
                )
            elif passes is None:
                log.info("Upscaling: %s @ %dx%d (target x%s) -> LANCZOS to target",
                         self.cfg.model, ctx.width, ctx.height, target_scale)
            else:
                log.info("Upscaling: %s @ %dx%d (target x%s)",
                         self.cfg.model, ctx.width, ctx.height, target_scale)
            # Pre-build the whole chain so the first frame isn't slow and each
            # later pass's VRAM estimate already accounts for earlier engines.
            cw, ch = ctx.width, ctx.height
            for _ in range(n):
                self._get_runner(weights, cw, ch)
                cw *= native_scale
                ch *= native_scale

        ctl = getattr(ctx, "control", None)
        # Chain of runners: pass i runs at source * native^i.
        n = passes or 1
        chain = [self._runners[(ctx.width * (native_scale ** i), ctx.height * (native_scale ** i))]
                 for i in range(n)]
        for frame in frames:
            check_control(ctl)  # M6: pause/cancel takes effect within one frame
            up = frame
            for runner in chain:
                up = runner(up)  # each pass upscales by the model's native scale
            if passes is None:
                # Not an exact power of the native scale -> resample to target.
                import cv2

                up = cv2.resize(up, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
            self.frames_processed += 1
            yield up


def _fps_from_str(fps: str) -> float:
    """Parse ffmpeg's rational fps string ('24/1', '30000/1001') to float."""
    if "/" in fps:
        num, den = fps.split("/")
        return float(num) / max(1.0, float(den))
    return float(fps)

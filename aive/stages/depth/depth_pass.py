"""Depth-map pass (M5): an independent output generated after enhance completes.

Per design.md §3.4, depth is NOT part of the frame chain: this pass decodes
the SOURCE frames per scene segment and writes a separate depth video/sequence
next to the enhanced file. Scene segments double as cut handling — streaming
windows never cross a hard cut because each segment gets fresh chunk state
(:func:`stream_depth_chunks`), so no cross-cut blending can occur.

Runs on source frames by default (depth models are trained at fixed input
resolutions, so upscaling first buys little and costs a lot); the advanced
``post_upscale`` option pushes every frame through the pipeline's SR runner
first for higher-resolution depth maps.

No checkpointing in v1: the pass always runs fresh after enhance (~5-7 min per
episode at TAS speeds on an RTX 3090-class GPU). A failure here never fails the
job — the enhanced output is already written; the orchestrator records the
error in the result and emits it as a stage-scoped event.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DepthPassConfig:
    """User-facing depth settings (M6 UI maps onto this)."""

    model: str = "limbo_v2"  # registry name (task=depth)
    output: str = "mp4_10bit"  # mp4_10bit | png16
    quality: str = "high"  # per-frame models only: high | medium | low
    window: int = 8  # streaming models: 4 | 8 | 16 | 32 (TAS default is 32)
    post_upscale: bool = False  # advanced: run frames through the SR runner first
    sr_model: str | None = None  # used with post_upscale; None -> pipeline's SR model
    sr_scale: int | None = None  # used with post_upscale; None -> model's native scale


def depth_output_path(output_path: str | Path, fmt: str) -> Path:
    """Depth output location next to the enhanced file.

    ``mp4_10bit`` -> ``<stem>_depth.mp4`` (file); ``png16`` -> ``<stem>_depth/``
    (folder of %05d.png).
    """
    out = Path(output_path)
    if fmt == "png16":
        return out.with_name(out.stem + "_depth")
    return out.with_name(out.stem + "_depth.mp4")


def _sr_frame_stream(
    frames: Iterator[np.ndarray], sr_runner, target_w: int, target_h: int
) -> Iterator[np.ndarray]:
    """Push decoded source frames through the SR runner (post-upscale option)."""
    import cv2

    for frame in frames:
        up = sr_runner(frame)  # HWC uint8 at native scale
        if up.shape[1] != target_w or up.shape[0] != target_h:
            up = cv2.resize(up, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        yield up


def run_depth_pass(
    source_path: str | Path,
    depth_out_path: str | Path,
    cfg: DepthPassConfig,
    info,
    scene_split: bool = True,
    emit: Callable[[object], None] | None = None,
    control=None,
) -> dict:
    """Generate the depth output for one source video. Returns a summary dict.

    Raises on hard failures (unknown model, no runner, writer error); the
    orchestrator catches and degrades so the enhanced file is never lost.
    """
    from ..scenes import detect_scenes
    from .chunk_stream import stream_depth_chunks
    from .stage import DEPTH_SPECS, disparity_stretch, get_depth_runner, spec_for_depth
    from .writer import DepthWriter

    source_path = str(source_path)
    depth_out_path = Path(depth_out_path)
    t0 = time.time()

    def _emit(ev):
        if emit is not None:
            emit(ev)


    # Lazy: the orchestrator module is always loaded by the time this runs
    # (it calls us), and it does not import this module at top level.
    from ...core.orchestrator import PipelineCancelled

    spec = spec_for_depth(cfg.model)
    if cfg.output not in ("mp4_10bit", "png16"):
        raise ValueError(f"Unknown depth output format '{cfg.output}'")

    # Effective frame size: post-upscale runs the SR leg first, so both the
    # runner's process resolution and the writer work at upscaled dims.
    in_w, in_h = info.width, info.height
    out_w, out_h = in_w, in_h
    sr_runner = None
    if cfg.post_upscale:
        from ..upscale.stage import SRRunner, spec_for as sr_spec_for

        from ...models.downloader import download_model
        from ...models.registry import get_model

        sr_name = cfg.sr_model or "shufflecugan_x2"
        sr_info = get_model(sr_name, task="sr")
        weights = download_model(sr_info)
        native_scale = sr_spec_for(sr_name).scale
        target_scale = cfg.sr_scale or native_scale
        out_w, out_h = in_w * target_scale, in_h * target_scale
        log.info("Depth post-upscale: %s x%d -> frames at %dx%d", sr_name, target_scale, out_w, out_h)
        sr_runner = SRRunner(weights, in_w, in_h, model_name=sr_name)

    runner = get_depth_runner(cfg.model, out_w, out_h, quality=cfg.quality, window=cfg.window)
    if runner is None:
        raise RuntimeError(
            f"Depth model '{cfg.model}' unavailable (no CUDA device, download or build failed — "
            "see log). The enhanced output is unaffected."
        )

    segments = detect_scenes(source_path) if scene_split else [(0, info.num_frames)]
    total_frames = sum(e - s for s, e in segments)
    _emit(_stage_started(total_frames))
    log.info(
        "Depth pass: %s (%s) -> %s [%s], %d frames in %d segment(s)",
        cfg.model, spec.mode, depth_out_path, cfg.output, total_frames, len(segments),
    )

    writer = DepthWriter(depth_out_path, out_w, out_h, fps=info.fps, fmt=cfg.output)
    done = 0
    try:
        for start, end in segments:
            if control is not None:
                control.wait_if_paused()
                if control.cancelled:
                    raise PipelineCancelled()

            from ..ingest import decode_frames

            frames = decode_frames(source_path, start_frame=start, count=end - start)
            if sr_runner is not None:
                frames = _sr_frame_stream(frames, sr_runner, out_w, out_h)

            if spec.mode == "streaming":
                infer_chunk = runner.infer_chunk  # type: ignore[attr-defined]
                for d in stream_depth_chunks(frames, infer_chunk, cfg.window):
                    gray = disparity_stretch(d)
                    if (gray.shape[0], gray.shape[1]) != (out_h, out_w):
                        import cv2

                        gray = cv2.resize(gray, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                    writer.write(gray)
                    done += 1
                    _progress(done, total_frames, t0, emit)
                    if control is not None and control.cancelled:
                        raise PipelineCancelled()
            else:
                for frame in frames:
                    gray = runner(frame)  # type: ignore[operator]
                    writer.write(gray)
                    done += 1
                    _progress(done, total_frames, t0, emit)
                    if control is not None and control.cancelled:
                        raise PipelineCancelled()
    finally:
        try:
            writer.close()
        except Exception as close_exc:  # pragma: no cover - ffmpeg died mid-write
            log.warning("Depth writer close failed: %s", close_exc)

    elapsed = time.time() - t0
    result = {
        "path": str(depth_out_path),
        "format": cfg.output,
        "model": cfg.model,
        "frames": done,
        "width": out_w,
        "height": out_h,
        "elapsed_s": round(elapsed, 1),
    }
    log.info("Depth pass finished: %s (%.1fs, %.2f fps)", depth_out_path, elapsed, done / max(elapsed, 1e-6))
    _emit(_stage_finished(elapsed))
    return result


# -- event helpers (lazy imports keep this module light) ---------------------------

_PROGRESS_EVERY = 3  # frames between StageProgress events (depth is slow; stay responsive)


def _stage_started(total_frames: int):
    from ...core.events import StageStarted

    return StageStarted(stage="depth", total_frames=total_frames)


def _stage_finished(elapsed_s: float):
    from ...core.events import StageFinished

    return StageFinished(stage="depth", elapsed_s=elapsed_s)


def _progress(done: int, total: int, t0: float, emit) -> None:
    if done % _PROGRESS_EVERY != 0 and done != total:
        return
    from ...core.events import StageProgress

    elapsed = time.time() - t0
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else 0.0
    if emit is not None:
        emit(StageProgress(stage="depth", frames_done=done, frames_total=total, fps=rate, eta_s=eta))


__all__ = ["DepthPassConfig", "depth_output_path", "run_depth_pass"]

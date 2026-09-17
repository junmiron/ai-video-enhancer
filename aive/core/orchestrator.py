"""Pipeline orchestrator: runs stages per scene segment with checkpointing.

M1 stage chain is identity (decode -> re-encode lossless chunks -> final encode).
M2+ plug in InterpolateStage / UpscaleStage / QAStage via build_stage_chain().
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from ..config import Settings, workspaces_dir
from ..stages.encode import concat_encode, encode_lossless
from ..stages.ingest import MediaInfo, decode_frames, probe
from ..stages.scenes import detect_scenes
from .checkpoint import CheckpointStore, Segment
from .events import (
    ErrorEvent,
    JobFinished,
    JobStarted,
    StageFinished,
    StageProgress,
    StageStarted,
)
from .jobqueue import ControlToken

log = logging.getLogger(__name__)


class PipelineCancelled(Exception):
    pass


@dataclass
class PipelineConfig:
    scene_split: bool = True
    output_codec: str = "x265-10bit-anime"
    crf: int = 19
    container: str = "mp4"
    audio_passthrough: bool = True
    keep_workspace: bool = False

    @classmethod
    def from_settings(cls, s: Settings) -> "PipelineConfig":
        return cls(
            scene_split=s.scene_split,
            output_codec=s.output_codec,
            crf=s.crf,
            container=s.container,
            audio_passthrough=s.audio_passthrough,
            keep_workspace=s.keep_workspace,
        )


@dataclass
class StageContext:
    width: int
    height: int
    fps: str  # exact rational string for ffmpeg -r
    segment_index: int = 0
    start_frame: int = 0
    end_frame: int = 0
    settings: dict = field(default_factory=dict)


class Stage:
    """A pipeline stage transforms a frame iterator into another frame iterator."""

    name = "stage"

    def process(self, frames: Iterator[np.ndarray], ctx: StageContext) -> Iterator[np.ndarray]:
        raise NotImplementedError


class IdentityStage(Stage):
    """M1 placeholder: passes frames through unchanged (re-encode only)."""

    name = "identity"

    def process(self, frames, ctx):
        yield from frames


def build_stage_chain(cfg: PipelineConfig) -> list[Stage]:
    # M2+: append InterpolateStage / UpscaleStage / QAStage here based on cfg.
    return [IdentityStage()]


def _throttled_frames(frames: Iterator[np.ndarray], every: int, on_frame: Callable[[int], None]) -> Iterator[np.ndarray]:
    i = 0
    for f in frames:
        if i % every == 0:
            on_frame(i)
        yield f
        i += 1


def run_pipeline(
    job_id: str,
    input_path: str | Path,
    output_path: str | Path,
    cfg: PipelineConfig,
    emit: Callable[[object], None] | None = None,
    control: ControlToken | None = None,
) -> dict:
    """Run the full pipeline for one job. Returns a result summary dict."""
    input_path, output_path = str(input_path), str(output_path)
    t_start = time.time()

    def _emit(ev):
        if emit is not None:
            ev.job_id = job_id
            emit(ev)

    control = control or ControlToken()
    _emit(JobStarted(input_path=input_path))

    info = probe(input_path)
    log.info(
        "Input: %s %dx%d @%sfps %d frames (%.1fs), audio=%s",
        Path(input_path).name, info.width, info.height, info.fps, info.num_frames, info.duration_s,
        [a["codec"] for a in info.audio_streams] or "none",
    )

    if cfg.scene_split:
        raw_segments = detect_scenes(input_path)
    else:
        raw_segments = [(0, info.num_frames)]
    segments = [Segment(i, s, e) for i, (s, e) in enumerate(raw_segments)]

    store = CheckpointStore(workspaces_dir() / job_id)
    store.init_job(job_id, input_path, info.width, info.height, info.fps, segments)
    resumed_from = store.done_count
    if resumed_from:
        log.info("Resuming: %d/%d segment(s) already done.", resumed_from, len(segments))

    stages = build_stage_chain(cfg)
    total_frames = sum(seg.frame_count for seg in segments)
    frames_done_global = 0

    try:
        for seg in segments:
            if store.is_done(seg.index):
                continue
            control.wait_if_paused()
            if control.cancelled:
                raise PipelineCancelled()

            t_seg = time.time()
            _emit(StageStarted(stage="process", total_frames=seg.frame_count))
            log.info("Segment %d/%d frames [%d:%d]", seg.index + 1, len(segments), seg.start_frame, seg.end_frame)

            raw = decode_frames(input_path, start_frame=seg.start_frame, count=seg.frame_count)
            ctx = StageContext(info.width, info.height, info.fps, seg.index, seg.start_frame, seg.end_frame)
            stream: Iterator[np.ndarray] = raw
            for st in stages:
                stream = st.process(stream, ctx)

            def _on_frame(i: int):
                nonlocal frames_done_global
                done = frames_done_global + i
                frac = done / total_frames if total_frames else 1.0
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (total_frames - done) / rate if rate > 0 else 0.0
                _emit(StageProgress(stage="process", frames_done=done, frames_total=total_frames, fps=rate, eta_s=eta))

            seg_file = store.segment_file(seg.index)
            encode_lossless(_throttled_frames(stream, 30, _on_frame), seg_file, info.width, info.height, info.fps)
            frames_done_global += seg.frame_count
            store.mark_done(seg.index)
            _emit(StageFinished(stage="process", elapsed_s=time.time() - t_seg))

        # Final assembly: concat lossless chunks -> encode with chosen preset + audio.
        control.wait_if_paused()
        if control.cancelled:
            raise PipelineCancelled()
        t_final = time.time()
        _emit(StageStarted(stage="encode", total_frames=total_frames))
        log.info("Final encode: %s crf=%d -> %s", cfg.output_codec, cfg.crf, output_path)
        audio_src = input_path if cfg.audio_passthrough else None
        concat_encode(
            store.concat_list_file(),
            output_path,
            fps=info.fps,
            preset=cfg.output_codec,
            crf=cfg.crf,
            container=cfg.container,
            audio_src=audio_src,
        )
        _emit(StageFinished(stage="encode", elapsed_s=time.time() - t_final))

    except PipelineCancelled:
        log.warning("Pipeline cancelled by user.")
        raise
    except Exception as exc:
        _emit(ErrorEvent(message=str(exc)))
        raise

    if not cfg.keep_workspace:
        store.clear()

    result = {
        "output_path": output_path,
        "elapsed_s": time.time() - t_start,
        "segments_total": len(segments),
        "resumed_from": resumed_from,
        "report": None,  # M4 fills in the artifact report
    }
    _emit(JobFinished(ok=True, output_path=output_path, elapsed_s=result["elapsed_s"], report=None))
    return result

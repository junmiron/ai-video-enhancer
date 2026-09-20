"""Pipeline orchestrator: runs stages per scene segment with checkpointing.

Stage order follows design.md (TAS-proven): interpolate -> upscale -> QA/restore.
M2 plugs in InterpolateStage; M3/M4 append UpscaleStage / QAStage via
build_stage_chain(). When interpolation is enabled the output frame rate differs
from the source: per-segment lossless chunks and the final encode both use the
output fps, and progress accounting uses exact gap_plan slot counts.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from ..config import Settings, workspaces_dir
from ..stages.encode import concat_encode, encode_lossless
from ..stages.ingest import MediaInfo, decode_frames, probe
from ..stages.interpolate.gapplan import estimate_output_frames  # pure Python, no torch
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

    # Frame interpolation (M2)
    fi_enabled: bool = False
    fi_model: str = "rife_425"  # registry name
    fi_target_fps: float | None = None  # wins over multiplier when set
    fi_multiplier: int | None = None  # used when target fps is unset (default x2)
    fi_dedup_threshold: float = 0.0  # mean abs diff in [0,1]; 0 disables de-duplication

    # Super-resolution (M3)
    sr_enabled: bool = False
    sr_model: str = "shufflecugan_x2"  # registry name (task=sr)
    sr_scale: int | None = None  # target output scale; None -> model's native scale

    # Artifact QA / restore (M4) — design.md §4.4 user controls
    qa_enabled: bool = True
    qa_mode: str = "auto"  # auto | detect_only | manual
    qa_proxy_height: int = 720
    qa_window_size: int = 7
    qa_min_severity: float = 0.15
    qa_sensitivity: float = 0.5
    qa_ai_repairs: bool = True

    # Depth map pass (M5) — independent output, runs after enhance completes
    depth_enabled: bool = False
    depth_model: str = "limbo_v2"  # registry name (task=depth)
    depth_output: str = "mp4_10bit"  # mp4_10bit | png16
    depth_quality: str = "high"  # per-frame models only: high | medium | low
    depth_window: int = 8  # streaming models: 4 | 8 | 16 | 32 (TAS default is 32)
    depth_post_upscale: bool = False  # advanced: run frames through the SR runner first

    @classmethod
    def from_settings(cls, s: Settings) -> "PipelineConfig":
        return cls(
            scene_split=s.scene_split,
            output_codec=s.output_codec,
            crf=s.crf,
            container=s.container,
            audio_passthrough=s.audio_passthrough,
            keep_workspace=s.keep_workspace,
            fi_enabled=s.fi_enabled,
            fi_model=s.fi_model,
            fi_target_fps=s.fi_target_fps,
            fi_multiplier=s.fi_multiplier,
            fi_dedup_threshold=s.fi_dedup_threshold,
            sr_enabled=s.sr_enabled,
            sr_model=s.sr_model,
            sr_scale=s.sr_scale,
            qa_enabled=s.qa_enabled,
            qa_mode=s.qa_mode,
            qa_proxy_height=s.qa_proxy_height,
            qa_window_size=s.qa_window_size,
            qa_min_severity=s.qa_min_severity,
            qa_sensitivity=s.qa_sensitivity,
            qa_ai_repairs=s.qa_ai_repairs,
            depth_enabled=s.depth_enabled,
            depth_model=s.depth_model,
            depth_output=s.depth_output,
            depth_quality=s.depth_quality,
            depth_window=s.depth_window,
            depth_post_upscale=s.depth_post_upscale,
        )

    def interpolate_config(self):
        """InterpolateConfig for this pipeline, or None when FI is disabled."""
        if not self.fi_enabled:
            return None
        from ..stages.interpolate.stage import InterpolateConfig  # lazy: pulls torch deps

        return InterpolateConfig(
            enabled=True,
            model=self.fi_model,
            target_fps=self.fi_target_fps,
            multiplier=self.fi_multiplier,
            dedup_threshold=self.fi_dedup_threshold,
        )

    def upscale_config(self):
        """UpscaleConfig for this pipeline, or None when SR is disabled."""
        if not self.sr_enabled:
            return None
        from ..stages.upscale.stage import UpscaleConfig  # lazy: pulls torch deps

        return UpscaleConfig(enabled=True, model=self.sr_model, scale=self.sr_scale)

    def qa_config(self):
        """QAConfig for this pipeline, or None when QA is disabled."""
        if not self.qa_enabled:
            return None
        from ..stages.qa.findings import ArtifactType  # cheap (no cv2/torch)
        from ..stages.qa.stage import QAConfig  # lazy: pulls cv2 deps

        sens = {a.value: float(self.qa_sensitivity) for a in ArtifactType.all()}
        return QAConfig(
            enabled=True,
            ai_repairs=self.qa_ai_repairs,
            mode=self.qa_mode,
            proxy_height=self.qa_proxy_height,
            window_size=self.qa_window_size,
            min_severity=self.qa_min_severity,
            sensitivity=sens,
        )

    def depth_config(self):
        """DepthPassConfig for this pipeline, or None when the depth pass is disabled."""
        if not self.depth_enabled:
            return None
        from ..stages.depth.depth_pass import DepthPassConfig  # lazy: pulls cv2 deps

        sr_model = self.sr_model if self.sr_enabled else None
        return DepthPassConfig(
            model=self.depth_model,
            output=self.depth_output,
            quality=self.depth_quality,
            window=self.depth_window,
            post_upscale=self.depth_post_upscale and self.sr_enabled,
            sr_model=sr_model,
            sr_scale=self.sr_scale if self.sr_enabled else None,
        )

    # Fields that do NOT affect lossless segment-chunk content (final-encode / UX only).
    # Depth settings are excluded too: the depth pass is an independent output
    # generated after enhance, so toggling it must not invalidate GPU checkpoints.
    _NON_CONTENT_FIELDS = frozenset({
        "output_codec", "crf", "container", "audio_passthrough", "keep_workspace",
        "depth_enabled", "depth_model", "depth_output", "depth_quality",
        "depth_window", "depth_post_upscale",
    })

    def content_fingerprint(self) -> str:
        """Stable hash of every setting that affects lossless chunk content.

        New PipelineConfig fields are included automatically, so future stages
        (M3 upscale, M4 QA) invalidate checkpoints correctly without touching
        this code. Encode-only settings stay excluded on purpose: changing the
        final codec/CRF must not force a full re-run of the GPU stages.
        """
        import hashlib
        import json
        from dataclasses import fields as _dc_fields

        payload = {f.name: getattr(self, f.name) for f in _dc_fields(self) if f.name not in self._NON_CONTENT_FIELDS}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:32]


@dataclass
class StageContext:
    width: int
    height: int
    fps: str  # exact rational string for ffmpeg -r
    segment_index: int = 0
    start_frame: int = 0
    end_frame: int = 0
    settings: dict = field(default_factory=dict)
    # M4 QA ground-truth access (optional; default None keeps earlier stages unchanged).
    input_path: str | None = None  # source path so QA can decode proxy refs for detection
    expected_out_frames: int | None = None  # processed-frame count for this segment


class Stage:
    """A pipeline stage transforms a frame iterator into another frame iterator."""

    name = "stage"

    def process(self, frames: Iterator[np.ndarray], ctx: StageContext) -> Iterator[np.ndarray]:
        raise NotImplementedError

    def output_dims(self, ctx: StageContext) -> tuple[int, int]:
        """Output (width, height) for a context with the given input dims.

        Declarative so the orchestrator can compute final frame dimensions
        before encoding lossless chunks (stages are lazy iterators — dims
        cannot be read off the stream mid-run). Default: identity.
        """
        return (ctx.width, ctx.height)


class IdentityStage(Stage):
    """M1 placeholder: passes frames through unchanged (re-encode only)."""

    name = "identity"

    def process(self, frames, ctx):
        yield from frames


def _fps_to_float(fps: str) -> float:
    """Parse ffmpeg's rational fps string ('24/1', '30000/1001') to float."""
    if "/" in fps:
        num, den = fps.split("/")
        return float(num) / max(1e-9, float(den))
    return float(fps)


def _fps_to_str(fps: float) -> str:
    """Float fps -> ffmpeg rational string (exact for common rates like 59.94)."""
    from fractions import Fraction

    fr = Fraction(round(fps, 6)).limit_denominator(1000)
    if fr.denominator == 1:
        return str(fr.numerator)
    return f"{fr.numerator}/{fr.denominator}"


def build_stage_chain(cfg: PipelineConfig) -> list[Stage]:
    """Assemble the stage chain (design.md order: interpolate -> upscale -> QA)."""
    stages: list[Stage] = []
    fi_cfg = cfg.interpolate_config()
    if fi_cfg is not None:
        from ..stages.interpolate.stage import InterpolateStage  # lazy: pulls torch deps

        stages.append(InterpolateStage(fi_cfg))
    sr_cfg = cfg.upscale_config()
    if sr_cfg is not None:
        from ..stages.upscale.stage import UpscaleStage  # lazy: pulls torch deps

        stages.append(UpscaleStage(sr_cfg))
    qa_cfg = cfg.qa_config()
    if qa_cfg is not None:
        from ..stages.qa.stage import QAStage  # lazy: pulls cv2 deps

        stages.append(QAStage(qa_cfg))
    return stages or [IdentityStage()]


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

    # Resume identity: same input file (path + size + mtime) AND same
    # content-affecting settings. Anything else discards stale chunks.
    st = Path(input_path).stat()
    fingerprint = hashlib.sha256(
        f"{input_path}|{st.st_size}|{st.st_mtime_ns}|{cfg.content_fingerprint()}".encode()
    ).hexdigest()[:32]

    store = CheckpointStore(workspaces_dir() / job_id)
    store.init_job(job_id, input_path, info.width, info.height, info.fps, segments, fingerprint=fingerprint)
    resumed_from = store.done_count
    if resumed_from:
        log.info("Resuming: %d/%d segment(s) already done.", resumed_from, len(segments))

    fi_cfg = cfg.interpolate_config()
    if fi_cfg is not None:
        src_fps_f = _fps_to_float(info.fps)
        out_fps_str = _fps_to_str(fi_cfg.output_fps(src_fps_f))
        num, den = fi_cfg.ratio(src_fps_f)
        log.info(
            "Interpolation enabled: %s -> output fps %s (ratio %d/%d)",
            cfg.fi_model, out_fps_str, num, den,
        )
    else:
        out_fps_str = info.fps
        num = den = 1

    stages = build_stage_chain(cfg)
    qa_stage = next((s for s in stages if getattr(s, "name", "") == "qa"), None)

    # Final frame dimensions: fold the declarative output_dims over the chain.
    out_w, out_h = info.width, info.height
    for st in stages:
        probe_ctx = StageContext(out_w, out_h, info.fps)
        out_w, out_h = st.output_dims(probe_ctx)
    if (out_w, out_h) != (info.width, info.height):
        log.info("Output resolution: %dx%d -> %dx%d", info.width, info.height, out_w, out_h)

    # Exact per-segment output frame counts: gap_plan slotting is deterministic
    # and de-duplication fills slots with copies, so the count never changes.
    seg_out_counts = [estimate_output_frames(seg.frame_count, num, den) for seg in segments]
    total_frames = sum(seg_out_counts)
    frames_done_global = 0

    try:
        for seg in segments:
            if store.is_done(seg.index):
                continue
            control.wait_if_paused()
            if control.cancelled:
                raise PipelineCancelled()

            t_seg = time.time()
            _emit(StageStarted(stage="process", total_frames=seg_out_counts[seg.index]))
            log.info("Segment %d/%d frames [%d:%d]", seg.index + 1, len(segments), seg.start_frame, seg.end_frame)

            raw = decode_frames(input_path, start_frame=seg.start_frame, count=seg.frame_count)
            ctx = StageContext(
                info.width, info.height, info.fps, seg.index, seg.start_frame, seg.end_frame,
                input_path=input_path,
                expected_out_frames=seg_out_counts[seg.index],
            )
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
            # Chunks carry the FINAL (post-stage) dimensions — e.g. upscaled.
            encode_lossless(_throttled_frames(stream, 30, _on_frame), seg_file, out_w, out_h, out_fps_str)
            frames_done_global += seg_out_counts[seg.index]
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
            fps=out_fps_str,
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

    # Depth map pass (M5): independent output generated after enhance completes.
    # A failure here never fails the job — the enhanced file is already written;
    # record the error in the result and surface it as a stage-scoped event.
    depth_result = None
    dcfg = cfg.depth_config()
    if dcfg is not None:
        from ..stages.depth.depth_pass import depth_output_path, run_depth_pass  # lazy: pulls cv2 deps

        try:
            depth_result = run_depth_pass(
                input_path,
                depth_output_path(output_path, dcfg.output),
                dcfg,
                info,
                scene_split=cfg.scene_split,
                emit=_emit,
                control=control,
            )
        except PipelineCancelled:
            log.warning("Depth pass cancelled by user (enhanced output is complete).")
            raise
        except Exception as exc:
            log.warning("Depth pass failed (enhanced output is unaffected): %s", exc)
            _emit(ErrorEvent(message=f"Depth map generation failed: {exc}", stage="depth"))
            depth_result = {"error": str(exc)}

    if not cfg.keep_workspace:
        store.clear()

    result = {
        "output_path": output_path,
        "elapsed_s": time.time() - t_start,
        "segments_total": len(segments),
        "resumed_from": resumed_from,
        "output_fps": out_fps_str,
        "output_frames": total_frames,
        "output_width": out_w,
        "output_height": out_h,
        "depth": depth_result,  # M5: depth pass summary (or {"error": ...})
        "report": None,  # M4 fills in the artifact report
    }

    # Collect the artifact-QA report (M4) if a QA stage ran.
    report_dict = None
    if qa_stage is not None and getattr(qa_stage, "report", None) is not None:
        qa_stage.report.output_path = str(output_path)
        report_dict = qa_stage.report.to_dict()
        try:
            from ..stages.qa.report import save_report

            saved = save_report(qa_stage.report, workspaces_dir() / job_id, base_name="artifact-report")
            log.info("Artifact report written: %s", saved)
        except Exception as exc:  # never fail the job over a report export
            log.warning("Failed to write artifact report files: %s", exc)
    result["report"] = report_dict
    _emit(JobFinished(ok=True, output_path=output_path, elapsed_s=result["elapsed_s"], report=report_dict))
    return result

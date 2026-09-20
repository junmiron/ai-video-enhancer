"""Artifact QA / restore pipeline stage (M4) — detect → repair → verify.

The core differentiator (design.md §4): after upscale + interpolation, run the v1
detector set on a ~720p proxy of the processed stream, apply *targeted* repairs to
flagged regions at full resolution, re-measure each metric (verify), and accumulate
a :class:`~aive.stages.qa.findings.QAReport`.

Architecture — **streaming windowed**, not whole-segment buffering:
* A sliding buffer of the last ``window_size`` *full-res* frames bounds RAM even for
  long/4K segments. Each frame is emitted once it has enough future neighbors to form
  a centered temporal window (a small fixed delay), with a one-sided flush at the tail.
* Detectors run on proxy-resolution windows (cheap); repairs apply to full res using
  masks upsampled from the proxy (design.md §4.5 "proxy-first").
* Source ground truth is decoded lazily at *proxy* resolution in lockstep with the
  processed stream (only halo / line-art need it; they degrade gracefully when absent).

The stage is duck-typed like :class:`~aive.stages.upscale.stage.UpscaleStage` — it
exposes ``name``, ``output_dims(ctx)`` (identity: QA never changes resolution) and
``process(frames, ctx)``. Heavy imports (cv2/numpy) are pulled lazily inside
``process`` so importing this module stays cheap.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from ...core.jobqueue import check_control  # M6: per-frame pause/cancel polling

log = logging.getLogger(__name__)


@dataclass
class QAConfig:
    """User-facing artifact-QA settings (M6 UI maps onto this)."""

    enabled: bool = True
    mode: str = "auto"  # auto | detect_only | manual
    proxy_height: int = 720  # detector resolution target (design.md §4.5)
    window_size: int = 7  # temporal frames for detectors/repairs (kept odd)
    min_severity: float = 0.15  # record a finding only at/above this severity
    ai_repairs: bool = True  # allow AI-model repairs (SCUNet / deepDeband-f); CV fallback when off

    # Per-artifact sensitivity in [0,1] and enable flag (design.md §4.4). Defaults
    # are uniform; the M6 UI exposes per-artifact sliders that override these.
    sensitivity: dict = field(default_factory=dict)
    enabled_artifacts: dict = field(default_factory=dict)

    def __post_init__(self):
        from .findings import ArtifactType

        if self.window_size % 2 == 0 or self.window_size < 3:
            self.window_size = 7
        all_arts = [a.value for a in ArtifactType.all()]
        self.sensitivity = {k: float(v) for k, v in self.sensitivity.items()}
        self.enabled_artifacts = {k: bool(v) for k, v in self.enabled_artifacts.items()}
        # Fill any missing per-artifact keys with sensible defaults.
        for a in all_arts:
            self.sensitivity.setdefault(a, 0.5)
            self.enabled_artifacts.setdefault(a, True)

    def sens_for(self, artifact_value: str) -> float:
        return float(np.clip(self.sensitivity.get(artifact_value, 0.5), 0.0, 1.0))

    def enabled_for(self, artifact_value: str) -> bool:
        if self.mode == "detect_only":
            return True  # detect everything, just don't repair
        return bool(self.enabled_artifacts.get(artifact_value, True))


class _SourceProvider:
    """Lazily decodes proxy-resolution source frames aligned to the processed stream.

    The processed→source time mapping is monotonic (linear), so a single forward
    decode of the segment suffices; only a handful of proxy source frames are cached
    at a time, keeping RAM bounded regardless of segment length.
    """

    def __init__(self, input_path: str, start_frame: int, n_src: int, expected_out_frames: int, pw: int, ph: int):
        from ..ingest import decode_frames  # lazy: pulls ffmpeg subprocess machinery

        self._it = decode_frames(input_path, start_frame=start_frame, count=n_src)
        self.pw, self.ph = pw, ph
        self.n_src = n_src
        self.exp = max(1, expected_out_frames)
        self.cache: dict[int, np.ndarray] = {}
        self.next_avail = 0

    def for_proc_index(self, e: int):
        """Proxy source frame nearest processed index ``e`` (or None if unavailable)."""
        from .proxy import to_proxy

        sidx = int(round(e * self.n_src / self.exp))
        sidx = max(0, min(self.n_src - 1, sidx))
        if sidx in self.cache:
            return self.cache[sidx]
        while self.next_avail <= sidx:
            try:
                f = next(self._it)
            except StopIteration:
                break
            self.cache[self.next_avail] = to_proxy(f, self.pw, self.ph)
            self.next_avail += 1
        # Evict source frames well behind the current position (mapping is monotonic).
        for k in [k for k in self.cache if k < sidx - 4]:
            del self.cache[k]
        return self.cache.get(sidx)

    def close(self):
        try:
            self._it.close()
        except Exception:
            pass


class QAStage:
    """Pipeline stage: processed frames -> (repaired) frames + artifact report."""

    name = "qa"

    def __init__(self, cfg: QAConfig):
        self.cfg = cfg
        self.report = None  # set by process(); read by the orchestrator after the run
        self.frames_processed = 0

    # -- stage interface ------------------------------------------------------

    def output_dims(self, ctx) -> tuple[int, int]:
        """QA never changes resolution — identity."""
        return (ctx.width, ctx.height)

    def process(self, frames: Iterator[np.ndarray], ctx) -> Iterator[np.ndarray]:
        import cv2  # noqa: F401 - ensure the CV backend is up before first use

        from .detectors import detect as run_detector
        from .detectors.base import FrameWindow, luma
        from .findings import ArtifactType, Finding, QAReport
        from .motion import flow_magnitude
        from .proxy import proxy_dims, to_proxy, upsample_mask
        from .repairs import repair as run_repair
        from .repairs.base import RepairContext

        cfg = self.cfg
        W = cfg.window_size
        D = W // 2

        # --- source ground-truth provider (proxy res), if available ----------
        n_src = None
        if getattr(ctx, "input_path", None) and getattr(ctx, "end_frame", 0) > getattr(ctx, "start_frame", 0):
            n_src = ctx.end_frame - ctx.start_frame
        exp_out = getattr(ctx, "expected_out_frames", None)
        # Interpolation is on when the processed stream has more frames than source.
        self._fi_on_flag = bool(exp_out and n_src is not None and int(exp_out) > n_src)
        src_provider = None
        # Proxy dims are derived from the first real frame (ctx carries source dims).
        pw = ph = None

        buf_full: list[np.ndarray] = []
        buf_proxy: list[np.ndarray] = []
        buf_gidx: list[int] = []
        scene_ref = None  # first processed proxy frame of this segment (color drift ref)
        iqa_hist: deque[float] = deque(maxlen=32)  # prior per-frame IQA scores (regression)
        mag_cache: dict[int, np.ndarray] = {}
        sampled_arts: set[str] = set()  # artifact types already captured as a report thumbnail
        seg_frames = 0  # frames processed in THIS segment (report accumulates across segments)

        # Job-level report: the stage chain is built once and reused for every
        # segment, so accumulate findings/frames into one persistent report.
        if self.report is None:
            self.report = QAReport(
                input_path=getattr(ctx, "input_path", "") or "",
                fps=_fps_float(getattr(ctx, "fps", "0")),
                mode=cfg.mode,
            )
        report = self.report

        def _mags(wp: list[np.ndarray], wg: list[int]) -> list[np.ndarray]:
            out = []
            for i in range(len(wp) - 1):
                key = wg[i]
                m = mag_cache.get(key)
                if m is None:
                    m = flow_magnitude(luma(wp[i]), luma(wp[i + 1]))
                    mag_cache[key] = m
                out.append(m)
            minkey = (wg[0] - 2) if wg else 0
            for k in [k for k in mag_cache if k < minkey]:
                del mag_cache[k]
            return out

        def _process_center(center_pos: int, gidx: int):
            nonlocal scene_ref, seg_frames
            wp = list(buf_proxy)
            wf = list(buf_full)
            center_frame = wf[center_pos]  # full-res HWC uint8
            center_proxy = wp[center_pos]  # proxy [0,1] HWC
            fw, fh = int(center_frame.shape[1]), int(center_frame.shape[0])

            if scene_ref is None:
                scene_ref = center_proxy.copy()

            src_center = src_provider.for_proc_index(gidx) if src_provider is not None else None

            win = FrameWindow(
                center=center_proxy,
                prev=wp[center_pos - 1] if center_pos > 0 else None,
                next=wp[center_pos + 1] if center_pos < len(wp) - 1 else None,
                window=wp,
                src_center=src_center,
                scene_ref=scene_ref,
                mags=_mags(wp, buf_gidx),
                is_interpolated=self._fi_on,
                recent_iqa=list(iqa_hist),
            )

            out_frame = center_frame
            # Detect every enabled artifact; repair (unless detect-only) and verify.
            for art in ArtifactType.all():
                av = art.value
                if not cfg.enabled_for(av):
                    continue
                sens = cfg.sens_for(av)
                mask_proxy, severity = run_detector(art, win, sens)
                if severity < cfg.min_severity:
                    continue

                before_metric = float(severity)
                repaired = False
                method = "detect-only"
                after_metric = None

                if cfg.mode != "detect_only":
                    mask_full = upsample_mask(mask_proxy, fw, fh)
                    rctx = RepairContext(
                        frame=center_frame,
                        mask=mask_full,
                        window=wf,
                        center_idx=center_pos,
                        scene_ref_proxy=scene_ref,
                        center_proxy=center_proxy,
                        ai_repairs=cfg.ai_repairs,
                    )
                    new_frame, method = run_repair(art, rctx)
                    if method != "none":
                        out_frame = new_frame
                        repaired = True
                        # Verify: re-run the same detector with ONLY this frame replaced
                        # by its repaired version (neighbors unchanged), so the delta is
                        # attributable to the repair.
                        wp2 = list(wp)
                        rep_proxy = to_proxy(new_frame, pw, ph)
                        wp2[center_pos] = rep_proxy
                        win2 = FrameWindow(
                            center=rep_proxy,
                            prev=wp2[center_pos - 1] if center_pos > 0 else None,
                            next=wp2[center_pos + 1] if center_pos < len(wp2) - 1 else None,
                            window=wp2,
                            src_center=src_center,
                            scene_ref=scene_ref,
                            mags=win.mags,
                            is_interpolated=self._fi_on,
                            recent_iqa=list(iqa_hist),
                        )
                        _, after_metric = run_detector(art, win2, sens)

                finding = Finding(
                    artifact=art,
                    frame_index=gidx,
                    severity=float(severity),
                    bbox=_bbox_full(mask_proxy, fw, fh),
                    region_area_frac=float(np.clip(mask_proxy, 0.0, 1.0).mean()),
                    repaired=repaired,
                    method=method,
                    before_metric=before_metric,
                    after_metric=None if after_metric is None else float(after_metric),
                )
                report.findings.append(finding)

                # Keep one representative thumbnail per artifact type for the PNG.
                if art.value not in sampled_arts and len(report.sample_frames) < 16:
                    sampled_arts.add(art.value)
                    report.sample_frames.append(
                        {
                            "frame_index": gidx,
                            "artifact": av,
                            "thumb": _small_thumb(center_frame, width=320),
                            "bbox": finding.bbox,
                            "fw": fw,
                            "fh": fh,
                        }
                    )

            # Rolling IQA history for the regression detector: record what actually
            # went out (repaired frame if it was repaired, else the original center).
            iqa_hist.append(_iqa(to_proxy(out_frame, pw, ph)))

            self.frames_processed += 1
            seg_frames += 1
            return out_frame

        # --- main streaming loop ---------------------------------------------
        ctl = getattr(ctx, "control", None)
        for i, frame in enumerate(frames):
            check_control(ctl)  # M6: pause/cancel takes effect within one frame
            if pw is None:
                h0, w0 = int(frame.shape[0]), int(frame.shape[1])
                pw, ph = proxy_dims(w0, h0, target_h=cfg.proxy_height)
                if src_provider is None and n_src is not None and getattr(ctx, "expected_out_frames", None):
                    src_provider = _SourceProvider(
                        ctx.input_path, ctx.start_frame, n_src, int(ctx.expected_out_frames), pw, ph
                    )
            buf_full.append(frame)
            buf_proxy.append(to_proxy(frame, pw, ph))
            buf_gidx.append(i)
            if len(buf_full) > W:
                buf_full.pop(0)
                buf_proxy.pop(0)
                buf_gidx.pop(0)

            if len(buf_full) < D + 1:
                continue
            center_pos = len(buf_full) - 1 - D
            yield _process_center(center_pos, buf_gidx[center_pos])

        # --- tail flush (one-sided windows for the last D frames) -------------
        start_pos = max(0, len(buf_full) - D)
        for pos in range(start_pos, len(buf_full)):
            yield _process_center(pos, buf_gidx[pos])

        # --- finalize this segment's contribution to the job report -----------
        report.num_frames += seg_frames
        if report.fps > 0:
            report.duration_s = report.num_frames / report.fps
        if src_provider is not None:
            src_provider.close()
        log.info(
            "QA stage: %d frames, %d finding(s) across %d artifact type(s), mode=%s",
            seg_frames, len(report.findings), len({f.artifact for f in report.findings}), cfg.mode,
        )

    # -- helpers --------------------------------------------------------------

    @property
    def _fi_on(self) -> bool:
        return getattr(self, "_fi_on_flag", False)


def _fps_float(fps: str | float) -> float:
    try:
        if isinstance(fps, (int, float)):
            return float(fps)
        s = str(fps)
        if "/" in s:
            num, den = s.split("/")
            return float(num) / max(1e-9, float(den))
        return float(s)
    except Exception:
        return 0.0


def _iqa(frame_proxy: np.ndarray) -> float:
    """Lightweight no-ref IQA score for the regression rolling window."""
    from .detectors.regression import iqa_feature

    return iqa_feature(frame_proxy)


def _small_thumb(frame: np.ndarray, width: int = 320) -> np.ndarray:
    """Downscale a full-res HWC uint8 frame to ~`width` px wide (for report thumbs)."""
    import cv2

    h, w = frame.shape[:2]
    if w <= width:
        return frame.copy()
    nh = max(1, int(h * width / w))
    return cv2.resize(frame, (width, nh), interpolation=cv2.INTER_AREA).astype(np.uint8)


def _bbox_full(mask_proxy: np.ndarray, fw: int, fh: int) -> tuple[int, int, int, int]:
    """Tight full-res bounding box of a proxy mask (x,y,w,h); (0,0,0,0) if empty."""
    ys, xs = np.where(mask_proxy > 0.15)
    if len(ys) == 0:
        return (0, 0, 0, 0)
    ph, pw = mask_proxy.shape[:2]
    sx = fw / max(1, pw)
    sy = fh / max(1, ph)
    x0, y0 = int(xs.min() * sx), int(ys.min() * sy)
    x1, y1 = int((xs.max() + 1) * sx), int((ys.max() + 1) * sy)
    return (x0, y0, max(0, x1 - x0), max(0, y1 - y0))

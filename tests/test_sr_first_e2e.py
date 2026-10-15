"""Orchestrator-level regression guard for sr_first (Upscale -> Interpolate) order.

Bug class: InterpolateStage once built its FI runner from ctx dims, which always
carry SOURCE resolution. In sr_first order the upscale stage runs first, so the
FI stage receives already-upscaled frames and the first interpolated pair crashed
GMFSSRunner._pad with a numpy broadcast error (320x240 runner fed 640x480 frames;
real run: OPENING.AVI + GMFSS x3 + span_v2 x2). Fixed in 85166b7 by building the
runner lazily from the first real input frame.

This test drives ``run_pipeline`` end-to-end in sr_first order with stubbed model
runners (no GPU, no weights) and asserts the dimension flow through the whole
chain: SR sees source-res frames, FI sees upscaled frames, and the encoded output
carries the final dims + interpolated frame count. It also covers the runner-level
dim-mismatch guard added alongside 85166b7's follow-up hardening.
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from aive.config import workspaces_dir
from aive.core.orchestrator import PipelineConfig, run_pipeline
from aive.stages.interpolate.gapplan import estimate_output_frames
from aive.stages.ingest import probe


class _FakeSRRunner:
    """Stands in for SRRunner: 2x LANCZOS upscale, records input dims."""

    scale = 2

    def __init__(self, *args, **kwargs):
        self.input_dims: list[tuple[int, int]] = []

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        self.input_dims.append((w, h))
        return cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)


class _FakeFIRunner:
    """Stands in for an FI runner: midpoint blend, records input dims."""

    def __init__(self):
        self.input_dims: list[tuple[int, int]] = []
        self.calls = 0

    def __call__(self, img0: np.ndarray, img1: np.ndarray, t: float) -> np.ndarray:
        h, w = img0.shape[:2]
        self.input_dims.append((w, h))
        self.calls += 1
        return np.clip(
            (1 - t) * img0.astype(np.float32) + t * img1.astype(np.float32), 0, 255
        ).astype(np.uint8)


def test_run_pipeline_sr_first_dim_flow(static_clip: str, tmp_path: Path, monkeypatch):
    """sr_first end-to-end: SR at source res -> FI at upscaled res -> final encode."""
    import aive.models.downloader as dl
    import aive.models.registry as reg
    import aive.stages.interpolate.stage as fi_stage
    import aive.stages.upscale.stage as sr_stage

    job_id = f"srfirst-{uuid.uuid4().hex[:8]}"
    out = tmp_path / "out.mp4"
    ws = workspaces_dir() / job_id

    sr_runner = _FakeSRRunner()
    fi_runner = _FakeFIRunner()
    monkeypatch.setattr(sr_stage, "SRRunner", lambda *a, **k: sr_runner)
    monkeypatch.setattr(fi_stage, "build_fi_runner", lambda name, weights, w, h: fi_runner)
    monkeypatch.setattr(dl, "download_model", lambda info: None)  # no real downloads
    monkeypatch.setattr(reg, "get_model", lambda *a, **k: SimpleNamespace(name="stub"))

    cfg = PipelineConfig(
        scene_split=False,            # deterministic single segment
        fi_enabled=True, fi_model="gmfss", fi_multiplier=2,
        sr_enabled=True, sr_model="span_v2", sr_scale=2,
        qa_enabled=False,             # keep the smoke CPU-light
        stage_order="sr_first",       # Upscale -> Interpolate (the order under test)
    )
    try:
        result = run_pipeline(job_id, static_clip, out, cfg)

        src = probe(static_clip)
        assert result["output_width"] == src.width * 2
        assert result["output_height"] == src.height * 2

        # The regression itself: in sr_first order the FI runner must receive
        # UPSCALED frames (pre-fix code built it at source dims and crashed).
        assert fi_runner.calls > 0, "FI runner was never called"
        bad = {d for d in fi_runner.input_dims if d != (src.width * 2, src.height * 2)}
        assert not bad, f"FI runner saw non-upscaled frames: {sorted(bad)[:3]}"

        # ...while SR always sees source-res frames in both orders.
        assert sr_runner.input_dims, "SR runner was never called"
        bad = {d for d in sr_runner.input_dims if d != (src.width, src.height)}
        assert not bad, f"SR runner saw non-source-res frames: {sorted(bad)[:3]}"

        # Encoded output carries final dims + interpolated frame count.
        expected_frames = estimate_output_frames(src.num_frames, 2, 1)
        assert result["output_frames"] == expected_frames
        info = probe(out)
        assert (info.width, info.height) == (src.width * 2, src.height * 2)
        # Container metadata may estimate the count from duration; allow +-1.
        assert abs(info.num_frames - expected_frames) <= 1
    finally:
        shutil.rmtree(ws, ignore_errors=True)


# -- runner-level dim-mismatch guard (no GPU/weights needed) -------------------


@pytest.mark.parametrize(
    ("runner_cls", "method"),
    [
        ("RIFERunner", "_pad"),
        ("GMFSSRunner", "_pad"),
        ("M2MRunner", "_to_gpu"),
    ],
)
def test_fi_runner_rejects_mismatched_frame_dims(runner_cls: str, method: str):
    """A frame whose dims differ from the runner's build dims must raise a clear,
    actionable error instead of a cryptic numpy broadcast / TRT shape failure."""
    import aive.stages.interpolate.gmfss_model as gmfss_mod
    import aive.stages.interpolate.m2m_model as m2m_mod
    import aive.stages.interpolate.stage as stage_mod

    cls = {
        "RIFERunner": stage_mod.RIFERunner,
        "GMFSSRunner": gmfss_mod.GMFSSRunner,
        "M2MRunner": m2m_mod.M2MRunner,
    }[runner_cls]

    self = SimpleNamespace(width=320, height=240)  # guard reads only build dims
    bad_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="does not match"):
        getattr(cls, method)(self, bad_frame)


def test_rife_pad_accepts_matching_dims():
    """Positive control: a frame at the runner's build dims passes the guard."""
    from aive.stages.interpolate.stage import RIFERunner

    self = SimpleNamespace(width=320, height=240, pw=320, ph=256)
    frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    out = RIFERunner._pad(self, frame)
    assert out.shape == (1, 3, 256, 320)

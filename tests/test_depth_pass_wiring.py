"""Wiring tests for the M5 depth pass — no GPU, no model weights.

* PipelineConfig -> DepthPassConfig mapping (incl. post-upscale gating on SR).
* run_pipeline degradation: a failing depth pass never fails the job; the error
  lands in result["depth"] and a stage-scoped ErrorEvent is emitted.
* The full pass with FAKE runners: real FFmpeg decode + real chunk streaming +
  real writer, proving exact frame counts, scene-cut isolation (windows never
  cross a hard cut), and both output formats.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pytest

from aive.core.events import ErrorEvent, JobFinished
from aive.core.orchestrator import PipelineConfig, run_pipeline
from aive.stages.depth.stage import DEPTH_SPECS


# -- config mapping -----------------------------------------------------------------


def test_depth_config_disabled_is_none():
    assert PipelineConfig(depth_enabled=False).depth_config() is None


def test_depth_config_mapping_without_sr():
    cfg = PipelineConfig(
        depth_enabled=True, depth_model="limbo_v1", depth_output="png16",
        depth_quality="medium", depth_window=16, depth_post_upscale=True,
        sr_enabled=False,
    )
    dc = cfg.depth_config()
    assert dc.model == "limbo_v1" and dc.output == "png16"
    assert dc.quality == "medium" and dc.window == 16
    # post-upscale is meaningless without SR: gated off.
    assert dc.post_upscale is False and dc.sr_model is None


def test_depth_config_mapping_with_sr():
    cfg = PipelineConfig(
        depth_enabled=True, depth_post_upscale=True,
        sr_enabled=True, sr_model="fallin_soft", sr_scale=2,
    )
    dc = cfg.depth_config()
    assert dc.post_upscale is True and dc.sr_model == "fallin_soft" and dc.sr_scale == 2


# -- run_pipeline degradation ---------------------------------------------------------


def _run(cfg: PipelineConfig, clip: str, tmp_path: Path) -> tuple[dict, list]:
    job_id = f"m5-wire-{uuid.uuid4().hex[:8]}"
    out = tmp_path / "out.mp4"
    events: list = []
    result = run_pipeline(job_id, clip, out, cfg, emit=events.append)
    return result, events


def test_depth_failure_never_fails_job(static_clip: str, tmp_path: Path, monkeypatch):
    import aive.stages.depth.depth_pass as dp

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(dp, "run_depth_pass", boom)
    cfg = PipelineConfig(
        scene_split=False, fi_enabled=False, sr_enabled=False, qa_enabled=False,
        depth_enabled=True, depth_model="limbo_v2",
    )
    result, events = _run(cfg, static_clip, tmp_path)

    assert (tmp_path / "out.mp4").exists(), "enhanced output must exist despite depth failure"
    assert result["depth"] == {"error": "boom"}
    finished = [e for e in events if isinstance(e, JobFinished)]
    assert finished and finished[-1].ok is True
    errs = [e for e in events if isinstance(e, ErrorEvent) and e.stage == "depth"]
    assert len(errs) == 1 and "boom" in errs[0].message


def test_depth_success_lands_in_result(static_clip: str, tmp_path: Path, monkeypatch):
    import aive.stages.depth.depth_pass as dp

    fake = {"path": "/fake/depth.mp4", "frames": 30}
    monkeypatch.setattr(dp, "run_depth_pass", lambda *a, **k: fake)
    cfg = PipelineConfig(
        scene_split=False, fi_enabled=False, sr_enabled=False, qa_enabled=False,
        depth_enabled=True,
    )
    result, events = _run(cfg, static_clip, tmp_path)
    assert result["depth"] == fake
    assert not [e for e in events if isinstance(e, ErrorEvent)]


# -- full pass with fake runners --------------------------------------------------------


class FakeStreamingRunner:
    """Stands in for StreamingDepthRunner: records chunks, returns synthetic depth."""

    def __init__(self):
        self.chunks: list[list[str]] = []  # per-chunk color labels (red/blue)

    def infer_chunk(self, frames):
        labels = []
        maps = []
        for f in frames:
            r, b = float(f[..., 0].mean()), float(f[..., 2].mean())
            labels.append("red" if r > b else "blue")
            # Distinct positive depth per frame index within the chunk.
            maps.append(np.full((8, 16), len(maps) + 1.0, dtype=np.float32))
        self.chunks.append(labels)
        return np.stack(maps)


class FakePerFrameRunner:
    """Stands in for PerFrameDepthRunner: one frame in, gray map at input size out."""

    def __init__(self):
        self.calls = 0

    def __call__(self, frame_hwc_u8):
        h, w = frame_hwc_u8.shape[:2]
        self.calls += 1
        return np.linspace(0.0, 1.0, w * h, dtype=np.float32).reshape(h, w)


def test_depth_pass_streaming_cut_isolation(cut_clip: str, tmp_path: Path, monkeypatch):
    """60-frame clip with a hard cut at frame 30 (red -> blue), window=8.

    Proves the scene-segment contract end-to-end: exact frame count in the
    output AND no streaming chunk ever mixes frames from both sides of the cut.
    """
    import aive.stages.depth.stage as stage_mod
    from aive.stages.depth.depth_pass import DepthPassConfig, run_depth_pass
    from aive.stages.ingest import probe

    fake = FakeStreamingRunner()
    monkeypatch.setattr(stage_mod, "get_depth_runner", lambda *a, **k: fake)

    info = probe(cut_clip)
    out = tmp_path / "cut_depth.mp4"
    result = run_depth_pass(
        cut_clip, out,
        DepthPassConfig(model="limbo_v2", output="mp4_10bit", window=8),
        info, scene_split=True,
    )

    assert result["frames"] == 60 and result["path"] == str(out)
    assert out.exists() and out.stat().st_size > 0

    # Every chunk is single-color: windows never crossed the cut.
    for labels in fake.chunks:
        assert set(labels) <= {"red"} or set(labels) <= {"blue"}, f"cross-cut chunk: {labels}"
    # Both colors were actually processed (both segments ran).
    all_labels = [l for c in fake.chunks for l in c]
    assert "red" in all_labels and "blue" in all_labels


def test_depth_pass_per_frame_png16(static_clip: str, tmp_path: Path, monkeypatch):
    import aive.stages.depth.stage as stage_mod
    from aive.stages.depth.depth_pass import DepthPassConfig, run_depth_pass
    from aive.stages.ingest import probe

    fake = FakePerFrameRunner()
    monkeypatch.setattr(stage_mod, "get_depth_runner", lambda *a, **k: fake)

    info = probe(static_clip)
    out_dir = tmp_path / "static_depth"  # png16 -> folder
    result = run_depth_pass(
        static_clip, out_dir,
        DepthPassConfig(model="depth_anything_v2_small", output="png16", quality="low"),
        info, scene_split=False,
    )

    assert result["frames"] == 30 and fake.calls == 30
    pngs = sorted(out_dir.glob("*.png"))
    assert len(pngs) == 30


def test_depth_pass_unknown_model_raises(tmp_path: Path):
    from aive.stages.depth.depth_pass import DepthPassConfig, run_depth_pass

    # spec_for_depth runs before any file access, so info may be None here.
    with pytest.raises(ValueError, match="Unknown depth model"):
        run_depth_pass("x.mp4", tmp_path / "d.mp4", DepthPassConfig(model="nope"), None)

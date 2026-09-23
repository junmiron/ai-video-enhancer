"""M4 stage-level integration tests (plan.md M4): streaming QAStage end-to-end.

Feeds synthetic frame streams through ``QAStage.process()`` with a fake StageContext
(``input_path=None`` → no source decode) and validates the streaming delay / tail-flush
bookkeeping, output fidelity, report accumulation across segments, detect-only mode,
and identity of ``output_dims`` — all without GPU or ffmpeg.
"""
from __future__ import annotations

import cv2
import numpy as np

from aive.stages.qa.stage import QAConfig, QAStage


class FakeCtx:
    """Duck-typed StageContext with no source (input_path=None → no decode)."""

    def __init__(self, w=640, h=360, fps="30/1", expected_out_frames=None):
        self.width = w
        self.height = h
        self.fps = fps
        self.input_path = None
        self.expected_out_frames = expected_out_frames
        self.start_frame = 0
        self.end_frame = 0


def _frames(n: int, seed: int = 7) -> list[np.ndarray]:
    """Static textured frames (no plateaus, no temporal change) → a clean stream."""
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 210, size=(360, 640, 3), dtype=np.uint8)
    # Light blur kills single-pixel spikes while keeping texture (no long plateaus).
    return [cv2.GaussianBlur(base, (3, 3), 0.8).copy() for _ in range(n)]


def test_streaming_yields_all_frames_in_order():
    stage = QAStage(QAConfig())
    inputs = _frames(10)

    out = list(stage.process(iter(inputs), FakeCtx()))

    assert len(out) == 10, f"expected 10 frames, got {len(out)}"
    for i, (o, f) in enumerate(zip(out, inputs)):
        # Same object ⇒ correct order AND no unintended mutation.
        assert o is f, f"frame {i} is not the input object (order or mutation broken)"
        assert o.dtype == np.uint8 and o.shape == (360, 640, 3)

    # Job report populated; a clean static stream must produce no findings.
    assert stage.report is not None
    assert stage.report.num_frames == 10
    assert stage.report.findings == []
    d = stage.report.to_dict()
    assert d["num_frames"] == 10 and d["findings"] == []


def test_short_stream_tail_flush():
    """N < window_size: the main loop yields nothing; tail flush must emit all frames."""
    stage = QAStage(QAConfig())
    inputs = _frames(3, seed=8)

    out = list(stage.process(iter(inputs), FakeCtx()))

    assert len(out) == 3
    for o, f in zip(out, inputs):
        assert o is f


def test_report_accumulates_across_segments():
    """The stage chain is built once and reused: the report accumulates across calls."""
    stage = QAStage(QAConfig())
    first = _frames(5, seed=9)
    second = _frames(4, seed=10)

    out1 = list(stage.process(iter(first), FakeCtx()))
    out2 = list(stage.process(iter(second), FakeCtx()))

    assert len(out1) == 5 and len(out2) == 4
    assert stage.report.num_frames == 9
    assert stage.frames_processed == 9


def test_detect_only_mode_reports_but_does_not_modify():
    from aive.stages.qa.inject import inject_flicker

    stage = QAStage(QAConfig(mode="detect_only"))
    inputs = _frames(7, seed=11)
    inj = inject_flicker(inputs, (200, 100, 160, 120), amp=48.0, seed=3)

    out = list(stage.process(iter(inj.frames), FakeCtx()))

    assert len(out) == 7
    for o, f in zip(out, inj.frames):
        assert o is f, "detect-only mode must never modify frames"

    flicker_findings = [f for f in stage.report.findings if f.artifact.value == "flicker"]
    assert flicker_findings, "injected flicker was not detected"
    assert all(not f.repaired and f.method == "detect-only" for f in flicker_findings)


def test_verify_gate_rejects_worsening_repair(monkeypatch):
    """Q2: a repair that fails to improve the metric is rejected — the original
    frame is emitted and the attempt is recorded transparently in the finding."""
    import aive.stages.qa.repairs as repairs_mod
    from aive.stages.qa.inject import inject_flicker

    stage = QAStage(QAConfig())
    inputs = _frames(7, seed=12)
    inj = inject_flicker(inputs, (200, 100, 160, 120), amp=48.0, seed=3)

    real_repair = repairs_mod.repair

    def bad_repair(art, ctx):
        if art.value == "flicker":
            # A "repair" that maximizes temporal noise inside the mask region —
            # guaranteed to worsen the flicker metric.
            f = ctx.frame.copy()
            rng = np.random.default_rng(99)
            m = ctx.mask > 0.5
            if m.any():
                f[m] = rng.integers(0, 256, size=(int(m.sum()), 3), dtype=np.uint8)
            return f, "bad-flicker-fix"
        return real_repair(art, ctx)

    monkeypatch.setattr(repairs_mod, "repair", bad_repair)
    out = list(stage.process(iter(inj.frames), FakeCtx()))

    flick = [f for f in stage.report.findings if f.artifact.value == "flicker"]
    assert flick, "injected flicker should be detected"
    # Every worsening attempt must be rejected: nothing applied, attempts recorded.
    assert all(not f.repaired and f.method == "none" for f in flick)
    assert all(f.rejected_methods == ["bad-flicker-fix"] for f in flick)
    assert all(f.before_metric is None and f.after_metric is None for f in flick)

    # The original frame must be emitted (except where another artifact's repair was kept).
    kept_elsewhere = {f.frame_index for f in stage.report.findings if f.repaired}
    for i in {f.frame_index for f in flick} - kept_elsewhere:
        assert out[i] is inj.frames[i], "worsening repair must not be emitted"


def test_output_dims_identity():
    stage = QAStage(QAConfig())
    ctx = FakeCtx(w=1920, h=1080)
    assert stage.output_dims(ctx) == (1920, 1080)

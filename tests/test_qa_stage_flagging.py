"""M7 A3: per-frame interpolation marking + flicker spatial-coherence calibration.

Two fixes for QA over-triggering on flat-color content (samples/m5-demo):
* Ghosting judged ORIGINAL source frames as interpolated (blanket flag) — with the
  FI slot ratio, only truly synthesized output slots are marked interpolated
  (source frame k lands on output slot floor(k*num/den), exact by gap_plan math and
  preserved under de-duplication).
* Flicker recorded findings from isolated sub-pixel edge shimmer (SR re-hallucinates
  edges per frame; flow can't track value flips on flat content) — connected
  components below MIN_COMPONENT_PX are dropped before severity.
"""
from __future__ import annotations

import numpy as np

from aive.stages.interpolate.gapplan import estimate_output_frames
from aive.stages.qa.detectors import detect, FrameWindow
from aive.stages.qa.detectors.flicker import MIN_COMPONENT_PX
from aive.stages.qa.findings import ArtifactType
from aive.stages.qa.stage import QAConfig, QAStage


class FlagCtx:
    """Duck-typed StageContext for flagging tests (no source decode)."""

    def __init__(self, n_src: int, exp_out: int | None = None, fi_ratio=None):
        self.width = 64
        self.height = 48
        self.fps = "30/1"
        self.input_path = None
        self.start_frame = 0
        self.end_frame = n_src
        self.expected_out_frames = exp_out if exp_out is not None else n_src
        self.fi_ratio = fi_ratio


def _static_frames(n: int, h: int = 48, w: int = 64) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    base = rng.integers(100, 150, size=(h, w, 3), dtype=np.uint8)
    return [base.copy() for _ in range(n)]


def test_x2_slots_even_original_odd_interpolated():
    """x2 on 4 source frames -> 7 output; source k lands on slot 2k."""
    n_src = 4
    exp_out = estimate_output_frames(n_src, 2, 1)
    assert exp_out == 7

    stage = QAStage(QAConfig())
    list(stage.process(iter(_static_frames(exp_out)), FlagCtx(n_src, exp_out, fi_ratio=(2, 1))))

    assert stage._fi_on_flag is True
    assert stage._orig_slots == {0, 2, 4, 6}
    for k in range(exp_out):
        assert stage._is_interp(k) == (k % 2 == 1), f"slot {k} mis-marked at x2"


def test_5_over_2_slot_pattern():
    """24->60 style ratio: source slots are floor(5k/2)."""
    n_src = 4
    num, den = 5, 2
    exp_out = estimate_output_frames(n_src, num, den)

    stage = QAStage(QAConfig())
    list(stage.process(iter(_static_frames(exp_out)), FlagCtx(n_src, exp_out, fi_ratio=(num, den))))

    expected_slots = {(k * num) // den for k in range(n_src)}
    assert expected_slots == {0, 2, 5, 7}
    assert stage._orig_slots == expected_slots
    # Every non-source slot is judged interpolated; source slots are not.
    n_interp = sum(stage._is_interp(k) for k in range(exp_out))
    assert n_interp == exp_out - len(expected_slots)


def test_no_fi_means_no_interpolated_frames():
    stage = QAStage(QAConfig())
    list(stage.process(iter(_static_frames(5)), FlagCtx(5, 5)))

    assert stage._fi_on_flag is False
    assert stage._orig_slots is None
    assert all(not stage._is_interp(k) for k in range(5))


def test_missing_ratio_falls_back_to_blanket_flag():
    """Legacy contexts without fi_ratio keep the old blanket behavior."""
    n_src = 4
    exp_out = estimate_output_frames(n_src, 2, 1)

    stage = QAStage(QAConfig())
    list(stage.process(iter(_static_frames(exp_out)), FlagCtx(n_src, exp_out)))  # no fi_ratio

    assert stage._fi_on_flag is True
    assert stage._orig_slots is None
    assert all(stage._is_interp(k) for k in range(exp_out))


# -- flicker spatial coherence -------------------------------------------------


def _win(frames, center_idx=3):
    c = frames[center_idx]
    return FrameWindow(
        center=c,
        prev=frames[center_idx - 1],
        next=frames[center_idx + 1],
        window=list(frames),
    )


def test_isolated_shimmer_not_flagged():
    """Scattered single-pixel temporal noise (edge shimmer) must not record a finding."""
    rng = np.random.default_rng(1)
    h, w = 96, 128
    base = np.full((h, w, 3), 0.5, dtype=np.float32)
    # ~40 scattered pixels, each an independent per-frame value flip (std_t ~0.11).
    n_px = 40
    ys, xs = rng.integers(0, h, size=n_px), rng.integers(0, w, size=n_px)
    frames = []
    for _ in range(7):
        f = base.copy()
        flip = rng.uniform(-0.2, 0.2, size=(n_px, 3))
        f[ys, xs] += flip
        frames.append(np.clip(f, 0, 1).astype(np.float32))

    mask, sev = detect(ArtifactType.FLICKER, _win(frames), sens=0.5)
    assert sev == 0.0, f"isolated shimmer recorded severity {sev}"
    # No surviving component may reach the coherence gate.
    import cv2

    n_cc, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0.3).astype(np.uint8), connectivity=8)
    assert all(a < MIN_COMPONENT_PX for a in stats[1:, cv2.CC_STAT_AREA])


def test_region_flicker_still_caught_after_coherence_filter():
    """A patch-like flicker region (the real artifact class) survives the filter."""
    rng = np.random.default_rng(2)
    base = np.full((96, 128, 3), 0.5, dtype=np.float32)
    frames = []
    for _ in range(7):
        f = base.copy()
        f[24:64, 32:96] += rng.normal(0, 0.15, size=(40, 64, 3))
        frames.append(np.clip(f, 0, 1).astype(np.float32))

    mask, sev = detect(ArtifactType.FLICKER, _win(frames), sens=0.7)
    assert sev > 0.3, f"region flicker severity {sev} below gate after coherence filter"
    assert mask[24:64, 32:96].mean() > 0.3

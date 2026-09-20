"""Tests for the interpolation stage: slotting, de-duplication and RIFE inference.

The logic tests inject a fake runner (no GPU/weights needed). The ``gpu``-marked
tests run real RIFE 4.25 at a tiny shape: torch path first, then the full
legacy-ONNX-export -> TensorRT-engine -> multi-input infer_multi path.
"""
from __future__ import annotations

import numpy as np
import pytest

from aive.stages.interpolate.gapplan import estimate_output_frames
from aive.stages.interpolate.stage import InterpolateConfig, InterpolateStage


class _FakeRunner:
    """Stands in for RIFERunner: records calls, returns a midpoint blend."""

    def __init__(self):
        self.calls = 0

    def __call__(self, img0, img1, t):
        self.calls += 1
        return np.clip(
            (1 - t) * img0.astype(np.float32) + t * img1.astype(np.float32), 0, 255
        ).astype(np.uint8)


class _Ctx:
    def __init__(self, width=96, height=64, fps="24/1"):
        self.width = width
        self.height = height
        self.fps = fps


def _moving_frames(n, w=96, h=64):
    """Deterministic moving pattern: a bright block shifts 7 px each frame."""
    frames = []
    for k in range(n):
        f = np.zeros((h, w, 3), dtype=np.uint8)
        x0 = (k * 7) % w
        f[:, x0 : x0 + 12] = 255
        frames.append(f)
    return frames


def _rife_weights():
    from aive.models.downloader import download_model
    from aive.models.registry import get_model

    return download_model(get_model("rife_425"))


# -- logic tests (fake runner, no GPU) ----------------------------------------


def test_stage_slotting_and_endpoints():
    stage = InterpolateStage(InterpolateConfig(enabled=True, model="rife_425", multiplier=2))
    fake = _FakeRunner()
    stage._runner = fake  # skip model download/build

    src = _moving_frames(8)
    out = list(stage.process(iter(src), _Ctx()))

    assert len(out) == estimate_output_frames(8, 2, 1) == 15
    # Endpoints land exactly on source frames (slot floor(k * f)).
    for k in range(8):
        np.testing.assert_array_equal(out[2 * k], src[k])
    # Intermediates are genuine blends of the neighbouring pair.
    mid = out[1]
    assert not np.array_equal(mid, src[0]) and not np.array_equal(mid, src[1])


def test_stage_target_fps_ratio():
    stage = InterpolateStage(InterpolateConfig(enabled=True, model="rife_425", target_fps=60.0))
    fake = _FakeRunner()
    stage._runner = fake

    src = _moving_frames(96)
    out = list(stage.process(iter(src), _Ctx(fps="24/1")))

    assert len(out) == estimate_output_frames(96, 5, 2) == 238
    # Source frame k lands on slot floor(k * 5 / 2).
    for k in (0, 1, 2, 47, 95):
        np.testing.assert_array_equal(out[(k * 5) // 2], src[k])


def test_stage_dedup_skips_model_calls():
    stage = InterpolateStage(
        InterpolateConfig(enabled=True, model="rife_425", multiplier=2, dedup_threshold=0.1)
    )
    fake = _FakeRunner()
    stage._runner = fake

    static = np.full((64, 96, 3), 128, dtype=np.uint8)
    src = [static.copy() for _ in range(8)]
    out = list(stage.process(iter(src), _Ctx()))

    assert len(out) == estimate_output_frames(8, 2, 1)
    assert fake.calls == 0
    assert stage.dedup_skips == 7
    # Slots are filled with copies of the previous frame.
    for f in out:
        np.testing.assert_array_equal(f, static)


def test_stage_dedup_threshold_zero_disables():
    stage = InterpolateStage(InterpolateConfig(enabled=True, model="rife_425", multiplier=2))
    fake = _FakeRunner()
    stage._runner = fake

    static = np.full((64, 96, 3), 128, dtype=np.uint8)
    src = [static.copy() for _ in range(4)]
    list(stage.process(iter(src), _Ctx()))

    assert fake.calls == 3  # every interval still costs a model call


def test_all_chain_stages_expose_output_dims():
    """Regression guard for run_pipeline's dimension folding.

    The orchestrator calls ``output_dims`` on every stage in the chain before
    encoding; any stage missing it crashes the whole job (surfaced by the M5
    demo re-run: InterpolateStage lacked the method)."""
    from aive.core.orchestrator import PipelineConfig, build_stage_chain

    cfg = PipelineConfig(fi_enabled=True, sr_enabled=True, qa_enabled=True)
    dims = (640, 360)
    for st in build_stage_chain(cfg):
        dims = st.output_dims(_Ctx(width=dims[0], height=dims[1]))
    assert dims == (1280, 720)  # SR x2; FI and QA preserve resolution


# -- GPU tests (real RIFE 4.25 weights) ----------------------------------------


@pytest.mark.gpu
def test_interpolate_stage_with_real_rife_torch():
    """Stage slotting + real RIFE torch inference at tiny shape (96x64 -> pad 128x64)."""
    from aive.stages.interpolate.stage import RIFERunner

    weights = _rife_weights()
    stage = InterpolateStage(InterpolateConfig(enabled=True, model="rife_425", multiplier=2))
    runner = RIFERunner(weights, width=96, height=64, use_trt=False)
    assert (runner.pw, runner.ph) == (128, 64)
    stage._runner = runner

    src = _moving_frames(8)
    out = list(stage.process(iter(src), _Ctx()))

    assert len(out) == estimate_output_frames(8, 2, 1) == 15
    for k in range(8):
        np.testing.assert_array_equal(out[2 * k], src[k])
    # Intermediates must not be exact copies of either endpoint (motion present).
    for i in range(1, len(out), 2):
        k = i // 2
        assert not np.array_equal(out[i], src[k])
        assert not np.array_equal(out[i], src[k + 1])


@pytest.mark.gpu
@pytest.mark.parametrize("variant", ["rife_425", "rife_425_heavy", "rife_425_lite"])
def test_rife_variants_torch_sanity(variant):
    """Every RIFE 4.25-family variant loads its weights and synthesizes a frame (torch mode)."""
    from aive.models.downloader import download_model
    from aive.models.registry import get_model
    from aive.stages.interpolate.stage import RIFERunner

    weights = download_model(get_model(variant))
    runner = RIFERunner(weights, width=96, height=64, use_trt=False, variant=variant)
    assert (runner.pw, runner.ph) == (128, 64)

    frames = _moving_frames(3)
    out = runner(frames[0], frames[2], 0.5)
    assert out.shape == (64, 96, 3) and out.dtype == np.uint8
    # The interpolated frame must differ from both endpoints (real motion synthesis).
    assert not np.array_equal(out, frames[0])
    assert not np.array_equal(out, frames[2])
    # Regression: __call__ must scale the [0,1] model output to uint8 — rounding
    # without *255 produced binary {0,1} frames that still passed the checks above.
    assert out.max() > 200 and len(np.unique(out)) > 4, (
        f"interpolated frame lacks dynamic range (max={out.max()}, "
        f"unique={len(np.unique(out))}) — check [0,1] -> uint8 scaling"
    )


@pytest.mark.gpu
def test_rife_trt_engine_tiny_shape():
    """Legacy ONNX export -> TRT engine build -> multi-input infer_multi.

    Compares the fp16 TensorRT output against the fp16 torch reference; both
    implement the same stateless graph so they must be near-identical.
    """
    from aive.stages.interpolate.stage import RIFERunner

    weights = _rife_weights()
    trt_runner = RIFERunner(weights, width=96, height=64, use_trt=True)
    assert trt_runner._trt_runner is not None, "TRT engine build failed"

    ref_runner = RIFERunner(weights, width=96, height=64, use_trt=False)

    src = _moving_frames(2)
    out_trt = trt_runner(src[0], src[1], 0.5)
    out_ref = ref_runner(src[0], src[1], 0.5)

    assert out_trt.shape == (64, 96, 3) and out_trt.dtype == np.uint8
    diff = np.abs(out_trt.astype(np.int16) - out_ref.astype(np.int16))
    # fp16 TRT vs fp16 torch: ~0.04 in [0,1] at high-contrast edges -> a few
    # gray levels after *255; the frame as a whole must be near-identical.
    assert diff.max() <= 16 and diff.mean() < 2.0, (
        f"TRT vs torch max abs diff {diff.max()} (mean {diff.mean():.3f})"
    )

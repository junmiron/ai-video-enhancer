"""CPU regression tests for SR cascade (Option A).

When a user requests an output scale that is an integer multiple of the model's
native scale, :class:`UpscaleStage` chains that many AI passes so every doubling
gets real super-resolution detail (true Nx) instead of one pass + Lanczos resample.
A fake nearest-x2 runner makes each pass exact, so we can assert both the final
output dims and the exact sequence of input resolutions the chain ran at.

Non-multiple targets and downscaling keep the old single-pass + Lanczos behavior;
a model run at its own native scale never cascades (one pass).
"""
from __future__ import annotations

import types

import numpy as np

from aive.stages.upscale.stage import UpscaleConfig, UpscaleStage


class _FakeSRRunner:
    """Stand-in for SRRunner: records its input (width,height) and does exact nearest-2x."""

    calls: list[tuple[int, int]] = []  # class-level so tests can inspect the chain order

    def __init__(self, weights_path, width, height, model_name=None, use_trt=True):
        self.width = width
        self.height = height
        self.scale = 2
        _FakeSRRunner.calls.append((width, height))

    def __call__(self, frame_hwc_u8: np.ndarray) -> np.ndarray:
        # Exact nearest-neighbor 2x spatial upscale (H and W only).
        return np.repeat(np.repeat(frame_hwc_u8, 2, axis=0), 2, axis=1)


def _make_stage(monkeypatch, model: str, scale):
    import aive.models.downloader as dl_mod
    import aive.models.registry as reg_mod

    from aive.stages.upscale import stage as sr_stage

    # process() imports download_model/get_model at call time and resolves SRRunner
    # through the module global, so patching these source attributes is enough.
    monkeypatch.setattr(sr_stage, "SRRunner", _FakeSRRunner)
    monkeypatch.setattr(dl_mod, "download_model", lambda info: "fake.pth")
    monkeypatch.setattr(
        reg_mod, "get_model", lambda name, task=None: types.SimpleNamespace(name=name)
    )
    return UpscaleStage(UpscaleConfig(enabled=True, model=model, scale=scale))


def _ctx(width: int, height: int):
    # process() only needs width/height; control is optional (getattr -> None).
    return types.SimpleNamespace(width=width, height=height)


def test_cascade_x4_two_passes(monkeypatch):
    # x2 model (ShuffleCUGAN), target x4 -> two chained AI passes (true x4).
    _FakeSRRunner.calls = []
    stage = _make_stage(monkeypatch, "shufflecugan_x2", scale=4)

    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, size=(24, 32, 3)).astype(np.uint8)  # H=24 W=32

    out = list(stage.process(iter([frame]), _ctx(width=32, height=24)))
    assert len(out) == 1
    assert out[0].shape == (96, 128, 3), f"cascade output {out[0].shape}"
    assert out[0].dtype == np.uint8
    # Chain ran at source then source*2: two real AI passes, no Lanczos.
    assert _FakeSRRunner.calls == [(32, 24), (64, 48)], f"chain calls {_FakeSRRunner.calls}"


def test_cascade_x8_three_passes(monkeypatch):
    # x2 model, target x8 -> three chained passes.
    _FakeSRRunner.calls = []
    stage = _make_stage(monkeypatch, "shufflecugan_x2", scale=8)

    rng = np.random.default_rng(10)
    frame = rng.integers(0, 256, size=(16, 16, 3)).astype(np.uint8)

    out = list(stage.process(iter([frame]), _ctx(width=16, height=16)))
    assert out[0].shape == (128, 128, 3), f"x8 output {out[0].shape}"
    assert _FakeSRRunner.calls == [(16, 16), (32, 32), (64, 64)]


def test_native_scale_single_pass_no_resize(monkeypatch):
    # target == native -> exactly one pass, no Lanczos.
    _FakeSRRunner.calls = []
    stage = _make_stage(monkeypatch, "shufflecugan_x2", scale=2)

    rng = np.random.default_rng(8)
    frame = rng.integers(0, 256, size=(24, 32, 3)).astype(np.uint8)

    out = list(stage.process(iter([frame]), _ctx(width=32, height=24)))
    assert out[0].shape == (48, 64, 3), f"native output {out[0].shape}"
    assert _FakeSRRunner.calls == [(32, 24)]


def test_nonmultiple_target_single_pass_lanczos(monkeypatch):
    # x2 model, target x3 -> not an integer multiple -> one pass + Lanczos to 3x.
    _FakeSRRunner.calls = []
    stage = _make_stage(monkeypatch, "shufflecugan_x2", scale=3)

    rng = np.random.default_rng(9)
    frame = rng.integers(0, 256, size=(24, 32, 3)).astype(np.uint8)

    out = list(stage.process(iter([frame]), _ctx(width=32, height=24)))
    # One AI pass (x2 -> 48x64) then Lanczos to target x3 (72x96).
    assert out[0].shape == (72, 96, 3), f"non-multiple output {out[0].shape}"
    assert _FakeSRRunner.calls == [(32, 24)]


def test_divisible_but_not_power_target_lanczos(monkeypatch):
    # x6 on an x2 model is divisible by the native scale but NOT a power of it.
    # The old code cascaded here (wrong output dims); correct behavior is one
    # pass + LANCZOS to exactly 3x... i.e. target x6, so output = source * 6.
    _FakeSRRunner.calls = []
    stage = _make_stage(monkeypatch, "shufflecugan_x2", scale=6)

    rng = np.random.default_rng(14)
    frame = rng.integers(0, 256, size=(24, 32, 3)).astype(np.uint8)

    out = list(stage.process(iter([frame]), _ctx(width=32, height=24)))
    # Single AI pass (x2 -> 48x64) then Lanczos to target x6 (144x192).
    assert out[0].shape == (144, 192, 3), f"divisible-not-power output {out[0].shape}"
    assert _FakeSRRunner.calls == [(32, 24)]


def test_native_x4_model_no_cascade(monkeypatch):
    # Real-ESRGAN is native x4; target x4 -> single pass (no cascade). The fake
    # always upscales by 2, so we assert the chain length/dims, not output shape.
    _FakeSRRunner.calls = []
    stage = _make_stage(monkeypatch, "realesrgan_x4plus", scale=4)

    rng = np.random.default_rng(11)
    frame = rng.integers(0, 256, size=(16, 16, 3)).astype(np.uint8)

    list(stage.process(iter([frame]), _ctx(width=16, height=16)))
    assert _FakeSRRunner.calls == [(16, 16)], f"native-x4 chain {_FakeSRRunner.calls}"


def test_runner_cache_reused_across_frames(monkeypatch):
    # The runner chain is built once and reused for every frame of the job.
    _FakeSRRunner.calls = []
    stage = _make_stage(monkeypatch, "shufflecugan_x2", scale=4)

    rng = np.random.default_rng(13)
    frames = [rng.integers(0, 256, size=(24, 32, 3)).astype(np.uint8) for _ in range(3)]

    out = list(stage.process(iter(frames), _ctx(width=32, height=24)))
    assert len(out) == 3 and all(o.shape == (96, 128, 3) for o in out)
    # Two runners built once total (not per frame).
    assert _FakeSRRunner.calls == [(32, 24), (64, 48)]

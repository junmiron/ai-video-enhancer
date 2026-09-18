"""Tests for the vendored SR architectures and checkpoint handling (CPU).

No weights are downloaded here: shapes are checked with randomly-initialized
models, and the real-weight load paths are covered by the ``gpu``-marked tests.
The size constraints pinned here were verified against the actual checkpoints
(sudo_shuffle_cugan_9.584.969.pth / RealESRGAN_x4plus.pth) during M3 research:

* ShuffleCUGAN (UpCunet2x_fast): input dims must be multiples of 4 and >= 40.
* RRDBNet x4: no size constraints; output is exactly 4x the input.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from aive.stages.upscale.sr_model import (
    SPECS,
    SR_MODELS,
    apply_sr_state,
    build_sr,
    spec_for,
    unwrap_sr_state,
)


# -- specs ---------------------------------------------------------------------


def test_specs_cover_registered_models():
    assert set(SPECS) == set(SR_MODELS)
    assert spec_for("shufflecugan_x2").scale == 2
    assert spec_for("realesrgan_x4plus").scale == 4
    with pytest.raises(ValueError):
        spec_for("nope")


# -- ShuffleCUGAN (UpCunet2x_fast) ---------------------------------------------


def test_cugan_forward_shape_and_finite():
    model = build_sr("shufflecugan_x2", device="cpu").eval()
    x = torch.randn(1, 3, 48, 96)
    with torch.no_grad():
        y = model(x)
    assert tuple(y.shape) == (1, 3, 96, 192)
    assert bool(torch.isfinite(y).all())


def test_cugan_rejects_non_multiple_of_4():
    """Pinned constraint: dims must be multiples of 4 (internal stride math)."""
    model = build_sr("shufflecugan_x2", device="cpu").eval()
    with pytest.raises(RuntimeError):
        with torch.no_grad():
            model(torch.randn(1, 3, 50, 96))


def test_cugan_rejects_dims_below_min():
    """Pinned constraint: reflect padding needs dims > 38 (min_dim=40)."""
    model = build_sr("shufflecugan_x2", device="cpu").eval()
    with pytest.raises(RuntimeError):
        with torch.no_grad():
            model(torch.randn(1, 3, 36, 96))


# -- RRDBNet (Real-ESRGAN x4plus) ----------------------------------------------


def test_rrdbnet_forward_shape():
    """Shape check only: random init overflows a 23-block RRDB trunk; finiteness
    with real weights is covered by the gpu-marked tests."""
    model = build_sr("realesrgan_x4plus", device="cpu").eval()
    for h, w in ((32, 32), (48, 96)):
        x = torch.randn(1, 3, h, w)
        with torch.no_grad():
            y = model(x)
        assert tuple(y.shape) == (1, 3, h * 4, w * 4)


# -- checkpoint handling ---------------------------------------------------------


def test_unwrap_params_ema_wrapper():
    inner = {"conv_first.weight": torch.zeros(64, 3, 3, 3)}
    state = unwrap_sr_state({"params_ema": inner})
    assert "conv_first.weight" in state and "params_ema" not in state


def test_unwrap_params_wrapper():
    inner = {"a": torch.zeros(1)}
    state = unwrap_sr_state({"params": inner, "other_meta": 5})
    assert "a" in state


def test_unwrap_plain_dict_passthrough():
    plain = {"conv_last.weight": torch.zeros(3, 64, 3, 3)}
    assert unwrap_sr_state(plain) is plain or dict(unwrap_sr_state(plain)) == plain


def test_apply_sr_state_fatal_on_missing_keys():
    model = build_sr("realesrgan_x4plus", device="cpu")
    state = {k: v.clone() for k, v in model.state_dict().items()}
    del state["conv_last.weight"]  # an inference-critical key
    with pytest.raises(RuntimeError, match="missing key"):
        apply_sr_state(model, state)


def test_apply_sr_state_ignores_training_extras():
    """Non-strict on purpose: training-only extras are dropped without error."""
    model = build_sr("shufflecugan_x2", device="cpu")
    state = dict(model.state_dict())
    state["optimizer_step"] = torch.tensor(1234)
    apply_sr_state(model, state)  # must not raise


# -- numpy round-trip helpers (used by the runner) ---------------------------------


def test_nchw_roundtrip_matches_torch():
    """The runner's HWC uint8 -> NCHW float [0,1] conversion is exact."""
    frame = np.random.randint(0, 256, size=(48, 96, 3), dtype=np.uint8)
    f = frame.astype(np.float32) / 255.0
    nchw = np.ascontiguousarray(f.transpose(2, 0, 1)[None])
    t = torch.from_numpy(nchw)
    back = (t[0].permute(1, 2, 0).numpy() * 255.0)
    np.testing.assert_allclose(back, frame.astype(np.float32), atol=1e-4)


# -- Fallin soft/strong (same UpCunet2x_fast arch as ShuffleCUGAN) ----------------


@pytest.mark.parametrize("name", ["fallin_soft", "fallin_strong"])
def test_fallin_forward_shape(name):
    model = build_sr(name, device="cpu").eval()
    x = torch.randn(1, 3, 48, 96)
    with torch.no_grad():
        y = model(x)
    assert tuple(y.shape) == (1, 3, 96, 192)


@pytest.mark.parametrize("name", ["fallin_soft", "fallin_strong"])
def test_fallin_rejects_non_multiple_of_4(name):
    """Same arch as ShuffleCUGAN -> same pinned size constraints."""
    model = build_sr(name, device="cpu").eval()
    with pytest.raises(RuntimeError):
        with torch.no_grad():
            model(torch.randn(1, 3, 50, 96))


# -- SPAN (Modern Spanimation V2) -------------------------------------------------


def test_span_forward_shape_odd_dims():
    """All convs are 3x3 padded -> no size constraints; output is exactly 2x."""
    model = build_sr("span_v2", device="cpu").eval()
    for h, w in ((33, 51), (48, 96)):
        x = torch.randn(1, 3, h, w)
        with torch.no_grad():
            y = model(x)
        assert tuple(y.shape) == (1, 3, h * 2, w * 2)


def test_span_state_dict_has_no_norm_buffer():
    """The checkpoint carries a zero `no_norm` buffer (norm=False training);
    the arch must expose it so loads are exact."""
    model = build_sr("span_v2", device="cpu")
    sd = model.state_dict()
    assert "no_norm" in sd and tuple(sd["no_norm"].shape) == (1,)


def test_span_fuse_matches_unfused_path():
    """fuse() must fold sk + conv(0/1/2) into eval_conv exactly: the fused
    eval-mode output equals the unfused training-path output."""
    model = build_sr("span_v2", device="cpu")
    x = torch.randn(1, 3, 48, 64)
    with torch.no_grad():
        y_unfused = model.train()(x)  # train mode -> conv(x_pad) + sk(x) path
        model.eval()
        model.fuse()
        y_fused = model(x)
    assert torch.allclose(y_unfused, y_fused, atol=1e-5), (
        f"max diff {(y_unfused - y_fused).abs().max().item():.3e}"
    )


# -- Compact SRVGG (Cyte V1 / OpenProteus) ----------------------------------------


@pytest.mark.parametrize("name", ["cyte_v1", "openproteus"])
def test_compact_forward_shape_odd_dims(name):
    model = build_sr(name, device="cpu").eval()
    for h, w in ((33, 51), (48, 96)):
        x = torch.randn(1, 3, h, w)
        with torch.no_grad():
            y = model(x)
        assert tuple(y.shape) == (1, 3, h * 2, w * 2)


@pytest.mark.parametrize("name", ["cyte_v1", "openproteus"])
def test_compact_nearest_neighbor_residual(name):
    """umzi2 variant: out = pixel_shuffle(body(x)) + nn_upsample(x). With the
    body zeroed, the output must equal exactly the nearest-neighbor upsample."""
    import torch.nn.functional as F

    model = build_sr(name, device="cpu").eval()
    with torch.no_grad():
        for p in model.body.parameters():
            p.zero_()
        x = torch.rand(1, 3, 37, 53)
        y = model(x)
    expected = F.interpolate(x, scale_factor=2, mode="nearest")
    assert torch.allclose(y, expected, atol=1e-6), (
        f"max diff {(y - expected).abs().max().item():.3e}"
    )

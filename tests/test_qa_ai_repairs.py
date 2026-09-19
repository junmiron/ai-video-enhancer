"""Unit tests for M4.x AI repair models — GPU- and weight-free by default.

Coverage:
* FastLineDarken port vs an inline torch reference of TAS's exact math, plus
  properties (darkening-only, flat-frame safety).
* Restore arch construction + the deepDeband state-dict remap (synthetic
  checkpoints built from the arch itself — no downloads).
* Runner wiring: monkeypatched ``get_restore_runner`` proves the CV fallbacks
  and the AI-first ordering without any model.
* RestoreRunner tiling/full-frame plumbing with a fake identity model.

Real-weight + TRT validation lives in the manual GPU script (see docs/plan.md,
M4.x entry).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from aive.stages.qa.findings import ArtifactType
from aive.stages.qa.repairs import RepairContext, repair
from aive.stages.qa.repairs import ai as ai_mod
from aive.stages.qa.repairs.fastlinedarken import fast_line_darken


# -- FastLineDarken port ---------------------------------------------------------


def _torch_reference(img_hwc_f32: np.ndarray, strength: float = 0.7) -> np.ndarray:
    """TAS's exact math in torch (CPU) — the cross-check for the numpy port."""
    x = torch.from_numpy(np.ascontiguousarray(img_hwc_f32.transpose(2, 0, 1))).to(torch.float32)
    weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    gray = torch.tensordot(x, weights, dims=([0], [0]))
    gx = torch.nn.functional.conv2d(gray[None, None], sobel_x, padding=1)[0, 0]
    gy = torch.nn.functional.conv2d(gray[None, None], sobel_y, padding=1)[0, 0]
    edges = torch.sqrt(gx ** 2 + gy ** 2)
    edges = (edges - edges.min()) / (edges.max() - edges.min())
    thinned = -torch.nn.functional.max_pool2d(-edges[None, None], kernel_size=3, stride=1, padding=1)[0, 0]

    kx = torch.exp(-(torch.arange(-1.0, 2.0) ** 2) / 2.0)
    kx = kx / kx.sum()
    k2d = (kx[:, None] * kx[None, :]).view(1, 1, 3, 3).expand(1, 1, 3, 3)
    softened = torch.nn.functional.conv2d(thinned[None], k2d, padding=1)[0]

    out = x - strength * softened[None]
    return torch.clamp(out, 0.0, 1.0).numpy().transpose(1, 2, 0)


def test_fastlinedarken_matches_torch_reference():
    rng = np.random.default_rng(42)
    # Structured anime-ish frame: smooth gradient + dark lines (one near the left
    # border so zero-padding semantics at the edge are exercised).
    h, w = 56, 80
    grad = np.linspace(0.15, 0.75, w)[None, :, None] * np.ones((h, 1, 3))
    frame = (grad + rng.normal(0, 0.02, size=(h, w, 3))).clip(0, 1)
    frame[:, 4:6, :] *= 0.25   # vertical dark line near the left border
    frame[28:30, :, :] *= 0.3  # horizontal dark line

    out = fast_line_darken((frame * 255).astype(np.uint8))
    ref = _torch_reference(frame)
    diff = np.abs(out.astype(np.float32) / 255.0 - ref)
    assert diff.max() <= 2.0 / 255.0, f"max diff {diff.max():.4f} (tolerance covers uint8 rounding)"


def test_fastlinedarken_darkens_only_and_flat_safe():
    rng = np.random.default_rng(7)
    frame = (rng.integers(0, 256, size=(32, 48, 3))).astype(np.uint8)
    out = fast_line_darken(frame)
    assert out.dtype == np.uint8 and out.shape == frame.shape
    # The filter only subtracts (clamped at 0): output never exceeds input (+1 for rounding).
    assert (out.astype(int) <= frame.astype(int) + 1).all()

    # Flat frame: no edges -> unchanged copy (TAS's unguarded divide would NaN here).
    flat = np.full((24, 32, 3), 120, dtype=np.uint8)
    assert np.array_equal(fast_line_darken(flat), flat)


# -- restore archs + weight handling ---------------------------------------------


def test_restore_specs():
    assert set(ai_mod.RESTORE_SPECS) == {"scunet", "deepdeband_f"}
    assert ai_mod.spec_for_restore("scunet").pad_mod == 64
    assert ai_mod.spec_for_restore("scunet").min_dim == 64
    assert ai_mod.spec_for_restore("deepdeband_f").pad_mod == 256
    assert ai_mod.spec_for_restore("deepdeband_f").min_dim == 256
    with pytest.raises(ValueError):
        ai_mod.spec_for_restore("nope")


def test_scunet_arch_builds_with_pinned_config():
    # config=[4]*7 is pinned to the color-real-psnr checkpoint (verified strict).
    m = ai_mod.build_restore("scunet", device="cpu")
    n_params = sum(p.numel() for p in m.parameters()) / 1e6
    assert abs(n_params - 17.95) < 0.05, f"SCUNet params {n_params:.2f}M != 17.95M"


def test_deepdeband_arch_builds():
    m = ai_mod.build_restore("deepdeband_f", device="cpu")
    n_params = sum(p.numel() for p in m.parameters()) / 1e6
    assert abs(n_params - 54.4) < 0.2, f"deepDeband-f params {n_params:.2f}M != 54.4M"


def test_deepdeband_state_remap_roundtrip():
    """The checkpoint is a plain UnetGenerator256 state dict ('model.*' keys);
    apply_restore_state must remap it onto the wrapper's .net attribute."""
    m = ai_mod.build_restore("deepdeband_f", device="cpu")
    fake_ckpt = {k[len("net."):]: v for k, v in m.state_dict().items()}
    assert all(k.startswith("model.") for k in fake_ckpt)
    ai_mod.apply_restore_state(m, "deepdeband_f", dict(fake_ckpt))  # must not raise


def test_scunet_state_load_exact_and_fatal_on_mismatch():
    m = ai_mod.build_restore("scunet", device="cpu")
    ai_mod.apply_restore_state(m, "scunet", dict(m.state_dict()))  # exact match passes

    bad = dict(m.state_dict())
    del bad["m_tail.0.weight"]
    with pytest.raises(RuntimeError):
        ai_mod.apply_restore_state(ai_mod.build_restore("scunet", device="cpu"), "scunet", bad)


def test_manifest_has_restore_models():
    from aive.models.registry import list_models

    names = {m.name for m in list_models(task="restore")}
    assert {"scunet", "deepdeband_f"} <= names


# -- repair wiring: AI-first with CV fallback ------------------------------------


def _ctx(frame, mask, ai_repairs=True):
    return RepairContext(
        frame=frame,
        mask=mask,
        window=[frame.copy() for _ in range(7)],
        center_idx=3,
        ai_repairs=ai_repairs,
    )


def test_flicker_uses_scunet_when_runner_available(monkeypatch):
    class FakeRunner:
        def __call__(self, frame):
            import cv2

            return cv2.GaussianBlur(frame.astype(np.float32), (5, 5), 1.0) / 255.0

    seen = []

    def fake_get(name, w, h):
        seen.append((name, w, h))
        return FakeRunner() if name == "scunet" else None

    monkeypatch.setattr(ai_mod, "get_restore_runner", fake_get)
    rng = np.random.default_rng(0)
    frame = (rng.integers(60, 200, size=(48, 64, 3))).astype(np.uint8)
    mask = np.zeros((48, 64), dtype=np.float32)
    mask[10:30, 20:40] = 1.0

    out, method = repair(ArtifactType.FLICKER, _ctx(frame, mask))
    assert method == "scunet-denoise" and seen == [("scunet", 64, 48)]
    # targeted: outside the mask the frame is untouched
    assert np.array_equal(out[0:5, 0:5], frame[0:5, 0:5])


def test_flicker_falls_back_to_cv_when_runner_unavailable(monkeypatch):
    monkeypatch.setattr(ai_mod, "get_restore_runner", lambda *a, **k: None)
    rng = np.random.default_rng(1)
    base = np.full((64, 96, 3), 128, dtype=np.int16)
    frames = []
    for _ in range(7):
        f = base.copy()
        f[16:48, 24:72] += rng.integers(-25, 25, size=(32, 48, 3))
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    mask = np.zeros((64, 96), dtype=np.float32)
    mask[16:48, 24:72] = 1.0

    ctx = _ctx(frames[3], mask)
    ctx.window = frames
    out, method = repair(ArtifactType.FLICKER, ctx)
    assert method == "temporal-smooth" and out.dtype == np.uint8


def test_ai_repairs_disabled_never_touches_runner(monkeypatch):
    seen = []
    monkeypatch.setattr(ai_mod, "get_restore_runner", lambda *a, **k: seen.append(a))
    rng = np.random.default_rng(2)
    frame = (rng.integers(60, 200, size=(48, 64, 3))).astype(np.uint8)
    mask = np.ones((48, 64), dtype=np.float32)

    out_f, m_f = repair(ArtifactType.FLICKER, _ctx(frame, mask, ai_repairs=False))
    out_b, m_b = repair(ArtifactType.BANDING, _ctx(frame, mask, ai_repairs=False))
    assert not seen  # runner must never be requested when the flag is off
    assert m_f == "temporal-smooth" or m_f == "none"
    assert m_b in ("deband-dither", "none")


def test_banding_uses_deepdeband_when_runner_available(monkeypatch):
    class FakeRunner:
        def __call__(self, frame):
            return (frame.astype(np.float32) / 255.0) + 0.01  # any [0,1]-ish output

    monkeypatch.setattr(ai_mod, "get_restore_runner", lambda name, w, h: FakeRunner() if name == "deepdeband_f" else None)
    rng = np.random.default_rng(3)
    frame = (rng.integers(60, 200, size=(48, 64, 3))).astype(np.uint8)
    mask = np.zeros((48, 64), dtype=np.float32)
    mask[10:30, 20:40] = 1.0

    out, method = repair(ArtifactType.BANDING, _ctx(frame, mask))
    assert method == "deepdeband-f" and out.dtype == np.uint8


def test_banding_falls_back_to_dither_when_runner_unavailable(monkeypatch):
    monkeypatch.setattr(ai_mod, "get_restore_runner", lambda *a, **k: None)
    w, h = 96, 48
    grad = np.linspace(0.2, 0.8, w)[None, :, None] * np.ones((h, 1, 3))
    frame = (np.round(grad / 0.1) * 0.1 * 255).astype(np.uint8)  # posterized gradient
    mask = np.ones((h, w), dtype=np.float32)

    out, method = repair(ArtifactType.BANDING, _ctx(frame, mask))
    assert method == "deband-dither" and out.dtype == np.uint8


def test_lineart_uses_fastlinedarken_label():
    rng = np.random.default_rng(4)
    frame = (rng.integers(0, 255, size=(48, 64, 3))).astype(np.uint8)
    mask = np.zeros((48, 64), dtype=np.float32)
    mask[10:30, 20:40] = 1.0

    out, method = repair(ArtifactType.LINEART, _ctx(frame, mask))
    assert method == "fastlinedarken" and out.dtype == np.uint8


# -- RestoreRunner plumbing (fake identity model, no GPU/weights) -----------------


class _IdentityModel(torch.nn.Module):
    """Local identity op: tiled blending must reproduce the input exactly."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))  # so next(parameters()) works

    def forward(self, x):
        return x


def _make_runner(monkeypatch, width: int, height: int, free_vram_bytes: int) -> ai_mod.RestoreRunner:
    import aive.inference.trt as trt_mod

    monkeypatch.setattr(ai_mod, "load_restore", lambda name, path, **kw: _IdentityModel())
    monkeypatch.setattr(trt_mod, "free_vram_bytes", lambda: free_vram_bytes)
    return ai_mod.RestoreRunner("fake.pth", width=width, height=height, model_name="scunet", use_trt=False)


def test_restore_runner_full_frame_identity(monkeypatch):
    runner = _make_runner(monkeypatch, 384, 256, free_vram_bytes=100 * (1 << 30))
    assert runner.tile_cover is None and (runner.engine_w, runner.engine_h) == (384, 256)

    rng = np.random.default_rng(5)
    frame = rng.integers(0, 256, size=(256, 384, 3)).astype(np.uint8)
    out = runner(frame)
    assert out.shape == (256, 384, 3) and out.dtype == np.float32
    assert np.abs(out - frame.astype(np.float32) / 255.0).max() < 1e-6


def test_restore_runner_tiled_identity(monkeypatch):
    # Tiny free VRAM forces tiling; the identity model makes the expected output exact.
    runner = _make_runner(monkeypatch, 384, 256, free_vram_bytes=1 << 20)
    assert runner.tile_cover is not None

    rng = np.random.default_rng(6)
    frame = rng.integers(0, 256, size=(256, 384, 3)).astype(np.uint8)
    out = runner(frame)
    assert out.shape == (256, 384, 3) and out.dtype == np.float32
    diff = np.abs(out - frame.astype(np.float32) / 255.0).max()
    assert diff < 1e-4, f"tiled identity max diff {diff:.2e}"


def test_restore_runner_pads_to_spec_constraints(monkeypatch):
    # 300x200 is not a multiple of 64 -> engine dims must be padded up.
    runner = _make_runner(monkeypatch, 300, 200, free_vram_bytes=100 * (1 << 30))
    assert runner.engine_w % 64 == 0 and runner.engine_h % 64 == 0
    assert runner.engine_w >= 300 and runner.engine_h >= 200

    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, size=(200, 300, 3)).astype(np.uint8)
    out = runner(frame)
    assert out.shape == (200, 300, 3) and out.dtype == np.float32
    # Content must stay at [0,0] after padding (identity model -> exact).
    diff = np.abs(out - frame.astype(np.float32) / 255.0).max()
    assert diff < 1e-6, f"full-frame padded identity max diff {diff:.2e}"

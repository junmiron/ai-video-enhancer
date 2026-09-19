"""CPU regression tests for SRRunner full-frame + tiled inference.

A fake nearest-x2 model makes the expected output exact: every padded region
must map back to its input pixels with zero shift. This pins down the
_pad_to_engine alignment contract (content at [0,0], bottom/right padding)
end-to-end — the centered-padding version silently shifted edge tiles and
non-multiple full-frame dims by deficit//2 px.
"""
from __future__ import annotations

import numpy as np
import torch

from aive.stages.upscale.stage import SRRunner


class _Nearest2x(torch.nn.Module):
    """Stand-in for an x2 SR model: nearest-neighbor 2x upscale (exact)."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))  # so next(parameters()) works

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")


def _make_runner(monkeypatch, width: int, height: int, free_vram_bytes: int) -> SRRunner:
    import aive.inference.trt as trt_mod

    from aive.stages.upscale import stage as sr_stage

    monkeypatch.setattr(sr_stage, "load_sr", lambda name, path, **kw: _Nearest2x())
    monkeypatch.setattr(trt_mod, "free_vram_bytes", lambda: free_vram_bytes)
    return SRRunner("fake.pth", width=width, height=height, model_name="shufflecugan_x2", use_trt=False)


def test_sr_full_frame_nonmultiple_dims_exact(monkeypatch):
    # 301x201 is not a multiple of pad_mod=4 -> engine 304x204; content must stay at [0,0].
    runner = _make_runner(monkeypatch, 301, 201, free_vram_bytes=100 * (1 << 30))
    assert runner.tile_cover is None and (runner.engine_w, runner.engine_h) == (304, 204)

    rng = np.random.default_rng(11)
    frame = rng.integers(0, 256, size=(201, 301, 3)).astype(np.uint8)
    out = runner(frame)
    exp = frame.repeat(2, axis=0).repeat(2, axis=1)  # nearest-2x (spatial only)
    assert out.shape == (402, 602, 3) and out.dtype == np.uint8
    assert np.array_equal(out, exp), "full-frame padded output is shifted"


def test_sr_tiled_identity_exact(monkeypatch):
    # Tiny free VRAM forces tiling; edge tiles get padded -> exactness proves alignment.
    runner = _make_runner(monkeypatch, 384, 256, free_vram_bytes=1 << 20)
    assert runner.tile_cover is not None

    rng = np.random.default_rng(12)
    frame = rng.integers(0, 256, size=(256, 384, 3)).astype(np.uint8)
    out = runner(frame)
    exp = frame.repeat(2, axis=0).repeat(2, axis=1)  # nearest-2x (spatial only)
    assert out.shape == (512, 768, 3) and out.dtype == np.uint8
    diff = int(np.abs(out.astype(np.int16) - exp.astype(np.int16)).max())
    assert diff == 0, f"tiled nearest-2x max diff {diff}"

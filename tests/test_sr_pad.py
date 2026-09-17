"""CPU tests for SRRunner._pad_to_engine (pure numpy; no GPU needed).

Regression: padding once replicated each edge by the FULL deficit, producing
engine_h + deficit rows instead of engine_h — only visible when a region is
strictly smaller than the engine (tiled mode / non-matching frame dims).
"""
from __future__ import annotations

import numpy as np
import pytest

from aive.stages.upscale.stage import SRRunner


def _pad(engine_h: int, engine_w: int, region_nchw: np.ndarray) -> np.ndarray:
    """Call the unbound method with a minimal fake self."""

    class FakeSelf:
        pass

    fake = FakeSelf()
    fake.engine_h = engine_h
    fake.engine_w = engine_w
    return SRRunner._pad_to_engine(fake, region_nchw)


def _region(h: int, w: int) -> np.ndarray:
    """NCHW float32 with unique values per pixel (plus a channel offset)."""
    base = np.arange(h * w, dtype=np.float32).reshape(1, 1, h, w)
    return np.concatenate([base + c for c in range(3)], axis=1)


@pytest.mark.parametrize("rh,rw,eh,ew", [(160, 160, 192, 192), (102, 192, 192, 192),
                                         (192, 94, 192, 192), (37, 51, 192, 192)])
def test_pad_to_engine_exact_size(rh, rw, eh, ew):
    out = _pad(eh, ew, _region(rh, rw))
    assert out.shape == (1, 3, eh, ew)


def test_pad_content_preserved_and_edges_replicated():
    region = _region(160, 128)
    out = _pad(192, 192, region)
    # The original content sits centered: deficit split top/bottom and left/right.
    dh_top = (192 - 160) // 2          # 16
    dw_left = (192 - 128) // 2         # 32
    core = out[:, :, dh_top:dh_top + 160, dw_left:dw_left + 128]
    np.testing.assert_array_equal(core, region)
    # Replicated edges match the border rows/cols.
    np.testing.assert_array_equal(out[:, :, 0], out[:, :, dh_top])          # top = first row
    np.testing.assert_array_equal(out[:, :, -1], out[:, :, dh_top + 159])   # bottom = last row
    np.testing.assert_array_equal(out[:, :, :, 0], out[:, :, :, dw_left])   # left = first col
    np.testing.assert_array_equal(out[:, :, :, -1], out[:, :, :, dw_left + 127])


def test_pad_odd_deficit_still_exact():
    # Deficit 33 (odd): split 16/17, total still exactly the engine size.
    out = _pad(193, 193, _region(160, 160))
    assert out.shape == (1, 3, 193, 193)


def test_pad_passthrough_at_engine_size():
    region = _region(192, 192)
    out = _pad(192, 192, region)
    assert out is region

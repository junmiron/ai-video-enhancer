"""CPU tests for SRRunner._pad_to_engine (pure numpy; no GPU needed).

Regression history:
* Padding once replicated each edge by the FULL deficit, producing
  engine_h + deficit rows instead of engine_h — only visible when a region is
  strictly smaller than the engine (tiled mode / non-matching frame dims).
* It then CENTERED the content while both consumers (_run_full_frame and
  _run_tiled) crop from the top-left corner, silently shifting every padded
  region by deficit//2 px. Padding must keep content at [0,0] and fill only
  bottom/right — see test_sr_runner_identity.py for end-to-end coverage.
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


def test_pad_content_at_origin_and_bottom_right_replicated():
    region = _region(160, 128)
    out = _pad(192, 192, region)
    # Content stays at [0,0]; the deficit is filled on bottom/right only.
    np.testing.assert_array_equal(out[:, :, :160, :128], region)
    # Replicated edges match the border rows/cols of the content.
    np.testing.assert_array_equal(out[:, :, 191], out[:, :, 159])          # bottom = last row
    np.testing.assert_array_equal(out[:, :, :, 191], out[:, :, :, 127])    # right = last col


def test_pad_odd_deficit_still_exact():
    # Deficit 33 (odd): all of it goes to the bottom/right, total still exact.
    out = _pad(193, 193, _region(160, 160))
    assert out.shape == (1, 3, 193, 193)


def test_pad_passthrough_at_engine_size():
    region = _region(192, 192)
    out = _pad(192, 192, region)
    assert out is region

"""Foundation tests for the M4 QA stage: proxy helpers, motion masks, findings."""
from __future__ import annotations

import numpy as np
import pytest

from aive.stages.qa.findings import ArtifactType, Finding, QAReport, report_from_dict
from aive.stages.qa.motion import flow_inconsistency, flow_magnitude, static_mask
from aive.stages.qa.proxy import (
    downsample_mask,
    gray,
    proxy_dims,
    to_full,
    to_proxy,
    upsample_mask,
)


# -- proxy helpers ------------------------------------------------------------


def test_proxy_dims_downscales_and_keeps_aspect():
    pw, ph = proxy_dims(1920, 1080, target_h=720)
    assert ph == 720
    assert pw == 1280  # 16:9 preserved


def test_proxy_dims_never_upscales_small_input():
    pw, ph = proxy_dims(320, 240, target_h=720)
    assert (pw, ph) == (320, 240)


def test_proxy_dims_even():
    pw, ph = proxy_dims(1921, 1081, target_h=720)
    assert pw % 2 == 0 and ph % 2 == 0


def test_to_proxy_range_and_shape():
    frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    p = to_proxy(frame, 320, 240)
    assert p.shape == (240, 320, 3)
    assert p.dtype == np.float32
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_to_full_roundtrip_close():
    frame = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
    back = to_full(to_proxy(frame, 320, 240), 320, 240)
    assert back.shape == frame.shape and back.dtype == np.uint8
    assert np.abs(back.astype(int) - frame.astype(int)).mean() < 6


def test_mask_up_downsample_shape_range():
    m = np.random.rand(120, 160).astype(np.float32)
    up = upsample_mask(m, 640, 480)
    assert up.shape == (480, 640) and up.min() >= 0.0 and up.max() <= 1.0
    down = downsample_mask(up, 160, 120)
    assert down.shape == (120, 160)


def test_gray_from_uint8_and_float():
    white = np.full((4, 5, 3), 255, dtype=np.uint8)
    assert abs(gray(white)[0, 0] - 1.0) < 1e-3
    red = np.zeros((4, 5, 3), dtype=np.uint8)
    red[..., 0] = 255  # pure red -> luma weight ~0.299
    g_red = gray(red)
    assert g_red.shape == (4, 5) and abs(g_red[0, 0] - 0.299) < 1e-2


# -- motion masks -------------------------------------------------------------


def _moving_pair(shift: int):
    a = np.zeros((64, 96), dtype=np.float32)
    b = np.roll(a, shift=shift, axis=1)
    # put a bright blob so flow has something to track
    a[20:44, 30:50] = 1.0
    b = np.zeros_like(a)
    b[20:44, 30 + shift : 50 + shift] = 1.0
    return a, b


def test_flow_magnitude_static_is_zero():
    a = (np.random.rand(64, 96) * 0.1).astype(np.float32)
    mag = flow_magnitude(a, a.copy())
    assert mag.shape == (64, 96)
    assert np.median(mag) < 0.2


def test_flow_magnitude_detects_motion():
    a, b = _moving_pair(8)
    mag = flow_magnitude(a, b)
    # the moved blob region should show non-trivial displacement
    assert mag.max() > 1.0


def test_static_mask_high_for_still_window():
    base = (np.random.rand(48, 64) * 0.05).astype(np.float32)
    mags = [flow_magnitude(base, base.copy()) for _ in range(4)]
    sm = static_mask(mags, thresh=0.5)
    assert sm.mean() > 0.9


def test_static_mask_low_for_motion():
    a, b = _moving_pair(8)
    mags = [flow_magnitude(a, b), flow_magnitude(b, a)]
    sm = static_mask(mags, thresh=0.5)
    # the moving blob region should NOT be marked static
    assert sm[20:44, 30:60].mean() < 0.5


def test_flow_inconsistency_zero_when_equal():
    m = (np.random.rand(48, 64) * 2).astype(np.float32)
    inc = flow_inconsistency(m, m.copy())
    assert inc.max() < 1e-3


# -- findings model -----------------------------------------------------------


def test_finding_improved():
    f = Finding(artifact=ArtifactType.BANDING, frame_index=5, severity=0.8, before_metric=0.9, after_metric=0.2)
    assert f.improved
    g = Finding(artifact=ArtifactType.BANDING, frame_index=5, severity=0.8, before_metric=0.2, after_metric=0.9)
    assert not g.improved


def test_report_summary_and_roundtrip():
    rep = QAReport(input_path="in.mp4", output_path="out.mp4", fps=60.0, num_frames=120, duration_s=2.0, mode="auto")
    rep.findings.append(Finding(ArtifactType.FLICKER, 10, 0.7, (1, 2, 3, 4), 0.1, True, "temporal-smooth", 0.8, 0.3))
    rep.findings.append(Finding(ArtifactType.BANDING, 20, 0.5, repaired=False, method="detect-only"))

    s = rep.summary()
    assert set(s) == {"flicker", "banding"}
    assert s["flicker"]["count"] == 1 and s["flicker"]["repaired_count"] == 1
    assert s["banding"]["repaired_count"] == 0

    d = rep.to_dict()
    # sample_frames must NOT leak into JSON
    assert "sample_frames" not in d
    back = report_from_dict(d)
    assert len(back.findings) == 2
    assert back.findings[0].artifact is ArtifactType.FLICKER
    assert back.findings[0].repaired and back.findings[0].improved
    assert "00:00:0" in rep.frame_timestamp(10)

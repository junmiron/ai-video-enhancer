"""Unit tests for the M4 QA repairs on synthetic tensors.

The critical property is that a repair *reduces the very metric its detector
measures* (the verify pass). For the acceptance-critical repairs (flicker, ghosting,
banding) we run the real detector before and after; for the rest we check valid
output + correct method label + that only the masked region changed.
"""
from __future__ import annotations

import numpy as np
import pytest

from aive.stages.qa.detectors import detect, FrameWindow
from aive.stages.qa.findings import ArtifactType
from aive.stages.qa.proxy import to_proxy
from aive.stages.qa.repairs import repair, RepairContext


def _win_u8(frames, center_idx=3, **kw):
    pw, ph = frames[0].shape[1], frames[0].shape[0]
    prox = [to_proxy(f, pw, ph) for f in frames]
    win = FrameWindow(
        center=prox[center_idx],
        prev=prox[center_idx - 1] if center_idx > 0 else None,
        next=prox[center_idx + 1] if center_idx < len(prox) - 1 else None,
        window=prox,
        **kw,
    )
    return win, prox


def _rebuild(win, prox, center_idx, new_frame):
    pw, ph = prox[0].shape[1], prox[0].shape[0]
    p2 = list(prox)
    p2[center_idx] = to_proxy(new_frame, pw, ph)
    return FrameWindow(
        center=p2[center_idx],
        prev=p2[center_idx - 1] if center_idx > 0 else None,
        next=p2[center_idx + 1] if center_idx < len(p2) - 1 else None,
        window=p2,
    )


def test_flicker_repair_reduces_temporal_severity():
    rng = np.random.default_rng(0)
    base = np.full((64, 96, 3), 128, dtype=np.int16)
    frames = []
    for _ in range(7):
        f = base.copy()
        f[16:48, 24:72] += rng.integers(-25, 25, size=(32, 48, 3))
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    mask = np.zeros((64, 96), dtype=np.float32)
    mask[16:48, 24:72] = 1.0

    win, prox = _win_u8(frames)
    _, before = detect(ArtifactType.FLICKER, win, sens=0.7)
    ctx = RepairContext(frame=frames[3], mask=mask, window=frames, center_idx=3)
    out, method = repair(ArtifactType.FLICKER, ctx)
    assert method == "temporal-smooth" and out.shape == frames[3].shape and out.dtype == np.uint8
    _, after = detect(ArtifactType.FLICKER, _rebuild(win, prox, 3, out), sens=0.7)
    assert after < before


def test_ghosting_repair_reduces_dual_diff():
    rng = np.random.default_rng(1)
    prev = (rng.integers(96, 128, size=(48, 64, 3))).astype(np.uint8)
    nxt = (rng.integers(96, 128, size=(48, 64, 3))).astype(np.uint8)
    center = prev.copy()
    center[12:36, 16:48] = 250  # a ghost blob far from both neighbors
    mask = np.zeros((48, 64), dtype=np.float32)
    mask[12:36, 16:48] = 1.0

    win, prox = _win_u8([prev, center, nxt], center_idx=1, is_interpolated=True)
    _, before = detect(ArtifactType.GHOSTING, win, sens=0.6)
    ctx = RepairContext(frame=center, mask=mask, window=[prev, center, nxt], center_idx=1)
    out, method = repair(ArtifactType.GHOSTING, ctx)
    assert method == "neighbor-blend" and out.dtype == np.uint8
    _, after = detect(ArtifactType.GHOSTING, _rebuild(win, prox, 1, out), sens=0.6)
    assert before > 0.1 and after < before


def test_banding_repair_breaks_plateaus():
    w, h = 96, 48
    grad = np.linspace(0.2, 0.8, w)[None, :, None] * np.ones((h, 1, 3))
    quant = (np.round(grad / 0.1) * 0.1).astype(np.uint8) // 1  # posterized
    frame = (quant * 255).astype(np.uint8)
    mask = np.zeros((h, w), dtype=np.float32)
    mask[:] = 1.0

    win, prox = _win_u8([frame] * 7)
    _, before = detect(ArtifactType.BANDING, win, sens=0.6)
    ctx = RepairContext(frame=frame, mask=mask, window=[frame] * 7, center_idx=3)
    out, method = repair(ArtifactType.BANDING, ctx)
    assert method == "deband-dither" and out.dtype == np.uint8
    _, after = detect(ArtifactType.BANDING, _rebuild(win, prox, 3, out), sens=0.6)
    assert before > 0.1 and after < before


def test_halo_repair_softens_flagged_region_only():
    rng = np.random.default_rng(2)
    frame = (rng.integers(0, 255, size=(48, 64, 3))).astype(np.uint8)
    mask = np.zeros((48, 64), dtype=np.float32)
    mask[10:30, 20:40] = 1.0
    ctx = RepairContext(frame=frame, mask=mask, window=[frame.copy() for _ in range(7)], center_idx=3)
    out, method = repair(ArtifactType.HALO, ctx)
    assert method == "edge-soften" and out.dtype == np.uint8
    # masked region changed; far corner untouched
    assert not np.array_equal(out[10:30, 20:40], frame[10:30, 20:40])
    assert np.array_equal(out[0:5, 0:5], frame[0:5, 0:5])


def test_lineart_repair_returns_valid_darkened_output():
    rng = np.random.default_rng(3)
    frame = (rng.integers(0, 255, size=(48, 64, 3))).astype(np.uint8)
    mask = np.zeros((48, 64), dtype=np.float32)
    mask[10:30, 20:40] = 1.0
    ctx = RepairContext(frame=frame, mask=mask, window=[frame.copy() for _ in range(7)], center_idx=3)
    out, method = repair(ArtifactType.LINEART, ctx)
    assert method == "fastlinedarken" and out.dtype == np.uint8 and out.shape == frame.shape
    # targeted: pixels outside the mask are untouched
    assert np.array_equal(out[0:5, 0:5], frame[0:5, 0:5])


def test_color_repair_matches_scene_palette():
    rng = np.random.default_rng(4)
    ref = (rng.integers(60, 200, size=(48, 64, 3))).astype(np.float32) / 255.0
    # a drifted frame: boosted red, lowered blue vs the reference
    cur = ref.copy()
    cur[..., 0] *= 1.3
    cur[..., 2] -= 0.2
    frame = np.clip(cur * 255, 0, 255).astype(np.uint8)
    mask = np.ones((48, 64), dtype=np.float32)
    ctx = RepairContext(frame=frame, mask=mask, window=[frame.copy() for _ in range(7)], center_idx=3,
                        scene_ref_proxy=ref, center_proxy=to_proxy(frame, 64, 48))
    out, method = repair(ArtifactType.COLOR_DRIFT, ctx)
    assert method == "color-match" and out.dtype == np.uint8
    # after correction the per-channel means should be closer to the reference's
    def _means(a):
        return a.astype(np.float32).mean(axis=(0, 1))
    d_before = np.abs(_means(frame) / 255.0 - ref.mean(axis=(0, 1))).sum()
    d_after = np.abs(_means(out) / 255.0 - ref.mean(axis=(0, 1))).sum()
    assert d_after < d_before


def test_color_repair_coefficients_stable_across_window():
    # Q4: every frame in one temporal window must receive the SAME correction
    # (window-median coefficients). A per-frame fit chases a legitimate lighting
    # ramp — consecutive flagged frames get different gains and pump. Here two
    # centers of the same 7-frame ramp+noise window are repaired; the implied
    # per-channel slope (out = a*in + c, strength 0.7) must be nearly identical.
    rng = np.random.default_rng(11)
    base = rng.integers(40, 160, size=(48, 64, 3)).astype(np.float32) / 255.0
    frames = []
    for t in range(7):
        f = base * (1.0 + 0.06 * t)
        f += rng.normal(0, 0.01, size=f.shape).astype(np.float32)
        frames.append(np.clip(f, 0.0, 1.0))
    ref_u8 = np.clip(frames[0] * 255, 0, 255).astype(np.uint8)  # scene reference
    win_u8 = [np.clip(fr * 255, 0, 255).astype(np.uint8) for fr in frames]

    def repair_at(idx: int) -> np.ndarray:
        ctx = RepairContext(
            frame=win_u8[idx], mask=np.ones((48, 64), dtype=np.float32),
            window=list(win_u8), center_idx=idx,
            scene_ref_proxy=to_proxy(ref_u8, 64, 48),
            center_proxy=to_proxy(win_u8[idx], 64, 48),
        )
        out, method = repair(ArtifactType.COLOR_DRIFT, ctx)
        assert method == "color-match"
        return out

    slopes = []
    for idx in (2, 4):
        out = repair_at(idx)
        a = np.array([
            float(np.polyfit(win_u8[idx][..., c].ravel().astype(float), out[..., c].ravel().astype(float), 1)[0])
            for c in range(3)
        ])
        slopes.append(a)
    # Same window -> same median coefficients. A per-frame (legacy) fit would
    # differ by ~0.07 slope between these two ramp positions.
    assert np.abs(slopes[0] - slopes[1]).max() < 0.01, f"coefficients pump: {slopes}"


def test_regression_repair_stabilizes_outlier_frame():
    rng = np.random.default_rng(5)
    base = (rng.integers(80, 180, size=(48, 64, 3))).astype(np.uint8)
    frames = [base.copy() for _ in range(7)]
    import cv2

    frames[3] = cv2.GaussianBlur(base, (15, 15), 0)  # the bad frame
    mask = np.ones((48, 64), dtype=np.float32)
    ctx = RepairContext(frame=frames[3], mask=mask, window=frames, center_idx=3)
    out, method = repair(ArtifactType.REGRESSION, ctx)
    assert method == "temporal-stabilize" and out.dtype == np.uint8
    # the repaired frame should be closer to its neighbors than the blurred one was
    d_before = np.abs(frames[3].astype(int) - base.astype(int)).mean()
    d_after = np.abs(out.astype(int) - base.astype(int)).mean()
    assert d_after < d_before


def test_repairs_noop_when_mask_empty():
    frame = (np.random.default_rng(6).integers(0, 255, size=(32, 48, 3))).astype(np.uint8)
    empty = np.zeros((32, 48), dtype=np.float32)
    ctx = RepairContext(frame=frame, mask=empty, window=[frame.copy() for _ in range(7)], center_idx=3)
    out, method = repair(ArtifactType.FLICKER, ctx)
    assert method == "none" and np.array_equal(out, frame)

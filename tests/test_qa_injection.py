"""M4 acceptance gate: artifact-injection harness (plan.md M4).

Take a clean processed clip, inject synthetic flicker / ghosting / banding into known
regions, run the *real* detectors on the proxy stream, and verify each catches **≥90%**
of the injected region with **<10% false positives**. This is the key quality gate for
the differentiator.
"""
from __future__ import annotations

import numpy as np
import pytest

from aive.stages.qa.detectors import detect, FrameWindow
from aive.stages.qa.findings import ArtifactType
from aive.stages.qa.inject import (
    inject_banding,
    inject_flicker,
    inject_ghosting,
    make_clean_window,
    measure,
)
from aive.stages.qa.proxy import proxy_dims, to_proxy

H, W = 480, 640


def _proxy_win(frames_u8, center_idx=3, **kw):
    pw, ph = proxy_dims(W, H, target_h=720)
    prox = [to_proxy(f, pw, ph) for f in frames_u8]
    win = FrameWindow(
        center=prox[center_idx],
        prev=prox[center_idx - 1] if center_idx > 0 else None,
        next=prox[center_idx + 1] if center_idx < len(prox) - 1 else None,
        window=prox,
        **kw,
    )
    return win, pw, ph


def _textured_frames(n=7, seed=0):
    """Static textured background (no long plateaus) for the banding test."""
    rng = np.random.default_rng(seed)
    base = np.full((H, W, 3), 128, dtype=np.int16)
    tex = rng.integers(-25, 26, size=(H, W, 3))
    img = np.clip(base + tex, 0, 255).astype(np.uint8)
    for _ in range(4):
        cy, cx = int(rng.integers(H // 4, 3 * H // 4)), int(rng.integers(W // 4, 3 * W // 4))
        col = tuple(int(c) for c in rng.integers(60, 210, size=3))
        import cv2

        cv2.circle(img, (cx, cy), int(rng.integers(20, 50)), col, -1)
    return [img.copy() for _ in range(n)]


def test_injected_flicker_caught():
    frames = make_clean_window(n_frames=7, h=H, w=W, seed=0)
    bbox = (220, 140, 160, 130)
    inj = inject_flicker(frames, bbox, amp=48.0, seed=0)
    win, pw, ph = _proxy_win(inj.frames)
    mask, sev = detect(ArtifactType.FLICKER, win, sens=0.6)
    recall, fp = measure(mask, bbox, W, H, pw, ph, thresh=0.3)
    assert sev > 0.2
    assert recall >= 0.9, f"flicker recall {recall:.2f} < 0.9"
    assert fp < 0.10, f"flicker FP {fp:.2f} >= 0.10"


def test_injected_ghosting_caught():
    frames = make_clean_window(n_frames=7, h=H, w=W, seed=1)
    bbox = (240, 150, 150, 130)
    inj = inject_ghosting(frames[3], frames[2], frames[4], bbox, seed=0)
    win, pw, ph = _proxy_win(inj.frames, center_idx=1, is_interpolated=True)
    mask, sev = detect(ArtifactType.GHOSTING, win, sens=0.6)
    recall, fp = measure(mask, bbox, W, H, pw, ph, thresh=0.3)
    assert sev > 0.2
    assert recall >= 0.9, f"ghosting recall {recall:.2f} < 0.9"
    assert fp < 0.10, f"ghosting FP {fp:.2f} >= 0.10"


def test_injected_banding_caught():
    frames = _textured_frames(n=7, seed=0)
    bbox = (200, 150, 200, 140)
    banded = inject_banding(frames[3], bbox, levels=6).frames[0]
    win, pw, ph = _proxy_win([frames[i].copy() if i != 3 else banded for i in range(7)])
    mask, sev = detect(ArtifactType.BANDING, win, sens=0.6)
    recall, fp = measure(mask, bbox, W, H, pw, ph, thresh=0.3)
    # Severity is frame-wide while the injection covers only ~9% of the frame,
    # so scale the sanity floor by the injected area (recall/FP are the real gate).
    area_frac = (bbox[2] * bbox[3]) / (W * H)
    assert sev > 0.4 * area_frac, f"banding severity {sev:.3f} too low for a {area_frac:.1%} injection"
    assert recall >= 0.9, f"banding recall {recall:.2f} < 0.9"
    assert fp < 0.10, f"banding FP {fp:.2f} >= 0.10"


def test_clean_clip_has_low_false_positives():
    """A clean (un-injected) static clip should produce almost no findings."""
    frames = make_clean_window(n_frames=7, h=H, w=W, seed=2)
    win, pw, ph = _proxy_win(frames)
    for art in (ArtifactType.FLICKER, ArtifactType.GHOSTING, ArtifactType.BANDING):
        mask, sev = detect(art, win, sens=0.6)
        _, fp = measure(mask, (0, 0, W, H), W, H, pw, ph, thresh=0.35)
        assert fp < 0.10, f"clean {art.value} FP {fp:.2f} >= 0.10"

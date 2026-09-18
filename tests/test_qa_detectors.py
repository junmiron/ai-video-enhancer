"""Unit tests for each M4 QA detector on synthetic tensors (no GPU / no pipeline).

Each test crafts a scenario that should trigger its detector (high severity) and a
clean control that should not, asserting the relative ordering plus a sanity floor.
This mirrors the style of :mod:`test_qa_foundation` and keeps detectors honest as
their thresholds are tuned.
"""
from __future__ import annotations

import numpy as np
import pytest

from aive.stages.qa.detectors import detect, FrameWindow
from aive.stages.qa.findings import ArtifactType


def _win(frames, center_idx=3, **kw):
    c = frames[center_idx]
    return FrameWindow(
        center=c,
        prev=frames[center_idx - 1] if center_idx > 0 else None,
        next=frames[center_idx + 1] if center_idx < len(frames) - 1 else None,
        window=list(frames),
        **kw,
    )


def _u8(a):
    return np.clip(np.round(a * 255.0), 0, 255).astype(np.uint8)


# -- flicker ------------------------------------------------------------------


def test_flicker_flags_temporal_noise_in_static_region():
    rng = np.random.default_rng(1)
    base = np.full((96, 128, 3), 0.5, dtype=np.float32)
    frames = []
    for _ in range(7):
        f = base.copy()
        # independent per-frame noise only inside a box -> high temporal sigma there
        f[24:64, 32:96] += rng.normal(0, 0.15, size=(40, 64, 3))
        frames.append(np.clip(f, 0, 1).astype(np.float32))
    mask, sev = detect(ArtifactType.FLICKER, _win(frames), sens=0.7)
    assert sev > 0.3
    # the noisy box should be flagged more than a clean corner
    assert mask[24:64, 32:96].mean() > mask[0:16, 0:16].mean()


def test_flicker_quiet_for_identical_frames():
    base = np.full((96, 128, 3), 0.5, dtype=np.float32)
    frames = [base.copy() for _ in range(7)]
    _, sev = detect(ArtifactType.FLICKER, _win(frames), sens=0.7)
    assert sev < 0.1


# -- ghosting -----------------------------------------------------------------


def test_ghosting_flags_content_far_from_both_neighbors():
    rng = np.random.default_rng(2)
    prev = (rng.random((96, 128, 3)) * 0.2 + 0.4).astype(np.float32)
    nxt = (rng.random((96, 128, 3)) * 0.2 + 0.4).astype(np.float32)
    center = prev.copy()
    # a bright blob in the center that matches neither neighbor
    center[30:70, 40:90] = 1.0
    win = _win([prev, center, nxt], center_idx=1, is_interpolated=True)
    mask, sev = detect(ArtifactType.GHOSTING, win, sens=0.6)
    assert sev > 0.2
    assert mask[30:70, 40:90].mean() > mask.mean()


def test_ghosting_ignores_non_interpolated_frames():
    rng = np.random.default_rng(3)
    prev = (rng.random((96, 128, 3)) * 0.2 + 0.4).astype(np.float32)
    nxt = (rng.random((96, 128, 3)) * 0.2 + 0.4).astype(np.float32)
    center = prev.copy()
    center[30:70, 40:90] = 1.0
    win = _win([prev, center, nxt], center_idx=1, is_interpolated=False)
    mask, sev = detect(ArtifactType.GHOSTING, win, sens=0.6)
    assert sev == 0.0 and not mask.any()


# -- banding ------------------------------------------------------------------


def test_banding_flags_posterized_gradient():
    w, h = 128, 96
    grad = np.linspace(0.2, 0.8, w)[None, :, None] * np.ones((h, 1, 3))
    # posterize to 5 levels -> long plateaus
    quant = (np.round(grad / 0.1) * 0.1).astype(np.float32)
    _, sev_banded = detect(ArtifactType.BANDING, _win([quant] * 7), sens=0.6)

    # a smooth diagonal gradient changes by >=1 level per pixel in BOTH axes -> no plateaus
    xx = np.linspace(0.0, 1.0, w)[None, :, None] * np.ones((1, 1, 3))
    yy = np.linspace(0.0, 1.0, h)[:, None, None] * np.ones((1, 1, 3))
    fine = np.clip(0.2 + 0.6 * xx + 0.6 * yy, 0, 1).astype(np.float32)
    _, sev_fine = detect(ArtifactType.BANDING, _win([fine] * 7), sens=0.6)

    assert sev_banded > sev_fine + 0.1


# -- halo ---------------------------------------------------------------------


def test_halo_flags_overshoot_vs_source():
    h, w = 96, 128
    src = np.full((h, w, 3), 0.5, dtype=np.float32)
    # a strong vertical edge in the source
    src[:, 60:70] = 0.1
    proc = src.copy()
    # ringing overshoot lobes just outside the edge
    proc[:, 70:74] = 0.95
    proc[:, 56:60] = 0.05
    mask, sev = detect(ArtifactType.HALO, _win([proc] * 7, src_center=src), sens=0.6)
    assert sev > 0.2


def test_halo_quiet_when_matching_source():
    h, w = 96, 128
    rng = np.random.default_rng(4)
    src = (rng.random((h, w, 3)) * 0.3 + 0.3).astype(np.float32)
    _, sev = detect(ArtifactType.HALO, _win([src] * 7, src_center=src), sens=0.6)
    assert sev < 0.15


# -- lineart ------------------------------------------------------------------


def test_lineart_flags_blurred_lines_vs_crisp_source():
    h, w = 96, 128
    import cv2

    src = np.full((h, w, 3), 0.7, dtype=np.float32)
    for x in range(10, w - 10, 12):
        src[:, x : x + 2] = 0.05  # crisp thin vertical lines
    proc = cv2.GaussianBlur(_u8(src), (5, 5), 1.0).astype(np.float32) / 255.0  # blurred/thickened
    _, sev_drift = detect(ArtifactType.LINEART, _win([proc] * 7, src_center=src), sens=0.6)

    proc_same = src.copy()
    _, sev_same = detect(ArtifactType.LINEART, _win([proc_same] * 7, src_center=src), sens=0.6)

    assert sev_drift > sev_same + 0.05


# -- color drift --------------------------------------------------------------


def test_color_drift_flags_palette_shift_vs_scene_ref():
    rng = np.random.default_rng(5)
    ref = (rng.random((96, 128, 3)) * 0.4 + 0.3).astype(np.float32)
    shifted = ref.copy()
    shifted[..., 0] *= 1.3
    shifted[..., 2] -= 0.25
    win = _win([shifted] * 7, scene_ref=ref)
    mask, sev = detect(ArtifactType.COLOR_DRIFT, win, sens=0.6)
    assert sev > 0.2


def test_color_drift_quiet_when_matching_scene_ref():
    rng = np.random.default_rng(6)
    ref = (rng.random((96, 128, 3)) * 0.4 + 0.3).astype(np.float32)
    win = _win([ref] * 7, scene_ref=ref)
    _, sev = detect(ArtifactType.COLOR_DRIFT, win, sens=0.6)
    assert sev < 0.1


# -- regression ---------------------------------------------------------------


def test_regression_flags_iqa_outlier_frame():
    import cv2

    rng = np.random.default_rng(7)
    base = (rng.random((96, 128, 3)) * 0.5 + 0.3).astype(np.float32)
    frames = [base.copy() for _ in range(7)]
    # make the center frame a heavy-blur outlier vs its neighbors
    frames[3] = cv2.GaussianBlur(_u8(base), (15, 15), 0).astype(np.float32) / 255.0

    from aive.stages.qa.detectors.regression import iqa_feature

    recent = [iqa_feature(frames[i]) for i in range(7) if i != 3]
    win = _win(frames, center_idx=3, recent_iqa=recent[:6])
    mask, sev = detect(ArtifactType.REGRESSION, win, sens=0.5)
    assert sev > 0.2


def test_regression_quiet_for_consistent_frames():
    rng = np.random.default_rng(8)
    base = (rng.random((96, 128, 3)) * 0.5 + 0.3).astype(np.float32)
    frames = [base.copy() for _ in range(7)]
    from aive.stages.qa.detectors.regression import iqa_feature

    recent = [iqa_feature(frames[i]) for i in range(6)]
    win = _win(frames, center_idx=3, recent_iqa=recent)
    _, sev = detect(ArtifactType.REGRESSION, win, sens=0.5)
    assert sev < 0.1


def test_regression_needs_enough_history():
    base = np.full((48, 64, 3), 0.5, dtype=np.float32)
    _, sev = detect(ArtifactType.REGRESSION, _win([base] * 7, recent_iqa=[1.0]), sens=0.9)
    assert sev == 0.0

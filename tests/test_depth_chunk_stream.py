"""Unit tests for the depth streaming window (chunk_stream) — pure numpy, no GPU.

Covers the exact-frame-count contract that makes the pass safe to run per scene
segment: every input frame yields exactly one depth map, tail chunks use their
real length, and overlap alignment rescales + blends only repeated views.
"""
from __future__ import annotations

import numpy as np
import pytest

from aive.stages.depth.chunk_stream import align_depth_overlap, stream_depth_chunks


def _gradient_chunk(n: int) -> list[np.ndarray]:
    """N distinct [4,8] depth maps (frame index baked into the values)."""
    return [(np.full((4, 8), i + 1.0, dtype=np.float32)) for i in range(n)]


def test_yields_exactly_one_depth_per_frame_full_windows():
    frames = _gradient_chunk(32)
    out = list(stream_depth_chunks(iter(frames), lambda w: np.stack(w), chunk_size=8))
    assert len(out) == 32
    for i, d in enumerate(out):
        # No blending happened (each frame predicted once per its own scale).
        assert np.allclose(d, i + 1.0), f"frame {i} corrupted: {d[0, 0]}"


def test_tail_chunk_uses_real_length():
    """A short tail must not be padded with duplicates (that would change attention)."""
    frames = _gradient_chunk(30)
    seen_windows: list[np.ndarray] = []

    def infer(window):
        seen_windows.append(np.stack(window))
        return np.stack(window)

    out = list(stream_depth_chunks(iter(frames), infer, chunk_size=8))
    assert len(out) == 30
    # Sliding window (50% overlap): six full windows of 8, then the tail at its
    # real length. The last window must be exactly frames 24..29 — no duplicate
    # padding to reach chunk_size.
    assert [len(w) for w in seen_windows] == [8] * 6 + [6]
    assert np.allclose(seen_windows[-1], np.arange(25.0, 31.0)[:, None, None])


def test_short_segment_single_chunk():
    frames = _gradient_chunk(5)
    out = list(stream_depth_chunks(iter(frames), lambda w: np.stack(w), chunk_size=8))
    assert len(out) == 5
    for i, d in enumerate(out):
        assert np.allclose(d, i + 1.0)


def test_empty_input_yields_nothing():
    assert list(stream_depth_chunks(iter([]), lambda w: np.stack(w), chunk_size=8)) == []


def test_single_frame():
    out = list(stream_depth_chunks(iter(_gradient_chunk(1)), lambda w: np.stack(w), chunk_size=4))
    assert len(out) == 1 and np.allclose(out[0], 1.0)


def test_invalid_window_params_raise():
    with pytest.raises(ValueError):
        list(stream_depth_chunks(iter([]), lambda w: w, chunk_size=1))
    with pytest.raises(ValueError):
        list(stream_depth_chunks(iter([]), lambda w: w, chunk_size=8, overlap=4 + 1))


def test_chunk_must_return_one_map_per_frame():
    frames = _gradient_chunk(8)
    with pytest.raises(ValueError, match="one \\[H,W\\] depth map per frame"):
        list(stream_depth_chunks(iter(frames), lambda w: np.stack(w)[:3], chunk_size=8))


def test_non_finite_and_nonpositive_are_zeroed():
    frames = _gradient_chunk(4)

    def infer(window):
        out = np.stack(window)
        out[0, 0, 0] = float("nan")
        out[1, 0, 0] = -2.0
        return out

    out = list(stream_depth_chunks(iter(frames), infer, chunk_size=4))
    assert len(out) == 4
    assert out[0][0, 0] == 0.0 and out[1][0, 0] == 0.0


def test_alignment_rescales_by_median_ratio():
    """The new chunk is rescaled so its repeated views match the old ones (median ratio)."""
    previous = np.stack([np.full((4, 8), 2.0), np.full((4, 8), 4.0)])  # last 2 of prior chunk
    current = np.stack([np.full((4, 8), 1.0), np.full((4, 8), 2.0), np.full((4, 8), 3.0)])
    aligned = align_depth_overlap(previous.copy(), current)
    # Median ratio old/new = median(2/1, 4/2) = 2 -> whole chunk scaled by 2.
    # Exact blend: weight for overlap i is i/(overlap+1); i=1 -> 1/3 new, i=2 -> 2/3 new.
    # (align_depth_overlap rescales/blends in place; expected values use the
    # original pre-scale literals, not the mutated `current`.)
    assert np.allclose(aligned[0], previous[0] * (2 / 3) + 1.0 * 2.0 * (1 / 3))
    assert np.allclose(aligned[1], previous[1] * (1 / 3) + 2.0 * 2.0 * (2 / 3))
    # Non-overlapped views keep the rescaled value.
    assert np.allclose(aligned[2], 6.0)


def test_alignment_no_shared_scale_returns_current_unchanged():
    previous = np.zeros((2, 4, 8), dtype=np.float32)  # no positive overlap pixels
    current = np.stack([np.full((4, 8), 1.0), np.full((4, 8), 2.0), np.full((4, 8), 3.0)])
    aligned = align_depth_overlap(previous, current.copy())
    assert np.allclose(aligned[2], 3.0)


def test_alignment_shape_mismatch_raises():
    with pytest.raises(ValueError):
        align_depth_overlap(np.zeros((2, 4, 8)), np.zeros((2, 5, 8)))


def test_streaming_blend_is_seam_free_on_constant_scene():
    """Constant scene: every chunk predicts the same map; output must be flat."""
    n = 20

    def infer(window):
        return np.stack([np.full((4, 8), 5.0) for _ in window])

    out = list(stream_depth_chunks(iter(_gradient_chunk(n)), infer, chunk_size=8))
    assert len(out) == n
    assert all(np.allclose(d, 5.0) for d in out)


def test_windows_never_cross_segment_boundary():
    """The pass feeds one scene segment at a time; verify the consumer contract:
    two consecutive segments of 30 frames each produce independent chunk plans."""
    plan = []
    for seg in (30, 30):
        seen = []

        def infer(window, _seen=seen):
            _seen.append(len(window))
            return np.stack([np.full((4, 8), float(i + 1)) for i in range(len(window))])

        out = list(stream_depth_chunks(iter(_gradient_chunk(seg)), infer, chunk_size=8))
        plan.append(seen)
        assert len(out) == seg
    # Both segments: six full windows (8) then a real-length tail (6).
    assert plan[0] == [8] * 6 + [6]
    assert plan[1] == [8] * 6 + [6]

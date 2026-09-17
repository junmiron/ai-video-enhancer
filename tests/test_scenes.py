"""Scene detection: hard cuts found, static content yields one segment."""
from __future__ import annotations

from aive.stages.ingest import probe
from aive.stages.scenes import detect_scenes


def test_detects_hard_cut(cut_clip):
    info = probe(cut_clip)
    segments = detect_scenes(cut_clip, threshold=27.0, min_scene_len_s=0.5)
    assert len(segments) >= 2, f"expected a cut, got {segments}"
    # Segments must tile the whole video without gaps/overlaps.
    assert segments[0][0] == 0
    assert segments[-1][1] == info.num_frames
    for (s1, e1), (s2, e2) in zip(segments, segments[1:]):
        assert e1 == s2


def test_static_single_segment(static_clip):
    info = probe(static_clip)
    segments = detect_scenes(static_clip, threshold=27.0, min_scene_len_s=0.5)
    assert len(segments) == 1
    assert segments[0] == (0, info.num_frames)

"""grab_frame (M6 preview fast-seek decode) vs sequential decode.

The preview compare view relies on grab_frame returning the exact frame that a
sequential decode would produce at the same absolute index — verified here on a
generated CFR clip.
"""
from __future__ import annotations

import numpy as np

from aive.stages.ingest import decode_frames, grab_frame, probe


def test_grab_matches_sequential_decode(test_clip):
    info = probe(test_clip)
    idx = 20
    seq = next(decode_frames(test_clip, start_frame=idx, count=1))
    got = grab_frame(test_clip, idx)
    assert got is not None
    assert got.shape == (info.height, info.width, 3)
    diff = int(np.abs(got.astype(int) - seq.astype(int)).max())
    assert diff <= 2, f"grabbed frame differs from sequential decode by {diff}"


def test_grab_first_and_last_frames(test_clip):
    info = probe(test_clip)
    first = grab_frame(test_clip, 0)
    last = grab_frame(test_clip, info.num_frames - 1)
    assert first is not None and last is not None
    seq_first = next(decode_frames(test_clip, start_frame=0, count=1))
    assert int(np.abs(first.astype(int) - seq_first.astype(int)).max()) <= 2


def test_grab_out_of_range_returns_none(test_clip):
    info = probe(test_clip)
    assert grab_frame(test_clip, -1) is None
    assert grab_frame(test_clip, info.num_frames + 5) is None

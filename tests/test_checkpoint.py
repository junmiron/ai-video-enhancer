"""Checkpoint store: resume semantics and concat list generation."""
from __future__ import annotations

import json

from aive.core.checkpoint import CheckpointStore, Segment


def _segs():
    return [Segment(0, 0, 30), Segment(1, 30, 60), Segment(2, 60, 90)]


def test_init_and_resume(tmp_path):
    store = CheckpointStore(tmp_path / "ws")
    store.init_job("job1", "/in.mp4", 640, 360, "30/1", _segs())
    assert store.total_segments == 3
    assert store.pending_indices() == [0, 1, 2]

    # Simulate segment 0 completing (file must exist for is_done).
    f = store.segment_file(0)
    f.write_bytes(b"fake")
    store.mark_done(0)

    # Re-open from disk: done state survives.
    store2 = CheckpointStore(tmp_path / "ws")
    assert store2.is_done(0) and not store2.is_done(1)
    assert store2.pending_indices() == [1, 2]
    assert store2.done_count == 1

    # Re-init with same segments preserves done records.
    store3 = CheckpointStore(tmp_path / "ws")
    store3.init_job("job1", "/in.mp4", 640, 360, "30/1", _segs())
    assert store3.is_done(0)
    assert store3.pending_indices() == [1, 2]


def test_missing_file_not_done(tmp_path):
    store = CheckpointStore(tmp_path / "ws")
    store.init_job("job1", "/in.mp4", 640, 360, "30/1", _segs())
    # Mark done without the file existing -> must NOT count as done.
    store.mark_done(2)
    assert not store.is_done(2)


def test_concat_list_order(tmp_path):
    store = CheckpointStore(tmp_path / "ws")
    store.init_job("job1", "/in.mp4", 640, 360, "30/1", _segs())
    for i in range(3):
        store.segment_file(i).write_bytes(b"x")
    lst = store.concat_list_file()
    lines = [l for l in lst.read_text().splitlines() if l.strip()]
    assert len(lines) == 3
    # In-order segment files, ffmpeg-escaped paths.
    assert "seg_0000.mkv" in lines[0] and "seg_0001.mkv" in lines[1] and "seg_0002.mkv" in lines[2]


def test_manifest_json_shape(tmp_path):
    store = CheckpointStore(tmp_path / "ws")
    store.init_job("jobX", "/in.mp4", 640, 360, "24000/1001", _segs())
    data = json.loads((tmp_path / "ws" / "manifest.json").read_text())
    assert data["job_id"] == "jobX"
    assert data["fps"] == "24000/1001"
    assert len(data["segments"]) == 3

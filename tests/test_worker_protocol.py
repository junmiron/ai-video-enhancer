"""Headless tests for the M6 worker protocol (aive.ui.worker).

``worker_main`` runs in a thread of this process with plain ``queue.Queue``
objects — the wire format is identical to the real mp.Queue path used by
WorkerClient, so these validate command dispatch and event sequencing without
spawning OS processes.
"""
from __future__ import annotations

import os
import queue
import subprocess
import threading
import time

import pytest

from aive.ui.worker import worker_main


def _start_worker():
    cmd_q: queue.Queue = queue.Queue()
    evt_q: queue.Queue = queue.Queue()
    t = threading.Thread(target=worker_main, args=(cmd_q, evt_q), daemon=True)
    t.start()
    return cmd_q, evt_q, t


def _wait_for(evt_q: queue.Queue, types: set[str], timeout: float = 180.0):
    """Drain events until one of `types` arrives; return (all_events, matched)."""
    deadline = time.time() + timeout
    seen: list[dict] = []
    while time.time() < deadline:
        try:
            d = evt_q.get(timeout=0.5)
        except queue.Empty:
            continue
        seen.append(d)
        if d.get("type") in types:
            return seen, d
    return seen, None


# Passthrough pipeline (no models needed); QA detect-only exercises the report
# path without any weights or GPU inference.
PASSTHROUGH_CFG = {
    "scene_split": True,
    "output_codec": "h264",
    "crf": 23,
    "container": "mp4",
    "audio_passthrough": True,
    "keep_workspace": False,
    "fi_enabled": False,
    "sr_enabled": False,
    "qa_enabled": True,
    "qa_mode": "detect_only",
    "qa_ai_repairs": False,
    "depth_enabled": False,
}


@pytest.fixture()
def worker():
    cmd_q, evt_q, t = _start_worker()
    # The ready event carries gpu_info (imports torch lazily — allow time).
    _events, ready = _wait_for(evt_q, {"ready"}, timeout=180)
    assert ready is not None and "gpu" in ready
    yield cmd_q, evt_q
    cmd_q.put({"cmd": "shutdown"})
    t.join(timeout=30)


def test_run_passthrough_job(worker, test_clip, tmp_path):
    cmd_q, evt_q = worker
    out = str(tmp_path / "out.mp4")
    cfg = dict(PASSTHROUGH_CFG)
    cmd_q.put({"cmd": "run", "job_id": "t1", "input_path": test_clip, "output_path": out, "cfg": cfg})

    events, fin = _wait_for(evt_q, {"job_finished", "cancelled", "error"}, timeout=600)
    assert fin is not None and fin["type"] == "job_finished" and fin["ok"], [e.get("type") for e in events]

    types = [e["type"] for e in events]
    stages_started = [e["stage"] for e in events if e["type"] == "stage_started"]
    assert "job_started" in types
    assert "process" in stages_started and "encode" in stages_started
    # Per-stage progress (M6 orchestrator wrapper) must appear for the chain.
    stage_progress = [e["stage"] for e in events if e["type"] == "stage_progress"]
    assert "identity" in stage_progress or "qa" in stage_progress

    assert os.path.exists(out) and os.path.getsize(out) > 0
    # QA ran -> report dict + exported files delivered on JobFinished.
    assert fin.get("report") is not None
    rf = fin.get("report_files") or {}
    assert rf.get("json") and os.path.exists(rf["json"])


def test_cancel_mid_run(worker, tmp_path):
    cmd_q, evt_q = worker
    # 12 s clip + slow x265 preset: the encode outlives our cancel round-trip.
    clip = str(tmp_path / "long.mp4")
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=12",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", clip,
        ],
        check=True,
    )
    out = str(tmp_path / "out.mp4")
    cfg = dict(PASSTHROUGH_CFG)
    cfg["output_codec"] = "x265-10bit-anime"
    cmd_q.put({"cmd": "run", "job_id": "t2", "input_path": clip, "output_path": out, "cfg": cfg})

    # Wait until the job is demonstrably mid-run (first per-segment progress).
    _events, prog = _wait_for(evt_q, {"stage_progress"}, timeout=600)
    assert prog is not None and prog["stage"] == "process"
    cmd_q.put({"cmd": "cancel"})

    _events2, fin = _wait_for(evt_q, {"cancelled", "job_finished", "error"}, timeout=300)
    assert fin is not None and fin["type"] == "cancelled", [e.get("type") for e in _events2]


def test_list_models(worker):
    cmd_q, evt_q = worker
    cmd_q.put({"cmd": "list_models"})
    _events, ev = _wait_for(evt_q, {"models_listed"}, timeout=120)
    assert ev is not None and ev.get("task") is None
    names = {m["name"] for m in ev["models"]}
    assert {"shufflecugan_x2", "rife_425"} <= names
    assert all(isinstance(m.get("cached"), bool) for m in ev["models"])


def test_download_cached_model(worker):
    from aive.models.downloader import cached_weights_path
    from aive.models.registry import get_model

    info = get_model("shufflecugan_x2")
    if cached_weights_path(info) is None:
        pytest.skip("shufflecugan_x2 weights are not cached on this machine")
    cmd_q, evt_q = worker
    cmd_q.put({"cmd": "download", "names": ["shufflecugan_x2"]})
    _events, ev = _wait_for(evt_q, {"download_done"}, timeout=180)
    assert ev is not None and ev["name"] == "shufflecugan_x2" and ev["ok"]


def test_gpu_info_command(worker):
    cmd_q, evt_q = worker
    cmd_q.put({"cmd": "gpu_info"})
    _events, ev = _wait_for(evt_q, {"ready"}, timeout=180)
    assert ev is not None
    gpu = ev["gpu"]
    assert isinstance(gpu.get("cuda_ok"), bool)
    assert isinstance(gpu.get("vram_total_gb"), (int, float))


def test_friendly_error_hints():
    """M7: low-level failures get actionable hints; the raw message is kept."""
    from aive.ui.worker import friendly_error

    oom = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    msg = friendly_error(oom)
    assert "out of memory" in msg.lower() and "Not enough GPU memory" in msg

    disk_os = OSError("[Errno 28] No space left on device: 'C:\\ws\\seg.mkv'")
    assert "disk appears to be full" in friendly_error(disk_os)

    # ffmpeg surfaces the same condition inside a RuntimeError with stderr text.
    disk_ff = RuntimeError("ffmpeg encode failed (rc=1): convert: No space left on device")
    assert "disk appears to be full" in friendly_error(disk_ff)

    nogpu = RuntimeError("CUDA device not available")
    assert "nvidia-smi" in friendly_error(nogpu)

    plain = ValueError("some other failure")
    assert friendly_error(plain) == "some other failure"

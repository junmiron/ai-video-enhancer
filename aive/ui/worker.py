"""Worker process for the M6 UI (design.md §6).

Runs the pipeline orchestrator in a separate process so GPU work never blocks
the UI event loop. Communication is two ``multiprocessing.Queue``s of plain
dicts — no Qt, no pickled objects:

  cmd_q (UI -> worker):
      {"cmd": "run",        job_id, input_path, output_path, cfg(dict)}
      {"cmd": "pause" | "resume" | "cancel"}          # applied to the running job
      {"cmd": "list_models", task?: str}
      {"cmd": "download",   names: [str]}
      {"cmd": "make_sample", dest: str}
      {"cmd": "gpu_info"}
      {"cmd": "shutdown"}

  evt_q (worker -> UI):
      pipeline events via ``Event.to_dict()`` (job_started, stage_started,
      stage_progress, stage_finished, warning, error, job_finished) plus
      worker-specific dicts: ready / log / metrics / models_listed /
      model_download_progress / download_done / sample_ready / sample_error /
      control_ack.

While a job runs, a pump thread owns ``cmd_q`` so pause/resume/cancel reach the
job's :class:`~aive.core.jobqueue.ControlToken` immediately; non-control
commands are buffered and processed after the job ends. A shutdown request
cancels the running job first, then exits — closing the app mid-run is clean.

This module must stay Qt-free (the worker process never touches Qt).
"""
from __future__ import annotations

import ctypes
import dataclasses
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)


def _emit(evt_q, d: dict) -> None:
    try:
        evt_q.put(d)
    except Exception:  # pragma: no cover - queue closed during shutdown
        pass


def friendly_error(exc: BaseException) -> str:
    """Wrap low-level failures in actionable messages for the UI (M7).

    The raw message is always kept (truncated) so nothing is hidden; the hint
    only adds what the user can actually do about it.
    """
    msg = str(exc) or exc.__class__.__name__
    low = msg.lower()
    if "out of memory" in low or "cublas_status_alloc_failed" in low:
        return (
            f"{msg[:400]}\n\nNot enough GPU memory. Close other GPU applications, pick a smaller model/scale, "
            "or lower the QA proxy height — then retry (finished segments are kept for resume)."
        )
    if isinstance(exc, OSError) or "no space left on device" in low:
        return (
            f"{msg[:400]}\n\nThe disk appears to be full. Free up space on the drive holding the video and its "
            "workspace, then retry — finished segments are kept for resume."
        )
    if "cuda device not available" in low or "no cuda-capable" in low:
        return (
            f"{msg[:400]}\n\nNo usable NVIDIA GPU was found. AI stages require CUDA — check the driver with "
            "nvidia-smi and restart the app."
        )
    return msg


# ---------------------------------------------------------------------------
# System info (banner + meters)
# ---------------------------------------------------------------------------

def gpu_info() -> dict:
    """GPU/CUDA/TensorRT status for the first-run banner. Never raises."""
    info = {
        "cuda_ok": False,
        "gpu_name": "",
        "vram_total_gb": 0.0,
        "vram_free_gb": 0.0,
        "trt_version": None,
    }
    try:
        import torch

        if torch.cuda.is_available():
            info["cuda_ok"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            _free, total = torch.cuda.mem_get_info()
            info["vram_total_gb"] = round(total / 2**30, 1)
    except Exception:
        pass
    try:
        from ..inference.trt import free_vram_bytes

        info["vram_free_gb"] = round(free_vram_bytes() / 2**30, 1)
    except Exception:
        pass
    try:
        import tensorrt as trt

        info["trt_version"] = getattr(trt, "__version__", None) or str(getattr(trt, "BUILD_VERSION", ""))
    except Exception:
        pass
    return info


def _ram_used_gb() -> float | None:
    """Used system RAM in GiB (Windows via GlobalMemoryStatusEx, else /proc)."""
    try:
        if sys.platform == "win32":

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return (stat.ullTotalPhys - stat.ullAvailPhys) / 2**30
        else:
            info: dict[str, int] = {}
            with open("/proc/meminfo", encoding="ascii") as f:
                for line in f:
                    key, _, val = line.partition(":")
                    parts = val.split()
                    if parts:
                        info[key.strip()] = int(parts[0]) * 1024
            return (info["MemTotal"] - info.get("MemAvailable", 0)) / 2**30
    except Exception:
        pass
    return None


def system_metrics() -> dict:
    """VRAM/RAM snapshot for the processing-view meters. Never raises."""
    m: dict = {}
    try:
        import torch

        if torch.cuda.is_available():
            _free, total = torch.cuda.mem_get_info()
            m["vram_total_gb"] = round(total / 2**30, 1)
            from ..inference.trt import free_vram_bytes

            m["vram_free_gb"] = round(free_vram_bytes() / 2**30, 1)
    except Exception:
        pass
    ram = _ram_used_gb()
    if ram is not None:
        m["ram_used_gb"] = round(ram, 1)
    return m


# ---------------------------------------------------------------------------
# Log forwarding (collapsible live log in the processing view)
# ---------------------------------------------------------------------------

class _QueueLogHandler(logging.Handler):
    def __init__(self, evt_q):
        super().__init__(level=logging.INFO)
        self._q = evt_q
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(name)s: %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            if len(msg) > 2000:
                msg = msg[:2000] + "…"
            _emit(self._q, {"type": "log", "level": record.levelname, "message": msg})
        except Exception:  # pragma: no cover - never break logging
            pass


def _install_log_handler(evt_q) -> None:
    handler = _QueueLogHandler(evt_q)
    root = logging.getLogger()
    if not any(isinstance(h, _QueueLogHandler) for h in root.handlers):
        root.addHandler(handler)
    if root.level > logging.INFO or root.level == 0:
        root.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Model manager commands
# ---------------------------------------------------------------------------

def _model_rows(task: str | None = None) -> list[dict]:
    from ..models.downloader import cached_weights_path
    from ..models.registry import list_models

    rows = []
    for m in list_models(task=task):
        p = cached_weights_path(m)
        rows.append(
            {
                "name": m.name,
                "task": m.task,
                "version": m.version,
                "license": m.license,
                "experimental": bool(m.experimental),
                "backends": dict(m.backends or {}),
                "min_vram_gb": float(m.min_vram_gb or 0.0),
                "recommended_for": m.recommended_for,
                "notes": m.notes,
                "cached": p is not None,
            }
        )
    return rows


def _download_one(name: str, evt_q) -> bool:
    from ..core.events import ModelDownloadProgress
    from ..models.downloader import download_model
    from ..models.registry import get_model

    try:
        info = get_model(name)

        def cb(done: int, total: int) -> None:
            _emit(evt_q, ModelDownloadProgress(model=name, bytes_done=int(done), bytes_total=int(total)).to_dict())

        path = download_model(info, progress_cb=cb)
        _emit(evt_q, {"type": "download_done", "name": name, "ok": True, "path": str(path)})
        return True
    except Exception as exc:
        log.exception("Model download failed: %s", name)
        _emit(evt_q, {"type": "download_done", "name": name, "ok": False, "error": str(exc)})
        return False


def _make_sample(cmd: dict, evt_q) -> None:
    dest = cmd.get("dest") or ""
    if not dest:
        _emit(evt_q, {"type": "sample_error", "error": "no destination path given"})
        return
    script = Path(__file__).resolve().parent.parent.parent / "tools" / "make_sample_clip.py"
    try:
        proc = subprocess.run(
            [sys.executable, str(script), dest], capture_output=True, text=True, timeout=900
        )
        if proc.returncode != 0 or not os.path.exists(dest):
            raise RuntimeError((proc.stderr or proc.stdout or "sample generation failed")[:500])
        _emit(evt_q, {"type": "sample_ready", "path": dest})
    except Exception as exc:
        log.exception("Sample clip generation failed")
        _emit(evt_q, {"type": "sample_error", "error": str(exc)})


# ---------------------------------------------------------------------------
# Worker main loop
# ---------------------------------------------------------------------------

def worker_main(cmd_q, evt_q) -> None:  # noqa: C901 - command dispatch is inherently branchy
    """Entry point for the worker process. Blocks until shutdown."""
    _install_log_handler(evt_q)
    stop = threading.Event()
    pending: list[dict] = []  # non-control commands that arrived mid-job
    token_box: dict = {}

    def _pump() -> None:
        """Own cmd_q while a job runs; apply control, buffer everything else."""
        while not stop.is_set():
            try:
                c = cmd_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if c is None or c.get("cmd") == "shutdown":
                tok = token_box.get("token")
                if tok is not None:
                    tok.request_cancel()  # finish the job, then exit cleanly
                stop.set()
                return
            kind = c.get("cmd")
            tok = token_box.get("token")
            if kind == "pause" and tok is not None:
                tok.request_pause()
                _emit(evt_q, {"type": "control_ack", "state": "paused"})
            elif kind == "resume" and tok is not None:
                tok.resume()
                _emit(evt_q, {"type": "control_ack", "state": "running"})
            elif kind == "cancel" and tok is not None:
                tok.request_cancel()
            else:
                pending.append(c)

    def _metrics_loop() -> None:
        while not stop.is_set():
            time.sleep(2.0)
            if stop.is_set():
                break
            m = system_metrics()
            if m:
                _emit(evt_q, {"type": "metrics", **m})

    def _run_job(cmd: dict) -> None:
        from ..core.events import ErrorEvent
        from ..core.jobqueue import ControlToken, PipelineCancelled
        from ..core.orchestrator import PipelineConfig, run_pipeline

        tok = ControlToken()
        token_box["token"] = tok
        pump = threading.Thread(target=_pump, daemon=True)
        metrics = threading.Thread(target=_metrics_loop, daemon=True)
        pump.start()
        metrics.start()
        try:
            allowed = {f.name for f in dataclasses.fields(PipelineConfig)}
            cfg_dict = {k: v for k, v in (cmd.get("cfg") or {}).items() if k in allowed}
            cfg = PipelineConfig(**cfg_dict)
            run_pipeline(
                cmd["job_id"],
                cmd["input_path"],
                cmd["output_path"],
                cfg,
                emit=lambda ev: _emit(evt_q, ev.to_dict()),
                control=tok,
            )
        except PipelineCancelled:
            log.warning("Job %s cancelled by user.", cmd.get("job_id"))
            _emit(evt_q, {"type": "cancelled", "job_id": str(cmd.get("job_id", ""))})
        except Exception as exc:  # surfaced to the UI as an error event
            log.exception("Pipeline failed for job %s", cmd.get("job_id"))
            _emit(evt_q, ErrorEvent(message=friendly_error(exc), job_id=str(cmd.get("job_id", ""))).to_dict())
        finally:
            token_box.pop("token", None)
            pump.join(timeout=2.0)

    log.info("Worker process starting (pid=%s).", os.getpid())
    _emit(evt_q, {"type": "ready", "gpu": gpu_info()})

    while not stop.is_set():
        try:
            cmd = cmd_q.get(timeout=0.5)
        except queue.Empty:
            continue
        if cmd is None or cmd.get("cmd") == "shutdown":
            break
        kind = cmd.get("cmd")
        if kind == "run":
            _run_job(cmd)
        elif kind == "list_models":
            try:
                rows = _model_rows(cmd.get("task"))
            except Exception as exc:  # pragma: no cover - manifest corruption
                log.exception("Failed to list models")
                rows = []
            _emit(evt_q, {"type": "models_listed", "task": cmd.get("task"), "models": rows})
        elif kind == "download":
            for name in cmd.get("names", []):
                if stop.is_set():
                    break
                _download_one(name, evt_q)
        elif kind == "make_sample":
            _make_sample(cmd, evt_q)
        elif kind == "gpu_info":
            _emit(evt_q, {"type": "ready", "gpu": gpu_info()})
        else:
            log.warning("Unknown worker command: %r", cmd)

    # Drain buffered commands that are safe to answer after the last job.
    for c in list(pending):
        if c.get("cmd") == "list_models":
            _emit(evt_q, {"type": "models_listed", "task": c.get("task"), "models": _model_rows(c.get("task"))})

    log.info("Worker shutting down.")

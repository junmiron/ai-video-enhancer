"""UI-side worker client (design.md §6 process model).

Spawns the :mod:`aive.ui.worker` process and pumps its event queue into a plain
callback — Qt-independent on purpose, so the protocol is testable headless. The
main window wraps this in a QObject bridge to re-emit events as Qt signals from
the pump thread (queued connections keep everything on the GUI thread).
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import threading
from typing import Callable

log = logging.getLogger(__name__)


class WorkerClient:
    """Owns one worker process; delivers its events to ``on_event(dict)``."""

    def __init__(self, on_event: Callable[[dict], None]):
        self._on_event = on_event
        # spawn (Windows default): the child re-imports modules cleanly and never
        # shares Qt state with this process.
        ctx = mp.get_context()
        self._cmd_q = ctx.Queue()
        self._evt_q = ctx.Queue()
        from .worker import worker_main

        self._proc = ctx.Process(
            target=worker_main, args=(self._cmd_q, self._evt_q), name="aive-worker", daemon=True
        )
        self._stop = threading.Event()
        self._proc.start()
        self._pump_thread = threading.Thread(target=self._pump_loop, name="aive-event-pump", daemon=True)
        self._pump_thread.start()

    # -- lifecycle -----------------------------------------------------------

    @property
    def pid(self):
        return self._proc.pid

    @property
    def alive(self) -> bool:
        return self._proc.is_alive()

    def _pump_loop(self) -> None:
        while not self._stop.is_set():
            try:
                d = self._evt_q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._on_event(d)
            except Exception:  # a bad handler must never kill the pump
                log.exception("Error handling worker event %r", d.get("type"))

    def stop(self, timeout: float = 20.0) -> None:
        """Ask the worker to shut down (it cancels any running job first)."""
        self._stop.set()
        try:
            self.send({"cmd": "shutdown"})
        except RuntimeError:
            pass
        self._proc.join(timeout)
        if self._proc.is_alive():
            log.warning("Worker did not exit in %.0fs; terminating.", timeout)
            self._proc.terminate()
            self._proc.join(5.0)

    # -- commands ------------------------------------------------------------

    def send(self, cmd: dict) -> None:
        if not self.alive:
            raise RuntimeError("worker process is not running")
        self._cmd_q.put(cmd)

    def run_job(self, job_id: str, input_path: str, output_path: str, cfg_dict: dict) -> None:
        self.send(
            {"cmd": "run", "job_id": job_id, "input_path": input_path, "output_path": output_path, "cfg": cfg_dict}
        )

    def pause(self) -> None:
        self.send({"cmd": "pause"})

    def resume(self) -> None:
        self.send({"cmd": "resume"})

    def cancel(self) -> None:
        self.send({"cmd": "cancel"})

    def list_models(self, task: str | None = None) -> None:
        self.send({"cmd": "list_models", "task": task})

    def download(self, names: list[str]) -> None:
        self.send({"cmd": "download", "names": list(names)})

    def make_sample(self, dest: str) -> None:
        self.send({"cmd": "make_sample", "dest": dest})

    def gpu_info(self) -> None:
        self.send({"cmd": "gpu_info"})

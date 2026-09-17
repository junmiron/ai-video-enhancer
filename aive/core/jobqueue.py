"""Jobs and pause/cancel control.

M1: in-process queue + threading events (worker runs in a separate process
from M6 onward; the same Job/ControlToken types cross the boundary).
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    CANCELLING = "cancelling"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class Job:
    input_path: str
    output_path: str | None = None
    settings: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: JobStatus = JobStatus.QUEUED
    created_ts: float = field(default_factory=time.time)


@dataclass
class ControlToken:
    """Cooperative pause/cancel signal checked between frames and segments."""

    _pause: bool = False
    _cancel: bool = field(default=False, init=False)

    def __post_init__(self):
        self._cancel = False

    # -- control ---------------------------------------------------------
    def request_pause(self) -> None:
        self._pause = True

    def resume(self) -> None:
        self._pause = False

    def request_cancel(self) -> None:
        self._cancel = True

    @property
    def cancelled(self) -> bool:
        return self._cancel

    def wait_if_paused(self, poll_s: float = 0.25) -> None:
        """Block while paused; returns immediately if not paused."""
        import time as _t

        while self._pause and not self._cancel:
            _t.sleep(poll_s)


class JobQueue:
    """Minimal FIFO job queue (in-process for M1)."""

    def __init__(self):
        from collections import deque

        self._q: deque[Job] = deque()
        self._history: list[Job] = []

    def submit(self, job: Job) -> None:
        self._q.append(job)

    def pop(self) -> Job | None:
        if not self._q:
            return None
        job = self._q.popleft()
        self._history.append(job)
        return job

    @property
    def pending(self) -> int:
        return len(self._q)

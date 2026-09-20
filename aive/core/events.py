"""Pipeline event types.

Events are plain dataclasses so they can cross process boundaries (M6 UI)
via multiprocessing.Queue after to_dict()/from_dict() round-trips.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class Event:
    type: str = "event"
    job_id: str = ""
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


@dataclass
class JobStarted(Event):
    input_path: str = ""
    type: str = field(default="job_started", init=False)


@dataclass
class StageStarted(Event):
    stage: str = ""
    total_frames: int = 0
    type: str = field(default="stage_started", init=False)


@dataclass
class StageProgress(Event):
    stage: str = ""
    frames_done: int = 0
    frames_total: int = 0
    fps: float = 0.0
    eta_s: float = 0.0
    type: str = field(default="stage_progress", init=False)


@dataclass
class StageFinished(Event):
    stage: str = ""
    elapsed_s: float = 0.0
    type: str = field(default="stage_finished", init=False)


@dataclass
class WarningEvent(Event):
    message: str = ""
    type: str = field(default="warning", init=False)


@dataclass
class ErrorEvent(Event):
    message: str = ""
    stage: str = ""
    type: str = field(default="error", init=False)


@dataclass
class JobFinished(Event):
    ok: bool = False
    output_path: str = ""
    elapsed_s: float = 0.0
    report: dict | None = None
    # M6 UI: paths of the exported artifact-report files ("json", "png") when a QA
    # stage ran, so the report view can offer Export without re-deriving paths.
    report_files: dict | None = None
    type: str = field(default="job_finished", init=False)


@dataclass
class ModelDownloadProgress(Event):
    model: str = ""
    bytes_done: int = 0
    bytes_total: int = 0
    type: str = field(default="model_download_progress", init=False)


@dataclass
class EngineBuildStarted(Event):
    model: str = ""
    type: str = field(default="engine_build_started", init=False)


@dataclass
class EngineBuildFinished(Event):
    model: str = ""
    engine_path: str = ""
    elapsed_s: float = 0.0
    cached: bool = False
    type: str = field(default="engine_build_finished", init=False)

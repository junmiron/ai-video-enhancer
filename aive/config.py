"""Application configuration: paths, settings persistence."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

APP_NAME = "ai-video-enhancer"
APP_DISPLAY = "AI Video Enhancer"


def _appdata_dir() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home())
    return Path(base) / APP_NAME


def _localappdata_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / APP_NAME


@dataclass
class Settings:
    """User-facing settings (persisted as JSON in %APPDATA%)."""

    # Output encoding (design.md D4: H.265 anime profile, 10-bit default)
    output_codec: str = "x265-10bit-anime"  # x265-10bit-anime | nvenc-hevc-10bit | h264
    crf: int = 19
    container: str = "mp4"  # mp4 | mkv
    audio_passthrough: bool = True

    # Pipeline behavior
    scene_split: bool = True
    keep_workspace: bool = False  # keep FFV1 segment chunks after success

    # Frame interpolation (M2)
    fi_enabled: bool = False
    fi_model: str = "rife_425"  # registry name (heavy/lite variants arrive with M2.x)
    fi_target_fps: float | None = None  # e.g. 60.0 — wins over multiplier when set
    fi_multiplier: int | None = None  # e.g. 2, 3, 4 — used when target fps is unset
    fi_dedup_threshold: float = 0.0  # mean abs diff in [0,1]; 0 disables de-duplication

    # Super-resolution (M3)
    sr_enabled: bool = False
    sr_model: str = "shufflecugan_x2"  # registry name (task=sr)
    sr_scale: int | None = None  # target output scale; None -> model's native scale

    # Artifact QA / restore (M4) — design.md §4.4 user controls
    qa_enabled: bool = True
    qa_mode: str = "auto"  # auto | detect_only | manual
    qa_proxy_height: int = 720  # detector resolution target (§4.5 proxy-first)
    qa_window_size: int = 7  # temporal window for detectors/repairs (kept odd)
    qa_min_severity: float = 0.15  # record a finding only at/above this severity
    qa_sensitivity: float = 0.5  # global default per-artifact sensitivity in [0,1]

    log_level: str = "INFO"


DEFAULTS: dict = asdict(Settings())


def data_dir() -> Path:
    p = _appdata_dir()
    p.mkdir(parents=True, exist_ok=True)
    return p


def model_cache_dir() -> Path:
    p = _localappdata_dir() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def workspaces_dir() -> Path:
    p = data_dir() / "workspaces"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _settings_path(path: str | os.PathLike | None = None) -> Path:
    if path is not None:
        return Path(path)
    return data_dir() / "settings.json"


def load_settings(path: str | os.PathLike | None = None) -> Settings:
    p = _settings_path(path)
    merged = dict(DEFAULTS)
    if p.exists():
        try:
            stored = json.loads(p.read_text(encoding="utf-8"))
            known = {f.name for f in fields(Settings)}
            merged.update({k: v for k, v in stored.items() if k in known})
        except (json.JSONDecodeError, OSError):
            pass  # fall back to defaults on corrupt file
    return Settings(**merged)


def save_settings(settings: Settings, path: str | os.PathLike | None = None) -> Path:
    p = _settings_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
    return p

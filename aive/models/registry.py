"""Model registry: manifest-driven model metadata.

Adding a new model = adding a manifest entry (+ adapter class where the
architecture is not covered by an existing loader). This is how the app stays
"latest" without pipeline changes (design.md §5.5).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelInfo:
    name: str
    task: str  # sr | fi | restore | depth
    version: str
    urls: list[str] = field(default_factory=list)
    sha256: str | None = None
    min_vram_gb: float = 0.0
    backends: dict = field(default_factory=lambda: {"cuda": True, "tensorrt": False})
    license: str = ""
    recommended_for: str = ""
    notes: str = ""
    extract_member: str | None = None  # if set, the download is an archive; this member becomes the weights file


def default_manifest_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "assets" / "models" / "manifest.json"


def load_manifest(path: str | Path | None = None) -> dict[str, ModelInfo]:
    p = Path(path) if path else default_manifest_path()
    data = json.loads(p.read_text(encoding="utf-8"))
    models: dict[str, ModelInfo] = {}
    for entry in data.get("models", []):
        info = ModelInfo(**entry)
        models[info.name] = info
    return models


def get_model(name: str, task: str | None = None, path: str | Path | None = None) -> ModelInfo:
    models = load_manifest(path)
    if name not in models:
        raise KeyError(f"Unknown model '{name}'. Known: {sorted(models)}")
    info = models[name]
    if task is not None and info.task != task:
        raise ValueError(f"Model '{name}' has task '{info.task}', expected '{task}'")
    return info

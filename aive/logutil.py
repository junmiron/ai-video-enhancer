"""Logging setup: console + rotating file under the app data dir."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: str = "INFO", logfile: str | Path | None = None) -> None:
    root = logging.getLogger()
    if root.handlers:  # already configured
        return
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
    root.addHandler(console)

    target = Path(logfile) if logfile else None
    if target is None:
        try:
            from .config import data_dir

            target = data_dir() / "logs" / "aive.log"
        except Exception:  # pragma: no cover - config import edge case
            target = None
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        fileh = RotatingFileHandler(target, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        fileh.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(fileh)

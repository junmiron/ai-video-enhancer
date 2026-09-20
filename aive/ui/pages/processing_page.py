"""Processing view (design.md §7.4).

Per-stage progress bars (keyed by stage name from the orchestrator's events),
overall bar + ETA, collapsible live log, VRAM/RAM meters and Pause/Cancel.
Stage display labels: identity -> Process, encode -> Encode, depth -> Depth map;
everything else is capitalized as-is (Interpolate / Upscale / QA).
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

STAGE_LABELS = {"identity": "Process", "encode": "Encode", "depth": "Depth map"}


def stage_label(stage: str) -> str:
    return STAGE_LABELS.get(stage, stage.capitalize())


class ProcessingPage(QWidget):
    pause_clicked = Signal()
    resume_clicked = Signal()
    cancel_clicked = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 10)
        lay.setSpacing(10)

        # Status + overall progress.
        top = QHBoxLayout()
        self.status_label = QLabel("Idle")
        self.status_label.setProperty("role", "big")
        self.eta_label = QLabel("")
        self.eta_label.setProperty("role", "dim")
        top.addWidget(self.status_label)
        top.addStretch(1)
        top.addWidget(self.eta_label)
        lay.addLayout(top)

        self.overall_bar = QProgressBar()
        self.overall_bar.setRange(0, 100)
        self.overall_bar.setValue(0)
        lay.addWidget(self.overall_bar)

        # Per-stage bars (rows appended as stages start).
        self.stage_box = QVBoxLayout()
        self._stage_rows: dict[str, tuple[QHBoxLayout, QLabel, QProgressBar]] = {}
        placeholder = QLabel("No job running yet — add a video on the Home page and press “Enhance all”.")
        placeholder.setProperty("role", "dim")
        self._placeholder = placeholder
        self.stage_box.addWidget(placeholder)
        lay.addLayout(self.stage_box)

        # Meters.
        meters = QHBoxLayout()
        vram_wrap = QVBoxLayout()
        vram_title = QLabel("VRAM")
        self.vram_bar = QProgressBar()
        self.vram_bar.setRange(0, 100)
        self.vram_bar.setValue(0)
        vram_wrap.addWidget(vram_title)
        vram_wrap.addWidget(self.vram_bar)
        ram_wrap = QVBoxLayout()
        ram_title = QLabel("RAM")
        self.ram_label = QLabel("")
        self.ram_label.setProperty("role", "dim")
        ram_wrap.addWidget(ram_title)
        ram_wrap.addWidget(self.ram_label)
        meters.addLayout(vram_wrap, 3)
        meters.addLayout(ram_wrap, 2)
        lay.addLayout(meters)

        # Collapsible live log.
        self.log_group = QGroupBox("Live log")
        self.log_group.setCheckable(True)
        self.log_group.setChecked(False)
        lg = QVBoxLayout(self.log_group)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        lg.addWidget(self.log_view)
        lay.addWidget(self.log_group, stretch=1)

        # Controls.
        btns = QHBoxLayout()
        self.pause_btn = QPushButton("Pause")
        self.resume_btn = QPushButton("Resume")
        self.cancel_btn = QPushButton("Cancel job")
        self.pause_btn.clicked.connect(self.pause_clicked.emit)
        self.resume_btn.clicked.connect(self.resume_clicked.emit)
        self.cancel_btn.clicked.connect(self.cancel_clicked.emit)
        btns.addWidget(self.pause_btn)
        btns.addWidget(self.resume_btn)
        btns.addStretch(1)
        btns.addWidget(self.cancel_btn)
        lay.addLayout(btns)

        self.set_state("idle")

    # -- state -----------------------------------------------------------------

    def set_state(self, state: str) -> None:
        """state: idle | running | paused"""
        text = {"idle": "Idle", "running": "Processing…", "paused": "Paused"}[state]
        self.status_label.setText(text)
        self.pause_btn.setEnabled(state == "running")
        self.resume_btn.setEnabled(state == "paused")
        self.cancel_btn.setEnabled(state in ("running", "paused"))

    def reset(self) -> None:
        for row, name, bar in list(self._stage_rows.values()):
            self.stage_box.removeItem(row)
            name.setParent(None)
            name.deleteLater()
            bar.setParent(None)
            bar.deleteLater()
        self._stage_rows.clear()
        self.overall_bar.setValue(0)
        self.eta_label.setText("")
        self.vram_bar.setValue(0)
        self.ram_label.setText("")
        self.log_view.clear()
        self._placeholder.show()

    # -- stage events -------------------------------------------------------------

    def _stage_bar(self, stage: str) -> QProgressBar:
        entry = self._stage_rows.get(stage)
        if entry is not None:
            return entry[2]
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)
        row = QHBoxLayout()
        name = QLabel(f"{stage_label(stage):<12}")
        row.addWidget(name)
        row.addWidget(bar, 1)
        self._placeholder.hide()
        self.stage_box.addLayout(row)
        self._stage_rows[stage] = (row, name, bar)
        return bar

    def on_stage_started(self, stage: str, total_frames: int) -> None:
        if stage == "process":
            return  # drives the overall bar only
        bar = self._stage_bar(stage)
        bar.setValue(0)
        bar.setFormat(f"starting… ({total_frames} frames)")

    def on_stage_progress(self, stage: str, done: int, total: int, fps: float, eta_s: float) -> None:
        if stage == "process":
            pct = int(done * 100 / total) if total else 0
            self.overall_bar.setValue(pct)
            self.eta_label.setText(
                f"ETA {eta_s:.0f} s · {done}/{total} frames · {fps:.1f} fps" if eta_s > 0.5 else ""
            )
            return
        bar = self._stage_bar(stage)
        pct = int(done * 100 / total) if total else 0
        bar.setValue(pct)
        bar.setFormat(f"{pct}% · {done}/{total}")

    def on_stage_finished(self, stage: str, elapsed_s: float) -> None:
        if stage == "process":
            self.overall_bar.setValue(100)
            return
        entry = self._stage_rows.get(stage)
        if entry is not None:
            bar = entry[2]
            bar.setValue(100)
            bar.setFormat(f"done in {elapsed_s:.1f} s")

    # -- meters / log ---------------------------------------------------------------

    def set_metrics(self, m: dict) -> None:
        total = float(m.get("vram_total_gb") or 0.0)
        free = float(m.get("vram_free_gb") or 0.0)
        if total > 0:
            used = max(0.0, total - free)
            self.vram_bar.setValue(int(used * 100 / total))
            self.vram_bar.setFormat(f"{used:.1f} / {total:.1f} GiB")
        ram = m.get("ram_used_gb")
        if ram is not None:
            self.ram_label.setText(f"used {float(ram):.1f} GiB")

    def append_log(self, level: str, message: str) -> None:
        self.log_view.appendPlainText(message)

"""Home page (design.md §7.1): drop zone + queue table + first-run card."""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFrame,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..theme import ACCENT, DIM, ERR, OK, WARN

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".ts", ".m2ts", ".wmv", ".flv"}

_STATUS_COLORS = {
    "pending": DIM,
    "processing": ACCENT,
    "paused": WARN,
    "done": OK,
    "failed": ERR,
    "cancelled": DIM,
}


def is_video(path: str | os.PathLike) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


class DropZone(QFrame):
    """Dashed drop target; emits dropped file paths (video files only)."""

    files_dropped = Signal(list)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("dropZone")
        self.setAcceptDrops(True)
        self.setMinimumHeight(150)

        lay = QVBoxLayout(self)
        title = QLabel("Drop video here")
        title.setObjectName("dzTitle")
        title.setProperty("role", "big")
        title.setAlignment(Qt.AlignCenter)
        sub = QLabel("MP4 · MKV · MOV — or click Browse")
        sub.setProperty("role", "dim")
        sub.setAlignment(Qt.AlignCenter)

        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(browse)
        row.addStretch(1)

        lay.addStretch(1)
        lay.addWidget(title)
        lay.addWidget(sub)
        lay.addLayout(row)
        lay.addStretch(2)

    def _browse(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select videos", "", "Videos (*.mp4 *.mkv *.mov *.avi *.webm *.ts *.m2ts)"
        )
        if paths:
            self.files_dropped.emit(paths)

    # -- drag & drop -----------------------------------------------------------

    def _urls(self, ev) -> list[str]:
        return [u.toLocalFile() for u in (ev.mimeData().urls() if ev.mimeData() else [])]

    def dragEnterEvent(self, ev) -> None:  # noqa: N802 - Qt naming
        if ev.mimeData().hasUrls():
            self.setProperty("active", "true")
            self.style().unpolish(self)
            self.style().polish(self)
            ev.acceptProposedAction()

    def dragLeaveEvent(self, ev) -> None:  # noqa: N802
        self.setProperty("active", "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def dropEvent(self, ev) -> None:  # noqa: N802
        self.setProperty("active", "false")
        self.style().unpolish(self)
        self.style().polish(self)
        paths = [p for p in self._urls(ev) if is_video(p)]
        if paths:
            ev.acceptProposedAction()
            self.files_dropped.emit(paths)


class HomePage(QWidget):
    """Queue management + first-run model download card."""

    files_requested = Signal(list)          # browse/drop -> main window adds to queue
    enhance_all_clicked = Signal()
    remove_selected_clicked = Signal()
    clear_finished_clicked = Signal()
    get_started_clicked = Signal()          # first-run: download default models
    make_sample_clicked = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 10)
        lay.setSpacing(12)

        self.drop_zone = DropZone()
        self.drop_zone.files_dropped.connect(self.files_requested.emit)
        lay.addWidget(self.drop_zone)

        # Queue table.
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["File", "Source info", "Status"])
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self._rows: list[dict] = []
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        lay.addWidget(self.table, stretch=1)

        # Action row.
        btns = QHBoxLayout()
        self.enhance_btn = QPushButton("Enhance all")
        self.enhance_btn.setProperty("accent", "true")
        self.enhance_btn.clicked.connect(self.enhance_all_clicked.emit)
        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(self.remove_selected_clicked.emit)
        clear_btn = QPushButton("Clear finished")
        clear_btn.clicked.connect(self.clear_finished_clicked.emit)
        btns.addWidget(self.enhance_btn)
        btns.addStretch(1)
        btns.addWidget(remove_btn)
        btns.addWidget(clear_btn)
        lay.addLayout(btns)

        # First-run card (hidden until the main window decides it is needed).
        self.first_run = QFrame()
        fr_lay = QVBoxLayout(self.first_run)
        fr_title = QLabel("Welcome — get your models ready")
        fr_title.setProperty("role", "big")
        fr_text = QLabel(
            "The default models for the current preset will be downloaded once (cached afterwards). "
            "TensorRT engines build automatically at first use of each model."
        )
        fr_text.setWordWrap(True)
        self.download_bar = QProgressBar()
        self.download_bar.setRange(0, 100)
        self.download_bar.setValue(0)
        get_btn = QPushButton("Download default models")
        get_btn.setProperty("accent", "true")
        get_btn.clicked.connect(self.get_started_clicked.emit)
        sample_btn = QPushButton("Generate a sample clip (3 s test run)")
        sample_btn.clicked.connect(self.make_sample_clicked.emit)
        fr_row = QHBoxLayout()
        fr_row.addWidget(get_btn)
        fr_row.addStretch(1)
        fr_row.addWidget(sample_btn)
        fr_lay.addWidget(fr_title)
        fr_lay.addWidget(fr_text)
        fr_lay.addWidget(self.download_bar)
        fr_lay.addLayout(fr_row)
        self.first_run.hide()
        lay.addWidget(self.first_run)

    # -- state -----------------------------------------------------------------

    def set_queue(self, rows: list[dict]) -> None:
        """rows: [{path, status, detail}] in queue order."""
        self._rows = list(rows)
        self.table.setRowCount(len(rows))
        for r, q in enumerate(rows):
            name_item = QTableWidgetItem(Path(q["path"]).name)
            info_item = QTableWidgetItem(q.get("info_text", ""))
            status_item = QTableWidgetItem(f"{q['status'].capitalize()} — {q.get('detail', '')}")
            color = _STATUS_COLORS.get(q["status"], DIM)
            for it in (name_item, info_item, status_item):
                it.setForeground(QColor(color if it is not name_item else "#e9eaef"))
            self.table.setItem(r, 0, name_item)
            self.table.setItem(r, 1, info_item)
            self.table.setItem(r, 2, status_item)

    def selected_paths(self) -> list[str]:
        model = self.table.selectionModel()
        rows = {ix.row() for ix in (model.selectedRows() if model else [])}
        out = []
        for r in sorted(rows):
            if 0 <= r < len(self._rows):
                out.append(self._rows[r]["path"])
        return out

    def set_first_run(self, visible: bool) -> None:
        self.first_run.setVisible(visible)

    def set_download_progress(self, model: str | None, frac: float | None) -> None:
        if model is None or frac is None:
            self.download_bar.setFormat("Preparing…")
            return
        pct = int(frac * 100)
        self.download_bar.setValue(pct)
        self.download_bar.setFormat(f"Downloading {model} — {pct}%")

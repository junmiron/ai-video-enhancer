"""Artifact report view (design.md §7.3).

Table of per-artifact summary rows (Type / Count / Severity / Repair applied /
Before -> After) built from the QA stage's report dict, plus JSON/PNG export.
The "Repair applied" column shows the dominant repair method observed in the
findings for that artifact type.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...stages.qa.findings import ARTIFACT_META, ArtifactType  # pure Python (no cv2/torch)


def _sev_word(v: float | None) -> str:
    if v is None:
        return "—"
    if v >= 0.7:
        return "high"
    if v >= 0.5:
        return "medium"
    if v >= 0.3:
        return "low"
    return "minor"


def _metric_text(before, after) -> str:
    try:
        b = float(before)
        a = float(after)
        arrow = "↓" if a < b else ("↑" if a > b else "=")
        return f"{b:.3f} {arrow} {a:.3f}"
    except (TypeError, ValueError):
        return "—"


class ReportPage(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 10)
        lay.setSpacing(10)

        self.header_label = QLabel("No artifact report yet — run a job with the QA stage enabled.")
        self.header_label.setProperty("role", "dim")
        self.header_label.setWordWrap(True)
        lay.addWidget(self.header_label)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Type", "Count", "Severity", "Repair applied", "Before → After"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        lay.addWidget(self.table, stretch=1)

        btns = QHBoxLayout()
        self.export_json_btn = QPushButton("Export JSON")
        self.export_png_btn = QPushButton("Export PNG summary")
        self.export_json_btn.clicked.connect(self._export_json)
        self.export_png_btn.clicked.connect(self._export_png)
        btns.addWidget(self.export_json_btn)
        btns.addWidget(self.export_png_btn)
        btns.addStretch(1)
        lay.addLayout(btns)

        self._report: dict | None = None
        self._report_files: dict | None = None
        self._input_name = ""
        self._set_enabled(False)

    # -- data -----------------------------------------------------------------

    def set_report(self, report: dict | None, report_files: dict | None, input_name: str = "") -> None:
        """Populate the view. `report` is the QA stage's to_dict() payload."""
        self._report = report
        self._report_files = report_files or {}
        self._input_name = input_name
        if not report:
            self.header_label.setText("No artifact report yet — run a job with the QA stage enabled.")
            self.table.setRowCount(0)
            self._set_enabled(False)
            return

        dur = float(report.get("duration_s") or 0.0)
        mode = report.get("mode", "auto")
        self.header_label.setText(
            f"Artifact report — {input_name or Path(str(report.get('output_path', ''))).name} "
            f"({dur / 60:.1f} min · mode: {mode})"
        )

        summary = report.get("summary") or {}
        findings = report.get("findings") or []
        methods: dict[str, Counter] = {}
        for f in findings:
            if isinstance(f, dict) and f.get("repaired"):
                methods.setdefault(str(f.get("artifact")), Counter())[str(f.get("method", "?"))] += 1

        # Stable order: the detector set order from ARTIFACT_META.
        ordered = [a for a in ArtifactType.all() if a.value in summary]
        self.table.setRowCount(len(ordered))
        for r, art in enumerate(ordered):
            s = summary[art.value] or {}
            label, _color = ARTIFACT_META[art]
            c = methods.get(art.value)
            repair = c.most_common(1)[0][0] if c else ("—" if mode == "detect_only" else "—")
            items = [
                QTableWidgetItem(label),
                QTableWidgetItem(str(int(s.get("count", 0)))),
                QTableWidgetItem(_sev_word(s.get("max_severity"))),
                QTableWidgetItem(repair),
                QTableWidgetItem(_metric_text(s.get("before_metric"), s.get("after_metric"))),
            ]
            for col, it in enumerate(items):
                self.table.setItem(r, col, it)
        self._set_enabled(True)

    def _set_enabled(self, on: bool) -> None:
        self.export_json_btn.setEnabled(on and self._report is not None)
        has_png = bool((self._report_files or {}).get("png"))
        self.export_png_btn.setEnabled(on and has_png)

    # -- exports -----------------------------------------------------------------

    def _export_json(self) -> None:
        if not self._report:
            return
        default = f"{Path(self._input_name).stem or 'artifact'}_report.json"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export artifact report (JSON)", default, "JSON files (*.json)"
        )
        if not path:
            return
        Path(path).write_text(json.dumps(self._report, indent=2), encoding="utf-8")

    def _export_png(self) -> None:
        src = (self._report_files or {}).get("png")
        if not src or not Path(src).exists():
            return
        default = f"{Path(self._input_name).stem or 'artifact'}_report.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export artifact report (PNG)", default, "PNG images (*.png)"
        )
        if not path:
            return
        shutil.copyfile(src, path)

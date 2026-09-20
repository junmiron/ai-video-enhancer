"""Main window (design.md §7): GPU banner, left nav + page stack, worker client.

The pipeline runs in a separate process (:mod:`aive.ui.worker`); this class owns
the :class:`~aive.ui.client.WorkerClient`, dispatches its event dicts to the
right pages, and drives the queue state machine (auto-advance on job finish).
Worker events arrive on the pump thread; the ``_EventBridge`` signal re-emits
them as queued Qt signals so every handler runs on the GUI thread.
"""
from __future__ import annotations

import dataclasses
import logging
import uuid
from pathlib import Path

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..config import Settings, load_settings, save_settings
from ..core.orchestrator import PipelineConfig
from ..models.downloader import cached_weights_path
from ..models.registry import get_model
from ..stages.ingest import MediaInfo, probe
from .client import WorkerClient
from .pages.home_page import HomePage
from .pages.preview_page import PreviewPage
from .pages.processing_page import ProcessingPage
from .pages.report_page import ReportPage
from .pages.settings_page import SettingsPage
from .presets import default_models_for

log = logging.getLogger(__name__)


class _EventBridge(QObject):
    """Thread-safe hop from the worker pump thread to the GUI thread."""

    event_received = Signal(object)


def unique_output_path(input_path: str | Path, container: str) -> str:
    """<dir>/<stem>_enhanced.<container>, with a numeric suffix if it exists."""
    p = Path(input_path)
    out = p.with_name(f"{p.stem}_enhanced.{container}")
    n = 2
    while out.exists():
        out = p.with_name(f"{p.stem}_enhanced_{n}.{container}")
        n += 1
    return str(out)


def _info_text(info: MediaInfo | None) -> str:
    if info is None:
        return ""
    fps = f"{info.fps_float:g} fps"
    dur = f"{info.duration_s / 60:.1f} min" if info.duration_s else "?"
    return f"{info.width}×{info.height} · {fps} · {dur}"


class MainWindow(QMainWindow):
    def __init__(self, settings_path: str | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("AI Video Enhancer")
        self.resize(1280, 800)

        self._settings_path = settings_path
        self.settings: Settings = load_settings(settings_path)
        self.queue: list[dict] = []          # {path, info, status, detail}
        self._current: dict | None = None    # queue entry of the running job
        self.last_report: dict | None = None
        self.last_report_files: dict | None = None

        # Worker process (separate OS process; events via bridge signal).
        # NOTE: keep the bridge alive as an attribute — a PySide6 bound signal
        # method does not hold a strong reference to its QObject, and GC would
        # delete it mid-run ("Signal source has been deleted").
        self._bridge = _EventBridge()
        self._bridge.event_received.connect(self._on_worker_event)
        self.client = WorkerClient(on_event=self._bridge.event_received.emit)
        self._worker_ready = False
        self._gpu_ok = False  # set from the worker's ready event (M7 fail-fast)

        # -- layout -------------------------------------------------------------
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 6)
        root.setSpacing(8)

        self.banner = QLabel("Starting engine…")
        self.banner.setProperty("role", "banner")
        root.addWidget(self.banner)

        body = QHBoxLayout()
        self.nav = QListWidget()
        nav = self.nav
        nav.setObjectName("nav")
        for name in ("Home", "Settings", "Processing", "Report", "Preview"):
            nav.addItem(name)
        nav.setFixedWidth(170)
        self.stack = QStackedWidget()

        self.home_page = HomePage()
        self.settings_page = SettingsPage()
        self.processing_page = ProcessingPage()
        self.report_page = ReportPage()
        self.preview_page = PreviewPage()
        for page in (self.home_page, self.settings_page, self.processing_page, self.report_page, self.preview_page):
            self.stack.addWidget(page)

        nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        body.addWidget(nav)
        body.addWidget(self.stack, 1)
        root.addLayout(body, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready", 5000)

        # -- page wiring -----------------------------------------------------------
        self.home_page.files_requested.connect(self.add_files)
        self.home_page.enhance_all_clicked.connect(self.start_queue)
        self.home_page.remove_selected_clicked.connect(self._remove_selected)
        self.home_page.clear_finished_clicked.connect(self._clear_finished)
        self.home_page.get_started_clicked.connect(self.get_started)
        self.home_page.make_sample_clicked.connect(self.make_sample)

        self.settings_page.preset_changed.connect(self.on_preset_changed)
        self.settings_page.settings_changed.connect(self.on_settings_changed)

        self.processing_page.pause_clicked.connect(lambda: self.client.pause())
        self.processing_page.resume_clicked.connect(lambda: self.client.resume())
        self.processing_page.cancel_clicked.connect(self._cancel_job)

        # -- initial state -------------------------------------------------------------
        self.settings_page.load(self.settings)
        self._refresh_model_badges()
        self._update_first_run_card()

    # ------------------------------------------------------------------ queue

    def add_files(self, paths: list[str]) -> None:
        added = 0
        for p in paths:
            p = str(p)
            if not Path(p).exists():
                continue
            try:
                info = probe(p)
                status, detail = "pending", ""
            except Exception as exc:
                info, status, detail = None, "failed", f"Cannot read file: {exc}"[:200]
            self.queue.append({"path": p, "info": info, "status": status, "detail": detail})
            added += 1
        if added:
            self.statusBar().showMessage(f"Added {added} file(s) to the queue", 5000)
        self._refresh_queue_ui()

    def start_queue(self) -> None:
        if not any(q["status"] == "pending" for q in self.queue):
            return
        self._save_settings()
        self._start_next()

    def _start_next(self) -> None:
        # M7 fail-fast: without CUDA no AI job can run — mark pending items
        # failed with a clear message instead of letting them die deep in torch.
        if self._worker_ready and not self._gpu_ok:
            for q in self.queue:
                if q["status"] == "pending":
                    q["status"] = "failed"
                    q["detail"] = "No NVIDIA GPU detected — AI stages cannot run on this machine."
            self._refresh_queue_ui()
            return
        for q in self.queue:
            if q["status"] != "pending":
                continue
            out = unique_output_path(q["path"], self.settings.container)
            cfg = PipelineConfig.from_settings(self.settings)
            job_id = uuid.uuid4().hex[:12]
            q["status"] = "processing"
            q["detail"] = f"→ {Path(out).name}"
            self._current = q
            self.processing_page.reset()
            self.processing_page.set_state("running")
            try:
                self.client.run_job(job_id, q["path"], out, dataclasses.asdict(cfg))
            except RuntimeError as exc:  # worker died
                q["status"] = "failed"
                q["detail"] = str(exc)
                self._refresh_queue_ui()
                return
            self.stack.setCurrentIndex(2)  # Processing view
            self.statusBar().showMessage(f"Processing {Path(q['path']).name}…", 10000)
            break
        else:
            self.processing_page.set_state("idle")
        self._refresh_queue_ui()

    def _remove_selected(self) -> None:
        paths = set(self.home_page.selected_paths())
        if not paths:
            return
        kept = [q for q in self.queue if q["path"] not in paths or q["status"] == "processing"]
        removed = len(self.queue) - len(kept)
        self.queue[:] = kept
        self.statusBar().showMessage(f"Removed {removed} file(s)", 5000)
        self._refresh_queue_ui()

    def _clear_finished(self) -> None:
        before = len(self.queue)
        self.queue[:] = [q for q in self.queue if q["status"] not in ("done", "failed", "cancelled")]
        self.statusBar().showMessage(f"Cleared {before - len(self.queue)} finished job(s)", 5000)
        self._refresh_queue_ui()

    def _cancel_job(self) -> None:
        if self._current and self._current["status"] in ("processing", "paused"):
            self.client.cancel()
            self.statusBar().showMessage("Cancelling… (takes effect within one frame)", 5000)

    def _refresh_queue_ui(self) -> None:
        rows = []
        for q in self.queue:
            row = dict(q)
            row["info_text"] = _info_text(q["info"]) if q["status"] != "failed" else ""
            rows.append(row)
        self.home_page.set_queue(rows)
        pending = next((q for q in self.queue if q["status"] == "pending"), None)
        self.settings_page.set_source_info(pending["info"] if pending and pending["info"] else (self.queue[0]["info"] if self.queue else None))

    # ---------------------------------------------------------------- settings

    def _save_settings(self) -> None:
        try:
            save_settings(self.settings, self._settings_path)
        except OSError as exc:  # pragma: no cover - read-only profile dir
            log.warning("Could not save settings: %s", exc)

    def on_preset_changed(self, key: str) -> None:
        from .presets import apply_preset

        merged = apply_preset(dataclasses.asdict(self.settings), key)
        self.settings = Settings(**merged)
        self.settings_page.load(self.settings)
        self._save_settings()
        self.statusBar().showMessage(f"Preset “{key}” applied", 5000)
        self._update_first_run_card()

    def on_settings_changed(self) -> None:
        d = self.settings_page.collect()
        for k, v in d.items():
            if hasattr(self.settings, k):
                setattr(self.settings, k, v)
        self._save_settings()

    # ------------------------------------------------------------- first run

    def _missing_default_models(self) -> list[str]:
        names = default_models_for(self.settings_page.current_preset_key(), dataclasses.asdict(self.settings))
        missing = []
        for n in names:
            try:
                if cached_weights_path(get_model(n)) is None:
                    missing.append(n)
            except KeyError:
                missing.append(n)
        return missing

    def _update_first_run_card(self) -> None:
        self.home_page.set_first_run(bool(self._missing_default_models()))

    def get_started(self) -> None:
        names = self._missing_default_models()
        if not names:
            self.statusBar().showMessage("All default models are already downloaded.", 5000)
            return
        self.home_page.set_first_run(True)
        self.client.download(names)
        self.statusBar().showMessage(f"Downloading {len(names)} model(s)…", 10000)

    def make_sample(self) -> None:
        default = str(Path.home() / "Videos" / "sample_anime.mp4") if (Path.home() / "Videos").exists() else "sample_anime.mp4"
        dest, _ = QFileDialog.getSaveFileName(
            self, "Generate sample clip", default, "MP4 files (*.mp4)"
        )
        if not dest:
            return
        self.client.make_sample(dest)
        self.statusBar().showMessage("Generating sample clip… (a few seconds)", 10000)

    def _refresh_model_badges(self) -> None:
        rows = []
        try:
            from ..models.registry import list_models

            for m in list_models():
                rows.append({"name": m.name, "cached": cached_weights_path(m) is not None})
        except Exception:  # pragma: no cover - manifest corruption
            pass
        self.settings_page.refresh_models(rows)

    # --------------------------------------------------------------- events

    def _set_banner(self, text: str, tone: str = "") -> None:
        self.banner.setText(text)
        if tone:
            self.banner.setProperty("tone", tone)
        else:
            self.banner.setProperty("tone", "none")
        self.banner.style().unpolish(self.banner)
        self.banner.style().polish(self.banner)

    def _on_worker_event(self, d: dict) -> None:  # GUI thread (queued signal)
        t = d.get("type")

        if t == "ready":
            g = d.get("gpu") or {}
            self._worker_ready = True
            self._gpu_ok = bool(g.get("cuda_ok"))
            if g.get("cuda_ok"):
                trt = f" · TensorRT {g['trt_version']}" if g.get("trt_version") else ""
                vram = f" · VRAM {g.get('vram_total_gb', 0):g} GiB ({g.get('vram_free_gb', 0):g} free)" if g.get("vram_total_gb") else ""
                self._set_banner(f"{g.get('gpu_name', 'NVIDIA GPU')} — CUDA ready{trt}{vram}")
            else:
                self._set_banner("No NVIDIA GPU detected — AI models cannot run on this machine.", "warn")
            self.client.list_models()

        elif t == "log":
            self.processing_page.append_log(d.get("level", "INFO"), d.get("message", ""))

        elif t == "metrics":
            self.processing_page.set_metrics(d)

        elif t == "job_started":
            if self.stack.currentIndex() != 2:
                self.stack.setCurrentIndex(2)

        elif t in ("stage_started", "stage_progress", "stage_finished"):
            stage = d.get("stage", "")
            if t == "stage_started":
                self.processing_page.on_stage_started(stage, int(d.get("total_frames") or 0))
            elif t == "stage_progress":
                self.processing_page.on_stage_progress(
                    stage, int(d.get("frames_done") or 0), int(d.get("frames_total") or 0),
                    float(d.get("fps") or 0.0), float(d.get("eta_s") or 0.0),
                )
            else:
                self.processing_page.on_stage_finished(stage, float(d.get("elapsed_s") or 0.0))

        elif t == "warning":
            self.statusBar().showMessage(f"Warning: {d.get('message', '')}", 10000)

        elif t == "error":
            msg = d.get("message", "Unknown error")
            if self._current and self._current["status"] in ("processing", "paused"):
                self._current["status"] = "failed"
                self._current["detail"] = msg[:200]
                self.processing_page.set_state("idle")
                self.statusBar().showMessage(f"Job failed: {msg}", 15000)
                self._refresh_queue_ui()
                self._start_next()

        elif t == "job_finished":
            q = self._current
            if d.get("ok"):
                out = d.get("output_path", "")
                if q:
                    q["status"] = "done"
                    q["detail"] = f"→ {Path(out).name} · {float(d.get('elapsed_s') or 0):.0f}s"
                self.last_report = d.get("report")
                self.last_report_files = d.get("report_files")
                if q:
                    in_info = q["info"]
                    try:
                        out_info = probe(out) if out else None
                    except Exception:
                        out_info = None
                    self.preview_page.set_job(in_info, out_info)
                    report = d.get("report") or {}
                    self.preview_page.set_findings(report.get("findings"))
                self.report_page.set_report(
                    d.get("report"), d.get("report_files"),
                    Path(q["path"]).name if q else "",
                )
                self.processing_page.set_state("idle")
                self.statusBar().showMessage(f"Done: {Path(out).name}", 15000)
                if (d.get("report")) and self.settings.qa_show_report:
                    self.stack.setCurrentIndex(3)  # Report view
            else:
                if q:
                    q["status"] = "failed"
                    q["detail"] = "Job finished with errors"
                self.processing_page.set_state("idle")
            self._refresh_queue_ui()
            self._start_next()

        elif t == "cancelled":
            if self._current and self._current["status"] in ("processing", "paused"):
                self._current["status"] = "cancelled"
                self._current["detail"] = "Cancelled by user"
            self.processing_page.set_state("idle")
            self.statusBar().showMessage("Job cancelled.", 8000)
            self._refresh_queue_ui()
            self._start_next()

        elif t == "model_download_progress":
            total = max(int(d.get("bytes_total") or 0), 1)
            self.home_page.set_download_progress(d.get("model"), int(d.get("bytes_done") or 0) / total)

        elif t == "download_done":
            name, ok = d.get("name"), bool(d.get("ok"))
            if ok:
                self.statusBar().showMessage(f"Model ready: {name}", 8000)
            else:
                self._set_banner(f"Download failed for “{name}”: {d.get('error', '')[:160]}", "warn")
                self.statusBar().showMessage(f"Download failed: {name}", 15000)
            self.client.list_models()

        elif t == "models_listed":
            self.settings_page.refresh_models(d.get("models") or [])
            self._update_first_run_card()

        elif t == "sample_ready":
            self.add_files([d["path"]])
            self.statusBar().showMessage(f"Sample clip ready: {Path(d['path']).name}", 8000)

        elif t == "sample_error":
            QMessageBox.warning(self, "Sample clip", f"Could not generate the sample clip:\n{d.get('error', '')}")

        elif t == "control_ack":
            state = d.get("state")
            self.processing_page.set_state(state)
            if self._current:
                self._current["status"] = "paused" if state == "paused" else "processing"
                self._refresh_queue_ui()

        elif t in ("engine_build_started", "engine_build_finished"):
            model = d.get("model", "")
            if t == "engine_build_started":
                self.processing_page.append_log("INFO", f"Building TensorRT engine for {model}…")
            else:
                cached = " (cached)" if d.get("cached") else ""
                self.processing_page.append_log(
                    "INFO", f"TensorRT engine ready for {model}{cached} in {float(d.get('elapsed_s') or 0):.1f}s"
                )

    # ------------------------------------------------------------------ close

    def closeEvent(self, ev) -> None:  # noqa: N802 - Qt naming
        self._save_settings()
        try:
            self.client.stop(timeout=20)
        except Exception:  # pragma: no cover - worker already gone
            pass
        ev.accept()

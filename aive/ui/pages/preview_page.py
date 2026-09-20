"""Preview view (design.md §7.3): compare slider + frame picker + timeline markers.

Frames are grabbed asynchronously on a QThreadPool via ``ingest.grab_frame``
(fast input-seek decode), so the UI stays responsive even for long 4K files. A
small LRU cache keeps scrubbing back and forth cheap. The "before" frame index
maps output time onto the source timeline: in_idx = round(i * src_fps / out_fps).
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, Signal, QThreadPool
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from ..compare_widget import CompareWidget, TimelineStrip


class _GrabBridge(QObject):
    """Re-emits grab results as a signal (safe across threads)."""

    done = Signal(int, str, int, object)  # req_id, path, index, frame|None


class _GrabTask(QRunnable):
    def __init__(self, bridge: _GrabBridge, req_id: int, path: str, index: int):
        super().__init__()
        self._bridge = bridge
        self._req_id = req_id
        self._path = path
        self._index = index

    def run(self) -> None:  # worker thread
        frame = None
        try:
            from ...stages.ingest import grab_frame  # lazy: pulls ffmpeg deps

            frame = grab_frame(self._path, self._index)
        except Exception:
            frame = None
        self._bridge.done.emit(self._req_id, self._path, self._index, frame)


class PreviewPage(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 10)
        lay.setSpacing(8)

        self.compare = CompareWidget()
        lay.addWidget(self.compare, stretch=3)

        mid = QHBoxLayout()
        self.frame_label = QLabel("Frame —")
        self.frame_label.setProperty("role", "dim")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider)
        mid.addWidget(QLabel("Frame:"))
        mid.addWidget(self.slider, 1)
        mid.addWidget(self.frame_label)
        lay.addLayout(mid)

        self.strip = TimelineStrip()
        self.strip.frame_requested.connect(self._jump_to_frame)
        lay.addWidget(self.strip)

        self._bridge = _GrabBridge()
        self._bridge.done.connect(self._on_grab_done)
        self._pool = QThreadPool.globalInstance()
        self._cache: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        self._req_id = 0

        # Job state (filled by set_job from probed MediaInfo objects).
        self._in_path: str | None = None
        self._out_path: str | None = None
        self._src_fps = 30.0
        self._out_fps = 30.0
        self._in_num_frames = 0
        self._out_frames = 0

    # -- job setup ---------------------------------------------------------------

    def set_job(self, in_info, out_info) -> None:
        """in_info/out_info: MediaInfo (or None when the file is missing)."""
        self._cache.clear()
        self.compare.clear()
        self.strip.set_findings([], 1)
        self._in_path = str(in_info.path) if in_info else None
        self._out_path = str(out_info.path) if out_info else None
        self._src_fps = float(getattr(in_info, "fps_float", 0.0) or 30.0) if in_info else 30.0
        self._out_fps = float(getattr(out_info, "fps_float", 0.0) or 30.0) if out_info else 30.0
        self._in_num_frames = int(getattr(in_info, "num_frames", 0) or 0) if in_info else 0
        self._out_frames = int(getattr(out_info, "num_frames", 0) or 0) if out_info else 0

        ready = bool(self._in_path and self._out_path and self._out_frames > 0)
        self.slider.setRange(0, max(self._out_frames - 1, 0))
        self.slider.setEnabled(ready)
        if not ready:
            self.frame_label.setText("Run a job to enable the preview")
        else:
            self._on_slider(0)

    def set_findings(self, findings: list[dict] | None) -> None:
        self.strip.set_findings(findings or [], max(self._out_frames, 1))

    # -- frame picking ---------------------------------------------------------------

    def _jump_to_frame(self, i: int) -> None:
        if self.slider.isEnabled():
            self.slider.setValue(int(i))

    def _on_slider(self, i: int) -> None:
        if not (self._in_path and self._out_path):
            return
        t = i / self._out_fps if self._out_fps else 0.0
        mm, ss = divmod(t, 60.0)
        hh, mm2 = divmod(int(mm), 60)
        ts = f"{hh:02d}:{int(mm2):02d}:{ss:04.1f}" if hh else f"{int(mm2):02d}:{ss:04.1f}"
        self.frame_label.setText(f"Frame {i} / {max(self._out_frames - 1, 0)} · {ts}")
        self.strip.set_frame(i)
        self._request(i)

    def _in_index_for(self, out_idx: int) -> int:
        """Source frame aligned with output frame `out_idx` (same timestamp)."""
        if not (self._src_fps and self._out_fps and self._in_num_frames):
            return 0
        t = out_idx / self._out_fps
        in_idx = int(round(t * self._src_fps))
        return min(max(in_idx, 0), self._in_num_frames - 1)

    # -- async grabs -----------------------------------------------------------------

    def _request(self, out_idx: int) -> None:
        if not (self._in_path and self._out_path):
            return
        in_idx = self._in_index_for(out_idx)

        def _ensure(path: str, idx: int) -> None:
            key = (path, idx)
            if key in self._cache:
                self._refresh_compare()
                return
            self._req_id += 1
            task = _GrabTask(self._bridge, self._req_id, path, idx)
            self._pool.start(task)

        _ensure(self._out_path, out_idx)
        _ensure(self._in_path, in_idx)

    def _on_grab_done(self, req_id: int, path: str, index: int, frame) -> None:
        # Stale results are harmless: the compare view only shows frames for the
        # current slider position, and caching them makes scrubbing back free.
        if frame is None:
            return
        self._cache[(path, index)] = np.asarray(frame)
        while len(self._cache) > 24:
            self._cache.popitem(last=False)
        self._refresh_compare()

    def _refresh_compare(self) -> None:
        if not (self._in_path and self._out_path):
            return
        i = int(self.slider.value())
        in_idx = self._in_index_for(i)
        before = self._cache.get((self._in_path, in_idx))
        after = self._cache.get((self._out_path, i))
        if before is not None and after is not None:
            self.compare.set_images(before=before, after=after)

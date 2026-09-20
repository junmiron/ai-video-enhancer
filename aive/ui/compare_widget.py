"""Before/after comparison widgets (design.md §7.3).

CompareWidget: paint-based split slider — left of the divider shows the source
frame ("before"), right shows the enhanced frame ("after"). The two images are
aligned 1:1 in content space, so a 2x-upscaled output overlays its source
exactly at any divider position (the before image is drawn scaled to cover the
same displayed area as the after image).

TimelineStrip: horizontal strip under the compare view with color-coded artifact
markers (palette from ``ARTIFACT_META``); hover shows a tooltip, click emits
``frame_requested(frame_index)``.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ..stages.qa.findings import ARTIFACT_META, ArtifactType  # pure Python (no cv2/torch)
from .theme import ACCENT, BG, DIM


def _to_qimage(frame: np.ndarray | None) -> QImage | None:
    """HxWx3 uint8 RGB (or HxW gray) numpy array -> independent QImage."""
    if frame is None:
        return None
    f = np.ascontiguousarray(frame)
    h, w = f.shape[:2]
    ch = 1 if f.ndim == 2 else 3
    fmt = QImage.Format_Grayscale8 if ch == 1 else QImage.Format_RGB888
    return QImage(f.data, w, h, w * ch, fmt).copy()


class CompareWidget(QWidget):
    """Split-slider before/after view. Drag anywhere to move the divider."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._before: QImage | None = None
        self._after: QImage | None = None
        self._split = 0.5
        self.setMinimumSize(480, 270)

    # -- data ----------------------------------------------------------------

    def set_images(self, before: np.ndarray | None = None, after: np.ndarray | None = None) -> None:
        if before is not None:
            self._before = _to_qimage(before)
        if after is not None:
            self._after = _to_qimage(after)
        self.update()

    def clear(self) -> None:
        self._before = None
        self._after = None
        self.update()

    # -- geometry ------------------------------------------------------------

    def _layout(self) -> tuple[float, float, float, float]:
        """(ox, oy, disp_w, disp_h) of the reference image (after if present)."""
        ref = self._after or self._before
        if ref is None:
            return 0.0, 0.0, 0.0, 0.0
        s = min((self.width() - 8) / max(ref.width(), 1), (self.height() - 8) / max(ref.height(), 1))
        dw, dh = ref.width() * s, ref.height() * s
        return (self.width() - dw) / 2.0, (self.height() - dh) / 2.0, dw, dh

    def _divider_x(self) -> float:
        ox, _oy, dw, _dh = self._layout()
        return ox + self._split * dw

    # -- painting ------------------------------------------------------------

    def paintEvent(self, ev) -> None:  # noqa: N802 - Qt naming
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(BG))
        if self._after is None and self._before is None:
            p.setPen(QColor(DIM))
            p.drawText(self.rect(), Qt.AlignCenter, "No frames loaded yet — run a job or pick a frame")
            return

        ox, oy, dw, dh = self._layout()

        # After (enhanced) full, then before clipped to the left of the divider.
        if self._after is not None:
            p.drawImage(int(ox), int(oy), self._after, 0, 0, self._after.width(), self._after.height())
        if self._before is not None and dw > 0:
            scale_b = dw / max(self._before.width(), 1)  # content-aligned with after
            bw, bh = self._before.width() * scale_b, self._before.height() * scale_b
            p.save()
            p.setClipRect(int(ox), int(oy), int(self._split * dw), int(dh))
            p.drawImage(int(ox), int(oy), self._before, 0, 0, self._before.width(), self._before.height())
            p.restore()

        # Divider + handle.
        dx = ox + self._split * dw
        pen = QPen(QColor("#ffffff"), 2)
        p.setPen(pen)
        p.drawLine(int(dx), int(oy), int(dx), int(oy + dh))
        p.setBrush(QColor("#ffffff"))
        cy = oy + dh / 2.0
        p.drawEllipse(dx, cy, 9, 9)

        # Corner labels.
        p.setFont(self.font())
        if self._before is not None:
            p.setPen(QColor(255, 255, 255, 210))
            p.drawText(int(ox + 6), int(oy + 18), "Before")
        if self._after is not None:
            p.setPen(QColor(255, 255, 255, 210))
            p.drawText(int(ox + dw - 46), int(oy + 18), "After")

    # -- interaction -----------------------------------------------------------

    def _drag_to(self, x: float) -> None:
        ox, _, dw, _ = self._layout()
        if dw <= 0:
            return
        self._split = max(0.0, min(1.0, (x - ox) / dw))
        self.update()

    def mousePressEvent(self, ev) -> None:  # noqa: N802
        if ev.button() == Qt.LeftButton:
            self._drag_to(ev.position().x())

    def mouseMoveEvent(self, ev) -> None:  # noqa: N802
        dx = self._divider_x()
        near = abs(ev.position().x() - dx) < 10 and (self._after is not None or self._before is not None)
        self.setCursor(Qt.SplitHorCursor if near else Qt.ArrowCursor)
        if ev.buttons() & Qt.LeftButton:
            self._drag_to(ev.position().x())

    def leaveEvent(self, ev) -> None:  # noqa: N802
        self.setCursor(Qt.ArrowCursor)


def _fmt_ts(t: float | int | None) -> str:
    if t is None:
        return "??:??"
    m, s = divmod(float(t), 60.0)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:04.1f}" if h else f"{int(m):02d}:{s:04.1f}"


class TimelineStrip(QWidget):
    """Color-coded artifact markers along the output timeline (design.md §7.3)."""

    frame_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._findings: list[dict] = []
        self._total = 1
        self._frame = -1
        self._hover_idx = -1
        self.setFixedHeight(36)

    # -- data ----------------------------------------------------------------

    def set_findings(self, findings: list[dict] | None, total_frames: int) -> None:
        self._findings = [f for f in (findings or []) if isinstance(f.get("frame_index"), int)]
        self._total = max(int(total_frames), 1)
        self.update()

    def set_frame(self, i: int) -> None:
        self._frame = i
        self.update()

    # -- geometry ------------------------------------------------------------

    def _x_of(self, frame_idx: int) -> float:
        return 4.0 + (self.width() - 8.0) * (frame_idx + 0.5) / self._total

    def _nearest(self, x: float, max_px: float = 9.0) -> int:
        best, best_d = -1, max_px
        for i, f in enumerate(self._findings):
            d = abs(x - self._x_of(f["frame_index"]))
            if d <= best_d:
                best, best_d = i, d
        return best

    # -- painting ------------------------------------------------------------

    def paintEvent(self, ev) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#101116"))
        for i, f in enumerate(self._findings):
            try:
                color = ARTIFACT_META[ArtifactType(f["artifact"])][1]
            except (KeyError, ValueError):
                color = ACCENT
            x = self._x_of(f["frame_index"])
            pen_w = 5 if i == self._hover_idx else 3
            p.setPen(QPen(QColor(color), pen_w))
            p.drawLine(int(x), 6, int(x), self.height() - 6)
        if self._frame >= 0:
            x = self._x_of(self._frame)
            p.setPen(QPen(QColor(ACCENT), 2))
            p.drawLine(int(x), 0, int(x), self.height())

    # -- interaction -----------------------------------------------------------

    def _tooltip_for(self, f: dict) -> str:
        try:
            label = ARTIFACT_META[ArtifactType(f["artifact"])][0]
        except (KeyError, ValueError):
            label = str(f.get("artifact"))
        if f.get("repaired"):
            fix = f" — repaired by {f.get('method', 'repair')}"
        else:
            fix = " — not repaired"
        return f"{label} @ {_fmt_ts(f.get('timestamp'))}{fix}"

    def mouseMoveEvent(self, ev) -> None:  # noqa: N802
        i = self._nearest(ev.position().x(), max_px=7.0)
        if i != self._hover_idx:
            self._hover_idx = i
            self.update()
        if i >= 0:
            self.setToolTip(self._tooltip_for(self._findings[i]))
            self.setCursor(Qt.PointingHandCursor)
        else:
            self.clearToolTip()
            self.setCursor(Qt.ArrowCursor)

    def mousePressEvent(self, ev) -> None:  # noqa: N802
        if ev.button() == Qt.LeftButton:
            i = self._nearest(ev.position().x(), max_px=9.0)
            if i >= 0:
                self.frame_requested.emit(int(self._findings[i]["frame_index"]))

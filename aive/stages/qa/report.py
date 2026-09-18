"""Artifact-QA report generation (design.md §7.3): JSON export + PNG summary.

The JSON is produced by :meth:`QAReport.save_json` (compact findings, no thumbnails).
The PNG is a human-readable one-pager: a per-artifact rollup table plus marked-frame
thumbnails with color-coded bounding boxes using the :data:`ARTIFACT_META` palette.
Rendering uses only cv2/numpy so it works headless in the worker process and needs
no extra dependency.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .findings import ARTIFACT_META, ArtifactType, QAReport


def _hex_to_bgr(hexstr: str) -> tuple[int, int, int]:
    h = hexstr.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def _put(img: np.ndarray, text: str, x: int, y: int, color=(235, 235, 235), scale: float = 0.5):
    import cv2

    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _fmt_metric(v: float | None) -> str:
    return "—" if v is None else f"{v:.3f}"


def render_png(report: QAReport, path: str | Path) -> Path:
    """Render the report to a PNG (table + marked-frame thumbnails)."""
    import cv2

    W = 1040
    pad = 24
    row_h = 34
    header_h = 96
    table_top = header_h + 8

    # ---- compute the rows we will draw --------------------------------------
    summary = report.summary()
    arts_with_findings = [a for a in ArtifactType.all() if a.value in summary]
    n_rows = max(1, len(arts_with_findings))

    thumbs = (report.sample_frames or [])[:8]
    thumb_w = 240
    cols = 4
    rows_needed = (len(thumbs) + cols - 1) // cols if thumbs else 0
    thumb_block_h = 0
    if thumbs:
        thumb_block_h = 60 + rows_needed * (int(thumb_w * 9 / 16) + 34)

    H = table_top + n_rows * row_h + 24 + thumb_block_h + pad
    img = np.full((H, W, 3), 24, dtype=np.uint8)  # dark background (BGR)

    # ---- header -------------------------------------------------------------
    base = os.path.basename(report.input_path) or "input"
    _put(img, f"Artifact Report — {base}", pad, 40, color=(255, 255, 255), scale=0.8)
    meta = (
        f"{report.fps:.3g} fps   ·   {report.num_frames} frames   ·   "
        f"{report.duration_s:.1f}s   ·   mode: {report.mode}"
    )
    _put(img, meta, pad, 70, color=(160, 160, 160), scale=0.5)

    # ---- table --------------------------------------------------------------
    col_x = [pad, pad + 230, pad + 340, pad + 470, pad + 600, pad + 740]
    headers = ["Artifact", "Count", "Max sev", "Avg sev", "Repaired", "Before → After"]
    hy = table_top
    for cx, ht in zip(col_x, headers):
        _put(img, ht, cx, hy + 18, color=(120, 150, 200), scale=0.45)
    cv2.line(img, (pad, hy + 30), (W - pad, hy + 30), (70, 70, 70), 1)

    y = table_top + row_h
    for art in arts_with_findings:
        s = summary[art.value]
        label, color_hex = ARTIFACT_META[art]
        swatch = _hex_to_bgr(color_hex)
        cv2.rectangle(img, (col_x[0], y + 4), (col_x[0] + 16, y + row_h - 8), swatch, -1)
        _put(img, label, col_x[0] + 26, y + 22, scale=0.5)
        _put(img, str(s["count"]), col_x[1], y + 22, scale=0.5)
        _put(img, f"{s['max_severity']:.3f}", col_x[2], y + 22, scale=0.5)
        _put(img, f"{s['avg_severity']:.3f}", col_x[3], y + 22, scale=0.5)
        _put(img, str(s["repaired_count"]), col_x[4], y + 22, scale=0.5)
        ba = f"{_fmt_metric(s['before_metric'])} → {_fmt_metric(s['after_metric'])}"
        _put(img, ba, col_x[5], y + 22, scale=0.5)
        cv2.line(img, (pad, y + row_h - 4), (W - pad, y + row_h - 4), (48, 48, 48), 1)
        y += row_h

    # ---- marked-frame thumbnails --------------------------------------------
    if thumbs:
        ty = y + 20
        _put(img, "Flagged frames", pad, ty + 16, color=(120, 150, 200), scale=0.5)
        ty += 40
        for i, t in enumerate(thumbs):
            r, c = divmod(i, cols)
            x0 = pad + c * (thumb_w + 8)
            y0 = ty + r * (int(thumb_w * 9 / 16) + 34)
            thumb = t.get("thumb")
            th_h = int(thumb_w * 9 / 16)
            if thumb is not None and isinstance(thumb, np.ndarray):
                thh, tw = thumb.shape[:2]
                small = cv2.resize(thumb, (thumb_w, max(1, int(thumb_w * thh / max(1, tw)))), interpolation=cv2.INTER_AREA)
            else:
                small = np.full((th_h, thumb_w, 3), 40, dtype=np.uint8)
            img[y0 : y0 + small.shape[0], x0 : x0 + thumb_w] = small
            # bbox overlay in the artifact color (bbox is full-res; scale into thumb)
            art = ArtifactType(t["artifact"]) if isinstance(t.get("artifact"), str) else t.get("artifact")
            col = _hex_to_bgr(ARTIFACT_META[art][1]) if art in ARTIFACT_META else (255, 80, 80)
            bx, by, bw, bh = t.get("bbox", (0, 0, 0, 0))
            fw, fh = int(t.get("fw", thumb_w)), int(t.get("fh", small.shape[0]))
            if bw > 0 and bh > 0:
                sx, sy = thumb_w / max(1, fw), small.shape[0] / max(1, fh)
                cv2.rectangle(img, (x0 + int(bx * sx), y0 + int(by * sy)),
                              (x0 + int((bx + bw) * sx), y0 + int((by + bh) * sy)), col, 2)
            ts = report.frame_timestamp(t.get("frame_index", 0))
            _put(img, f"{ARTIFACT_META[art][0]} @ {ts}", x0, y0 + small.shape[0] + 18, scale=0.4)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img)
    return p


def save_report(report: QAReport, out_dir: str | Path, base_name: str = "artifact-report") -> dict:
    """Write both the JSON and PNG for a report; returns their paths."""
    out_dir = Path(out_dir)
    json_path = report.save_json(out_dir / f"{base_name}.json")
    png_path = render_png(report, out_dir / f"{base_name}.png")
    return {"json": str(json_path), "png": str(png_path)}

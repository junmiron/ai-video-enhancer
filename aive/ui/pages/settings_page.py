"""Settings page (design.md §7.2).

Presets drive all defaults; every field stays overridable. The page is a plain
dict <-> widgets bridge: ``load(Settings)`` populates the controls,
``collect()`` returns a dict of Settings fields. Model lists come straight from
the registry manifest (pure JSON — no worker round-trip needed); cached-weight
badges refresh when the worker reports download results.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...models.registry import list_models
from ..estimates import estimate_output_bytes, human_bytes
from ..presets import PRESETS, PRESET_ORDER
from ..theme import DIM

# (settings field suffix, display label) — order matches design.md §7.2 sliders.
ARTIFACTS: list[tuple[str, str]] = [
    ("flicker", "Flicker"),
    ("ghosting", "Ghosting"),
    ("banding", "Banding"),
    ("halo", "Halo / ringing"),
    ("lineart", "Line-art"),
    ("color_drift", "Color drift"),
    ("regression", "Regression frame"),
]

RIFE_QUALITY = {"Lite": "rife_425_lite", "Balanced": "rife_425", "Heavy": "rife_425_heavy"}
_RIFE_NAMES = set(RIFE_QUALITY.values())


def _group(title: str) -> tuple[QGroupBox, QVBoxLayout]:
    g = QGroupBox(title)
    lay = QVBoxLayout(g)
    return g, lay


class SettingsPage(QWidget):
    settings_changed = Signal()   # any control changed (main window collects + saves)
    preset_changed = Signal(str)  # preset combo picked a different key

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._loading = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(10)

        # -- Preset -----------------------------------------------------------
        preset_row = QHBoxLayout()
        plabel = QLabel("Preset:")
        self.preset_combo = QComboBox()
        for key in PRESET_ORDER:
            self.preset_combo.addItem(PRESETS[key]["label"], key)
        self.preset_desc = QLabel("")
        self.preset_desc.setProperty("role", "dim")
        self.preset_desc.setWordWrap(True)
        preset_row.addWidget(plabel)
        preset_row.addStretch(1)
        preset_row.addWidget(self.preset_combo, 0)
        lay.addLayout(preset_row)
        lay.addWidget(self.preset_desc)

        # -- Upscale ----------------------------------------------------------
        g, gl = _group("Upscale")
        self.sr_cb = QCheckBox("Enable AI upscaling")
        self.sr_combo = QComboBox()
        self.scale_combo = QComboBox()
        for s in (2, 3, 4):
            self.scale_combo.addItem(f"{s}×", s)
        row1 = QHBoxLayout()
        row1.addWidget(self.sr_cb)
        row1.addStretch(1)
        lay.addLayout(row1)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Model:"))
        row2.addWidget(self.sr_combo, 1)
        row2.addWidget(QLabel("Scale:"))
        row2.addWidget(self.scale_combo)
        gl.addLayout(row2)
        lay.addWidget(g)

        # -- Frame interpolation ----------------------------------------------
        g, gl = _group("Frame interpolation")
        self.fi_cb = QCheckBox("Enable frame interpolation")
        self.fi_mode_combo = QComboBox()
        self.fi_mode_combo.addItem("Target FPS", "fps")
        self.fi_mode_combo.addItem("Multiplier", "mult")
        self.fps_combo = QComboBox()
        for f in (24, 30, 50, 60, 120):
            self.fps_combo.addItem(f"{f} fps", float(f))
        self.mult_combo = QComboBox()
        for m in (2, 3, 4):
            self.mult_combo.addItem(f"×{m}", int(m))
        self.fi_model_combo = QComboBox()
        self.quality_combo = QComboBox()
        for q in RIFE_QUALITY:
            self.quality_combo.addItem(q)

        r1 = QHBoxLayout()
        r1.addWidget(self.fi_cb)
        r1.addStretch(1)
        gl.addLayout(r1)
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Mode:"))
        r2.addWidget(self.fi_mode_combo)
        r2.addWidget(QLabel("Target FPS:"))
        r2.addWidget(self.fps_combo)
        r2.addWidget(QLabel("Multiplier:"))
        r2.addWidget(self.mult_combo)
        gl.addLayout(r2)
        r3 = QHBoxLayout()
        r3.addWidget(QLabel("Model:"))
        r3.addWidget(self.fi_model_combo, 1)
        r3.addWidget(QLabel("Quality:"))
        r3.addWidget(self.quality_combo)
        gl.addLayout(r3)
        self.dedup_spin = QDoubleSpinBox()
        self.dedup_spin.setRange(0.0, 0.2)
        self.dedup_spin.setSingleStep(0.005)
        self.dedup_spin.setDecimals(3)
        dedup_row = QHBoxLayout()
        dedup_row.addWidget(QLabel("De-duplicate held frames (threshold):"))
        dedup_row.addWidget(self.dedup_spin)
        dedup_row.addStretch(1)
        gl.addLayout(dedup_row)
        lay.addWidget(g)

        # -- Artifact handling --------------------------------------------------
        g, gl = _group("Artifact detection & repair")
        self.qa_cb = QCheckBox("Enable artifact QA stage")
        r0 = QHBoxLayout()
        r0.addWidget(self.qa_cb)
        r0.addStretch(1)
        gl.addLayout(r0)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.mode_auto = QRadioButton("Auto (detect + fix)")
        self.mode_detect = QRadioButton("Detect only")
        self.mode_manual = QRadioButton("Manual")
        for rb in (self.mode_auto, self.mode_detect, self.mode_manual):
            mode_row.addWidget(rb)
        mode_row.addStretch(1)
        gl.addLayout(mode_row)

        sens_grid = QHBoxLayout()
        self._sens_sliders: dict[str, tuple[QSlider, QLabel]] = {}
        for key, label in ARTIFACTS:
            col = QVBoxLayout()
            head = QHBoxLayout()
            t = QLabel(label)
            val = QLabel("50%")
            val.setProperty("role", "dim")
            head.addWidget(t)
            head.addStretch(1)
            head.addWidget(val)
            sl = QSlider()
            sl.setRange(0, 100)
            sl.setValue(50)
            col.addLayout(head)
            col.addWidget(sl)
            self._sens_sliders[key] = (sl, val)
            sens_grid.addLayout(col)
        gl.addLayout(sens_grid)

        manual_row = QHBoxLayout()
        manual_row.addWidget(QLabel("Manual mode — active artifacts:"))
        self._manual_checks: dict[str, QCheckBox] = {}
        for key, label in ARTIFACTS:
            cb = QCheckBox(label)
            cb.setChecked(True)
            self._manual_checks[key] = cb
            manual_row.addWidget(cb)
        gl.addLayout(manual_row)

        r4 = QHBoxLayout()
        self.ai_cb = QCheckBox("AI-model repairs (SCUNet / deepDeband-f)")
        self.report_cb = QCheckBox("Show artifact report when done")
        r4.addWidget(self.ai_cb)
        r4.addStretch(1)
        r4.addWidget(self.report_cb)
        gl.addLayout(r4)
        lay.addWidget(g)

        # -- Depth map ----------------------------------------------------------
        g, gl = _group("Depth map (optional)")
        self.depth_cb = QCheckBox("Generate depth map alongside the enhanced video")
        drow = QHBoxLayout()
        drow.addWidget(self.depth_cb)
        drow.addStretch(1)
        gl.addLayout(drow)
        drow2 = QHBoxLayout()
        drow2.addWidget(QLabel("Model:"))
        self.depth_combo = QComboBox()
        drow2.addWidget(self.depth_combo, 1)
        drow2.addWidget(QLabel("Output:"))
        self.depth_out_combo = QComboBox()
        self.depth_out_combo.addItem("10-bit MP4", "mp4_10bit")
        self.depth_out_combo.addItem("PNG sequence (16-bit)", "png16")
        drow2.addWidget(self.depth_out_combo)
        gl.addLayout(drow2)
        lay.addWidget(g)

        # -- Output ---------------------------------------------------------------
        g, gl = _group("Output")
        orow = QHBoxLayout()
        orow.addWidget(QLabel("Container:"))
        self.container_combo = QComboBox()
        self.container_combo.addItem("MP4", "mp4")
        self.container_combo.addItem("MKV", "mkv")
        orow.addWidget(self.container_combo)
        orow.addStretch(1)
        gl.addLayout(orow)
        crow = QHBoxLayout()
        crow.addWidget(QLabel("Codec:"))
        self.codec_combo = QComboBox()
        self.codec_combo.addItem("H.265 10-bit (x265, anime profile)", "x265-10bit-anime")
        self.codec_combo.addItem("H.265 10-bit (NVENC — fast)", "nvenc-hevc-10bit")
        self.codec_combo.addItem("H.264 (compatibility)", "h264")
        crow.addWidget(self.codec_combo, 1)
        gl.addLayout(crow)
        qrow = QHBoxLayout()
        qrow.addWidget(QLabel("Quality (CRF):"))
        self.crf_spin = QSpinBox()
        self.crf_spin.setRange(0, 51)
        qrow.addWidget(self.crf_spin)
        self.estimate_label = QLabel("")
        self.estimate_label.setProperty("role", "dim")
        qrow.addStretch(1)
        qrow.addWidget(self.estimate_label)
        gl.addLayout(qrow)
        self.audio_cb = QCheckBox("Audio: passthrough (no re-encode)")
        gl.addWidget(self.audio_cb)
        lay.addWidget(g)

        # -- Advanced ---------------------------------------------------------------
        g, gl = _group("Advanced")
        self.scene_cb = QCheckBox("Scene-cut detection (never interpolate across cuts)")
        self.keep_cb = QCheckBox("Keep lossless workspace chunks after success")
        gl.addWidget(self.scene_cb)
        gl.addWidget(self.keep_cb)
        lay.addWidget(g)

        lay.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll)

        # -- model lists (registry is pure JSON — populate synchronously) --------
        self._fill_model_combo(self.sr_combo, "sr")
        self._fill_model_combo(self.fi_model_combo, "fi")
        self._fill_model_combo(self.depth_combo, "depth")

        # -- wiring -----------------------------------------------------------------
        self.preset_combo.currentIndexChanged.connect(self._on_preset)
        self.fi_mode_combo.currentIndexChanged.connect(lambda: self._sync_fi())
        self.fi_model_combo.currentIndexChanged.connect(lambda: self._sync_quality())
        for w in (
            self.sr_combo, self.scale_combo, self.fps_combo, self.mult_combo,
            self.quality_combo, self.depth_combo, self.depth_out_combo,
            self.container_combo, self.codec_combo,
        ):
            w.currentIndexChanged.connect(lambda _i: self._on_change())
        for w in (self.dedup_spin, self.crf_spin):
            w.valueChanged.connect(self._on_change)
        for w in (
            self.sr_cb, self.fi_cb, self.qa_cb, self.ai_cb, self.report_cb,
            self.depth_cb, self.audio_cb, self.scene_cb, self.keep_cb,
        ):
            w.toggled.connect(self._on_change)
        for rb in (self.mode_auto, self.mode_detect, self.mode_manual):
            rb.toggled.connect(self._on_change)
        for sl, val in self._sens_sliders.values():
            sl.valueChanged.connect(
                lambda v, lab=val: (lab.setText(f"{v}%"), self._on_change())
            )
        for cb in self._manual_checks.values():
            cb.toggled.connect(self._on_change)

    # -- model combos ---------------------------------------------------------

    def _fill_model_combo(self, combo: QComboBox, task: str) -> None:
        for m in list_models(task=task):
            idx = combo.count()
            combo.addItem(m.name, m.name)
            tip = []
            if m.recommended_for:
                tip.append(m.recommended_for)
            if m.license:
                tip.append(f"License: {m.license}")
            if m.experimental:
                tip.append("Experimental")
            if m.notes:
                tip.append(m.notes)
            combo.setItemData(idx, "\n".join(tip), Qt.ItemDataRole.ToolTipRole)

    def refresh_models(self, rows: list[dict]) -> None:
        """Update '✓ ready' badges from worker models_listed/download_done events."""
        cached = {r["name"]: bool(r.get("cached")) for r in rows}
        for combo in (self.sr_combo, self.fi_model_combo, self.depth_combo):
            for i in range(combo.count()):
                name = combo.itemData(i)
                if name in cached:
                    suffix = " ✓" if cached[name] else ""
                    cur = combo.itemText(i).rstrip(" ✓")
                    combo.setItemText(i, f"{cur}{suffix}")

    @staticmethod
    def _set_combo_by_data(combo: QComboBox, value) -> None:
        for i in range(combo.count()):
            if combo.itemData(i) == value:
                combo.setCurrentIndex(i)
                return

    # -- sync helpers -----------------------------------------------------------

    def _sync_fi(self) -> None:
        is_fps = self.fi_mode_combo.currentData() == "fps"
        self.fps_combo.setEnabled(is_fps)
        self.mult_combo.setEnabled(not is_fps)
        self._on_change()

    def _sync_quality(self) -> None:
        model = self.fi_model_combo.currentData()
        rife = model in _RIFE_NAMES
        self.quality_combo.setEnabled(rife)
        if rife:
            for i, name in enumerate(RIFE_QUALITY.values()):
                if name == model:
                    self.quality_combo.setCurrentIndex(i)
                    break

    def current_preset_key(self) -> str:
        return PRESET_ORDER[self.preset_combo.currentIndex()]

    # -- load / collect -----------------------------------------------------------

    def load(self, s) -> None:  # s: aive.config.Settings
        self._loading = True
        try:
            # Presets are not persisted as a Settings field — the combo keeps its
            # current selection (Anime by default) and re-applies on demand.
            self._update_preset_desc()

            self.sr_cb.setChecked(bool(s.sr_enabled))
            self._set_combo_by_data(self.sr_combo, s.sr_model)
            for i in range(self.scale_combo.count()):
                if int(self.scale_combo.itemData(i)) == (int(s.sr_scale) if s.sr_scale else 2):
                    self.scale_combo.setCurrentIndex(i)

            self.fi_cb.setChecked(bool(s.fi_enabled))
            self._set_combo_by_data(self.fi_model_combo, s.fi_model)
            if s.fi_target_fps:
                self.fi_mode_combo.setCurrentIndex(0)
                for i in range(self.fps_combo.count()):
                    if abs(float(self.fps_combo.itemData(i)) - float(s.fi_target_fps)) < 0.01:
                        self.fps_combo.setCurrentIndex(i)
            else:
                self.fi_mode_combo.setCurrentIndex(1)
                m = int(s.fi_multiplier or 2)
                for i in range(self.mult_combo.count()):
                    if int(self.mult_combo.itemData(i)) == m:
                        self.mult_combo.setCurrentIndex(i)
            self.dedup_spin.setValue(float(s.fi_dedup_threshold))

            self.qa_cb.setChecked(bool(s.qa_enabled))
            mode = s.qa_mode or "auto"
            for rb, key in ((self.mode_auto, "auto"), (self.mode_detect, "detect_only"), (self.mode_manual, "manual")):
                rb.setChecked(mode == key)
            self._sync_manual()
            for key, _label in ARTIFACTS:
                v = int(round(getattr(s, f"qa_sens_{key}") * 100))
                sl, val = self._sens_sliders[key]
                sl.setValue(v)
                val.setText(f"{v}%")
            manual_set = set(s.qa_manual_artifacts or [])
            for key, cb in self._manual_checks.items():
                cb.setChecked(key in manual_set)
            self.ai_cb.setChecked(bool(s.qa_ai_repairs))
            self.report_cb.setChecked(bool(s.qa_show_report))

            self.depth_cb.setChecked(bool(s.depth_enabled))
            self._set_combo_by_data(self.depth_combo, s.depth_model)
            for i in range(self.depth_out_combo.count()):
                if self.depth_out_combo.itemData(i) == s.depth_output:
                    self.depth_out_combo.setCurrentIndex(i)

            self.container_combo.setCurrentIndex(0 if s.container == "mp4" else 1)
            self._set_combo_by_data(self.codec_combo, s.output_codec)
            self.crf_spin.setValue(int(s.crf))
            self.audio_cb.setChecked(bool(s.audio_passthrough))
            self.scene_cb.setChecked(bool(s.scene_split))
            self.keep_cb.setChecked(bool(s.keep_workspace))

            self._sync_fi()
            self._sync_quality()
        finally:
            self._loading = False
        self._update_estimate()

    def collect(self) -> dict:
        """Current control values as a dict of Settings fields."""
        is_fps = self.fi_mode_combo.currentData() == "fps"
        fi_model = (
            RIFE_QUALITY[self.quality_combo.currentText()]
            if self.quality_combo.isEnabled()
            else self.fi_model_combo.currentData()
        )
        return {
            "output_codec": self.codec_combo.currentData(),
            "crf": int(self.crf_spin.value()),
            "container": self.container_combo.currentData(),
            "audio_passthrough": bool(self.audio_cb.isChecked()),
            "scene_split": bool(self.scene_cb.isChecked()),
            "keep_workspace": bool(self.keep_cb.isChecked()),
            "fi_enabled": bool(self.fi_cb.isChecked()),
            "fi_model": fi_model,
            "fi_target_fps": float(self.fps_combo.currentData()) if is_fps else None,
            "fi_multiplier": int(self.mult_combo.currentData()) if not is_fps else None,
            "fi_dedup_threshold": float(self.dedup_spin.value()),
            "sr_enabled": bool(self.sr_cb.isChecked()),
            "sr_model": self.sr_combo.currentData(),
            "sr_scale": int(self.scale_combo.currentData()),
            "qa_enabled": bool(self.qa_cb.isChecked()),
            "qa_mode": (
                "auto" if self.mode_auto.isChecked()
                else "detect_only" if self.mode_detect.isChecked()
                else "manual"
            ),
            **{f"qa_sens_{k}": sl.value() / 100.0 for k, (sl, _v) in self._sens_sliders.items()},
            "qa_manual_artifacts": [k for k, cb in self._manual_checks.items() if cb.isChecked()],
            "qa_ai_repairs": bool(self.ai_cb.isChecked()),
            "qa_show_report": bool(self.report_cb.isChecked()),
            "depth_enabled": bool(self.depth_cb.isChecked()),
            "depth_model": self.depth_combo.currentData(),
            "depth_output": self.depth_out_combo.currentData(),
        }

    # -- source info / estimate ------------------------------------------------------

    def set_source_info(self, info) -> None:  # MediaInfo | None
        self._src = info
        self._update_estimate()

    def _update_estimate(self) -> None:
        info = getattr(self, "_src", None)
        if info is None:
            self.estimate_label.setText("Est. size: — (add a video)")
            return
        d = self.collect()
        scale = int(d["sr_scale"]) if d["sr_enabled"] else 1
        out_w, out_h = info.width * scale, info.height * scale
        n = estimate_output_bytes(
            src_width=info.width,
            src_height=info.height,
            src_fps=info.fps_float,
            num_frames=max(info.num_frames, 1),
            sr_enabled=d["sr_enabled"],
            sr_scale=int(d["sr_scale"]),
            fi_enabled=d["fi_enabled"],
            fi_target_fps=d["fi_target_fps"],
            fi_multiplier=d["fi_multiplier"],
            codec=d["output_codec"],
            crf=int(d["crf"]),
        )
        from ..estimates import fi_ratio

        ratio = fi_ratio(d["fi_enabled"], d["fi_target_fps"], d["fi_multiplier"], info.fps_float)
        out_fps = info.fps_float * ratio
        self.estimate_label.setText(
            f"Est. size: ~{human_bytes(n)} · {out_w}×{out_h} @ {out_fps:g} fps"
        )

    # -- change plumbing ---------------------------------------------------------------

    def _on_preset(self, index: int) -> None:
        if self._loading:
            return
        key = self.preset_combo.itemData(index)
        self._update_preset_desc()
        self.preset_changed.emit(key)

    def _update_preset_desc(self) -> None:
        key = self.current_preset_key()
        self.preset_desc.setText(PRESETS.get(key, {}).get("description", ""))

    def _sync_manual(self) -> None:
        manual = self.mode_manual.isChecked()
        for cb in self._manual_checks.values():
            cb.setEnabled(manual)

    def _on_change(self) -> None:
        if self._loading:
            return
        self._sync_manual()
        self._update_estimate()
        self.settings_changed.emit()

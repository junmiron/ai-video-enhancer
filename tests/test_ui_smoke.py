"""Offscreen smoke tests for the M6 GUI (design.md §7).

Runs with QT_QPA_PLATFORM=offscreen and builds the real MainWindow (which also
spawns a real worker process — kept module-scoped so only one pair of windows /
workers is created per test run). Covers navigation, settings persistence,
report population, compare-widget rendering and page screenshots.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="module")
def window(qapp, tmp_path_factory):
    from aive.ui.main_window import MainWindow

    settings_file = str(tmp_path_factory.mktemp("settings") / "settings.json")
    win = MainWindow(settings_path=settings_file)
    yield win
    win.close()


def test_pages_navigate(window, qapp):
    for i in range(5):
        window.nav.setCurrentRow(i)
        qapp.processEvents()
        assert window.stack.currentIndex() == i


def test_settings_persist_across_restart(window, qapp):
    from aive.config import load_settings
    from aive.ui.main_window import MainWindow

    # Change CRF through the normal page -> collect -> save path.
    window.settings_page.crf_spin.setValue(21)
    window.on_settings_changed()
    assert load_settings(window._settings_path).crf == 21

    # A second window on the same file sees the persisted value (and the widget).
    win2 = MainWindow(settings_path=window._settings_path)
    try:
        assert win2.settings.crf == 21
        assert win2.settings_page.crf_spin.value() == 21
    finally:
        win2.close()


def test_preset_apply_round_trip(window):
    from aive.config import Settings

    window.on_preset_changed("fast_preview")
    s = window.settings
    assert s.output_codec == "nvenc-hevc-10bit"
    assert s.fi_enabled is False
    assert s.qa_mode == "detect_only"
    # The page reflects the applied preset.
    assert window.settings_page.codec_combo.currentData() == "nvenc-hevc-10bit"
    # Back to anime defaults.
    window.on_preset_changed("anime")
    assert window.settings.output_codec == "x265-10bit-anime"
    assert window.settings.fi_enabled is True


def test_report_table_population(window):
    report = {
        "input_path": "in.mp4",
        "output_path": "out.mp4",
        "fps": "30/1",
        "num_frames": 90,
        "duration_s": 3.0,
        "mode": "auto",
        "summary": {
            "flicker": {"label": "Flicker", "count": 5, "max_severity": 0.62,
                        "repaired_count": 4, "before_metric": 2.3, "after_metric": 0.6},
            "banding": {"label": "Banding", "count": 2, "max_severity": 0.21,
                        "repaired_count": 2, "before_metric": 5.0, "after_metric": 0.4},
        },
        "findings": [
            {"artifact": "flicker", "frame_index": 3, "timestamp": 0.1, "severity": 0.6,
             "repaired": True, "method": "scunet-denoise"},
            {"artifact": "banding", "frame_index": 7, "timestamp": 0.23, "severity": 0.2,
             "repaired": True, "method": "deepdeband-f"},
        ],
    }
    window.report_page.set_report(report, None, "in.mp4")
    t = window.report_page.table
    assert t.rowCount() == 2
    assert t.item(0, 0).text() == "Flicker"
    assert t.item(0, 1).text() == "5"
    assert t.item(0, 3).text() == "scunet-denoise"
    assert window.report_page.export_json_btn.isEnabled()
    # No PNG files -> export disabled.
    assert not window.report_page.export_png_btn.isEnabled()


def test_compare_widget_paint(window, qapp, tmp_path):
    cw = window.preview_page.compare
    before = (np.random.rand(180, 320, 3) * 255).astype("uint8")
    after = (np.random.rand(360, 640, 3) * 255).astype("uint8")
    cw.set_images(before=before, after=after)
    qapp.processEvents()
    assert window.grab().save(str(tmp_path / "compare.png"))


def test_page_screenshots(window, qapp, tmp_path):
    for i, name in enumerate(["home", "settings", "processing", "report", "preview"]):
        window.nav.setCurrentRow(i)
        qapp.processEvents()
        png = tmp_path / f"page_{name}.png"
        assert window.grab().save(str(png))

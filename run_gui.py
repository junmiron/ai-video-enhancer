"""AI Video Enhancer — GUI entry point (design.md §7).

Run with::

    python run_gui.py

All imports happen inside ``main()`` under the ``__main__`` guard: on Windows
the multiprocessing "spawn" start method re-imports this module in the worker
child process, and a top-level Qt import would create a second QApplication
there.
"""
from __future__ import annotations

import sys


def main() -> int:
    # Windows frozen-spawn contract (CPython multiprocessing.freeze_support):
    # worker children are launched as `exe --multiprocessing-fork ...` and must run
    # spawn_main instead of this script. Without this guard every child re-runs the
    # whole app and spawns its own worker — an unbounded process chain.
    import multiprocessing as mp
    import os

    mp.freeze_support()

    if getattr(sys, "frozen", False):
        # Packaged build (M7 B): Qt plugins ship inside the bundle at
        # <bundle>/PySide6/Qt/plugins — point Qt there before QApplication starts.
        from pathlib import Path

        meipass = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        qt_plugins = meipass / "PySide6" / "Qt" / "plugins"
        if qt_plugins.is_dir():
            os.environ.setdefault("QT_PLUGIN_PATH", str(qt_plugins))

    from PySide6.QtWidgets import QApplication

    from aive.config import APP_DISPLAY
    from aive.logutil import setup_logging
    from aive.ui.main_window import MainWindow
    from aive.ui.theme import apply_theme

    setup_logging("INFO")
    app = QApplication(sys.argv)
    app.setApplicationName(APP_DISPLAY)
    apply_theme(app)
    win = MainWindow()
    win.show()

    # Diagnostic (M7 B): AIVE_SMOKE_SHOT=<dir> -> offscreen screenshots of Home and
    # Settings (with the update check exercised), then quit. Used by packaging smoke tests.
    shot_dir = os.environ.get("AIVE_SMOKE_SHOT")
    if shot_dir:
        from pathlib import Path as _P

        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QScrollArea

        def _snap_settings() -> None:
            win.nav.setCurrentRow(1)  # Settings page (About group lives at the bottom)
            sp = win.settings_page
            for sa in sp.findChildren(QScrollArea):
                sb = sa.verticalScrollBar()
                sb.setValue(sb.maximum())
            try:  # exercise the update-check path (no server configured -> friendly message)
                sp._check_updates()
            except Exception as exc:
                print(f"smoke: update check failed: {exc!r}", file=sys.stderr)

        def _snap_home() -> None:
            try:
                win.grab().save(str(_P(shot_dir) / "home.png"))
                QTimer.singleShot(4000, _snap_settings)
                QTimer.singleShot(6500, lambda: (win.grab().save(str(_P(shot_dir) / "settings.png")), app.quit()))
            except Exception as exc:
                print(f"smoke shot failed: {exc!r}", file=sys.stderr)
                app.exit(1)

        QTimer.singleShot(20_000, _snap_home)

    return int(app.exec())


if __name__ == "__main__":
    sys.exit(main())

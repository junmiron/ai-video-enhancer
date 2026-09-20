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
    return int(app.exec())


if __name__ == "__main__":
    sys.exit(main())

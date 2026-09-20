"""PySide6 UI package (M6).

Importing this package must stay cheap: Qt is imported by the submodules that
need it, never here. The worker process (:mod:`aive.ui.worker`) imports only
the pipeline core — no Qt at all.
"""

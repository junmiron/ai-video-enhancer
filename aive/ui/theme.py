"""Dark theme for the M6 GUI (design.md §7: dark, minimal jargon).

A single QSS string plus the palette constants used by widgets that paint
themselves (compare view, timeline strip). No Qt imports beyond what a
stylesheet needs — this module is safe to import from tests.
"""
from __future__ import annotations

# Palette -----------------------------------------------------------------
BG = "#17181d"          # window background
PANEL = "#20222a"       # panels / group boxes
PANEL_2 = "#262933"     # raised elements (inputs, headers)
BORDER = "#343846"
TEXT = "#e9eaef"
DIM = "#9aa0b0"
ACCENT = "#4cc9f0"      # primary action / progress
ACCENT_DARK = "#1f5d75"
OK = "#57d98a"
WARN = "#f5a623"
ERR = "#ef476f"

DARK_QSS = f"""
* {{
    outline: 0;
}}
QMainWindow, QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 13px;
}}
QLabel[role="dim"] {{ color: {DIM}; }}
QLabel[role="big"] {{ font-size: 20px; font-weight: 600; }}
QLabel[role="banner"] {{
    background-color: {PANEL_2};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 10px 14px;
}}
QLabel[role="banner"][tone="warn"] {{ color: {WARN}; }}
QLabel[role="banner"][tone="err"]  {{ color: {ERR}; }}

/* Drop zone (home page) */
QFrame#dropZone {{
    background-color: {PANEL};
    border: 2px dashed {BORDER};
    border-radius: 10px;
}}
QFrame#dropZone[active="true"] {{
    border: 2px solid {ACCENT};
    background-color: #1d2733;
}}

/* Buttons */
QPushButton {{
    background-color: {PANEL_2};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 7px 14px;
}}
QPushButton:hover {{ background-color: #2e3140; }}
QPushButton:pressed {{ background-color: #23252f; }}
QPushButton:disabled {{ color: {DIM}; background-color: {PANEL}; border-color: {BORDER}; }}
QPushButton[accent="true"] {{
    background-color: {ACCENT_DARK};
    border: 1px solid {ACCENT};
    color: #ffffff;
    font-weight: 600;
}}
QPushButton[accent="true"]:hover {{ background-color: #27708f; }}
QPushButton[accent="true"]:disabled {{ background-color: {PANEL_2}; border-color: {BORDER}; color: {DIM}; }}

/* Inputs */
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background-color: {PANEL_2};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 4px 8px;
    selection-background-color: {ACCENT_DARK};
}}
QComboBox::drop-down {{ border: 0; width: 22px; }}
QComboBox QAbstractItemView, QListView {{
    background-color: {PANEL_2};
    color: {TEXT};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT_DARK};
}}

/* Sliders */
QSlider::groove:horizontal {{ height: 4px; background: {PANEL_2}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 14px; height: 14px; margin: -6px 0;
    border-radius: 7px; background: {TEXT};
}}

/* Progress bars */
QProgressBar {{
    background-color: {PANEL_2};
    border: 1px solid {BORDER};
    border-radius: 5px;
    text-align: center;
    color: {TEXT};
    min-height: 18px;
}}
QProgressBar::chunk {{ background-color: {ACCENT}; border-radius: 4px; }}

/* Tables */
QTableWidget, QTableView {{
    background-color: {PANEL};
    alternate-background-color: #23252e;
    gridline-color: {BORDER};
    border: 1px solid {BORDER};
    border-radius: 6px;
}}
QHeaderView::section {{
    background-color: {PANEL_2};
    color: {DIM};
    border: 0;
    padding: 6px 8px;
    font-weight: 600;
}}
QTableWidget::item:selected, QTableView::item:selected {{
    background-color: {ACCENT_DARK};
    color: #ffffff;
}}

/* Group boxes */
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 6px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {DIM};
}}

/* Left navigation */
QListWidget#nav {{
    background-color: transparent;
    border: 0;
    font-size: 14px;
}}
QListWidget#nav::item {{
    padding: 12px 16px;
    border-radius: 8px;
    color: {DIM};
}}
QListWidget#nav::item:hover {{ background-color: {PANEL_2}; }}
QListWidget#nav::item:selected {{ background-color: {ACCENT_DARK}; color: #ffffff; font-weight: 600; }}

/* Log view */
QPlainTextEdit {{
    background-color: #101116;
    border: 1px solid {BORDER};
    border-radius: 6px;
    font-family: "Cascadia Mono", Consolas, monospace;
    font-size: 12px;
}}

/* Scrollbars */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {BORDER}; border-radius: 5px; min-width: 30px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* Tooltips / status bar */
QToolTip {{
    background-color: {PANEL_2};
    color: {TEXT};
    border: 1px solid {BORDER};
    padding: 4px 8px;
}}
QStatusBar {{ background-color: {PANEL}; color: {DIM}; }}
"""


def apply_theme(app) -> None:
    """Apply the dark stylesheet to a QApplication."""
    app.setStyleSheet(DARK_QSS)

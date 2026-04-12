from __future__ import annotations

from typing import Dict

from PySide6 import QtGui


_LIGHT_THEME = {
    "window_bg": "#f4f7fb",
    "window_elevated": "#ffffff",
    "window_alt": "#e9eff8",
    "surface": "#ffffff",
    "surface_alt": "#f1f5fb",
    "surface_muted": "#f7f9fd",
    "surface_raised": "#fbfdff",
    "border": "#d4dfec",
    "border_soft": "#e3ebf4",
    "border_focus": "#5b8ef2",
    "text": "#1a2433",
    "text_muted": "#6a778d",
    "text_soft": "#41506a",
    "text_heading": "#0f2747",
    "text_invert": "#ffffff",
    "primary": "#215fcf",
    "primary_hover": "#3471de",
    "primary_pressed": "#174fac",
    "primary_disabled": "#9cb8e8",
    "primary_tint": "#eaf1ff",
    "primary_soft": "#bfd3fb",
    "selection_bg": "#2b6ee8",
    "selection_text": "#ffffff",
    "scroll_bg": "#ebf1f8",
    "scroll_handle": "#c4d3e9",
    "progress_bg": "#e7eef7",
    "progress_chunk": "#2f73e6",
    "splitter": "#dde7f3",
    "help_bg": "#f8fbff",
    "help_hover_bg": "#ecf3ff",
    "help_border": "#c7d7ef",
    "help_hover_border": "#8cb0ec",
    "dropzone_bg": "#f8fbff",
    "dropzone_border": "#c7d9f5",
    "menu_hover": "#ebf2fc",
}

_DARK_THEME = {
    "window_bg": "#0f1520",
    "window_elevated": "#151d29",
    "window_alt": "#1a2331",
    "surface": "#18212d",
    "surface_alt": "#1f2a38",
    "surface_muted": "#131b26",
    "surface_raised": "#1c2634",
    "border": "#2a3748",
    "border_soft": "#36465a",
    "border_focus": "#6aa2ff",
    "text": "#e6edf6",
    "text_muted": "#97a9bf",
    "text_soft": "#c2cfdf",
    "text_heading": "#f3f7ff",
    "text_invert": "#f8fbff",
    "primary": "#3c86ff",
    "primary_hover": "#5e9cff",
    "primary_pressed": "#2b70df",
    "primary_disabled": "#4b5f7c",
    "primary_tint": "#182846",
    "primary_soft": "#355786",
    "selection_bg": "#2f71e8",
    "selection_text": "#ffffff",
    "scroll_bg": "#141d29",
    "scroll_handle": "#42566f",
    "progress_bg": "#16202d",
    "progress_chunk": "#4b96ff",
    "splitter": "#243244",
    "help_bg": "#1a2532",
    "help_hover_bg": "#223244",
    "help_border": "#3a4b61",
    "help_hover_border": "#6c89af",
    "dropzone_bg": "#141d28",
    "dropzone_border": "#345277",
    "menu_hover": "#223042",
}


_APP_QSS_TEMPLATE = """
QWidget {
    color: %(text)s;
    font-family: "Segoe UI";
    font-size: 14px;
}
QMainWindow, QDialog {
    background: %(window_bg)s;
}
QMenuBar {
    background: transparent;
    border: none;
    spacing: 6px;
    padding: 8px 14px 2px 14px;
}
QMenuBar::item {
    padding: 8px 12px;
    border-radius: 10px;
    background: transparent;
    color: %(text_soft)s;
    font-weight: 600;
}
QMenuBar::item:selected {
    background: %(menu_hover)s;
    color: %(text_heading)s;
}
QMenuBar::item:pressed {
    background: %(surface_alt)s;
}
QStatusBar {
    background: transparent;
    color: %(text_muted)s;
    border-top: 1px solid %(border_soft)s;
    padding: 4px 10px 6px 10px;
}
QToolBar {
    background: transparent;
    border: none;
    border-bottom: 1px solid %(border_soft)s;
    spacing: 10px;
    padding: 8px 16px 12px 16px;
}
QToolBar QLabel {
    color: %(text_heading)s;
    font-weight: 600;
}
QTabWidget::pane {
    border: 1px solid %(border)s;
    border-radius: 18px;
    background: %(window_elevated)s;
    top: -2px;
}
QTabBar::tab {
    padding: 11px 18px;
    margin-right: 8px;
    margin-top: 10px;
    border: 1px solid %(border)s;
    border-bottom: none;
    border-top-left-radius: 14px;
    border-top-right-radius: 14px;
    background: %(window_alt)s;
    color: %(text_soft)s;
    font-weight: 600;
}
QTabBar::tab:hover:!selected {
    background: %(surface_alt)s;
    color: %(text_heading)s;
}
QTabBar::tab:selected {
    background: %(window_elevated)s;
    color: %(text_heading)s;
}
QGroupBox {
    border: 1px solid %(border)s;
    border-radius: 18px;
    margin-top: 16px;
    padding: 18px 18px 16px 18px;
    background: %(surface)s;
    color: %(text_heading)s;
    font-size: 15px;
    font-weight: 700;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 4px 11px;
    background: %(window_bg)s;
    color: %(text_heading)s;
    border: 1px solid %(border)s;
    border-radius: 11px;
}
QGroupBox[settingsSection="true"] {
    background: %(surface_muted)s;
    border-color: %(border_soft)s;
}
QGroupBox[heroCard="true"] {
    background: %(window_elevated)s;
    border-color: %(primary_soft)s;
}
QGroupBox[heroCard="true"]::title {
    background: %(primary_tint)s;
    border-color: %(primary_soft)s;
}
QPushButton {
    background: %(surface_alt)s;
    color: %(text_heading)s;
    border: 1px solid %(border)s;
    border-radius: 12px;
    padding: 9px 15px;
    font-weight: 600;
    min-height: 20px;
}
QPushButton:hover {
    background: %(window_alt)s;
    border-color: %(border_focus)s;
}
QPushButton:pressed {
    background: %(surface_muted)s;
}
QPushButton:checked {
    background: %(primary_tint)s;
    border-color: %(primary_soft)s;
    color: %(text_heading)s;
}
QPushButton:disabled {
    background: %(surface_muted)s;
    border-color: %(border_soft)s;
    color: %(text_muted)s;
}
QPushButton[role="primary"] {
    background: %(primary)s;
    color: %(text_invert)s;
    border: 1px solid %(primary)s;
}
QPushButton[role="primary"]:hover {
    background: %(primary_hover)s;
    border-color: %(primary_hover)s;
}
QPushButton[role="primary"]:pressed {
    background: %(primary_pressed)s;
    border-color: %(primary_pressed)s;
}
QPushButton[role="primary"]:disabled {
    background: %(primary_disabled)s;
    border-color: %(primary_disabled)s;
    color: %(text_invert)s;
}
QPushButton[role="secondary"] {
    background: %(primary_tint)s;
    border-color: %(primary_soft)s;
    color: %(text_heading)s;
}
QPushButton[role="secondary"]:hover {
    background: %(surface_alt)s;
}
QPushButton[role="ghost"] {
    background: transparent;
    color: %(text_soft)s;
    border-color: transparent;
}
QPushButton[role="ghost"]:hover {
    background: %(surface_alt)s;
    border-color: %(border_soft)s;
}
QFrame[navRail="true"] {
    background: %(surface)s;
    border: 1px solid %(border_soft)s;
    border-radius: 18px;
}
QFrame[navRail="true"][compactRail="true"] {
    border-radius: 16px;
}
QPushButton[navButton="true"] {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 12px;
    color: %(text_soft)s;
    font-weight: 600;
    min-height: 34px;
    padding: 8px 12px;
    text-align: left;
}
QPushButton[navButton="true"][verticalLabel="true"] {
    min-width: 40px;
    max-width: 40px;
    min-height: 88px;
    padding: 10px 4px;
}
QPushButton[navButton="true"]:hover {
    background: %(surface_alt)s;
    border-color: %(border_soft)s;
}
QPushButton[navButton="true"]:checked {
    background: %(primary_tint)s;
    border-color: %(primary_soft)s;
    color: %(text_heading)s;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QListWidget, QTableWidget {
    border: 1px solid %(border_soft)s;
    border-radius: 12px;
    background: %(surface_raised)s;
    color: %(text)s;
    selection-background-color: %(selection_bg)s;
    selection-color: %(selection_text)s;
    padding: 7px 11px;
    min-height: 34px;
}
QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QPlainTextEdit:hover, QListWidget:hover, QTableWidget:hover {
    border-color: %(border)s;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus, QListWidget:focus, QTableWidget:focus {
    border: 1px solid %(border_focus)s;
}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QPlainTextEdit:disabled, QListWidget:disabled, QTableWidget:disabled {
    background: %(surface_muted)s;
    color: %(text_muted)s;
}
QComboBox {
    padding-right: 28px;
}
QComboBox QAbstractItemView, QListView {
    background: %(window_elevated)s;
    color: %(text)s;
    selection-background-color: %(selection_bg)s;
    selection-color: %(selection_text)s;
    border: 1px solid %(border)s;
    padding: 4px;
}
QAbstractItemView {
    outline: none;
}
QCheckBox {
    color: %(text)s;
    font-size: 14px;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 1px solid %(border)s;
    border-radius: 6px;
    background: %(surface_raised)s;
}
QCheckBox::indicator:checked {
    background: %(primary)s;
    border-color: %(primary)s;
}
QCheckBox::indicator:hover {
    border-color: %(border_focus)s;
}
QPlainTextEdit, QListWidget, QTableWidget {
    min-height: 90px;
}
QListWidget::item {
    padding: 8px 10px;
    border-radius: 10px;
    margin: 2px 4px;
}
QListWidget::item:selected {
    background: %(selection_bg)s;
    color: %(selection_text)s;
}
QListWidget::item:hover:!selected {
    background: %(surface_alt)s;
}
QListWidget[dropZone="true"] {
    background: %(dropzone_bg)s;
    border: 2px dashed %(dropzone_border)s;
    padding: 10px;
}
QTableWidget {
    gridline-color: %(border_soft)s;
}
QHeaderView::section {
    background: %(surface_alt)s;
    color: %(text_heading)s;
    border: none;
    border-right: 1px solid %(border_soft)s;
    border-bottom: 1px solid %(border_soft)s;
    padding: 8px 10px;
    font-weight: 700;
}
QLabel {
    color: %(text)s;
}
QLabel#FieldTitle {
    color: %(text_heading)s;
    font-weight: 700;
    font-size: 14px;
}
QLabel#SectionHint {
    color: %(text_muted)s;
    font-size: 13px;
    padding-bottom: 4px;
}
QWidget[compactDialog="true"] {
    font-size: 13px;
}
QWidget[compactDialog="true"] QGroupBox {
    border-radius: 16px;
    margin-top: 14px;
    padding: 14px 14px 12px 14px;
    font-size: 14px;
}
QWidget[compactDialog="true"] QGroupBox::title {
    border-radius: 10px;
    left: 12px;
    padding: 3px 9px;
}
QWidget[compactDialog="true"] QPushButton {
    border-radius: 10px;
    min-height: 18px;
    padding: 7px 12px;
}
QWidget[compactDialog="true"] QPushButton[navButton="true"] {
    min-height: 32px;
    padding: 8px 10px;
}
QWidget[compactDialog="true"] QPushButton[navButton="true"][verticalLabel="true"] {
    min-width: 36px;
    max-width: 36px;
    min-height: 80px;
    padding: 8px 2px;
}
QWidget[compactDialog="true"] QLineEdit,
QWidget[compactDialog="true"] QComboBox,
QWidget[compactDialog="true"] QSpinBox,
QWidget[compactDialog="true"] QDoubleSpinBox,
QWidget[compactDialog="true"] QPlainTextEdit,
QWidget[compactDialog="true"] QListWidget,
QWidget[compactDialog="true"] QTableWidget {
    border-radius: 10px;
    min-height: 28px;
    padding: 5px 8px;
}
QWidget[compactDialog="true"] QCheckBox {
    font-size: 13px;
    spacing: 6px;
}
QWidget[compactDialog="true"] QCheckBox::indicator {
    width: 16px;
    height: 16px;
}
QWidget[compactDialog="true"] QHeaderView::section {
    padding: 6px 8px;
}
QWidget[compactDialog="true"] QLabel#FieldTitle {
    font-size: 13px;
}
QWidget[compactDialog="true"] QLabel#SectionHint {
    font-size: 12px;
}
QLabel#QueueSummary {
    color: %(text_soft)s;
    font-size: 13px;
    font-weight: 600;
    padding-top: 2px;
}
QLabel#HeroTitle {
    color: %(text_heading)s;
    font-size: 21px;
    font-weight: 700;
}
QLabel#HeroSubtitle {
    color: %(text_muted)s;
    font-size: 13px;
}
QToolButton {
    border: 1px solid transparent;
    border-radius: 10px;
    background: transparent;
    color: %(text_soft)s;
    padding: 4px 8px;
}
QToolButton:hover {
    background: %(surface_alt)s;
    border-color: %(border_soft)s;
}
QToolButton[helpButton="true"] {
    min-width: 22px;
    max-width: 22px;
    min-height: 22px;
    max-height: 22px;
    border: 1px solid %(help_border)s;
    border-radius: 11px;
    background: %(help_bg)s;
    color: %(text_soft)s;
    font-weight: 700;
    font-size: 12px;
    padding: 0;
}
QToolButton[helpButton="true"]:hover {
    background: %(help_hover_bg)s;
    border-color: %(help_hover_border)s;
}
QScrollArea {
    border: none;
    background: transparent;
}
QScrollBar:vertical {
    background: %(scroll_bg)s;
    width: 12px;
    border-radius: 6px;
    margin: 2px;
}
QScrollBar:horizontal {
    background: %(scroll_bg)s;
    height: 12px;
    border-radius: 6px;
    margin: 2px;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: %(scroll_handle)s;
    min-height: 32px;
    min-width: 32px;
    border-radius: 6px;
}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
    background: %(border_focus)s;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical, QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
    height: 0px;
}
QSplitter::handle {
    background: %(splitter)s;
    border-radius: 3px;
}
QSplitter::handle:horizontal {
    width: 6px;
}
QSplitter::handle:vertical {
    height: 6px;
}
QProgressBar {
    border: 1px solid %(border_soft)s;
    border-radius: 10px;
    text-align: center;
    background: %(progress_bg)s;
    color: %(text_soft)s;
    min-height: 18px;
    font-weight: 600;
}
QProgressBar::chunk {
    border-radius: 9px;
    background: %(progress_chunk)s;
}
QProgressBar[footerProgress="true"] {
    border: none;
    border-radius: 3px;
    background: %(progress_bg)s;
    min-height: 6px;
    max-height: 6px;
    padding: 0;
}
QProgressBar[footerProgress="true"]::chunk {
    border-radius: 3px;
}
QProgressBar[stepProgress="true"] {
    min-height: 10px;
    max-height: 10px;
    border-radius: 6px;
    border: 1px solid %(border_soft)s;
    background: %(surface_alt)s;
    padding: 0;
}
QProgressBar[stepProgress="true"]::chunk {
    border-radius: 5px;
    background: %(progress_chunk)s;
}
QMenu {
    background: %(window_elevated)s;
    color: %(text)s;
    border: 1px solid %(border)s;
    border-radius: 12px;
    padding: 6px;
}
QMenu::item {
    padding: 8px 12px;
    border-radius: 8px;
}
QMenu::item:selected {
    background: %(selection_bg)s;
    color: %(selection_text)s;
}
QToolTip {
    background: %(surface_alt)s;
    color: %(text)s;
    border: 1px solid %(border)s;
    padding: 6px 8px;
}
QPlainTextEdit[logView="true"] {
    background: %(surface_muted)s;
    border: 1px solid %(border)s;
    font-family: "Cascadia Mono";
    font-size: 13px;
}
QWidget[stepStatusCard="true"] {
    background: %(surface_muted)s;
    border: 1px solid %(border)s;
    border-radius: 18px;
}
QWidget[stepStatusItem="true"] {
    background: %(surface_raised)s;
    border: 1px solid %(border_soft)s;
    border-radius: 14px;
}
QLabel#StepStatusRunName {
    font-size: 16px;
    font-weight: 700;
    color: %(text_heading)s;
}
QLabel#StepStatusTitle {
    font-size: 13px;
    font-weight: 700;
    color: %(text_heading)s;
}
QLabel#StepStatusPercent {
    font-size: 12px;
    font-weight: 700;
    color: %(text_soft)s;
}
"""


def _theme_tokens(theme: str) -> Dict[str, str]:
    if str(theme).lower().startswith("dark"):
        return dict(_DARK_THEME)
    return dict(_LIGHT_THEME)


def build_app_qss(theme: str) -> str:
    return _APP_QSS_TEMPLATE % _theme_tokens(theme)


def build_app_palette(theme: str) -> QtGui.QPalette:
    tokens = _theme_tokens(theme)
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(tokens["window_bg"]))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(tokens["text"]))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(tokens["surface_raised"]))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(tokens["surface_alt"]))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(tokens["surface_alt"]))
    palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(tokens["text"]))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor(tokens["text"]))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(tokens["surface_alt"]))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(tokens["text_heading"]))
    palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor(tokens["selection_text"]))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(tokens["selection_bg"]))
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(tokens["selection_text"]))
    palette.setColor(QtGui.QPalette.Link, QtGui.QColor(tokens["primary"]))
    palette.setColor(QtGui.QPalette.LinkVisited, QtGui.QColor(tokens["primary_hover"]))
    palette.setColor(QtGui.QPalette.PlaceholderText, QtGui.QColor(tokens["text_muted"]))

    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, QtGui.QColor(tokens["text_muted"]))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, QtGui.QColor(tokens["text_muted"]))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, QtGui.QColor(tokens["text_muted"]))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Highlight, QtGui.QColor(tokens["border"]))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.HighlightedText, QtGui.QColor(tokens["text_muted"]))
    return palette


APP_QSS = build_app_qss("Light")

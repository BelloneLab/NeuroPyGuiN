"""Self-contained phy plugin: game-controller curation + gamification.

Installed to ``~/.phy/plugins/`` by neuropyguin's ``phy_integration.py``.
When phy launches, this plugin:

1. Initialises pygame for gamepad polling (50 ms QTimer)
2. Maps controller buttons/axes/hats to Supervisor curation actions
3. Adds an ethoscore-style gamification HUD overlay (score, combo, stats)
4. All keyboard + mouse shortcuts remain untouched — this is additive

Controller layout (Xbox-style, remappable):
    A (green)  → Good       B (red)    → Noise
    X (blue)   → MUA        Y (yellow) → Merge
    D-Up/Down  → Prev/Next best cluster
    D-Left/Rt  → Prev/Next similar
    Left stick → Browse clusters (analog speed)
    Right stick→ Browse similar  (analog speed)
    LB → Undo   RB → Redo
    LT (full)  → Split      Start → Save
    Back       → Toggle HUD  R3 → Unselect similar
"""

# ── source code returned by _gamepad_plugin_source() ──────────────────────
# This function is called from phy_integration.py; its return value is
# written verbatim into ~/.phy/plugins/neuropyguin_gamepad_controller.py


def _gamepad_plugin_source() -> str:
    """Return the full plugin source as a string (phy discovers .py files)."""
    return r'''
import json
import logging
import math
import time
from enum import Enum, auto
from functools import partial

import numpy as np
from phy import IPlugin, connect

logger = logging.getLogger("phy.gamepad")

# ── safely import Qt (phy uses PyQt5 — must use the same binding) ─────────
try:
    # From phy's re-exports
    from phy.gui.qt import (
        QWidget, QLabel, QVBoxLayout, QHBoxLayout,
        QPushButton, QTimer, Qt, QColor,
        QObject, QEvent,
    )
    # Additional PyQt5 imports not re-exported by phy.gui.qt
    from PyQt5.QtWidgets import (
        QProgressBar, QDialog, QTableWidget, QTableWidgetItem,
        QHeaderView, QGroupBox, QSizePolicy,
    )
    from PyQt5.QtGui import QFont, QPainter, QPainterPath, QPen, QLinearGradient, QRadialGradient
    from PyQt5.QtCore import QPointF, QRect, QRectF, QSettings
    _HAS_QT = True
except ImportError:
    _HAS_QT = False

# ── safely import pygame ──────────────────────────────────────────────────
try:
    import pygame
    _HAS_PYGAME = True
except ImportError:
    _HAS_PYGAME = False


# =========================================================================
# GamepadAction enum
# =========================================================================

class GamepadAction(Enum):
    MARK_GOOD      = auto()
    MARK_NOISE     = auto()
    MARK_MUA       = auto()
    MERGE          = auto()
    NEXT_BEST      = auto()
    PREV_BEST      = auto()
    NEXT_SIMILAR   = auto()
    PREV_SIMILAR   = auto()
    UNDO           = auto()
    REDO           = auto()
    SPLIT          = auto()
    SAVE           = auto()
    TOGGLE_HUD     = auto()
    UNSELECT_SIM   = auto()
    SPLIT_KMEANS   = auto()   # K-means cluster split
    SPLIT_ISI      = auto()   # Split short ISI (plugin action)
    SELECT_SIMILAR = auto()   # Select similar clusters
    MARK_GOOD_NEXT  = auto()  # Mark good + advance to next best
    MARK_NOISE_NEXT = auto()  # Mark noise + advance to next best

# Human-readable labels for the HUD
_ACTION_LABELS = {
    GamepadAction.MARK_GOOD:    "GOOD",
    GamepadAction.MARK_NOISE:   "NOISE",
    GamepadAction.MARK_MUA:     "MUA",
    GamepadAction.MERGE:        "MERGE",
    GamepadAction.NEXT_BEST:    "Next",
    GamepadAction.PREV_BEST:    "Prev",
    GamepadAction.NEXT_SIMILAR: "Next similar",
    GamepadAction.PREV_SIMILAR: "Prev similar",
    GamepadAction.UNDO:         "Undo",
    GamepadAction.REDO:         "Redo",
    GamepadAction.SPLIT:        "Split",
    GamepadAction.SAVE:         "Save",
    GamepadAction.TOGGLE_HUD:    "Toggle HUD",
    GamepadAction.UNSELECT_SIM:  "Unselect similar",
    GamepadAction.SPLIT_KMEANS:  "K-means Split",
    GamepadAction.SPLIT_ISI:     "Split short ISI",
    GamepadAction.SELECT_SIMILAR: "Select similar",
    GamepadAction.MARK_GOOD_NEXT:  "Good + Next",
    GamepadAction.MARK_NOISE_NEXT: "Noise + Next",
}


# =========================================================================
# Default mapping (Xbox layout — SDL button IDs)
# =========================================================================

DEFAULT_BUTTON_MAP = {
    0: GamepadAction.MARK_GOOD,      # A
    1: GamepadAction.MARK_NOISE,     # B
    2: GamepadAction.MARK_MUA,       # X
    3: GamepadAction.MERGE,          # Y
    4: GamepadAction.UNDO,           # LB
    5: GamepadAction.REDO,           # RB
    6: GamepadAction.TOGGLE_HUD,     # Back / Select
    7: GamepadAction.SAVE,           # Start
    9: GamepadAction.UNSELECT_SIM,   # R3
}

DEFAULT_HAT_MAP = {
    (0,  1): GamepadAction.PREV_BEST,
    (0, -1): GamepadAction.NEXT_BEST,
    (-1, 0): GamepadAction.PREV_SIMILAR,
    ( 1, 0): GamepadAction.NEXT_SIMILAR,
}

_BUTTON_MAP_SETTINGS_KEY = "gamepad/button_map_v2"

_BUTTON_SPECS = {
    0: {
        "label": "A",
        "short": "A",
        "shape": "round",
        "size": 0.092,
        "color": "#59d16f",
        "pos": (0.74, 0.69),
        "callout": (0.93, 0.82),
        "align": "left",
    },
    1: {
        "label": "B",
        "short": "B",
        "shape": "round",
        "size": 0.092,
        "color": "#ff6464",
        "pos": (0.83, 0.58),
        "callout": (0.95, 0.60),
        "align": "left",
    },
    2: {
        "label": "X",
        "short": "X",
        "shape": "round",
        "size": 0.092,
        "color": "#46b6ff",
        "pos": (0.65, 0.58),
        "callout": (0.07, 0.57),
        "align": "right",
    },
    3: {
        "label": "Y",
        "short": "Y",
        "shape": "round",
        "size": 0.092,
        "color": "#f4d957",
        "pos": (0.74, 0.47),
        "callout": (0.93, 0.34),
        "align": "left",
    },
    4: {
        "label": "LB",
        "short": "LB",
        "shape": "pill",
        "size": 0.088,
        "color": "#d6dbe4",
        "pos": (0.26, 0.15),
        "callout": (0.07, 0.20),
        "align": "right",
    },
    5: {
        "label": "RB",
        "short": "RB",
        "shape": "pill",
        "size": 0.088,
        "color": "#d6dbe4",
        "pos": (0.74, 0.15),
        "callout": (0.93, 0.20),
        "align": "left",
    },
    6: {
        "label": "View",
        "short": "View",
        "shape": "small",
        "size": 0.070,
        "color": "#bac2d0",
        "pos": (0.44, 0.43),
        "callout": (0.38, 0.08),
        "align": "center",
    },
    7: {
        "label": "Menu",
        "short": "Menu",
        "shape": "small",
        "size": 0.070,
        "color": "#bac2d0",
        "pos": (0.56, 0.43),
        "callout": (0.62, 0.08),
        "align": "center",
    },
    8: {
        "label": "L3",
        "short": "L3",
        "shape": "small",
        "size": 0.072,
        "color": "#94a3b8",
        "pos": (0.26, 0.42),
        "callout": (0.08, 0.82),
        "align": "right",
    },
    9: {
        "label": "R3",
        "short": "R3",
        "shape": "small",
        "size": 0.072,
        "color": "#94a3b8",
        "pos": (0.60, 0.69),
        "callout": (0.93, 0.92),
        "align": "left",
    },
}

_FIXED_CONTROL_HINTS = (
    "D-pad Up/Down -> Prev/Next best",
    "D-pad Left/Right -> Prev/Next similar",
    "LT -> Split (always active)",
)

# Axis IDs
_LEFT_STICK_Y   = 1
_RIGHT_STICK_X  = 2   # often axis 3 on some pads — auto-detected
_LEFT_TRIGGER   = 4


def _button_display_name(btn_id):
    if btn_id is None:
        return "(unassigned)"
    spec = _BUTTON_SPECS.get(int(btn_id))
    return spec["label"] if spec else f"Button {int(btn_id)}"


def _serialize_button_map(button_map):
    payload = {str(int(btn_id)): action.name for btn_id, action in (button_map or {}).items()}
    return json.dumps(payload, sort_keys=True)


def _deserialize_button_map(raw_value):
    if not raw_value:
        return dict(DEFAULT_BUTTON_MAP)
    try:
        data = json.loads(str(raw_value))
    except Exception:
        return dict(DEFAULT_BUTTON_MAP)
    if not isinstance(data, dict):
        return dict(DEFAULT_BUTTON_MAP)

    out = {}
    seen_actions = set()
    for btn_id, action_name in data.items():
        try:
            button_id = int(btn_id)
            action = GamepadAction[str(action_name)]
        except Exception:
            continue
        if action in seen_actions:
            continue
        out[button_id] = action
        seen_actions.add(action)
    return out or dict(DEFAULT_BUTTON_MAP)


# =========================================================================
# Response-curve helpers (ethoscore-style)
# =========================================================================

def _apply_curve(raw, mode="quadratic", expo=0.3, super_rate=0.5):
    if mode == "linear":
        return raw
    if mode == "quadratic":
        return raw ** 2
    # Betaflight expo
    y = raw * (1.0 + expo * (raw ** 2 - 1.0))
    d = max(1.0 - super_rate * raw, 1e-6)
    return y / d


# =========================================================================
# Gamification engine (lightweight, ethoscore-inspired)
# =========================================================================

class _GamificationEngine:
    """Score / combo / streak tracker for spike-sorting curation."""

    def __init__(self):
        self.enabled = True
        self.total_score = 0
        self.high_score = 0
        self.base_points = 10
        self.spike_bonus = True
        self.combo_threshold = 3
        self.combo_timeout_ms = 3000
        self.combo_across_groups = True

        self._combo_count = 0
        self._last_group = None

        # Session stats
        self.total_curated = 0
        self.good_count = 0
        self.noise_count = 0
        self.mua_count = 0
        self.merge_count = 0
        self.split_count = 0
        self.undo_count = 0
        self.longest_streak = 0
        self._current_streak = 0
        self._session_start = time.time()
        self._last_curated_time = 0.0

        # Combo timer state (driven externally by QTimer)
        self._combo_remaining_ms = 0
        self._combo_active = False

        # Callbacks (set by HUD)
        self.on_score_updated = None    # (total, gained, combo_text)
        self.on_combo_progress = None   # (float 0-1)
        self.on_combo_visible = None    # (bool)
        self.on_stats_updated = None    # ()

    @property
    def clusters_per_minute(self):
        elapsed = max((time.time() - self._session_start) / 60.0, 0.01)
        return self.total_curated / elapsed

    @property
    def elapsed_min(self):
        return (time.time() - self._session_start) / 60.0

    def cluster_curated(self, group, n_spikes=100):
        if not self.enabled:
            return 0
        self.total_curated += 1
        if group == "good":
            self.good_count += 1
        elif group == "noise":
            self.noise_count += 1
        elif group == "mua":
            self.mua_count += 1

        self._current_streak += 1
        if self._current_streak > self.longest_streak:
            self.longest_streak = self._current_streak

        # Combo
        if group == self._last_group or self.combo_across_groups:
            self._combo_count += 1
        else:
            self._combo_count = 1

        eff = self._combo_count if self._combo_count >= self.combo_threshold else 1
        spike_mult = max(1, int(math.log2(max(n_spikes, 1)))) if self.spike_bonus else 1
        gained = self.base_points * spike_mult * eff
        self.total_score += gained
        if self.total_score > self.high_score:
            self.high_score = self.total_score

        if self._combo_count >= self.combo_threshold:
            combo_text = f"{spike_mult}x {self._combo_count} Combo!"
        else:
            combo_text = f"+{gained}"

        self._last_group = group
        self._last_curated_time = time.time()

        # Start combo timer
        self._combo_remaining_ms = self.combo_timeout_ms
        self._combo_active = True

        if self.on_score_updated:
            self.on_score_updated(self.total_score, gained, combo_text)
        if self.on_combo_visible:
            self.on_combo_visible(True)
        if self.on_stats_updated:
            self.on_stats_updated()
        return gained

    def tick_combo(self, dt_ms=50):
        """Called by QTimer to count down combo window."""
        if not self._combo_active:
            return
        self._combo_remaining_ms -= dt_ms
        if self._combo_remaining_ms <= 0:
            self._reset_combo()
        elif self.on_combo_progress:
            self.on_combo_progress(
                max(0.0, self._combo_remaining_ms / self.combo_timeout_ms)
            )

    def _reset_combo(self):
        self._combo_count = 0
        self._last_group = None
        self._combo_active = False
        self._current_streak = 0
        if self.on_combo_visible:
            self.on_combo_visible(False)

    def action_undone(self):
        self.undo_count += 1
        self._current_streak = 0
        self._reset_combo()
        if self.on_stats_updated:
            self.on_stats_updated()

    def merged(self):
        self.merge_count += 1
        if self.on_stats_updated:
            self.on_stats_updated()

    def split(self):
        self.split_count += 1
        if self.on_stats_updated:
            self.on_stats_updated()

    def reset_session(self):
        self.total_score = 0
        self._combo_count = 0
        self._last_group = None
        self._combo_active = False
        self.total_curated = 0
        self.good_count = self.noise_count = self.mua_count = 0
        self.merge_count = self.split_count = self.undo_count = 0
        self.longest_streak = self._current_streak = 0
        self._session_start = time.time()
        if self.on_score_updated:
            self.on_score_updated(0, 0, "")
        if self.on_stats_updated:
            self.on_stats_updated()


# =========================================================================
# HUD overlay widget
# =========================================================================

if _HAS_QT:

    class _GamepadHUD(QWidget):
        """Floating HUD overlay: score, combo bar, controller status, stats."""

        _STYLE = """
            QWidget#gamepadHUD {
                background: rgba(20, 20, 35, 200);
                border: 1px solid rgba(124, 58, 237, 150);
                border-radius: 8px;
            }
            QLabel { color: #cdd6f4; background: transparent; border: none; }
            QProgressBar {
                border: none; border-radius: 3px;
                background: rgba(255,255,255,60);
                max-height: 6px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #f9e2af, stop:1 #fab387);
                border-radius: 3px;
            }
        """

        def __init__(self, gam, parent_widget=None):
            super().__init__(parent=parent_widget)
            self.gam = gam
            self.setObjectName("gamepadHUD")
            self.setStyleSheet(self._STYLE)
            self.setFixedWidth(260)

            lay = QVBoxLayout(self)
            lay.setContentsMargins(12, 10, 12, 10)
            lay.setSpacing(4)

            # ── Controller status ─────────────────────────────────────
            self._lbl_controller = QLabel("Controller: scanning...")
            self._lbl_controller.setStyleSheet(
                "color: #a6adc8; font-size: 10px;"
            )
            lay.addWidget(self._lbl_controller)

            # ── Score ─────────────────────────────────────────────────
            self._lbl_score = QLabel("Score: 0")
            self._lbl_score.setFont(QFont("Segoe UI", 18, QFont.Bold))
            self._lbl_score.setStyleSheet("color: #cba6f7;")
            lay.addWidget(self._lbl_score)

            self._lbl_high = QLabel("High: 0")
            self._lbl_high.setStyleSheet("color: #6c7086; font-size: 10px;")
            lay.addWidget(self._lbl_high)

            # ── Combo progress bar ────────────────────────────────────
            self._combo_bar = QProgressBar()
            self._combo_bar.setRange(0, 100)
            self._combo_bar.setValue(100)
            self._combo_bar.setTextVisible(False)
            self._combo_bar.setVisible(False)
            lay.addWidget(self._combo_bar)

            # ── Floating combo text (painted in paintEvent) ───────────
            self._float_text = ""
            self._float_opacity = 0.0
            self._float_y = 0

            # ── Session stats ─────────────────────────────────────────
            self._lbl_stats = QLabel("")
            self._lbl_stats.setStyleSheet(
                "color: #a6adc8; font-size: 10px; margin-top: 4px;"
            )
            self._lbl_stats.setWordWrap(True)
            lay.addWidget(self._lbl_stats)

            # ── Last action feedback ──────────────────────────────────
            self._lbl_action = QLabel("")
            self._lbl_action.setStyleSheet(
                "color: #f9e2af; font-size: 12px; font-weight: bold;"
            )
            self._lbl_action.setAlignment(Qt.AlignCenter)
            lay.addWidget(self._lbl_action)

            # ── Button legend (compact) ───────────────────────────────
            self._lbl_mapping = QLabel("")
            self._lbl_mapping.setStyleSheet("font-size: 10px; margin-top: 6px;")
            self._lbl_mapping.setWordWrap(True)
            lay.addWidget(self._lbl_mapping)

            nav_legend = QLabel(
                "D-pad / sticks navigate  |  LT split  |  Configure controller actions from the Gamepad menu"
            )
            nav_legend.setStyleSheet("color: #585b70; font-size: 9px;")
            nav_legend.setWordWrap(True)
            lay.addWidget(nav_legend)

            # Wire gamification callbacks
            gam.on_score_updated = self._on_score
            gam.on_combo_progress = self._on_combo_progress
            gam.on_combo_visible = self._on_combo_visible
            gam.on_stats_updated = self._on_stats

            # Fade-out timer for floating text
            self._fade_timer = QTimer(self)
            self._fade_timer.setInterval(50)
            self._fade_timer.timeout.connect(self._tick_fade)

            self.set_button_map({})

        # ── Gamification callbacks ────────────────────────────────────

        def _on_score(self, total, gained, combo_text):
            self._lbl_score.setText(f"Score: {total:,}")
            self._lbl_high.setText(f"High: {self.gam.high_score:,}")
            if gained > 0:
                self._float_text = combo_text
                self._float_opacity = 1.0
                self._float_y = 0
                self._fade_timer.start()
            self.update()

        def _on_combo_progress(self, p):
            self._combo_bar.setValue(int(p * 100))

        def _on_combo_visible(self, vis):
            self._combo_bar.setVisible(vis)

        def _on_stats(self):
            g = self.gam
            txt = (
                f"Curated: {g.total_curated}  "
                f"({g.good_count}G / {g.noise_count}N / {g.mua_count}M)\n"
                f"Merges: {g.merge_count}  Splits: {g.split_count}  "
                f"Undos: {g.undo_count}\n"
                f"Streak: {g.longest_streak}  "
                f"Rate: {g.clusters_per_minute:.1f}/min  "
                f"Time: {g.elapsed_min:.0f}m"
            )
            self._lbl_stats.setText(txt)

        def set_action_text(self, text):
            self._lbl_action.setText(text)
            # Auto-clear after 2s
            QTimer.singleShot(2000, lambda: self._lbl_action.setText(""))

        def set_controller_name(self, name):
            self._lbl_controller.setText(f"Controller: {name}")
            self._lbl_controller.setStyleSheet("color: #a6e3a1; font-size: 10px;")

        def set_controller_disconnected(self):
            self._lbl_controller.setText("Controller: not connected")
            self._lbl_controller.setStyleSheet("color: #f38ba8; font-size: 10px;")

        def set_button_map(self, button_map):
            inverse = {action: _button_display_name(btn_id) for btn_id, action in (button_map or {}).items()}
            parts = [
                f"<span style='color:#59d16f;'>{inverse.get(GamepadAction.MARK_GOOD, 'A')}</span> Good",
                f"<span style='color:#ff6464;'>{inverse.get(GamepadAction.MARK_NOISE, 'B')}</span> Noise",
                f"<span style='color:#46b6ff;'>{inverse.get(GamepadAction.MARK_MUA, 'X')}</span> MUA",
                f"<span style='color:#f4d957;'>{inverse.get(GamepadAction.MERGE, 'Y')}</span> Merge",
                f"<span style='color:#cbd5e1;'>{inverse.get(GamepadAction.SAVE, 'Menu')}</span> Save",
            ]
            self._lbl_mapping.setText("  |  ".join(parts))

        # ── Floating text animation ───────────────────────────────────

        def _tick_fade(self):
            self._float_opacity -= 0.02
            self._float_y -= 1
            if self._float_opacity <= 0:
                self._float_opacity = 0
                self._fade_timer.stop()
            self.update()

        def paintEvent(self, event):
            super().paintEvent(event)
            if self._float_opacity <= 0 or not self._float_text:
                return
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setOpacity(self._float_opacity)

            # Color: green → red based on combo count
            cc = self.gam._combo_count
            ct = self.gam.combo_threshold
            if cc >= ct:
                ratio = min((cc - ct) / 5.0, 1.0)
                r = int(255 * ratio)
                g = int(255 * (1 - ratio))
                color = QColor(r, g, 0, int(255 * self._float_opacity))
            else:
                color = QColor(0, 255, 0, int(255 * self._float_opacity))

            painter.setPen(QPen(color, 2))
            painter.setFont(QFont("Segoe UI", 14, QFont.Bold))

            bar_rect = self._combo_bar.geometry()
            text_rect = QRect(
                bar_rect.x(), bar_rect.y() + self._float_y - 20,
                bar_rect.width(), 20,
            )
            painter.drawText(text_rect, Qt.AlignCenter, self._float_text)
            painter.setOpacity(1.0)
            painter.end()


# =========================================================================
# Resize event filter (QObject — needed because IPlugin is not a QObject)
# =========================================================================

if _HAS_QT:

    class _ResizeFilter(QObject):
        """Lightweight QObject that calls *callback* on parent resize."""
        def __init__(self, parent, callback):
            super().__init__(parent)
            self._callback = callback

        def eventFilter(self, obj, event):
            if event.type() == QEvent.Resize:
                self._callback()
            return False


# =========================================================================
# Controller-status dialog
# =========================================================================

if _HAS_QT:

    class _ControllerStatusDialog(QDialog):
        """Read-only controller info with a Re-detect button."""

        _STYLE = """
            QDialog { background: #1e1e2e; }
            QLabel { color: #cdd6f4; background: transparent; }
            QPushButton {
                background: #313244; color: #cdd6f4;
                border: 1px solid #585b70; border-radius: 4px;
                padding: 6px 14px;
            }
            QPushButton:hover { background: #45475a; }
            QGroupBox {
                color: #cba6f7; font-weight: bold;
                border: 1px solid #45475a; border-radius: 4px;
                margin-top: 10px; padding-top: 8px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
        """

        def __init__(self, plugin, parent=None):
            super().__init__(parent)
            self._plugin = plugin
            self.setWindowTitle("Gamepad \u2014 Controller Status")
            self.setStyleSheet(self._STYLE)
            self.setMinimumWidth(360)
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

            lay = QVBoxLayout(self)
            lay.setSpacing(10)
            lay.setContentsMargins(14, 14, 14, 14)

            grp = QGroupBox("Detected Controller")
            grp_lay = QVBoxLayout(grp)
            grp_lay.setSpacing(4)
            self._lbl_pygame  = QLabel()
            self._lbl_name    = QLabel()
            self._lbl_axes    = QLabel()
            self._lbl_buttons = QLabel()
            self._lbl_hats    = QLabel()
            for lbl in (self._lbl_pygame, self._lbl_name,
                        self._lbl_axes, self._lbl_buttons, self._lbl_hats):
                grp_lay.addWidget(lbl)
            lay.addWidget(grp)

            btn_row = QHBoxLayout()
            btn_redetect = QPushButton("Re-detect")
            btn_redetect.clicked.connect(self._refresh)
            btn_row.addWidget(btn_redetect)
            btn_row.addStretch()
            btn_close = QPushButton("Close")
            btn_close.clicked.connect(self.accept)
            btn_row.addWidget(btn_close)
            lay.addLayout(btn_row)

            self._refresh()

        def _refresh(self):
            plugin = self._plugin
            if not _HAS_PYGAME:
                self._lbl_pygame.setText("pygame: NOT installed  (pip install pygame)")
                self._lbl_pygame.setStyleSheet("color: #f38ba8;")
                for lbl in (self._lbl_name, self._lbl_axes,
                             self._lbl_buttons, self._lbl_hats):
                    lbl.setText("")
                return

            self._lbl_pygame.setText(f"pygame {pygame.version.ver}  \u2713")
            self._lbl_pygame.setStyleSheet("color: #a6e3a1;")

            plugin._detect_controller()
            js = plugin._joystick
            if js is None:
                self._lbl_name.setText("Status:  not connected")
                self._lbl_name.setStyleSheet("color: #f38ba8;")
                for lbl in (self._lbl_axes, self._lbl_buttons, self._lbl_hats):
                    lbl.setText("")
            else:
                self._lbl_name.setText(f"Name:    {js.get_name()}")
                self._lbl_name.setStyleSheet("color: #a6e3a1;")
                self._lbl_axes.setText(f"Axes:    {js.get_numaxes()}")
                self._lbl_buttons.setText(f"Buttons: {js.get_numbuttons()}")
                self._lbl_hats.setText(f"Hats (D-pad): {js.get_numhats()}")
                for lbl in (self._lbl_axes, self._lbl_buttons, self._lbl_hats):
                    lbl.setStyleSheet("color: #cdd6f4;")


# =========================================================================
# Button-configuration dialog
# =========================================================================

if _HAS_QT:

    class _ControllerMapWidget(QWidget):
        """Stylized Xbox controller view with clickable hotspots and live callouts."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setObjectName("controllerCanvas")
            self.setMinimumSize(560, 360)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            self._mapping = {}
            self._focused_action = None
            self._learning_action = None
            self._button_positions = {}
            self.on_button_clicked = None

            self._buttons = {}
            for button_id, spec in _BUTTON_SPECS.items():
                btn = QPushButton(spec["short"], self)
                btn.setFocusPolicy(Qt.NoFocus)
                btn.setCursor(Qt.PointingHandCursor)
                btn.clicked.connect(partial(self._emit_button_clicked, button_id))
                self._buttons[button_id] = btn

            self._refresh_button_styles()

        def _emit_button_clicked(self, button_id):
            if callable(self.on_button_clicked):
                self.on_button_clicked(button_id)

        def set_mapping(self, mapping):
            self._mapping = dict(mapping or {})
            self._refresh_button_styles()
            self.update()

        def set_focus_action(self, action):
            self._focused_action = action
            self._refresh_button_styles()
            self.update()

        def set_learning_action(self, action):
            self._learning_action = action
            self._refresh_button_styles()
            self.update()

        def resizeEvent(self, event):
            super().resizeEvent(event)
            base = max(26, int(min(self.width(), self.height()) * 0.11))
            for button_id, btn in self._buttons.items():
                spec = _BUTTON_SPECS[button_id]
                cx = int(self.width() * spec["pos"][0])
                cy = int(self.height() * spec["pos"][1])
                size = max(18, int(min(self.width(), self.height()) * spec.get("size", 0.09)))
                shape = spec.get("shape", "round")
                if shape == "pill":
                    width = int(size * 1.9)
                    height = int(size * 0.78)
                elif shape == "small":
                    width = int(size * 0.95)
                    height = int(size * 0.95)
                else:
                    width = height = size
                btn.setGeometry(cx - width // 2, cy - height // 2, width, height)
                self._button_positions[button_id] = QPointF(cx, cy)
            self._refresh_button_styles()

        def _refresh_button_styles(self):
            for button_id, btn in self._buttons.items():
                spec = _BUTTON_SPECS[button_id]
                mapped_action = self._mapping.get(button_id)
                active = mapped_action is not None and mapped_action == self._focused_action
                learning = mapped_action is not None and mapped_action == self._learning_action
                accent = QColor(spec["color"])
                border = accent.name() if (active or learning) else "#c7d2e0"
                background = accent.darker(140).name() if mapped_action is not None else "#2a3142"
                text_color = "#f8fafc" if mapped_action is not None else "#d7dce5"
                shadow = accent.lighter(135).name() if active else accent.name()
                radius = max(10, btn.height() // 2)
                btn.setStyleSheet(
                    "QPushButton {"
                    f"background: {background};"
                    f"color: {text_color};"
                    f"border: 2px solid {border};"
                    f"border-radius: {radius}px;"
                    "font-weight: 700;"
                    f"font-size: {max(8, min(btn.width(), btn.height()) // 3)}px;"
                    "padding: 0px 6px;"
                    "}"
                    "QPushButton:hover {"
                    f"border-color: {shadow};"
                    "}"
                )
                if mapped_action is None:
                    tip = f"{spec['label']} is available for assignment."
                else:
                    tip = f"{spec['label']} -> {_ACTION_LABELS.get(mapped_action, str(mapped_action))}"
                btn.setToolTip(tip)

        def _draw_stick(self, painter, center, radius):
            painter.save()
            painter.setPen(QPen(QColor(255, 255, 255, 40), 1.3))
            painter.setBrush(QColor(18, 20, 28))
            painter.drawEllipse(center, radius, radius)
            painter.setBrush(QColor(30, 33, 45))
            painter.drawEllipse(center, radius * 0.62, radius * 0.62)
            painter.restore()

        def _draw_dpad(self, painter, center, size):
            painter.save()
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(22, 25, 36))
            arm = QRectF(center.x() - size * 0.14, center.y() - size * 0.5, size * 0.28, size)
            bar = QRectF(center.x() - size * 0.5, center.y() - size * 0.14, size, size * 0.28)
            painter.drawRoundedRect(arm, 6, 6)
            painter.drawRoundedRect(bar, 6, 6)
            painter.restore()

        def paintEvent(self, event):
            super().paintEvent(event)
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)

            background = QLinearGradient(0, 0, self.width(), self.height())
            background.setColorAt(0, QColor("#10131d"))
            background.setColorAt(1, QColor("#161c2a"))
            painter.fillRect(self.rect(), background)

            glow = QRadialGradient(self.width() * 0.5, self.height() * 0.44, self.width() * 0.36)
            glow.setColorAt(0.0, QColor(255, 255, 255, 20))
            glow.setColorAt(1.0, QColor(255, 255, 255, 0))
            painter.fillRect(self.rect(), glow)

            body_path = QPainterPath()
            body_path.addRoundedRect(QRectF(self.width() * 0.28, self.height() * 0.18, self.width() * 0.44, self.height() * 0.36), 62, 62)
            body_path.addEllipse(QRectF(self.width() * 0.13, self.height() * 0.25, self.width() * 0.28, self.height() * 0.48))
            body_path.addEllipse(QRectF(self.width() * 0.59, self.height() * 0.25, self.width() * 0.28, self.height() * 0.48))
            body_path.addRoundedRect(QRectF(self.width() * 0.31, self.height() * 0.44, self.width() * 0.38, self.height() * 0.18), 54, 54)

            painter.setPen(QPen(QColor(255, 255, 255, 28), 1.2))
            painter.setBrush(QColor("#0d1018"))
            painter.drawPath(body_path)

            painter.setBrush(QColor("#141a24"))
            painter.drawRoundedRect(QRectF(self.width() * 0.17, self.height() * 0.11, self.width() * 0.18, self.height() * 0.07), 18, 18)
            painter.drawRoundedRect(QRectF(self.width() * 0.65, self.height() * 0.11, self.width() * 0.18, self.height() * 0.07), 18, 18)

            self._draw_stick(painter, QPointF(self.width() * 0.26, self.height() * 0.42), max(18, int(min(self.width(), self.height()) * 0.085)))
            self._draw_stick(painter, QPointF(self.width() * 0.58, self.height() * 0.68), max(18, int(min(self.width(), self.height()) * 0.085)))
            self._draw_dpad(painter, QPointF(self.width() * 0.30, self.height() * 0.67), max(34, int(min(self.width(), self.height()) * 0.11)))

            painter.setPen(QPen(QColor(255, 255, 255, 30), 1.0))
            painter.setBrush(QColor("#1d2430"))
            painter.drawEllipse(QPointF(self.width() * 0.50, self.height() * 0.29), 18, 18)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor("#d4d4d8"))
            painter.drawEllipse(QPointF(self.width() * 0.50, self.height() * 0.29), 8, 8)

            for button_id, spec in _BUTTON_SPECS.items():
                anchor = self._button_positions.get(button_id)
                if anchor is None:
                    continue
                mapped_action = self._mapping.get(button_id)
                is_active = mapped_action is not None and mapped_action == self._focused_action
                accent = QColor(spec["color"]) if mapped_action is not None else QColor("#6b7280")
                if is_active:
                    accent = accent.lighter(140)
                end_point = QPointF(self.width() * spec["callout"][0], self.height() * spec["callout"][1])
                elbow_x = anchor.x() + (28 if spec["align"] == "left" else -28 if spec["align"] == "right" else 0)
                elbow = QPointF(elbow_x, end_point.y())

                painter.setPen(QPen(accent, 1.7 if is_active else 1.1, Qt.DotLine))
                painter.drawLine(anchor, elbow)
                painter.drawLine(elbow, end_point)
                painter.setBrush(accent)
                painter.drawRect(QRectF(anchor.x() - 2.2, anchor.y() - 2.2, 4.4, 4.4))

                box_width = self.width() * (0.19 if spec["align"] != "center" else 0.16)
                if spec["align"] == "right":
                    rect = QRectF(end_point.x() - box_width, end_point.y() - 18, box_width - 6, 36)
                    align = Qt.AlignRight | Qt.AlignVCenter
                elif spec["align"] == "center":
                    rect = QRectF(end_point.x() - box_width * 0.5, end_point.y() - 18, box_width, 36)
                    align = Qt.AlignHCenter | Qt.AlignVCenter
                else:
                    rect = QRectF(end_point.x() + 6, end_point.y() - 18, box_width - 6, 36)
                    align = Qt.AlignLeft | Qt.AlignVCenter

                title = spec["label"]
                subtitle = _ACTION_LABELS.get(mapped_action, "Unassigned")
                title_color = accent if mapped_action is not None else QColor("#9ca3af")
                subtitle_color = QColor("#f8fafc") if mapped_action is not None else QColor("#6b7280")

                painter.setFont(QFont("Segoe UI", 8, QFont.DemiBold))
                painter.setPen(title_color)
                painter.drawText(rect, align | Qt.AlignTop, title)
                painter.setFont(QFont("Segoe UI", 9, QFont.Bold if mapped_action is not None else QFont.DemiBold))
                painter.setPen(subtitle_color)
                painter.drawText(rect.adjusted(0, 10, 0, 0), align | Qt.AlignBottom, subtitle)

            painter.end()

    class _ButtonConfigDialog(QDialog):
        """Remap controller buttons to curation actions (learn mode)."""

        _STYLE = """
            QDialog { background: #1e1e2e; }
            QLabel { color: #cdd6f4; background: transparent; }
            QWidget#controllerCanvas {
                border: 1px solid #2b3346;
                border-radius: 20px;
            }
            QTableWidget {
                background: #181825; color: #cdd6f4;
                gridline-color: #313244; border: 1px solid #45475a;
                selection-background-color: #45475a;
                selection-color: #cdd6f4;
            }
            QHeaderView::section {
                background: #313244; color: #cba6f7;
                border: none; padding: 4px 8px; font-weight: bold;
            }
            QGroupBox {
                color: #cba6f7;
                font-weight: bold;
                border: 1px solid #313244;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
            }
            QPushButton {
                background: #313244; color: #cdd6f4;
                border: 1px solid #585b70; border-radius: 4px;
                padding: 5px 12px; min-width: 60px;
            }
            QPushButton:hover { background: #45475a; }
            QPushButton:disabled { color: #585b70; border-color: #313244; }
            QPushButton#learnBtn {
                background: #1e3a5f; color: #89b4fa;
                border-color: #89b4fa; min-width: 60px;
            }
            QPushButton#learnBtn:hover { background: #2a4a70; }
            QPushButton#learnBtn[active="true"] {
                background: #3d1a2a; color: #f38ba8; border-color: #f38ba8;
            }
        """

        _ACTIONS = [
            # ── Curation ──────────────────────────────────────────────
            GamepadAction.MARK_GOOD,        GamepadAction.MARK_NOISE,
            GamepadAction.MARK_MUA,         GamepadAction.MERGE,
            GamepadAction.MARK_GOOD_NEXT,   GamepadAction.MARK_NOISE_NEXT,
            # ── Splitting ─────────────────────────────────────────────
            GamepadAction.SPLIT,
            GamepadAction.SPLIT_KMEANS,
            GamepadAction.SPLIT_ISI,
            # ── Navigation ────────────────────────────────────────────
            GamepadAction.NEXT_BEST,        GamepadAction.PREV_BEST,
            GamepadAction.NEXT_SIMILAR,     GamepadAction.PREV_SIMILAR,
            GamepadAction.SELECT_SIMILAR,   GamepadAction.UNSELECT_SIM,
            # ── Editing ───────────────────────────────────────────────
            GamepadAction.UNDO,             GamepadAction.REDO,
            GamepadAction.SAVE,             GamepadAction.TOGGLE_HUD,
        ]

        def __init__(self, plugin, parent=None):
            super().__init__(parent)
            self._plugin       = plugin
            self._learning_row = None
            self._learn_timer  = None

            self.setWindowTitle("Gamepad \u2014 Configure Buttons")
            self.setStyleSheet(self._STYLE)
            self.setMinimumSize(980, 620)
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

            lay = QVBoxLayout(self)
            lay.setSpacing(12)
            lay.setContentsMargins(14, 14, 14, 14)

            tip = QLabel(
                "Select an action, then click Learn. Press a controller button or click a control on the Xbox layout to assign it."
            )
            tip.setStyleSheet("color: #a6adc8; font-size: 11px;")
            tip.setWordWrap(True)
            lay.addWidget(tip)

            content = QHBoxLayout()
            content.setSpacing(14)
            lay.addLayout(content, 1)

            left = QVBoxLayout()
            left.setSpacing(10)
            content.addLayout(left, 3)

            self._controller_map = _ControllerMapWidget(self)
            self._controller_map.on_button_clicked = self._controller_button_clicked
            left.addWidget(self._controller_map, 1)

            fixed_box = QGroupBox("Always-on defaults")
            fixed_lay = QVBoxLayout(fixed_box)
            fixed_lay.setContentsMargins(10, 12, 10, 10)
            fixed_lay.setSpacing(4)
            for hint in _FIXED_CONTROL_HINTS:
                lbl = QLabel(hint)
                lbl.setStyleSheet("color: #a6adc8; font-size: 10px;")
                fixed_lay.addWidget(lbl)
            left.addWidget(fixed_box)

            right = QGroupBox("Action assignments")
            right_lay = QVBoxLayout(right)
            right_lay.setContentsMargins(12, 14, 12, 12)
            right_lay.setSpacing(8)
            content.addWidget(right, 2)

            self._tbl = QTableWidget(len(self._ACTIONS), 3)
            self._tbl.setHorizontalHeaderLabels(["Action", "Xbox control", ""])
            hdr = self._tbl.horizontalHeader()
            hdr.setSectionResizeMode(0, QHeaderView.Stretch)
            hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
            hdr.setSectionResizeMode(2, QHeaderView.Fixed)
            self._tbl.setColumnWidth(2, 72)
            self._tbl.verticalHeader().setVisible(False)
            self._tbl.setSelectionBehavior(QTableWidget.SelectRows)
            self._tbl.setEditTriggers(QTableWidget.NoEditTriggers)
            self._tbl.setFocusPolicy(Qt.NoFocus)
            self._tbl.setAlternatingRowColors(False)
            self._tbl.currentCellChanged.connect(self._on_row_selected)
            right_lay.addWidget(self._tbl, 1)

            self._status_lbl = QLabel("")
            self._status_lbl.setStyleSheet(
                "color: #f9e2af; font-size: 11px; min-height: 18px;"
            )
            self._status_lbl.setAlignment(Qt.AlignCenter)
            lay.addWidget(self._status_lbl)

            bot = QHBoxLayout()
            btn_defaults = QPushButton("Reset to Defaults")
            btn_defaults.clicked.connect(self._reset_defaults)
            bot.addWidget(btn_defaults)
            bot.addStretch()
            btn_apply = QPushButton("Apply")
            btn_apply.clicked.connect(self._apply)
            bot.addWidget(btn_apply)
            btn_ok = QPushButton("OK")
            btn_ok.setDefault(True)
            btn_ok.clicked.connect(self._ok)
            bot.addWidget(btn_ok)
            btn_cancel = QPushButton("Cancel")
            btn_cancel.clicked.connect(self.reject)
            bot.addWidget(btn_cancel)
            lay.addLayout(bot)

            self._work_map = dict(plugin._button_map)
            self._build_table()
            self._controller_map.set_mapping(self._work_map)
            self._tbl.selectRow(0)
            self._on_row_selected(0, 0, -1, -1)

        # ── helpers ───────────────────────────────────────────────────

        def _btn_for_action(self, action):
            for bid, act in self._work_map.items():
                if act == action:
                    return bid
            return None

        def _btn_label(self, bid):
            return _button_display_name(bid)

        def _build_table(self):
            for row, action in enumerate(self._ACTIONS):
                item_name = QTableWidgetItem(_ACTION_LABELS.get(action, str(action)))
                item_name.setFlags(item_name.flags() & ~Qt.ItemIsEditable)
                self._tbl.setItem(row, 0, item_name)

                bid = self._btn_for_action(action)
                item_btn = QTableWidgetItem(self._btn_label(bid))
                item_btn.setFlags(item_btn.flags() & ~Qt.ItemIsEditable)
                if bid is None:
                    item_btn.setForeground(QColor("#585b70"))
                self._tbl.setItem(row, 1, item_btn)

                learn_btn = QPushButton("Learn")
                learn_btn.setObjectName("learnBtn")
                learn_btn.clicked.connect(partial(self._toggle_learn, row))
                self._tbl.setCellWidget(row, 2, learn_btn)

        def _refresh_row(self, row):
            action = self._ACTIONS[row]
            bid = self._btn_for_action(action)
            item = self._tbl.item(row, 1)
            if item:
                item.setText(self._btn_label(bid))
                item.setForeground(
                    QColor("#cdd6f4") if bid is not None else QColor("#585b70")
                )
            self._controller_map.set_mapping(self._work_map)

        def _set_learn_btn(self, row, active):
            btn = self._tbl.cellWidget(row, 2)
            if btn is None:
                return
            btn.setText("Cancel" if active else "Learn")
            btn.setProperty("active", "true" if active else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        def _focus_action(self, action):
            self._controller_map.set_focus_action(action)
            self._controller_map.set_learning_action(self._ACTIONS[self._learning_row] if self._learning_row is not None else None)

        def _action_at_row(self, row):
            if row < 0 or row >= len(self._ACTIONS):
                return None
            return self._ACTIONS[row]

        def _find_row_for_button(self, button_id):
            action = self._work_map.get(button_id)
            if action is None:
                return None
            try:
                return self._ACTIONS.index(action)
            except ValueError:
                return None

        def _on_row_selected(self, row, _current_col, _prev_row, _prev_col):
            action = self._action_at_row(row)
            self._focus_action(action)
            if action is None:
                return
            control = self._btn_label(self._btn_for_action(action))
            self._status_lbl.setText(f"Selected {_ACTION_LABELS.get(action, str(action))} -> {control}")

        def _controller_button_clicked(self, button_id):
            if self._learning_row is not None:
                self._assign(button_id, self._learning_row)
                self._stop_learn(cancelled=False)
                return

            row = self._find_row_for_button(button_id)
            if row is None:
                self._status_lbl.setText(f"{_button_display_name(button_id)} is currently unassigned. Select an action and click Learn to map it.")
                return

            self._tbl.selectRow(row)
            action = self._ACTIONS[row]
            self._status_lbl.setText(f"{_button_display_name(button_id)} currently triggers {_ACTION_LABELS.get(action, str(action))}.")

        # ── learn mode ────────────────────────────────────────────────

        def _toggle_learn(self, row):
            if self._learning_row == row:
                self._stop_learn(cancelled=True)
            else:
                if self._learning_row is not None:
                    self._stop_learn(cancelled=True)
                self._start_learn(row)

        def _start_learn(self, row):
            self._learning_row = row
            label = _ACTION_LABELS.get(self._ACTIONS[row], str(self._ACTIONS[row]))
            self._status_lbl.setText(f"Assign {label}: press a controller button or click a control on the Xbox layout.")
            self._set_learn_btn(row, active=True)
            self._focus_action(self._ACTIONS[row])

            if _HAS_PYGAME and self._plugin._joystick is not None:
                self._learn_timer = QTimer(self)
                self._learn_timer.setInterval(50)
                self._learn_timer.timeout.connect(self._learn_poll)
                self._learn_timer.start()

        def _learn_poll(self):
            js = self._plugin._joystick
            if js is None:
                return
            try:
                pygame.event.pump()
                for bid in range(js.get_numbuttons()):
                    if js.get_button(bid):
                        self._assign(bid, self._learning_row)
                        self._stop_learn(cancelled=False)
                        return
            except Exception:
                self._stop_learn(cancelled=True)

        def _assign(self, btn_id, row):
            action = self._ACTIONS[row]
            # Clear old action for this button (display update)
            old_act = self._work_map.get(btn_id)
            if old_act is not None and old_act != action:
                try:
                    self._refresh_row(self._ACTIONS.index(old_act))
                except ValueError:
                    pass
            # Remove this action from any previous button
            for b in list(self._work_map):
                if self._work_map[b] == action:
                    del self._work_map[b]
            self._work_map[btn_id] = action
            self._refresh_row(row)
            label = _ACTION_LABELS.get(action, str(action))
            self._status_lbl.setText(f"Assigned Button {btn_id} \u2192 {label}")

        def _stop_learn(self, cancelled=False):
            if self._learn_timer:
                self._learn_timer.stop()
                self._learn_timer = None
            if self._learning_row is not None:
                self._set_learn_btn(self._learning_row, active=False)
            self._learning_row = None
            self._controller_map.set_learning_action(None)
            if cancelled:
                self._status_lbl.setText("")
            current_row = self._tbl.currentRow()
            self._focus_action(self._action_at_row(current_row))

        def _reset_defaults(self):
            self._stop_learn(cancelled=True)
            self._work_map = dict(DEFAULT_BUTTON_MAP)
            for row in range(len(self._ACTIONS)):
                self._refresh_row(row)
            self._status_lbl.setText("Reset to default Xbox layout.")
            self._controller_map.set_mapping(self._work_map)

        def _apply(self):
            self._stop_learn(cancelled=False)
            self._plugin._button_map = dict(self._work_map)
            self._plugin._save_button_map(self._plugin._button_map)
            self._plugin._update_mapping_surfaces()

        def _ok(self):
            self._apply()
            self.accept()

        def closeEvent(self, event):
            self._stop_learn(cancelled=True)
            super().closeEvent(event)


# =========================================================================
# The phy IPlugin
# =========================================================================

class NeuroPyGuiNGamepadController(IPlugin):
    """Game-controller curation + gamification for phy."""

    def attach_to_controller(self, controller):

        @connect
        def on_gui_ready(sender, gui):
            supervisor = controller.supervisor
            self._supervisor = supervisor
            self._controller = controller
            self._gui = gui

            # ── Gamification engine ───────────────────────────────────
            self._gam = _GamificationEngine()

            # ── HUD overlay ───────────────────────────────────────────
            self._hud = None
            self._hud_visible = True
            if _HAS_QT:
                self._hud = _GamepadHUD(self._gam, parent_widget=gui)
                self._hud.show()
                self._hud.raise_()
                self._position_hud()

                # Reposition HUD on window resize via a lightweight QObject filter
                self._resize_filter = _ResizeFilter(gui, self._position_hud)
                gui.installEventFilter(self._resize_filter)

            # ── Gamepad polling ───────────────────────────────────────
            self._joystick = None
            self._button_map = self._load_button_map()
            self._button_states = {}
            self._hat_states = {}
            self._trigger_pressed = False
            self._stick_accum_cluster = 0.0
            self._stick_accum_similar = 0.0
            self._deadzone = 0.12
            self._sensitivity = 5.0
            self._curve_mode = "quadratic"

            self._update_mapping_surfaces()

            if _HAS_PYGAME:
                pygame.init()
                pygame.joystick.init()
                self._detect_controller()

                # Main poll timer (50ms = 20Hz, same as ethoscore)
                self._poll_timer = QTimer(gui)
                self._poll_timer.setInterval(50)
                self._poll_timer.timeout.connect(self._poll_gamepad)
                self._poll_timer.start()

                # Combo tick timer
                self._combo_timer = QTimer(gui)
                self._combo_timer.setInterval(50)
                self._combo_timer.timeout.connect(
                    lambda: self._gam.tick_combo(50)
                )
                self._combo_timer.start()

                # Reconnect timer (check every 2s)
                self._reconnect_timer = QTimer(gui)
                self._reconnect_timer.setInterval(2000)
                self._reconnect_timer.timeout.connect(self._detect_controller)
                self._reconnect_timer.start()

                logger.info(
                    "Gamepad controller plugin active "
                    "(poll=50ms, curve=%s, deadzone=%.0f%%)",
                    self._curve_mode, self._deadzone * 100,
                )
            else:
                logger.warning(
                    "pygame not installed — gamepad plugin loaded "
                    "but controller input disabled. pip install pygame"
                )
                if self._hud:
                    self._hud.set_controller_name(
                        "pygame not installed"
                    )

            # ── Gamepad menu ──────────────────────────────────────────
            self._setup_menu(gui)

    def _position_hud(self):
        if not self._hud or not self._gui:
            return
        gui = self._gui
        hud = self._hud
        x = gui.width() - hud.width() - 12
        y = 12
        hud.move(x, y)
        hud.raise_()

    # ── Controller detection ──────────────────────────────────────────

    def _detect_controller(self):
        if not _HAS_PYGAME:
            return
        try:
            pygame.joystick.quit()
            pygame.joystick.init()
        except Exception:
            pass
        if pygame.joystick.get_count() > 0:
            js = pygame.joystick.Joystick(0)
            js.init()
            self._joystick = js
            self._button_states.clear()
            self._hat_states.clear()
            name = js.get_name()
            logger.info("Gamepad connected: %s", name)
            if self._hud:
                self._hud.set_controller_name(name)

            # Auto-detect right-stick X axis
            # Some controllers use axis 2, others axis 3
            global _RIGHT_STICK_X
            n = js.get_numaxes()
            if n >= 4:
                _RIGHT_STICK_X = 3  # common for Xbox on Windows
            elif n >= 3:
                _RIGHT_STICK_X = 2
        else:
            if self._joystick is not None:
                self._joystick = None
                logger.info("Gamepad disconnected")
                if self._hud:
                    self._hud.set_controller_disconnected()

    # ── Main poll loop ────────────────────────────────────────────────

    def _poll_gamepad(self):
        if not _HAS_PYGAME or self._joystick is None:
            return
        try:
            pygame.event.pump()
        except Exception:
            return

        js = self._joystick

        # ── Buttons (edge-detect) ─────────────────────────────────
        try:
            for btn_id, action in self._button_map.items():
                if btn_id >= js.get_numbuttons():
                    continue
                pressed = js.get_button(btn_id)
                was = self._button_states.get(btn_id, False)
                if pressed and not was:
                    self._dispatch_action(action)
                self._button_states[btn_id] = pressed
        except Exception:
            self._handle_disconnect()
            return

        # ── Hat / D-pad (edge-detect) ─────────────────────────────
        try:
            for hat_id in range(js.get_numhats()):
                val = js.get_hat(hat_id)
                prev = self._hat_states.get(hat_id, (0, 0))
                if val != prev and val != (0, 0):
                    action = DEFAULT_HAT_MAP.get(val)
                    if action:
                        self._dispatch_action(action)
                self._hat_states[hat_id] = val
        except Exception:
            self._handle_disconnect()
            return

        # ── Analog sticks ─────────────────────────────────────────
        try:
            speed = self._sensitivity / 2.0

            # Left stick Y → cluster navigation
            if _LEFT_STICK_Y < js.get_numaxes():
                ry = js.get_axis(_LEFT_STICK_Y)
                self._process_stick(ry, "cluster", speed)

            # Right stick X → similarity browsing
            if _RIGHT_STICK_X < js.get_numaxes():
                rx = js.get_axis(_RIGHT_STICK_X)
                self._process_stick(rx, "similar", speed)

            # Left trigger → Split (full pull)
            if _LEFT_TRIGGER < js.get_numaxes():
                lt = js.get_axis(_LEFT_TRIGGER)
                lt_on = lt > 0.9
                if lt_on and not self._trigger_pressed:
                    self._dispatch_action(GamepadAction.SPLIT)
                self._trigger_pressed = lt_on
        except Exception:
            self._handle_disconnect()

    def _process_stick(self, raw, direction, speed_factor):
        dz = self._deadzone
        if abs(raw) < dz:
            return
        sign = 1.0 if raw > 0 else -1.0
        norm = (abs(raw) - dz) / (1.0 - dz)
        curved = _apply_curve(norm, self._curve_mode)
        val = sign * curved * speed_factor

        if direction == "cluster":
            self._stick_accum_cluster += val
            while abs(self._stick_accum_cluster) >= 1.0:
                if self._stick_accum_cluster > 0:
                    self._supervisor.next_best()
                    self._stick_accum_cluster -= 1.0
                else:
                    self._supervisor.previous_best()
                    self._stick_accum_cluster += 1.0
        elif direction == "similar":
            self._stick_accum_similar += val
            while abs(self._stick_accum_similar) >= 1.0:
                if self._stick_accum_similar > 0:
                    self._supervisor.next()
                    self._stick_accum_similar -= 1.0
                else:
                    self._supervisor.previous()
                    self._stick_accum_similar += 1.0

    def _handle_disconnect(self):
        self._joystick = None
        self._button_states.clear()
        self._hat_states.clear()
        if self._hud:
            self._hud.set_controller_disconnected()

    def _settings(self):
        if not _HAS_QT:
            return None
        try:
            return QSettings("NeuroPyGuiN", "NeuroPyGuiN")
        except Exception:
            return None

    def _load_button_map(self):
        settings = self._settings()
        if settings is None:
            return dict(DEFAULT_BUTTON_MAP)
        raw_value = settings.value(_BUTTON_MAP_SETTINGS_KEY, "")
        return _deserialize_button_map(raw_value)

    def _save_button_map(self, button_map):
        settings = self._settings()
        if settings is None:
            return
        settings.setValue(_BUTTON_MAP_SETTINGS_KEY, _serialize_button_map(button_map))
        settings.sync()

    def _update_mapping_surfaces(self):
        if self._hud:
            self._hud.set_button_map(self._button_map)

    def _run_actions(self, action_group, *names):
        if action_group is None:
            return False
        for name in names:
            try:
                action_group.run(name)
                return True
            except Exception:
                continue
        return False

    def _trigger_qaction(self, qaction):
        if qaction is None:
            return False
        try:
            qaction.trigger()
            return True
        except Exception:
            return False

    def _run_edit_action(self, *names):
        return self._run_actions(getattr(self._supervisor, "actions", None), *names)

    def _run_select_action(self, *names):
        return self._run_actions(getattr(self._supervisor, "select_actions", None), *names)

    def _run_file_action(self, *names):
        return self._run_actions(getattr(self._gui, "file_actions", None), *names)

    def _selected_cluster_ids(self, include_similar=False):
        attr = "selected" if include_similar else "selected_clusters"
        cluster_ids = list(getattr(self._supervisor, attr, []) or [])
        return [int(cluster_id) for cluster_id in cluster_ids]

    def _advance_next_best(self):
        if not self._run_select_action("next_best"):
            self._supervisor.next_best()

    # ── Action dispatch ───────────────────────────────────────────────

    def _dispatch_action(self, action):
        sup = self._supervisor
        label = _ACTION_LABELS.get(action, "")

        try:
            if action is GamepadAction.MARK_GOOD:
                self._move_cluster("good")
            elif action is GamepadAction.MARK_NOISE:
                self._move_cluster("noise")
            elif action is GamepadAction.MARK_MUA:
                self._move_cluster("mua")
            elif action is GamepadAction.MERGE:
                self._do_merge()
            elif action is GamepadAction.NEXT_BEST:
                if not self._run_select_action("next_best"):
                    sup.next_best()
                self._show_action(label)
            elif action is GamepadAction.PREV_BEST:
                if not self._run_select_action("previous_best"):
                    sup.previous_best()
                self._show_action(label)
            elif action is GamepadAction.NEXT_SIMILAR:
                if not self._run_select_action("next"):
                    sup.next()
                self._show_action(label)
            elif action is GamepadAction.PREV_SIMILAR:
                if not self._run_select_action("previous"):
                    sup.previous()
                self._show_action(label)
            elif action is GamepadAction.UNDO:
                if not self._run_edit_action("undo"):
                    sup.undo()
                self._gam.action_undone()
                self._show_action(label)
            elif action is GamepadAction.REDO:
                if not self._run_edit_action("redo"):
                    sup.redo()
                self._show_action(label)
            elif action is GamepadAction.SPLIT:
                self._do_split()
            elif action is GamepadAction.SAVE:
                if not self._run_file_action("save"):
                    sup.save()
                self._show_action("Saved!")
            elif action is GamepadAction.TOGGLE_HUD:
                self._toggle_hud()
            elif action is GamepadAction.UNSELECT_SIM:
                if not self._run_select_action("unselect_similar"):
                    sup.unselect_similar()
                self._show_action(label)
            elif action is GamepadAction.SPLIT_KMEANS:
                self._do_split_kmeans()
            elif action is GamepadAction.SPLIT_ISI:
                self._do_split_isi()
            elif action is GamepadAction.SELECT_SIMILAR:
                self._do_select_similar()
            elif action is GamepadAction.MARK_GOOD_NEXT:
                if self._move_cluster("good"):
                    self._advance_next_best()
            elif action is GamepadAction.MARK_NOISE_NEXT:
                if self._move_cluster("noise"):
                    self._advance_next_best()
        except Exception as exc:
            logger.warning("Gamepad action %s failed: %s", action, exc)
            self._show_action(f"Error: {exc}")

    def _move_cluster(self, group):
        sup = self._supervisor
        cluster_ids = self._selected_cluster_ids(include_similar=False)
        if not cluster_ids:
            self._show_action("No cluster selected")
            return False

        n_spikes = sum(sup.n_spikes(c) for c in cluster_ids)
        if not self._run_edit_action(f"move_best_to_{group}"):
            sup.move(group, "best")

        ids_str = ", ".join(str(c) for c in cluster_ids[:3])
        if len(cluster_ids) > 3:
            ids_str += f" (+{len(cluster_ids) - 3})"
        self._show_action(f"{ids_str} -> {group.upper()}")
        self._gam.cluster_curated(group, n_spikes)
        return True

    def _do_merge(self):
        sup = self._supervisor
        sel = self._selected_cluster_ids(include_similar=True)
        if len(sel) < 2:
            self._show_action("Select 2+ to merge")
            return
        if not self._run_edit_action("merge"):
            sup.merge(cluster_ids=sel)
        self._gam.merged()
        self._show_action(f"Merged {len(sel)} clusters")

    def _do_split(self):
        try:
            if not self._run_edit_action("split"):
                self._supervisor.split()
            self._gam.split()
            self._show_action("Split")
        except Exception as exc:
            self._show_action(f"Split: {exc}")

    def _toggle_hud(self):
        if self._hud:
            self._hud_visible = not self._hud_visible
            self._hud.setVisible(self._hud_visible)

    # ── Plugin-action helpers ─────────────────────────────────────────

    def _do_split_kmeans(self):
        """K-means split — tries the phy action system first."""
        if self._run_edit_action(
            "split_init", "split_kmeans", "k_means_split", "K_means_clustering"
        ):
            self._show_action("K-means split")
            self._gam.split()
            return
        finder = getattr(self._supervisor, "_find_cluster_context_action", None)
        if callable(finder):
            if self._trigger_qaction(finder("K_means_clustering")):
                self._show_action("K-means split")
                self._gam.split()
                return
        # Fallback: controller-level method (some phy builds)
        try:
            ctrl = self._controller
            if hasattr(ctrl, "split_init"):
                ctrl.split_init()
                self._show_action("K-means split")
                self._gam.split()
                return
        except Exception as exc:
            logger.debug("split_init failed: %s", exc)
        # Last resort: basic split
        try:
            self._supervisor.split()
            self._show_action("Split (K-means N/A)")
            self._gam.split()
        except Exception as exc:
            self._show_action(f"K-means: {exc}")

    def _do_split_isi(self):
        """ISI-based split — triggers the registered phy plugin action."""
        if self._run_edit_action(
            "split_short_isi", "Split short ISI", "split_short_isi_1", "split_isi", "isi_split"
        ):
            self._show_action("Split ISI")
            self._gam.split()
        else:
            self._show_action("Split ISI: plugin not loaded")

    def _do_select_similar(self):
        """Select the first or next similar cluster."""
        if not self._selected_cluster_ids(include_similar=False):
            self._show_action("No cluster selected")
            return
        try:
            if not self._run_select_action("next"):
                self._supervisor.next()
            self._show_action("Select similar")
        except Exception as exc:
            self._show_action(f"Select similar: {exc}")

    def _show_action(self, text):
        if self._hud:
            self._hud.set_action_text(text)

    # ── Gamepad menu ──────────────────────────────────────────────────

    def _setup_menu(self, gui):
        """Add a Gamepad top-level menu to phy's menubar."""
        if not _HAS_QT:
            return
        try:
            menubar = gui.menuBar()
            gp_menu = menubar.addMenu("&Gamepad")

            act_status = gp_menu.addAction("Controller Status\u2026")
            act_status.triggered.connect(
                lambda: self._show_controller_status(gui)
            )

            act_config = gp_menu.addAction("Configure Buttons\u2026")
            act_config.triggered.connect(
                lambda: self._show_button_config(gui)
            )

            gp_menu.addSeparator()

            act_hud = gp_menu.addAction("Toggle HUD")
            act_hud.triggered.connect(self._toggle_hud)

            act_reset = gp_menu.addAction("Reset Session Stats")
            act_reset.triggered.connect(self._gam.reset_session)

            logger.info("Gamepad menu added to phy GUI")
        except Exception as exc:
            logger.warning("Could not add Gamepad menu: %s", exc)

    def _show_controller_status(self, parent):
        if not _HAS_QT:
            return
        dlg = _ControllerStatusDialog(self, parent=parent)
        dlg.exec_()

    def _show_button_config(self, parent):
        if not _HAS_QT:
            return
        dlg = _ButtonConfigDialog(self, parent=parent)
        dlg.exec_()
'''

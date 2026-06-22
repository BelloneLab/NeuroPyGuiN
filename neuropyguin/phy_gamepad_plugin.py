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

from pathlib import Path


_ASSET_DIR = Path(__file__).resolve().parent / "assets"


def _gamepad_plugin_source() -> str:
    """Return the full plugin source as a string (phy discovers .py files)."""
    xbox_image = repr(str((_ASSET_DIR / "xbox.png").resolve()))
    switch_image = repr(str((_ASSET_DIR / "switch.png").resolve()))
    source = r'''
import json
import logging
import math
import ctypes
import os
import sys
import time
from collections import deque
from ctypes import wintypes
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
        QFrame, QProgressBar, QDialog, QScrollArea,
        QTableWidget, QTableWidgetItem, QComboBox,
        QHeaderView, QGroupBox, QSizePolicy,
    )
    from PyQt5.QtGui import QFont, QImage, QPainter, QPen, QPixmap
    from PyQt5.QtCore import QRect, QRectF, QSettings
    _HAS_QT = True
except ImportError:
    _HAS_QT = False

# ── safely import pygame ──────────────────────────────────────────────────
# Force SDL's dummy video and audio drivers BEFORE pygame is initialized. A full
# pygame.init() otherwise brings up SDL's real video subsystem, whose graphics
# context collides with phy's Qt/OpenGL context and crashes the whole process
# with a native access violation (exit code 0xC0000005). The gamepad plugin only
# needs joystick polling, which works fine under the dummy drivers (no window,
# GPU, or audio device is created).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
try:
    import pygame
    _HAS_PYGAME = True
except ImportError:
    _HAS_PYGAME = False

_IS_WINDOWS = sys.platform.startswith("win")
_XINPUT_DLL = None
if _IS_WINDOWS:
    for _dll_name in ("xinput1_4.dll", "xinput1_3.dll", "xinput9_1_0.dll", "xinput1_2.dll", "xinput1_1.dll"):
        try:
            _XINPUT_DLL = ctypes.WinDLL(_dll_name)
            break
        except Exception:
            continue
_HAS_XINPUT = _XINPUT_DLL is not None

if _HAS_XINPUT:
    class _XINPUT_GAMEPAD(ctypes.Structure):
        # Layout must match the Win32 XINPUT_GAMEPAD struct exactly. BYTE in the
        # Windows headers is unsigned (0..255), so the triggers use c_ubyte, not
        # wintypes.BYTE (which is signed c_byte). A wrong field type here makes
        # XInputGetState read into a mismatched buffer, which is the kind of
        # ctypes signature error that can corrupt memory.
        _fields_ = [
            ("wButtons", wintypes.WORD),
            ("bLeftTrigger", ctypes.c_ubyte),
            ("bRightTrigger", ctypes.c_ubyte),
            ("sThumbLX", ctypes.c_short),
            ("sThumbLY", ctypes.c_short),
            ("sThumbRX", ctypes.c_short),
            ("sThumbRY", ctypes.c_short),
        ]


    class _XINPUT_STATE(ctypes.Structure):
        _fields_ = [
            ("dwPacketNumber", wintypes.DWORD),
            ("Gamepad", _XINPUT_GAMEPAD),
        ]


    _XINPUT_GET_STATE = _XINPUT_DLL.XInputGetState
    _XINPUT_GET_STATE.argtypes = [wintypes.DWORD, ctypes.POINTER(_XINPUT_STATE)]
    _XINPUT_GET_STATE.restype = wintypes.DWORD
else:
    _XINPUT_GET_STATE = None


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


LAYOUT_XBOX = "xbox"
LAYOUT_SWITCH = "switch"

XBOX_LAYOUT_IMAGE = __XBOX_LAYOUT_IMAGE__
SWITCH_LAYOUT_IMAGE = __SWITCH_LAYOUT_IMAGE__


# =========================================================================
# Default mappings and layout metadata
# =========================================================================

DEFAULT_XBOX_BUTTON_MAP = {
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

DEFAULT_SWITCH_BUTTON_MAP = {
    0: GamepadAction.MARK_GOOD,      # B
    1: GamepadAction.MARK_NOISE,     # A
    2: GamepadAction.MARK_MUA,       # Y
    3: GamepadAction.MERGE,          # X
    4: GamepadAction.UNDO,           # L
    5: GamepadAction.REDO,           # R
    6: GamepadAction.TOGGLE_HUD,     # Minus
    7: GamepadAction.SAVE,           # Plus
    9: GamepadAction.UNSELECT_SIM,   # R3
}

DEFAULT_HAT_MAP = {
    (0,  1): GamepadAction.PREV_BEST,
    (0, -1): GamepadAction.NEXT_BEST,
    (-1, 0): GamepadAction.PREV_SIMILAR,
    ( 1, 0): GamepadAction.NEXT_SIMILAR,
}

_LAYOUT_SETTINGS_KEY = "gamepad/layout_v1"
_BUTTON_MAP_SETTINGS_KEY_TEMPLATE = "gamepad/button_map_{layout}_v1"
_LEGACY_BUTTON_MAP_SETTINGS_KEY = "gamepad/button_map_v2"

_LAYOUTS = {
    LAYOUT_XBOX: {
        "title": "Xbox",
        "image_path": XBOX_LAYOUT_IMAGE,
        "control_column": "Xbox control",
        "default_button_map": DEFAULT_XBOX_BUTTON_MAP,
        "button_labels": {
            0: "A", 1: "B", 2: "X", 3: "Y", 4: "LB",
            5: "RB", 6: "View", 7: "Menu", 8: "L3", 9: "R3",
        },
        "hotspots": {
            0: {"center": (0.730, 0.555), "size": (0.088, 0.088), "shape": "round", "text": "", "accent": "#59d16f"},
            1: {"center": (0.788, 0.490), "size": (0.085, 0.085), "shape": "round", "text": "", "accent": "#ff6464"},
            2: {"center": (0.670, 0.490), "size": (0.085, 0.085), "shape": "round", "text": "", "accent": "#46b6ff"},
            3: {"center": (0.728, 0.424), "size": (0.085, 0.085), "shape": "round", "text": "", "accent": "#f4d957"},
            4: {"center": (0.235, 0.172), "size": (0.150, 0.055), "shape": "pill", "text": "LB", "accent": "#d6dbe4"},
            5: {"center": (0.762, 0.172), "size": (0.150, 0.055), "shape": "pill", "text": "RB", "accent": "#d6dbe4"},
            6: {"center": (0.442, 0.382), "size": (0.060, 0.060), "shape": "round", "text": "View", "accent": "#cbd5e1"},
            7: {"center": (0.560, 0.382), "size": (0.060, 0.060), "shape": "round", "text": "Menu", "accent": "#cbd5e1"},
            8: {"center": (0.245, 0.385), "size": (0.110, 0.110), "shape": "round", "text": "L3", "accent": "#94a3b8"},
            9: {"center": (0.630, 0.588), "size": (0.108, 0.108), "shape": "round", "text": "R3", "accent": "#94a3b8"},
        },
    },
    LAYOUT_SWITCH: {
        "title": "Nintendo Switch",
        "image_path": SWITCH_LAYOUT_IMAGE,
        "control_column": "Switch control",
        "default_button_map": DEFAULT_SWITCH_BUTTON_MAP,
        "button_labels": {
            0: "B", 1: "A", 2: "Y", 3: "X", 4: "L",
            5: "R", 6: "-", 7: "+", 8: "L3", 9: "R3",
        },
        "hotspots": {
            0: {"center": (0.744, 0.466), "size": (0.088, 0.088), "shape": "round", "text": "", "accent": "#ff8a8a"},
            1: {"center": (0.826, 0.381), "size": (0.088, 0.088), "shape": "round", "text": "", "accent": "#9fdf8d"},
            2: {"center": (0.659, 0.381), "size": (0.088, 0.088), "shape": "round", "text": "", "accent": "#d8d266"},
            3: {"center": (0.744, 0.294), "size": (0.088, 0.088), "shape": "round", "text": "", "accent": "#80c7ff"},
            4: {"center": (0.200, 0.155), "size": (0.132, 0.055), "shape": "pill", "text": "L", "accent": "#d6dbe4"},
            5: {"center": (0.793, 0.155), "size": (0.132, 0.055), "shape": "pill", "text": "R", "accent": "#d6dbe4"},
            6: {"center": (0.395, 0.246), "size": (0.062, 0.062), "shape": "round", "text": "-", "accent": "#cbd5e1"},
            7: {"center": (0.590, 0.246), "size": (0.062, 0.062), "shape": "round", "text": "+", "accent": "#cbd5e1"},
            8: {"center": (0.152, 0.327), "size": (0.118, 0.118), "shape": "round", "text": "L3", "accent": "#94a3b8"},
            9: {"center": (0.584, 0.528), "size": (0.118, 0.118), "shape": "round", "text": "R3", "accent": "#94a3b8"},
        },
    },
}

_FIXED_CONTROL_HINTS = (
    "D-pad Up/Down -> Prev/Next best",
    "D-pad Left/Right -> Prev/Next similar",
    "LT -> Split (always active)",
)

_ACTION_SECTIONS = (
    ("Curation", (
        GamepadAction.MARK_GOOD,
        GamepadAction.MARK_NOISE,
        GamepadAction.MARK_MUA,
        GamepadAction.MERGE,
        GamepadAction.MARK_GOOD_NEXT,
        GamepadAction.MARK_NOISE_NEXT,
    )),
    ("Splitting", (
        GamepadAction.SPLIT,
        GamepadAction.SPLIT_KMEANS,
        GamepadAction.SPLIT_ISI,
    )),
    ("Navigation", (
        GamepadAction.NEXT_BEST,
        GamepadAction.PREV_BEST,
        GamepadAction.NEXT_SIMILAR,
        GamepadAction.PREV_SIMILAR,
        GamepadAction.SELECT_SIMILAR,
        GamepadAction.UNSELECT_SIM,
    )),
    ("Editing", (
        GamepadAction.UNDO,
        GamepadAction.REDO,
        GamepadAction.SAVE,
        GamepadAction.TOGGLE_HUD,
    )),
)

_CONTROLLER_IMAGE_CACHE = {}

# Axis IDs
_LEFT_STICK_Y   = 1
_RIGHT_STICK_X  = 2   # often axis 3 on some pads — auto-detected
_LEFT_TRIGGER   = 4

_PYGAME_BACKEND = "pygame"
_XINPUT_BACKEND = "xinput"

_XINPUT_ERROR_SUCCESS = 0
_XINPUT_GAMEPAD_DPAD_UP = 0x0001
_XINPUT_GAMEPAD_DPAD_DOWN = 0x0002
_XINPUT_GAMEPAD_DPAD_LEFT = 0x0004
_XINPUT_GAMEPAD_DPAD_RIGHT = 0x0008
_XINPUT_GAMEPAD_START = 0x0010
_XINPUT_GAMEPAD_BACK = 0x0020
_XINPUT_GAMEPAD_LEFT_THUMB = 0x0040
_XINPUT_GAMEPAD_RIGHT_THUMB = 0x0080
_XINPUT_GAMEPAD_LEFT_SHOULDER = 0x0100
_XINPUT_GAMEPAD_RIGHT_SHOULDER = 0x0200
_XINPUT_GAMEPAD_A = 0x1000
_XINPUT_GAMEPAD_B = 0x2000
_XINPUT_GAMEPAD_X = 0x4000
_XINPUT_GAMEPAD_Y = 0x8000

_XINPUT_BUTTON_BITS = {
    0: _XINPUT_GAMEPAD_A,
    1: _XINPUT_GAMEPAD_B,
    2: _XINPUT_GAMEPAD_X,
    3: _XINPUT_GAMEPAD_Y,
    4: _XINPUT_GAMEPAD_LEFT_SHOULDER,
    5: _XINPUT_GAMEPAD_RIGHT_SHOULDER,
    6: _XINPUT_GAMEPAD_BACK,
    7: _XINPUT_GAMEPAD_START,
    8: _XINPUT_GAMEPAD_LEFT_THUMB,
    9: _XINPUT_GAMEPAD_RIGHT_THUMB,
}


def _normalize_thumb(value):
    if value >= 0:
        return min(1.0, float(value) / 32767.0)
    return max(-1.0, float(value) / 32768.0)


def _normalize_trigger(value):
    return max(0.0, min(1.0, float(value) / 255.0))


class _XInputController:
    """Thin XInput wrapper that mimics the pygame joystick interface we use."""

    def __init__(self, user_index):
        self._user_index = int(user_index)
        self._state = _XINPUT_STATE()
        self._name = f"XInput Controller {self._user_index + 1}"

    def refresh_state(self):
        if not _HAS_XINPUT or _XINPUT_GET_STATE is None:
            return False
        state = _XINPUT_STATE()
        try:
            result = _XINPUT_GET_STATE(self._user_index, ctypes.byref(state))
        except Exception:
            # Any ctypes failure here means input is simply unavailable. Degrade
            # quietly to a disconnect rather than letting it bubble up.
            logger.exception("XInputGetState raised for slot %s", self._user_index)
            return False
        if result != _XINPUT_ERROR_SUCCESS:
            return False
        self._state = state
        return True

    def init(self):
        return self.refresh_state()

    def get_name(self):
        return self._name

    def get_numaxes(self):
        return 5

    def get_numbuttons(self):
        return len(_XINPUT_BUTTON_BITS)

    def get_numhats(self):
        return 1

    def get_button(self, button_id):
        bit = _XINPUT_BUTTON_BITS.get(int(button_id))
        if bit is None:
            return 0
        return 1 if (self._state.Gamepad.wButtons & bit) else 0

    def get_hat(self, hat_id):
        if int(hat_id) != 0:
            return (0, 0)
        buttons = self._state.Gamepad.wButtons
        x = int(bool(buttons & _XINPUT_GAMEPAD_DPAD_RIGHT)) - int(bool(buttons & _XINPUT_GAMEPAD_DPAD_LEFT))
        y = int(bool(buttons & _XINPUT_GAMEPAD_DPAD_UP)) - int(bool(buttons & _XINPUT_GAMEPAD_DPAD_DOWN))
        return (x, y)

    def get_axis(self, axis_id):
        axis_id = int(axis_id)
        pad = self._state.Gamepad
        if axis_id == 0:
            return _normalize_thumb(pad.sThumbLX)
        if axis_id == 1:
            return -_normalize_thumb(pad.sThumbLY)
        if axis_id == 2:
            return _normalize_thumb(pad.sThumbRX)
        if axis_id == 3:
            return -_normalize_thumb(pad.sThumbRY)
        if axis_id == 4:
            return _normalize_trigger(pad.bLeftTrigger)
        return 0.0


def _layout_config(layout_key):
    return _LAYOUTS.get(layout_key, _LAYOUTS[LAYOUT_XBOX])


def _button_map_settings_key(layout_key):
    return _BUTTON_MAP_SETTINGS_KEY_TEMPLATE.format(layout=layout_key)


def _default_button_map(layout_key):
    return dict(_layout_config(layout_key).get("default_button_map", DEFAULT_XBOX_BUTTON_MAP))


def _button_display_name(btn_id, layout_key=LAYOUT_XBOX):
    if btn_id is None:
        return "(unassigned)"
    label = _layout_config(layout_key).get("button_labels", {}).get(int(btn_id))
    return label or f"Button {int(btn_id)}"


def _serialize_button_map(button_map):
    payload = {str(int(btn_id)): action.name for btn_id, action in (button_map or {}).items()}
    return json.dumps(payload, sort_keys=True)


def _deserialize_button_map(raw_value, default_button_map):
    if not raw_value:
        return dict(default_button_map)
    try:
        data = json.loads(str(raw_value))
    except Exception:
        return dict(default_button_map)
    if not isinstance(data, dict):
        return dict(default_button_map)

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
    return out or dict(default_button_map)


def _is_background_candidate(color):
    if color.alpha() <= 0:
        return False
    rgb_min = min(color.red(), color.green(), color.blue())
    rgb_max = max(color.red(), color.green(), color.blue())
    return rgb_max >= 185 and (rgb_max - rgb_min) <= 26


def _processed_controller_asset(image_path):
    cached = _CONTROLLER_IMAGE_CACHE.get(image_path)
    if cached is not None:
        return cached
    if not _HAS_QT:
        result = (QPixmap(), QRect())
        _CONTROLLER_IMAGE_CACHE[image_path] = result
        return result

    pixmap = QPixmap(image_path)
    if pixmap.isNull():
        result = (pixmap, QRect())
        _CONTROLLER_IMAGE_CACHE[image_path] = result
        return result

    image = pixmap.toImage().convertToFormat(QImage.Format_ARGB32)
    width = image.width()
    height = image.height()
    visited = bytearray(width * height)
    queue = deque()

    def _visit(x, y):
        if x < 0 or x >= width or y < 0 or y >= height:
            return
        index = y * width + x
        if visited[index]:
            return
        visited[index] = 1
        color = image.pixelColor(x, y)
        if _is_background_candidate(color):
            queue.append((x, y))

    for x in range(width):
        _visit(x, 0)
        _visit(x, height - 1)
    for y in range(height):
        _visit(0, y)
        _visit(width - 1, y)

    while queue:
        x, y = queue.popleft()
        color = image.pixelColor(x, y)
        if _is_background_candidate(color):
            color.setAlpha(0)
            image.setPixelColor(x, y, color)
            _visit(x - 1, y)
            _visit(x + 1, y)
            _visit(x, y - 1)
            _visit(x, y + 1)

    left = width
    top = height
    right = -1
    bottom = -1
    for y in range(height):
        for x in range(width):
            if image.pixelColor(x, y).alpha() <= 0:
                continue
            if x < left:
                left = x
            if y < top:
                top = y
            if x > right:
                right = x
            if y > bottom:
                bottom = y

    if right < left or bottom < top:
        source_rect = QRect(0, 0, width, height)
    else:
        pad_x = max(8, int((right - left + 1) * 0.04))
        pad_y = max(8, int((bottom - top + 1) * 0.04))
        rect_x = max(0, left - pad_x)
        rect_y = max(0, top - pad_y)
        rect_right = min(width - 1, right + pad_x)
        rect_bottom = min(height - 1, bottom + pad_y)
        source_rect = QRect(
            rect_x,
            rect_y,
            rect_right - rect_x + 1,
            rect_bottom - rect_y + 1,
        )

    result = (QPixmap.fromImage(image), source_rect)
    _CONTROLLER_IMAGE_CACHE[image_path] = result
    return result


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

            self.set_button_profile(LAYOUT_XBOX, {})

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

        def set_button_profile(self, layout_key, button_map):
            layout = _layout_config(layout_key)
            inverse = {
                action: _button_display_name(btn_id, layout_key)
                for btn_id, action in (button_map or {}).items()
            }
            parts = [
                f"<span style='color:#59d16f;'>{inverse.get(GamepadAction.MARK_GOOD, _button_display_name(0, layout_key))}</span> Good",
                f"<span style='color:#ff6464;'>{inverse.get(GamepadAction.MARK_NOISE, _button_display_name(1, layout_key))}</span> Noise",
                f"<span style='color:#46b6ff;'>{inverse.get(GamepadAction.MARK_MUA, _button_display_name(2, layout_key))}</span> MUA",
                f"<span style='color:#f4d957;'>{inverse.get(GamepadAction.MERGE, _button_display_name(3, layout_key))}</span> Merge",
                f"<span style='color:#cbd5e1;'>{inverse.get(GamepadAction.SAVE, _button_display_name(7, layout_key))}</span> Save",
            ]
            self._lbl_mapping.setText(f"{layout.get('title', 'Xbox')} layout  |  " + "  |  ".join(parts))

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
            self._lbl_backend = QLabel()
            self._lbl_name    = QLabel()
            self._lbl_axes    = QLabel()
            self._lbl_buttons = QLabel()
            self._lbl_hats    = QLabel()
            for lbl in (self._lbl_pygame, self._lbl_backend, self._lbl_name,
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
            if _HAS_PYGAME:
                self._lbl_pygame.setText(f"pygame {pygame.version.ver}  \u2713")
                self._lbl_pygame.setStyleSheet("color: #a6e3a1;")
            else:
                self._lbl_pygame.setText("pygame: not installed")
                self._lbl_pygame.setStyleSheet("color: #f38ba8;")

            if not (_HAS_PYGAME or _HAS_XINPUT):
                self._lbl_backend.setText("Backends: none available")
                self._lbl_backend.setStyleSheet("color: #f38ba8;")
                for lbl in (self._lbl_name, self._lbl_axes, self._lbl_buttons, self._lbl_hats):
                    lbl.setText("")
                return

            plugin._detect_controller()
            js = plugin._joystick
            if js is None:
                self._lbl_name.setText("Status:  not connected")
                self._lbl_name.setStyleSheet("color: #f38ba8;")
                backends = []
                if _HAS_PYGAME:
                    backends.append("pygame joystick")
                if _HAS_XINPUT:
                    backends.append("Windows XInput")
                self._lbl_backend.setText("Backends: " + (", ".join(backends) if backends else "none"))
                self._lbl_backend.setStyleSheet("color: #a6adc8;")
                for lbl in (self._lbl_axes, self._lbl_buttons, self._lbl_hats):
                    lbl.setText("")
            else:
                backend = getattr(plugin, "_controller_backend", "")
                if backend == _XINPUT_BACKEND:
                    backend_label = "Windows XInput"
                elif backend == _PYGAME_BACKEND:
                    backend_label = "pygame joystick"
                else:
                    backend_label = "unknown"
                self._lbl_backend.setText(f"Backend: {backend_label}")
                self._lbl_backend.setStyleSheet("color: #cdd6f4;")
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

    class _AssignmentRow(QFrame):
        """Compact row showing an action, its current control, and a learn button."""

        def __init__(self, action, parent=None):
            super().__init__(parent)
            self._action = action
            self._selected = False
            self._learning = False
            self.on_selected = None
            self.on_learn_clicked = None

            self.setObjectName("assignmentRow")
            self.setCursor(Qt.PointingHandCursor)

            layout = QHBoxLayout(self)
            layout.setContentsMargins(12, 10, 12, 10)
            layout.setSpacing(10)

            self._name_lbl = QLabel(_ACTION_LABELS.get(action, str(action)))
            self._name_lbl.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            self._name_lbl.setStyleSheet("font-size: 12px; font-weight: 600;")
            layout.addWidget(self._name_lbl, 1)

            self._control_chip = QLabel("(unassigned)")
            self._control_chip.setAlignment(Qt.AlignCenter)
            self._control_chip.setMinimumWidth(92)
            self._control_chip.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            layout.addWidget(self._control_chip)

            self._learn_btn = QPushButton("Learn")
            self._learn_btn.setObjectName("learnBtn")
            self._learn_btn.clicked.connect(self._emit_learn_clicked)
            layout.addWidget(self._learn_btn)

            self._refresh_style()

        @property
        def action(self):
            return self._action

        def mousePressEvent(self, event):
            if event.button() == Qt.LeftButton and callable(self.on_selected):
                self.on_selected(self._action)
            super().mousePressEvent(event)

        def _emit_learn_clicked(self):
            if callable(self.on_learn_clicked):
                self.on_learn_clicked(self._action)

        def set_control(self, text, assigned):
            self._control_chip.setText(text)
            if assigned:
                self._control_chip.setStyleSheet(
                    "background: rgba(137, 180, 250, 0.16);"
                    "color: #dce9ff;"
                    "border: 1px solid rgba(137, 180, 250, 0.50);"
                    "border-radius: 10px;"
                    "padding: 4px 10px;"
                    "font-size: 11px; font-weight: 700;"
                )
            else:
                self._control_chip.setStyleSheet(
                    "background: rgba(148, 163, 184, 0.10);"
                    "color: #7f8ba3;"
                    "border: 1px solid rgba(148, 163, 184, 0.24);"
                    "border-radius: 10px;"
                    "padding: 4px 10px;"
                    "font-size: 11px; font-weight: 600;"
                )

        def set_selected(self, selected):
            self._selected = bool(selected)
            self._refresh_style()

        def set_learning(self, learning):
            self._learning = bool(learning)
            self._learn_btn.setText("Cancel" if self._learning else "Learn")
            self._learn_btn.setProperty("active", "true" if self._learning else "false")
            self._learn_btn.style().unpolish(self._learn_btn)
            self._learn_btn.style().polish(self._learn_btn)
            self._refresh_style()

        def _refresh_style(self):
            if self._learning:
                background = "rgba(243, 139, 168, 0.14)"
                border = "rgba(243, 139, 168, 0.42)"
            elif self._selected:
                background = "rgba(137, 180, 250, 0.13)"
                border = "rgba(137, 180, 250, 0.34)"
            else:
                background = "rgba(255, 255, 255, 0.03)"
                border = "rgba(148, 163, 184, 0.14)"
            self.setStyleSheet(
                "QFrame#assignmentRow {"
                f"background: {background};"
                f"border: 1px solid {border};"
                "border-radius: 12px;"
                "}"
            )

    class _ControllerMapWidget(QWidget):
        """Controller image view with cleaned artwork and low-noise hotspots."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setObjectName("controllerCanvas")
            self.setMinimumSize(540, 380)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.setAttribute(Qt.WA_StyledBackground, True)

            self._layout_key = LAYOUT_XBOX
            self._mapping = {}
            self._focused_action = None
            self._learning_action = None
            self._pixmap = QPixmap()
            self._source_rect = QRect()
            self._display_rect = QRect()
            self.on_button_clicked = None

            self._buttons = {}
            for button_id in range(10):
                btn = QPushButton(self)
                btn.setFocusPolicy(Qt.NoFocus)
                btn.setCursor(Qt.PointingHandCursor)
                btn.clicked.connect(partial(self._emit_button_clicked, button_id))
                self._buttons[button_id] = btn

            self.set_layout_key(LAYOUT_XBOX)
            self._refresh_button_styles()

        def _emit_button_clicked(self, button_id):
            if callable(self.on_button_clicked):
                self.on_button_clicked(button_id)

        def set_layout_key(self, layout_key):
            self._layout_key = layout_key if layout_key in _LAYOUTS else LAYOUT_XBOX
            image_path = _layout_config(self._layout_key).get("image_path", "")
            self._pixmap, self._source_rect = _processed_controller_asset(image_path)
            self._update_button_labels()
            self._update_overlay_geometry()
            self.update()

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
            self._update_overlay_geometry()

        def _update_button_labels(self):
            layout = _layout_config(self._layout_key)
            labels = layout.get("button_labels", {})
            hotspots = layout.get("hotspots", {})
            for button_id, btn in self._buttons.items():
                spec = hotspots.get(button_id)
                if spec is None:
                    btn.hide()
                    continue
                btn.setText(spec.get("text") or labels.get(button_id, ""))
                btn.show()

        def _update_overlay_geometry(self):
            margin = 18
            canvas = QRect(
                margin,
                margin,
                max(1, self.width() - margin * 2),
                max(1, self.height() - margin * 2),
            )
            source_rect = self._source_rect if self._source_rect.isValid() else QRect(0, 0, max(1, self._pixmap.width()), max(1, self._pixmap.height()))
            if self._pixmap.isNull() or source_rect.width() <= 0 or source_rect.height() <= 0:
                self._display_rect = canvas
            else:
                scale = min(canvas.width() / float(source_rect.width()), canvas.height() / float(source_rect.height()))
                draw_w = max(1, int(source_rect.width() * scale))
                draw_h = max(1, int(source_rect.height() * scale))
                self._display_rect = QRect(
                    canvas.x() + (canvas.width() - draw_w) // 2,
                    canvas.y() + (canvas.height() - draw_h) // 2,
                    draw_w,
                    draw_h,
                )

            hotspots = _layout_config(self._layout_key).get("hotspots", {})
            for button_id, btn in self._buttons.items():
                spec = hotspots.get(button_id)
                if spec is None:
                    btn.hide()
                    continue
                cx, cy = self._hotspot_center(spec)
                width, height = self._hotspot_size(spec)
                btn.setGeometry(cx - width // 2, cy - height // 2, width, height)
                btn.show()
            self._refresh_button_styles()

        def _hotspot_center(self, spec):
            if self._pixmap.isNull() or not self._display_rect.isValid():
                return self.width() // 2, self.height() // 2
            source_rect = self._source_rect if self._source_rect.isValid() else QRect(0, 0, max(1, self._pixmap.width()), max(1, self._pixmap.height()))
            source_w = max(1, source_rect.width())
            source_h = max(1, source_rect.height())
            orig_x = self._pixmap.width() * float(spec["center"][0])
            orig_y = self._pixmap.height() * float(spec["center"][1])
            norm_x = max(0.0, min(1.0, (orig_x - source_rect.x()) / source_w))
            norm_y = max(0.0, min(1.0, (orig_y - source_rect.y()) / source_h))
            cx = self._display_rect.x() + int(self._display_rect.width() * norm_x)
            cy = self._display_rect.y() + int(self._display_rect.height() * norm_y)
            return cx, cy

        def _hotspot_size(self, spec):
            display_w = max(1, self._display_rect.width())
            display_h = max(1, self._display_rect.height())
            shape = spec.get("shape", "round")
            if shape == "pill":
                width = max(34, int(display_w * spec["size"][0] * 0.62))
                height = max(18, int(display_h * spec["size"][1] * 0.56))
            else:
                width = max(24, int(display_w * spec["size"][0] * 0.58))
                height = max(24, int(display_h * spec["size"][1] * 0.58))
            return width, height

        def _refresh_button_styles(self):
            hotspots = _layout_config(self._layout_key).get("hotspots", {})
            for button_id, btn in self._buttons.items():
                spec = hotspots.get(button_id)
                if spec is None:
                    continue
                mapped_action = self._mapping.get(button_id)
                focused = mapped_action is not None and mapped_action == self._focused_action
                learning = mapped_action is not None and mapped_action == self._learning_action
                accent = QColor(spec.get("accent", "#cbd5e1"))
                border = accent.name() if mapped_action is not None else "rgba(148, 163, 184, 0.45)"
                border_width = 3 if (focused or learning) else 2 if mapped_action is not None else 1
                fill_alpha = 126 if (focused or learning) else 28 if mapped_action is not None else 10
                if focused or learning:
                    border = accent.lighter(130).name()
                background = f"rgba({accent.red()}, {accent.green()}, {accent.blue()}, {fill_alpha})"
                hover_alpha = min(190, fill_alpha + 26)
                hover = f"rgba({accent.red()}, {accent.green()}, {accent.blue()}, {hover_alpha})"
                if btn.text().strip():
                    text_alpha = 255 if (focused or learning) else 205 if mapped_action is not None else 170
                    text_color = f"rgba(248, 250, 252, {text_alpha})"
                else:
                    text_color = "transparent"
                shadow = accent.lighter(118).name()
                radius = max(10, btn.height() // 2)
                btn.setStyleSheet(
                    "QPushButton {"
                    f"background: {background};"
                    f"color: {text_color};"
                    f"border: {border_width}px solid {border};"
                    f"border-radius: {radius}px;"
                    "font-weight: 700;"
                    f"font-size: {max(9, min(btn.width(), btn.height()) // (6 if len(btn.text()) > 2 else 3))}px;"
                    "padding: 0px 6px;"
                    "}"
                    "QPushButton:hover {"
                    f"border-color: {shadow};"
                    f"background: {hover};"
                    "}"
                )
                if mapped_action is None:
                    tip = f"{_button_display_name(button_id, self._layout_key)} is available for assignment."
                else:
                    tip = f"{_button_display_name(button_id, self._layout_key)} -> {_ACTION_LABELS.get(mapped_action, str(mapped_action))}"
                btn.setToolTip(tip)

        def paintEvent(self, event):
            super().paintEvent(event)
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)

            painter.fillRect(self.rect(), QColor("#0d1220"))
            card = QRectF(1, 1, self.width() - 2, self.height() - 2)
            painter.setPen(QPen(QColor("#2b3346"), 1.0))
            painter.setBrush(QColor("#101826"))
            painter.drawRoundedRect(card, 20, 20)

            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(116, 190, 255, 20))
            painter.drawEllipse(QRectF(self.width() * 0.16, self.height() * 0.08, self.width() * 0.68, self.height() * 0.72))
            painter.setBrush(QColor(255, 255, 255, 14))
            painter.drawEllipse(QRectF(self.width() * 0.28, self.height() * 0.68, self.width() * 0.44, self.height() * 0.10))

            if self._pixmap.isNull():
                painter.setPen(QColor("#f38ba8"))
                painter.drawText(self.rect(), Qt.AlignCenter, "Controller image unavailable")
            else:
                painter.setRenderHint(QPainter.SmoothPixmapTransform)
                painter.drawPixmap(self._display_rect, self._pixmap, self._source_rect)

            painter.end()

    class _ButtonConfigDialog(QDialog):
        """Remap controller buttons to curation actions (learn mode)."""

        _STYLE = """
            QDialog { background: #1e1e2e; }
            QLabel { color: #cdd6f4; background: transparent; }
            QFrame#panelCard, QFrame#sectionCard {
                background: #141b2b;
                border: 1px solid #2b3346;
                border-radius: 16px;
            }
            QFrame#sectionCard {
                background: #121928;
                border-radius: 14px;
            }
            QWidget#controllerCanvas {
                border: 1px solid #2b3346;
                border-radius: 20px;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
            QLabel#eyebrow {
                color: #cba6f7;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.06em;
            }
            QLabel#cardTitle {
                color: #f8fafc;
                font-size: 18px;
                font-weight: 700;
            }
            QLabel#muted {
                color: #94a3b8;
                font-size: 11px;
            }
            QLabel#selectedAction {
                color: #f8fafc;
                font-size: 20px;
                font-weight: 700;
            }
            QLabel#selectedHelp {
                color: #a6adc8;
                font-size: 11px;
            }
            QPushButton#layoutButton {
                background: rgba(255, 255, 255, 0.03);
                color: #cdd6f4;
                border: 1px solid #313b52;
                border-radius: 12px;
                padding: 8px 16px;
                min-width: 110px;
                font-weight: 700;
            }
            QPushButton#layoutButton:hover {
                background: rgba(137, 180, 250, 0.10);
                border-color: rgba(137, 180, 250, 0.36);
            }
            QPushButton#layoutButton:checked {
                background: rgba(137, 180, 250, 0.16);
                color: #eff6ff;
                border: 1px solid rgba(137, 180, 250, 0.56);
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

        _ACTIONS = [action for _, actions in _ACTION_SECTIONS for action in actions]

        def __init__(self, plugin, parent=None):
            super().__init__(parent)
            self._plugin = plugin
            self._learning_action = None
            self._learn_timer = None
            self._work_layout_key = plugin._layout_key
            self._work_maps = {
                layout_key: dict(plugin._button_maps.get(layout_key, _default_button_map(layout_key)))
                for layout_key in _LAYOUTS
            }
            self._selected_action = self._ACTIONS[0]
            self._rows = {}
            self._layout_buttons = {}

            self.setWindowTitle("Gamepad \u2014 Configure Buttons")
            self.setStyleSheet(self._STYLE)
            self.setMinimumSize(1060, 700)
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

            lay = QVBoxLayout(self)
            lay.setSpacing(12)
            lay.setContentsMargins(14, 14, 14, 14)

            header = QFrame(self)
            header.setObjectName("panelCard")
            header_lay = QVBoxLayout(header)
            header_lay.setContentsMargins(16, 14, 16, 14)
            header_lay.setSpacing(10)

            title = QLabel("Controller Mapping")
            title.setObjectName("cardTitle")
            header_lay.addWidget(title)

            self._tip_lbl = QLabel("")
            self._tip_lbl.setObjectName("muted")
            self._tip_lbl.setWordWrap(True)
            header_lay.addWidget(self._tip_lbl)

            toggle_row = QHBoxLayout()
            toggle_row.setSpacing(8)
            toggle_row.addWidget(self._eyebrow_label("Layout"))
            for layout_key in (LAYOUT_XBOX, LAYOUT_SWITCH):
                button = QPushButton(_layout_config(layout_key).get("title", layout_key.title()))
                button.setObjectName("layoutButton")
                button.setCheckable(True)
                button.clicked.connect(partial(self._choose_layout, layout_key))
                self._layout_buttons[layout_key] = button
                toggle_row.addWidget(button)
            toggle_row.addStretch()
            header_lay.addLayout(toggle_row)
            lay.addWidget(header)

            content = QHBoxLayout()
            content.setSpacing(14)
            lay.addLayout(content, 1)

            left = QVBoxLayout()
            left.setSpacing(12)
            content.addLayout(left, 5)

            preview_card = QFrame(self)
            preview_card.setObjectName("panelCard")
            preview_lay = QVBoxLayout(preview_card)
            preview_lay.setContentsMargins(14, 14, 14, 14)
            preview_lay.setSpacing(10)
            preview_lay.addWidget(self._eyebrow_label("Controller Preview"))

            self._controller_map = _ControllerMapWidget(self)
            self._controller_map.on_button_clicked = self._controller_button_clicked
            preview_lay.addWidget(self._controller_map, 1)
            left.addWidget(preview_card, 1)

            lower_left = QHBoxLayout()
            lower_left.setSpacing(12)
            left.addLayout(lower_left)

            self._selection_card = QFrame(self)
            self._selection_card.setObjectName("panelCard")
            selection_lay = QVBoxLayout(self._selection_card)
            selection_lay.setContentsMargins(14, 14, 14, 14)
            selection_lay.setSpacing(8)
            selection_lay.addWidget(self._eyebrow_label("Selected Assignment"))
            self._selected_name_lbl = QLabel("")
            self._selected_name_lbl.setObjectName("selectedAction")
            selection_lay.addWidget(self._selected_name_lbl)
            self._selected_control_lbl = QLabel("")
            selection_lay.addWidget(self._selected_control_lbl)
            self._selected_help_lbl = QLabel("")
            self._selected_help_lbl.setObjectName("selectedHelp")
            self._selected_help_lbl.setWordWrap(True)
            selection_lay.addWidget(self._selected_help_lbl)
            lower_left.addWidget(self._selection_card, 1)

            fixed_card = QFrame(self)
            fixed_card.setObjectName("panelCard")
            fixed_lay = QVBoxLayout(fixed_card)
            fixed_lay.setContentsMargins(14, 14, 14, 14)
            fixed_lay.setSpacing(6)
            fixed_lay.addWidget(self._eyebrow_label("Always-On Controls"))
            for hint in _FIXED_CONTROL_HINTS:
                lbl = QLabel(hint)
                lbl.setObjectName("muted")
                lbl.setWordWrap(True)
                fixed_lay.addWidget(lbl)
            lower_left.addWidget(fixed_card, 1)

            assignment_card = QFrame(self)
            assignment_card.setObjectName("panelCard")
            assignment_lay = QVBoxLayout(assignment_card)
            assignment_lay.setContentsMargins(14, 14, 14, 14)
            assignment_lay.setSpacing(10)
            assignment_lay.addWidget(self._eyebrow_label("Assignments"))
            assignment_hint = QLabel("Pick an action, then click Learn or click a highlighted controller marker.")
            assignment_hint.setObjectName("muted")
            assignment_hint.setWordWrap(True)
            assignment_lay.addWidget(assignment_hint)

            scroll = QScrollArea(self)
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setWidget(self._build_assignment_sections())
            assignment_lay.addWidget(scroll, 1)
            content.addWidget(assignment_card, 4)

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

            self._apply_layout_state()
            self._select_action(self._selected_action)

        # ── helpers ───────────────────────────────────────────────────

        def _eyebrow_label(self, text):
            label = QLabel(text)
            label.setObjectName("eyebrow")
            return label

        def _control_pill_html(self, text, assigned):
            if assigned:
                return (
                    "<span style=\"display:inline-block; padding:4px 10px; border-radius:10px; "
                    "background: rgba(137, 180, 250, 0.16); border: 1px solid rgba(137, 180, 250, 0.50); "
                    "color: #eff6ff; font-weight: 700;\">"
                    f"{text}</span>"
                )
            return (
                "<span style=\"display:inline-block; padding:4px 10px; border-radius:10px; "
                "background: rgba(148, 163, 184, 0.10); border: 1px solid rgba(148, 163, 184, 0.24); "
                "color: #94a3b8; font-weight: 600;\">"
                f"{text}</span>"
            )

        def _build_assignment_sections(self):
            container = QWidget(self)
            container_lay = QVBoxLayout(container)
            container_lay.setContentsMargins(0, 0, 0, 0)
            container_lay.setSpacing(10)

            for section_title, actions in _ACTION_SECTIONS:
                section = QFrame(container)
                section.setObjectName("sectionCard")
                section_lay = QVBoxLayout(section)
                section_lay.setContentsMargins(12, 12, 12, 12)
                section_lay.setSpacing(8)
                section_lay.addWidget(self._eyebrow_label(section_title))
                for action in actions:
                    row = _AssignmentRow(action, parent=section)
                    row.on_selected = self._select_action
                    row.on_learn_clicked = self._toggle_learn
                    self._rows[action] = row
                    section_lay.addWidget(row)
                container_lay.addWidget(section)

            container_lay.addStretch()
            return container

        def _current_map(self):
            return self._work_maps[self._work_layout_key]

        def _choose_layout(self, layout_key, _checked=False):
            if layout_key != self._work_layout_key:
                if self._learning_action is not None:
                    self._stop_learn(cancelled=True)
                self._work_layout_key = layout_key
            self._apply_layout_state()

        def _apply_layout_state(self):
            layout = _layout_config(self._work_layout_key)
            self._tip_lbl.setText(
                f"Choose a {layout.get('title', 'Xbox')} layout, then select an action on the right. The controller preview stays readable while its markers remain clickable for direct assignment."
            )
            for candidate_key, button in self._layout_buttons.items():
                button.blockSignals(True)
                button.setChecked(candidate_key == self._work_layout_key)
                button.blockSignals(False)
            self._controller_map.set_layout_key(self._work_layout_key)
            self._controller_map.set_mapping(self._current_map())
            self._refresh_all_rows()
            self._update_selection_card()

        def _btn_for_action(self, action):
            for bid, act in self._current_map().items():
                if act == action:
                    return bid
            return None

        def _btn_label(self, bid):
            return _button_display_name(bid, self._work_layout_key)

        def _focus_action(self, action):
            self._controller_map.set_focus_action(action)
            self._controller_map.set_learning_action(self._learning_action)

        def _refresh_action_row(self, action):
            row = self._rows.get(action)
            if row is None:
                return
            bid = self._btn_for_action(action)
            row.set_control(self._btn_label(bid), bid is not None)
            row.set_selected(action == self._selected_action)
            row.set_learning(action == self._learning_action)

        def _refresh_all_rows(self):
            for action in self._ACTIONS:
                self._refresh_action_row(action)
            self._controller_map.set_mapping(self._current_map())

        def _find_row_for_button(self, button_id):
            action = self._current_map().get(button_id)
            if action is None:
                return None
            try:
                return self._ACTIONS.index(action)
            except ValueError:
                return None

        def _update_selection_card(self, message=None):
            action = self._selected_action
            label = _ACTION_LABELS.get(action, str(action)) if action is not None else "No action selected"
            bid = self._btn_for_action(action) if action is not None else None
            control = self._btn_label(bid)
            self._selected_name_lbl.setText(label)
            self._selected_control_lbl.setText(self._control_pill_html(control, bid is not None))
            if message:
                help_text = message
            elif self._learning_action is action:
                help_text = "Press a controller button or click a marker on the preview to finish learning this action."
            else:
                help_text = "Select Learn to remap this action, or click one of the controller markers to inspect its current binding."
            self._selected_help_lbl.setText(help_text)

        def _select_action(self, action):
            if action is None:
                return
            self._selected_action = action
            self._refresh_all_rows()
            self._focus_action(action)
            self._update_selection_card()

        def _controller_button_clicked(self, button_id):
            if self._learning_action is not None:
                self._assign(button_id, self._learning_action)
                self._stop_learn(cancelled=False)
                return

            mapped_action = self._current_map().get(button_id)
            if mapped_action is None:
                self._update_selection_card(
                    f"{_button_display_name(button_id, self._work_layout_key)} is currently unassigned. Select an action and click Learn to attach it."
                )
                return

            self._select_action(mapped_action)
            self._update_selection_card(
                f"{_button_display_name(button_id, self._work_layout_key)} currently triggers {_ACTION_LABELS.get(mapped_action, str(mapped_action))}."
            )

        # ── learn mode ────────────────────────────────────────────────

        def _toggle_learn(self, action):
            if self._learning_action == action:
                self._stop_learn(cancelled=True)
            else:
                if self._learning_action is not None:
                    self._stop_learn(cancelled=True)
                self._start_learn(action)

        def _start_learn(self, action):
            self._learning_action = action
            self._select_action(action)
            label = _ACTION_LABELS.get(action, str(action))
            layout_title = _layout_config(self._work_layout_key).get("title", "Xbox")
            self._refresh_all_rows()
            self._update_selection_card(
                f"Assign {label}: press a controller button or click a hotspot on the {layout_title} preview."
            )
            self._focus_action(action)

            if self._plugin._joystick is not None:
                self._learn_timer = QTimer(self)
                self._learn_timer.setInterval(50)
                self._learn_timer.timeout.connect(self._learn_poll)
                self._learn_timer.start()

        def _learn_poll(self):
            js = self._plugin._joystick
            if js is None:
                return
            backend = getattr(self._plugin, "_controller_backend", None)
            try:
                # Refresh controller state per backend. Only the pygame backend
                # needs pygame.event.pump(); the XInput backend reads state via a
                # direct ctypes call and must NOT touch SDL (that is the segfault
                # path). Guarding on the active backend keeps Learn-mode safe.
                if backend == _XINPUT_BACKEND:
                    if not js.refresh_state():
                        return
                elif backend == _PYGAME_BACKEND:
                    if not _HAS_PYGAME:
                        return
                    pygame.event.pump()
                else:
                    return
                for bid in range(js.get_numbuttons()):
                    if js.get_button(bid):
                        self._assign(bid, self._learning_action)
                        self._stop_learn(cancelled=False)
                        return
            except Exception:
                self._stop_learn(cancelled=True)

        def _assign(self, btn_id, action):
            button_map = self._current_map()
            old_act = button_map.get(btn_id)
            for button_id, mapped_action in list(button_map.items()):
                if mapped_action == action and button_id != btn_id:
                    del button_map[button_id]
            button_map[btn_id] = action
            self._refresh_all_rows()
            if old_act is not None and old_act != action:
                self._refresh_action_row(old_act)
            label = _ACTION_LABELS.get(action, str(action))
            self._select_action(action)
            self._update_selection_card(f"Assigned {_button_display_name(btn_id, self._work_layout_key)} -> {label}.")

        def _stop_learn(self, cancelled=False):
            if self._learn_timer:
                self._learn_timer.stop()
                self._learn_timer = None
            self._learning_action = None
            self._controller_map.set_learning_action(None)
            self._refresh_all_rows()
            if cancelled:
                self._update_selection_card()
            self._focus_action(self._selected_action)

        def _reset_defaults(self):
            self._stop_learn(cancelled=True)
            self._work_maps[self._work_layout_key] = _default_button_map(self._work_layout_key)
            self._refresh_all_rows()
            layout_title = _layout_config(self._work_layout_key).get("title", "Xbox")
            self._update_selection_card(f"Reset to the default {layout_title} layout.")

        def _apply(self):
            self._stop_learn(cancelled=False)
            self._plugin._layout_key = self._work_layout_key
            self._plugin._button_maps = {
                layout_key: dict(button_map)
                for layout_key, button_map in self._work_maps.items()
            }
            self._plugin._button_map = dict(self._plugin._button_maps[self._plugin._layout_key])
            self._plugin._save_layout_key(self._plugin._layout_key)
            for layout_key, button_map in self._plugin._button_maps.items():
                self._plugin._save_button_map(layout_key, button_map)
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
            self._controller_backend = None
            self._layout_key = self._load_layout_key()
            self._button_maps = {
                layout_key: self._load_button_map(layout_key)
                for layout_key in _LAYOUTS
            }
            self._button_map = dict(self._button_maps.get(self._layout_key, _default_button_map(self._layout_key)))
            self._button_states = {}
            self._hat_states = {}
            self._trigger_pressed = False
            self._stick_accum_cluster = 0.0
            self._stick_accum_similar = 0.0
            self._deadzone = 0.12
            self._sensitivity = 5.0
            self._curve_mode = "quadratic"

            self._update_mapping_surfaces()

            # Only spin up pygame/SDL when XInput is not available. On Windows with
            # XInput present we deliberately keep SDL completely out of the picture:
            # initializing pygame's joystick subsystem makes SDL open the physical
            # device and start feeding it through the SDL message loop, which is
            # exactly the interaction that segfaults phy. XInput needs no SDL state.
            if _HAS_PYGAME and not _HAS_XINPUT:
                pygame.init()
                try:
                    pygame.joystick.init()
                except Exception:
                    logger.exception("Failed to initialize pygame joystick subsystem")

            if _HAS_PYGAME or _HAS_XINPUT:
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
                    "(poll=50ms, curve=%s, deadzone=%.0f%%, xinput=%s)",
                    self._curve_mode, self._deadzone * 100, "yes" if _HAS_XINPUT else "no",
                )
            else:
                logger.warning(
                    "pygame not installed and Windows XInput unavailable — "
                    "gamepad plugin loaded but controller input disabled"
                )
                if self._hud:
                    self._hud.set_controller_name(
                        "no controller backend"
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

    def _detect_pygame_controller(self):
        if not _HAS_PYGAME:
            return None
        try:
            pygame.joystick.quit()
            pygame.joystick.init()
            count = pygame.joystick.get_count()
        except Exception:
            logger.exception("Unable to enumerate pygame joysticks")
            return None
        for joystick_index in range(count):
            try:
                js = pygame.joystick.Joystick(joystick_index)
                js.init()
                return js
            except Exception:
                logger.exception("Failed to initialize pygame joystick %s", joystick_index)
        return None

    def _detect_xinput_controller(self):
        if not _HAS_XINPUT:
            return None
        for user_index in range(4):
            controller = _XInputController(user_index)
            try:
                if controller.init():
                    return controller
            except Exception:
                logger.exception("Failed to initialize XInput controller slot %s", user_index)
        return None

    def _detect_controller(self):
        # Prefer the Windows XInput backend whenever it is available. XInput is
        # polled directly through XInputGetState (a pure ctypes call), so it does
        # NOT require pygame.event.pump() and never touches SDL's event/message
        # loop. Driving pygame.event.pump() from a Qt QTimer while a real pad is
        # connected under SDL_VIDEODRIVER=dummy crashes phy natively with an
        # access violation (0xC0000005): SDL's joystick state on Windows is fed
        # by the SDL message loop, which fights Qt's own loop. Using XInput as the
        # primary backend sidesteps that conflict entirely. pygame is only used
        # as a fallback (non-Windows, or when XInput reports no controller).
        js = self._detect_xinput_controller()
        backend = _XINPUT_BACKEND if js is not None else None
        if js is None:
            js = self._detect_pygame_controller()
            backend = _PYGAME_BACKEND if js is not None else None
        if js is None:
            if self._joystick is not None:
                self._handle_disconnect()
                logger.info("Gamepad disconnected")
            return

        previous_name = self._joystick.get_name() if self._joystick is not None else None
        previous_backend = getattr(self, "_controller_backend", None)
        self._joystick = js
        self._controller_backend = backend
        self._button_states.clear()
        self._hat_states.clear()

        name = js.get_name()
        if previous_name != name or previous_backend != backend:
            logger.info("Gamepad connected via %s: %s", backend, name)
        if self._hud:
            self._hud.set_controller_name(name)

        global _RIGHT_STICK_X
        n = js.get_numaxes()
        if backend == _XINPUT_BACKEND:
            _RIGHT_STICK_X = 2
        elif n >= 4:
            _RIGHT_STICK_X = 3
        elif n >= 3:
            _RIGHT_STICK_X = 2

    # ── Main poll loop ────────────────────────────────────────────────

    def _poll_gamepad(self):
        if self._joystick is None:
            return

        js = self._joystick
        backend = getattr(self, "_controller_backend", None)
        try:
            if backend == _PYGAME_BACKEND:
                if not _HAS_PYGAME:
                    self._handle_disconnect()
                    return
                pygame.event.pump()
            elif backend == _XINPUT_BACKEND:
                if not js.refresh_state():
                    self._handle_disconnect()
                    return
            else:
                self._handle_disconnect()
                return
        except Exception:
            self._handle_disconnect()
            return

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
        self._controller_backend = None
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

    def _load_layout_key(self):
        settings = self._settings()
        if settings is None:
            return LAYOUT_XBOX
        raw_value = str(settings.value(_LAYOUT_SETTINGS_KEY, LAYOUT_XBOX) or LAYOUT_XBOX).strip().lower()
        if raw_value not in _LAYOUTS:
            return LAYOUT_XBOX
        return raw_value

    def _save_layout_key(self, layout_key):
        settings = self._settings()
        if settings is None:
            return
        settings.setValue(_LAYOUT_SETTINGS_KEY, layout_key)
        settings.sync()

    def _load_button_map(self, layout_key):
        default_button_map = _default_button_map(layout_key)
        settings = self._settings()
        if settings is None:
            return default_button_map
        raw_value = settings.value(_button_map_settings_key(layout_key), "")
        if (not raw_value) and layout_key == LAYOUT_XBOX:
            raw_value = settings.value(_LEGACY_BUTTON_MAP_SETTINGS_KEY, "")
        return _deserialize_button_map(raw_value, default_button_map)

    def _save_button_map(self, layout_key, button_map):
        settings = self._settings()
        if settings is None:
            return
        serialized = _serialize_button_map(button_map)
        settings.setValue(_button_map_settings_key(layout_key), serialized)
        if layout_key == LAYOUT_XBOX:
            settings.setValue(_LEGACY_BUTTON_MAP_SETTINGS_KEY, serialized)
        settings.sync()

    def _update_mapping_surfaces(self):
        if self._hud:
            self._hud.set_button_profile(self._layout_key, self._button_map)

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
    return (
        source.replace("__XBOX_LAYOUT_IMAGE__", xbox_image)
        .replace("__SWITCH_LAYOUT_IMAGE__", switch_image)
    )

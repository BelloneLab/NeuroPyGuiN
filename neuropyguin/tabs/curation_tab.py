from __future__ import annotations

import ast
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import signal as sps
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from ..bombcell_core import (
    bombcell_get_default_thresholds,
    bombcell_label_units_from_metrics,
    run_bombcell_on_folder_with_thresholds,
    sync_phy_cluster_group,
)
from ..ecephys_runtime import ecephys_subprocess_env
from ..ks_output_resolver import find_kilosort_output_dir, find_metrics_file
from ..phy_integration import ensure_phy_short_isi_plugin
from ..preprocessing import (
    find_concat_splitinfo_for_ks_folder,
    infer_completed_run_name,
    split_concatenated_sort,
)
from ..processes import tracked_run
from ..pybombcell_integration import (
    PYBOMBCELL_SETTINGS_SCHEMA,
    launch_pybombcell_gui,
    normalize_pybombcell_settings,
    pybombcell_default_settings,
    run_pybombcell_on_folder,
    run_pybombcell_on_folders,
    summarize_saved_pybombcell_results,
)
from ..side_nav import SideNavStack
from ..workers import FunctionWorker


def _find_modules_input_json(json_root: Path, ks_folder: Path) -> Optional[Path]:
    target_dir = find_kilosort_output_dir(ks_folder, max_depth=4) or ks_folder
    target = target_dir.resolve().as_posix().lower()
    # Per-run JSONs now live in a 'pipeline_json' folder inside the run's mirrored
    # processed root (an ancestor of the KS folder). Search there as well as the
    # legacy flat json_root so quality-metrics recompute keeps working either way.
    candidate_roots: List[Path] = []
    if json_root and str(json_root).strip():
        candidate_roots.append(json_root)
    for ancestor in [ks_folder, *ks_folder.parents][:6]:
        pj = ancestor / "pipeline_json"
        if pj not in candidate_roots:
            candidate_roots.append(pj)
    for root in candidate_roots:
        if not root.exists():
            continue
        for p in sorted(root.glob("*_modules-input.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                d = data.get("directories", {})
                kdir = str(d.get("kilosort_output_directory", "")).lower()
                if not kdir:
                    continue
                resolved_kdir = find_kilosort_output_dir(kdir, max_depth=4) or Path(kdir)
                if resolved_kdir.resolve().as_posix().lower() == target:
                    return p
            except Exception:
                continue
    return None


def _recompute_quality_metrics(ks_folder: str, json_root: str) -> str:
    folder = Path(ks_folder)
    jr = Path(json_root)
    inp = _find_modules_input_json(jr, folder)
    if inp is None:
        return "No matching modules-input.json found. Cannot recompute quality metrics automatically."

    out = inp.with_name(inp.name.replace("-input.json", "-qm-output.json"))
    cmd = [
        sys.executable,
        "-W",
        "ignore",
        "-m",
        "ecephys_spike_sorting.modules.quality_metrics",
        "--input_json",
        str(inp),
        "--output_json",
        str(out),
    ]
    proc = tracked_run(cmd, cwd=str(folder), capture_output=True, text=True, env=ecephys_subprocess_env())
    if proc.returncode != 0:
        raise RuntimeError(f"quality_metrics failed: {proc.stdout}\n{proc.stderr}")
    return "Quality metrics recomputed successfully."


def _parse_phy_dat_path(params_path: Path) -> object | None:
    if not params_path.exists():
        return None
    text = params_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"^dat_path\s*=\s*(.+)$", text, flags=re.MULTILINE)
    if not match:
        return None
    raw = match.group(1).strip()
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw.strip().strip("'").strip('"')


def _resolve_phy_dat_candidate(ks_folder: Path, raw_value: object | None) -> Optional[Path]:
    raw_candidates: List[str] = []
    if isinstance(raw_value, (list, tuple)):
        raw_candidates.extend(str(value).strip() for value in raw_value if str(value).strip())
    elif raw_value is not None and str(raw_value).strip():
        raw_candidates.append(str(raw_value).strip())

    basenames: List[str] = []
    for candidate in raw_candidates:
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = (ks_folder / path).resolve()
        if path.exists():
            return path.resolve()
        if path.name and path.name not in basenames:
            basenames.append(path.name)

    sibling_bins = sorted(ks_folder.parent.glob("*.ap.bin"))
    for name in basenames:
        direct = ks_folder.parent / name
        if direct.exists():
            return direct.resolve()
    if len(sibling_bins) == 1:
        return sibling_bins[0].resolve()

    search_roots = [ks_folder.parent, ks_folder.parent.parent, ks_folder.parent.parent.parent]
    for root in search_roots:
        if not root.exists():
            continue
        for name in basenames:
            matches = sorted(root.rglob(name))
            if matches:
                return matches[0].resolve()

    if sibling_bins:
        return sibling_bins[0].resolve()
    return None


def _repair_phy_params_path(ks_folder: Path) -> Optional[str]:
    params_path = ks_folder / "params.py"
    if not params_path.exists():
        return None

    current_value = _parse_phy_dat_path(params_path)
    resolved = _resolve_phy_dat_candidate(ks_folder, current_value)
    if resolved is None:
        old_params_path = ks_folder / "old_params.py"
        if old_params_path.exists():
            resolved = _resolve_phy_dat_candidate(ks_folder, _parse_phy_dat_path(old_params_path))

    if resolved is None:
        return "Could not resolve params.py dat_path automatically."

    params_text = params_path.read_text(encoding="utf-8", errors="ignore")
    absolute_path = resolved.as_posix()
    expected_line = f"dat_path = '{absolute_path}'"
    if re.search(rf"^dat_path\s*=\s*['\"]{re.escape(absolute_path)}['\"]$", params_text, flags=re.MULTILINE):
        return None

    new_text, count = re.subn(
        r"^dat_path\s*=\s*.+$",
        expected_line,
        params_text,
        count=1,
        flags=re.MULTILINE,
    )
    if count == 0:
        return "Could not rewrite params.py dat_path automatically."
    params_path.write_text(new_text, encoding="utf-8")
    return f"Repaired params.py dat_path -> {absolute_path}"


def _preferred_metrics_file(ks_folder: Path) -> Optional[Path]:
    bombcell_metrics = ks_folder / "bombcell" / "templates._bc_qMetrics.csv"
    if bombcell_metrics.exists():
        return bombcell_metrics
    return find_metrics_file(ks_folder, max_depth=4)


class PyBombcellSettingsDialog(QtWidgets.QDialog):
    def __init__(self, settings: Dict[str, object], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._values = normalize_pybombcell_settings(settings)
        self.setWindowTitle("py_bombcell Defaults")
        self.resize(840, 760)
        self._build_ui()

    @staticmethod
    def _format_value(value: object) -> str:
        try:
            if np.isnan(value):
                return "NaN"
        except Exception:
            pass
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    @staticmethod
    def _type_label(value: object) -> str:
        try:
            if np.isnan(value):
                return "float"
        except Exception:
            pass
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int) and not isinstance(value, bool):
            return "int"
        if isinstance(value, float):
            return "float"
        return "str"

    @staticmethod
    def _parse_value(text: str, default: object) -> object:
        raw = (text or "").strip()
        if isinstance(default, bool):
            lowered = raw.lower()
            if lowered in {"true", "1", "yes", "y", "on"}:
                return True
            if lowered in {"false", "0", "no", "n", "off"}:
                return False
            raise ValueError("Expected a boolean value")
        if isinstance(default, int) and not isinstance(default, bool):
            return int(float(raw))
        if isinstance(default, float):
            if raw.lower() == "nan":
                return float("nan")
            return float(raw)
        return raw

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        hint = QtWidgets.QLabel(
            "These values are merged into py_bombcell's bundled defaults for every run. "
            "Use `true`/`false` for booleans and `NaN` for undefined float thresholds."
        )
        hint.setObjectName("SectionHint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setAlternatingRowColors(True)
        self.table.setHorizontalHeaderLabels(["Parameter", "Value", "Type"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        layout.addWidget(self.table, 1)

        self._populate(self._values)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=self,
        )
        btn_reset = buttons.addButton("Reset bundled defaults", QtWidgets.QDialogButtonBox.ResetRole)
        btn_reset.clicked.connect(lambda: self._populate(pybombcell_default_settings()))
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _populate(self, values: Dict[str, object]) -> None:
        self.table.setRowCount(0)
        for key, default in PYBOMBCELL_SETTINGS_SCHEMA:
            row = self.table.rowCount()
            self.table.insertRow(row)
            value = values.get(key, default)
            key_item = QtWidgets.QTableWidgetItem(str(key))
            key_item.setFlags(key_item.flags() & ~QtCore.Qt.ItemIsEditable)
            type_item = QtWidgets.QTableWidgetItem(self._type_label(default))
            type_item.setFlags(type_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(row, 0, key_item)
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(self._format_value(value)))
            self.table.setItem(row, 2, type_item)
        self.table.resizeColumnsToContents()

    def _accept(self) -> None:
        values: Dict[str, object] = {}
        for row, (key, default) in enumerate(PYBOMBCELL_SETTINGS_SCHEMA):
            item = self.table.item(row, 1)
            raw = item.text() if item is not None else ""
            try:
                values[key] = self._parse_value(raw, default)
            except Exception as exc:
                self.table.setCurrentCell(row, 1)
                QtWidgets.QMessageBox.warning(self, "Invalid py_bombcell Value", f"{key}: {exc}")
                return
        self._values = normalize_pybombcell_settings(values)
        self.accept()

    def values(self) -> Dict[str, object]:
        return dict(self._values)


class CurationTab(QtWidgets.QWidget):
    def __init__(self, thread_pool: QtCore.QThreadPool) -> None:
        super().__init__()
        self.pool = thread_pool
        self.settings = QtCore.QSettings("NeuroPyGuiN", "NeuroPyGuiN")
        self.metrics_df = pd.DataFrame()
        self.preview_labels = pd.DataFrame()
        self.ks_folders: List[str] = []
        self._pybombcell_settings = pybombcell_default_settings()
        self._updating_table = False
        self._plot_lines: List[pg.PlotDataItem] = []
        self._metric_plots: List[pg.PlotItem] = []
        self._min_line: Optional[pg.InfiniteLine] = None
        self._max_line: Optional[pg.InfiniteLine] = None
        self._selected_unit_id: Optional[int] = None
        self._busy_count = 0
        self._plots_detached = False
        self._psd_metrics_cache: Dict[str, pd.DataFrame] = {}
        self._plots_dialog: Optional[QtWidgets.QDialog] = None
        self._body_split: Optional[QtWidgets.QSplitter] = None
        self._unit_split: Optional[QtWidgets.QSplitter] = None
        self._right_metrics: Optional[QtWidgets.QWidget] = None
        self._right_metrics_l: Optional[QtWidgets.QVBoxLayout] = None
        self._body_sizes_before_detach: List[int] = []
        self._plot_theme = "Light"
        self._show_grid = True
        self._updating_pybomb_table = False

        self.phy_process = QtCore.QProcess(self)
        self.phy_process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.phy_process.readyReadStandardOutput.connect(self._on_phy_output)
        self.phy_process.errorOccurred.connect(self._on_phy_error)
        self.phy_process.started.connect(lambda: self._log("Phy process started."))
        self.phy_process.finished.connect(lambda code, status: self._log(f"Phy process finished: code={code}, status={status}"))

        self.watcher = QtCore.QFileSystemWatcher(self)
        self.watcher.fileChanged.connect(self._on_metrics_changed)

        self._build_ui()
        self._reset_thresholds()
        self._restore_settings()

    def _build_ui(self) -> None:
        main = QtWidgets.QVBoxLayout(self)
        main.setContentsMargins(18, 16, 18, 18)
        main.setSpacing(14)

        def _wrap_page(widget: QtWidgets.QWidget, *, stretch: bool = True) -> QtWidgets.QWidget:
            page = QtWidgets.QWidget()
            page_l = QtWidgets.QVBoxLayout(page)
            page_l.setContentsMargins(0, 0, 0, 0)
            page_l.setSpacing(12)
            page_l.addWidget(widget, 1 if stretch else 0)
            if not stretch:
                page_l.addStretch(1)
            return page

        grp_phy = QtWidgets.QGroupBox("Phy")
        grp_phy.setProperty("settingsSection", True)
        phy_layout = QtWidgets.QVBoxLayout(grp_phy)
        phy_layout.setSpacing(10)
        phy_hint = QtWidgets.QLabel(
            "Keep a working list of Kilosort folders here. Select one folder to open it in Phy or review its py_bombcell outputs."
        )
        phy_hint.setObjectName("SectionHint")
        phy_hint.setWordWrap(True)
        phy_layout.addWidget(phy_hint)
        self.ed_phy_folder = QtWidgets.QLineEdit()
        self.ed_phy_folder.setReadOnly(True)
        self.ed_bomb_folder = QtWidgets.QLineEdit()
        self.ed_bomb_folder.setReadOnly(True)
        self.list_ks_folders = QtWidgets.QListWidget()
        self.list_ks_folders.setAlternatingRowColors(True)
        self.list_ks_folders.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.list_ks_folders.setMinimumHeight(420)
        self.list_ks_folders.setSpacing(6)
        self.list_ks_folders.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.list_ks_folders.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        phy_layout.addWidget(self.list_ks_folders, 1)

        folder_actions = QtWidgets.QHBoxLayout()
        self.btn_add_ks_folder = QtWidgets.QPushButton("Add folder")
        self.btn_remove_ks_folder = QtWidgets.QPushButton("Remove selected")
        self.btn_clear_ks_folders = QtWidgets.QPushButton("Clear list")
        self.btn_run_selected_pybomb = QtWidgets.QPushButton("Run py_bombcell on selected runs")
        self.btn_split_sessions = QtWidgets.QPushButton("Split concat → sessions")
        self.btn_open_selected_folder = QtWidgets.QPushButton("Open in Explorer")
        self.btn_open_figures = QtWidgets.QPushButton("Open py_bombcell figures")
        self.btn_open_bombcell_gui = QtWidgets.QPushButton("Open BombCell GUI")
        self.btn_add_ks_folder.setProperty("role", "secondary")
        self.btn_remove_ks_folder.setProperty("role", "ghost")
        self.btn_clear_ks_folders.setProperty("role", "ghost")
        self.btn_run_selected_pybomb.setProperty("role", "primary")
        self.btn_split_sessions.setProperty("role", "secondary")
        self.btn_split_sessions.setToolTip(
            "For a joint sort built from a concatenated recording: split the result back into one "
            "phy folder per session. This cuts the spike trains only (a view on the sort), never the "
            "binary, so the shared unit identities and templates are preserved. Each session folder "
            "gets session-local spike times, the same cluster IDs, params.py pointing at the "
            "original recording, and that session's TPrime-aligned NI event files."
        )
        self.btn_open_selected_folder.setProperty("role", "ghost")
        self.btn_open_figures.setProperty("role", "ghost")
        self.btn_open_bombcell_gui.setProperty("role", "secondary")
        folder_actions.addWidget(self.btn_add_ks_folder)
        folder_actions.addWidget(self.btn_remove_ks_folder)
        folder_actions.addWidget(self.btn_clear_ks_folders)
        folder_actions.addWidget(self.btn_run_selected_pybomb)
        folder_actions.addWidget(self.btn_split_sessions)
        folder_actions.addStretch(1)
        phy_layout.addLayout(folder_actions)

        phy_row = QtWidgets.QHBoxLayout()
        phy_row.setSpacing(10)
        self.btn_launch_phy = QtWidgets.QPushButton("Open in Phy")
        self.btn_stop_phy = QtWidgets.QPushButton("Stop")
        self.btn_launch_phy.setProperty("role", "primary")
        self.btn_stop_phy.setProperty("role", "ghost")
        phy_row.addWidget(QtWidgets.QLabel("Current folder"))
        phy_row.addWidget(self.ed_phy_folder, 1)
        phy_row.addWidget(self.btn_open_selected_folder)
        phy_row.addWidget(self.btn_open_figures)
        phy_row.addWidget(self.btn_open_bombcell_gui)
        phy_row.addWidget(self.btn_launch_phy)
        phy_row.addWidget(self.btn_stop_phy)
        phy_layout.addLayout(phy_row)

        grp_bomb = QtWidgets.QGroupBox("Bombcell: live QC")
        grp_bomb.setProperty("settingsSection", True)
        bomb_layout = QtWidgets.QVBoxLayout(grp_bomb)
        bomb_layout.setSpacing(10)

        bomb_top = QtWidgets.QHBoxLayout()
        self.btn_load_metrics = QtWidgets.QPushButton("Load active metrics")
        self.btn_run_pybomb = QtWidgets.QPushButton("Run py_bombcell on active folder")
        self.btn_load_metrics.setProperty("role", "secondary")
        self.btn_run_pybomb.setProperty("role", "primary")
        self.ck_extract_raw = QtWidgets.QCheckBox("Extract raw waveforms (SNR)")
        self.ck_extract_raw.setToolTip(
            "Opt-in: extract raw spike waveforms from the recording to compute signal-to-noise "
            "ratio and raw amplitude. Slower, and it recomputes metrics. Works on a concatenated "
            "joint sort too (waveforms are pulled from the fused binary). Leave off for the faster "
            "template-only metrics."
        )
        self.ck_extract_raw.setChecked(
            bool(self.settings.value("curation/pybomb_extract_raw", False, type=bool))
        )
        self.ck_extract_raw.toggled.connect(
            lambda checked: self.settings.setValue("curation/pybomb_extract_raw", bool(checked))
        )
        bomb_top.addWidget(self.btn_load_metrics)
        bomb_top.addWidget(self.btn_run_pybomb)
        bomb_top.addWidget(self.ck_extract_raw)

        review_folder_row = QtWidgets.QHBoxLayout()
        review_folder_row.setSpacing(10)
        review_folder_row.addWidget(QtWidgets.QLabel("Active folder"))
        review_folder_row.addWidget(self.ed_bomb_folder, 1)

        self.tbl_thresh = QtWidgets.QTableWidget(0, 5)
        self.tbl_thresh.setAlternatingRowColors(True)
        self.tbl_thresh.setHorizontalHeaderLabels(["Category", "Metric", "Min", "Max", "Abs"])
        self.tbl_thresh.horizontalHeader().setStretchLastSection(True)
        self.tbl_thresh.verticalHeader().setVisible(False)
        self.tbl_thresh.setMinimumWidth(460)

        metric_row = QtWidgets.QHBoxLayout()
        list_col = QtWidgets.QWidget()
        list_col_l = QtWidgets.QVBoxLayout(list_col)
        list_col_l.setContentsMargins(0, 0, 0, 0)
        self.list_metrics = QtWidgets.QListWidget()
        self.list_metrics.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.btn_metrics_all = QtWidgets.QPushButton("All")
        self.btn_metrics_clear = QtWidgets.QPushButton("Clear")
        self.btn_reset = QtWidgets.QPushButton("Reset defaults")
        self.btn_apply = QtWidgets.QPushButton("Apply settings")
        self.ck_live_apply = QtWidgets.QCheckBox("Live update")
        self.ck_live_apply.setChecked(True)
        self.btn_reset.setProperty("role", "ghost")
        self.btn_apply.setProperty("role", "secondary")
        metric_row.addWidget(self.btn_metrics_all)
        metric_row.addWidget(self.btn_metrics_clear)
        metric_row.addStretch(1)
        metric_row.addWidget(self.btn_reset)
        metric_row.addWidget(self.btn_apply)
        metric_row.addWidget(self.ck_live_apply)
        list_col_l.addWidget(QtWidgets.QLabel("Metrics (multi-select)"))
        list_col_l.addWidget(self.list_metrics, 1)
        list_col_l.addLayout(metric_row)

        threshold_box = QtWidgets.QGroupBox("Threshold settings")
        threshold_box.setProperty("settingsSection", True)
        threshold_l = QtWidgets.QVBoxLayout(threshold_box)
        threshold_l.addWidget(self.tbl_thresh, 1)

        defaults_box = QtWidgets.QGroupBox("py_bombcell default parameters")
        defaults_box.setProperty("settingsSection", True)
        defaults_l = QtWidgets.QVBoxLayout(defaults_box)
        defaults_hint = QtWidgets.QLabel(
            "These are the default py_bombcell parameters used for reruns. They are separate from the live review thresholds."
        )
        defaults_hint.setObjectName("SectionHint")
        defaults_hint.setWordWrap(True)
        defaults_l.addWidget(defaults_hint)
        self.tbl_pybomb_defaults = QtWidgets.QTableWidget(0, 3)
        self.tbl_pybomb_defaults.setAlternatingRowColors(True)
        self.tbl_pybomb_defaults.setHorizontalHeaderLabels(["Parameter", "Value", "Type"])
        self.tbl_pybomb_defaults.horizontalHeader().setStretchLastSection(True)
        self.tbl_pybomb_defaults.verticalHeader().setVisible(False)
        defaults_l.addWidget(self.tbl_pybomb_defaults, 1)
        defaults_actions = QtWidgets.QHBoxLayout()
        self.btn_reset_pybomb_defaults = QtWidgets.QPushButton("Reset bundled defaults")
        self.btn_apply_pybomb_defaults = QtWidgets.QPushButton("Apply default parameters")
        self.btn_reset_pybomb_defaults.setProperty("role", "ghost")
        self.btn_apply_pybomb_defaults.setProperty("role", "secondary")
        defaults_actions.addStretch(1)
        defaults_actions.addWidget(self.btn_reset_pybomb_defaults)
        defaults_actions.addWidget(self.btn_apply_pybomb_defaults)
        defaults_l.addLayout(defaults_actions)
        metric_box = QtWidgets.QGroupBox("Metric selection")
        metric_box.setProperty("settingsSection", True)
        metric_box_l = QtWidgets.QVBoxLayout(metric_box)
        metric_box_l.addWidget(list_col, 1)

        self.metrics_grid = pg.GraphicsLayoutWidget()
        self.metrics_grid.setMinimumHeight(420)
        plots_panel = QtWidgets.QWidget()
        plots_panel_l = QtWidgets.QVBoxLayout(plots_panel)
        plots_panel_l.setContentsMargins(0, 0, 0, 0)
        plots_panel_l.setSpacing(0)
        plots_panel_l.addWidget(self.metrics_grid, 1)

        self.btn_save_labels = QtWidgets.QPushButton("Save bombcell_labels.csv")
        self.btn_export = QtWidgets.QPushButton("Export plotted data")
        self.btn_detach_plots = QtWidgets.QPushButton("Detach plots")
        self.btn_detach_plots.setCheckable(True)
        self.btn_save_labels.setProperty("role", "primary")
        self.btn_export.setProperty("role", "ghost")
        self.btn_detach_plots.setProperty("role", "ghost")
        self.lbl_good = QtWidgets.QLabel("good: 0")
        self.lbl_noise = QtWidgets.QLabel("noise: 0")
        self.lbl_mua = QtWidgets.QLabel("mua: 0")
        self.lbl_non_soma = QtWidgets.QLabel("non_soma: 0")
        self.report_plot = pg.PlotWidget()
        self.report_plot.setMinimumHeight(180)
        self.report_plot.hideAxis("left")
        self.report_plot.showGrid(x=False, y=False)
        self.report_plot.setMouseEnabled(x=False, y=False)
        self.report_plot.setMenuEnabled(False)
        status_row = QtWidgets.QHBoxLayout()
        for w in [self.lbl_good, self.lbl_noise, self.lbl_mua, self.lbl_non_soma]:
            status_row.addWidget(w)
        status_row.addStretch(1)

        action_row = QtWidgets.QHBoxLayout()
        action_row.addStretch(1)
        action_row.addWidget(self.btn_detach_plots)
        action_row.addWidget(self.btn_export)
        action_row.addWidget(self.btn_save_labels)

        self.tabs_units = QtWidgets.QTabWidget()
        self.list_good = QtWidgets.QListWidget()
        self.list_noise = QtWidgets.QListWidget()
        self.list_mua = QtWidgets.QListWidget()
        self.list_non_soma = QtWidgets.QListWidget()
        self.tabs_units.addTab(self.list_good, "good")
        self.tabs_units.addTab(self.list_noise, "noise")
        self.tabs_units.addTab(self.list_mua, "mua")
        self.tabs_units.addTab(self.list_non_soma, "non_soma")
        self.list_good.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list_noise.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list_mua.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list_non_soma.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

        unit_inspector = QtWidgets.QGroupBox("Unit inspector")
        unit_inspector_l = QtWidgets.QVBoxLayout(unit_inspector)
        self.lbl_selected_unit = QtWidgets.QLabel("Selected unit: -")
        self.lbl_selected_label = QtWidgets.QLabel("Label: -")
        self.tbl_unit_metrics = QtWidgets.QTableWidget(0, 2)
        self.tbl_unit_metrics.setAlternatingRowColors(True)
        self.tbl_unit_metrics.setHorizontalHeaderLabels(["Metric", "Value"])
        self.tbl_unit_metrics.horizontalHeader().setStretchLastSection(True)
        self.tbl_unit_metrics.verticalHeader().setVisible(False)
        self.tbl_unit_metrics.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl_unit_metrics.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        unit_inspector_l.addWidget(self.lbl_selected_unit)
        unit_inspector_l.addWidget(self.lbl_selected_label)
        unit_inspector_l.addWidget(self.tbl_unit_metrics, 1)

        unit_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self._unit_split = unit_split
        unit_split.setChildrenCollapsible(False)
        unit_split.addWidget(self.tabs_units)
        unit_split.addWidget(unit_inspector)
        unit_split.setStretchFactor(0, 2)
        unit_split.setStretchFactor(1, 3)
        unit_split.setSizes([430, 320])

        review_box = QtWidgets.QGroupBox("Review controls")
        review_box.setProperty("settingsSection", True)
        review_box_l = QtWidgets.QVBoxLayout(review_box)
        review_box_l.setSpacing(10)
        review_hint = QtWidgets.QLabel(
            "Load metrics, run py_bombcell, then use the Thresholds and Metrics subsections to refine the review."
        )
        review_hint.setObjectName("SectionHint")
        review_hint.setWordWrap(True)
        review_box_l.addWidget(review_hint)
        review_box_l.addLayout(review_folder_row)
        review_box_l.addLayout(bomb_top)
        review_box_l.addLayout(status_row)
        review_box_l.addWidget(self.report_plot)
        review_box_l.addLayout(action_row)

        units_box = QtWidgets.QGroupBox("Units")
        units_box.setProperty("settingsSection", True)
        units_box_l = QtWidgets.QVBoxLayout(units_box)
        units_box_l.setSpacing(10)
        units_hint = QtWidgets.QLabel("Inspect labelled units on the left while the metric distributions stay visible on the right.")
        units_hint.setObjectName("SectionHint")
        units_hint.setWordWrap(True)
        units_box_l.addWidget(units_hint)
        units_box_l.addWidget(unit_split, 1)

        bomb_subsections = SideNavStack(
            vertical_labels=True,
            compact_rail=True,
        )
        self._bomb_subsections = bomb_subsections
        bomb_subsections.add_page("Review", _wrap_page(review_box, stretch=False))
        bomb_subsections.add_page("Units", _wrap_page(units_box, stretch=True))
        bomb_subsections.add_page("Thresholds", _wrap_page(threshold_box, stretch=True))
        bomb_subsections.add_page("Defaults", _wrap_page(defaults_box, stretch=True))
        bomb_subsections.add_page("Metrics", _wrap_page(metric_box, stretch=True))
        bomb_subsections.setCurrentIndex(0)

        left_panel = QtWidgets.QWidget()
        left_panel.setMinimumWidth(380)
        left_panel_l = QtWidgets.QVBoxLayout(left_panel)
        left_panel_l.setContentsMargins(0, 0, 0, 0)
        left_panel_l.setSpacing(0)
        left_panel_l.addWidget(bomb_subsections, 1)

        plots_box = QtWidgets.QGroupBox("Visualisation")
        plots_box.setProperty("heroCard", True)
        plots_box_l = QtWidgets.QVBoxLayout(plots_box)
        plots_box_l.setSpacing(10)
        plots_hint = QtWidgets.QLabel("Selected metric distributions and unit overlays. This panel keeps the largest share of the tab width.")
        plots_hint.setObjectName("SectionHint")
        plots_hint.setWordWrap(True)
        plots_box_l.addWidget(plots_hint)
        plots_panel.setMinimumWidth(720)
        plots_box_l.addWidget(plots_panel, 1)

        self._right_metrics = plots_box
        self._right_metrics_l = plots_panel_l

        body_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self._body_split = body_split
        body_split.setChildrenCollapsible(False)
        body_split.addWidget(left_panel)
        body_split.addWidget(plots_box)
        body_split.setStretchFactor(0, 1)
        body_split.setStretchFactor(1, 1)
        body_split.setSizes([860, 980])
        bomb_layout.addWidget(body_split, 1)

        self.btn_copy_log = QtWidgets.QPushButton("Copy log")
        self.btn_copy_log.setProperty("role", "secondary")
        log_box = QtWidgets.QGroupBox("Curation log")
        log_box.setProperty("settingsSection", True)
        log_layout = QtWidgets.QVBoxLayout(log_box)
        log_layout.setSpacing(8)
        log_header = QtWidgets.QHBoxLayout()
        log_hint = QtWidgets.QLabel("Live output from Phy, py_bombcell, and metrics refresh actions.")
        log_hint.setObjectName("SectionHint")
        log_hint.setWordWrap(True)
        log_header.addWidget(log_hint, 1)
        log_header.addWidget(self.btn_copy_log, 0)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setProperty("logView", True)
        self.log.setPlaceholderText("Curation and Phy output will appear here.")
        self.log.setMinimumHeight(260)
        self.log.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        log_layout.addLayout(log_header)
        log_layout.addWidget(self.log, 1)

        main_sections = SideNavStack(
            "Sections",
            "Switch between Phy launch, Bombcell review, and the curation log.",
            vertical_labels=True,
            compact_rail=True,
        )
        main_sections.add_page("Phy", _wrap_page(grp_phy, stretch=True))
        main_sections.add_page("Bombcell", _wrap_page(grp_bomb, stretch=True))
        main_sections.add_page("Log", _wrap_page(log_box, stretch=True))
        main_sections.setCurrentIndex(1)
        self._main_sections = main_sections
        main.addWidget(main_sections, 1)

        self.btn_add_ks_folder.clicked.connect(self._add_ks_folder)
        self.btn_remove_ks_folder.clicked.connect(self._remove_selected_ks_folders)
        self.btn_clear_ks_folders.clicked.connect(self._clear_ks_folders)
        self.btn_run_selected_pybomb.clicked.connect(self._run_pybombcell_on_selected_folders)
        self.btn_split_sessions.clicked.connect(self._split_selected_concatenated_sort)
        self.btn_open_selected_folder.clicked.connect(self._open_selected_folder_in_explorer)
        self.btn_open_figures.clicked.connect(self._open_selected_figures_in_explorer)
        self.btn_open_bombcell_gui.clicked.connect(self._open_selected_bombcell_gui)
        self.list_ks_folders.itemSelectionChanged.connect(self._on_folder_selection_changed)
        self.btn_launch_phy.clicked.connect(self._launch_phy)
        self.btn_stop_phy.clicked.connect(self._stop_phy)

        self.btn_load_metrics.clicked.connect(self._load_metrics)
        self.btn_run_pybomb.clicked.connect(self._run_pybombcell)
        self.btn_reset_pybomb_defaults.clicked.connect(self._reset_pybombcell_defaults_table)
        self.btn_apply_pybomb_defaults.clicked.connect(self._apply_pybombcell_defaults)
        self.btn_copy_log.clicked.connect(self._copy_log)
        self.btn_reset.clicked.connect(self._reset_thresholds)
        self.btn_apply.clicked.connect(self._apply_settings)
        self.btn_save_labels.clicked.connect(self._save_labels)
        self.btn_export.clicked.connect(self._export_plotted_data)
        self.btn_detach_plots.toggled.connect(self._toggle_plot_detach)
        self.tbl_thresh.itemChanged.connect(self._on_threshold_changed)
        self.tbl_thresh.currentCellChanged.connect(self._on_threshold_row_selected)
        self.list_metrics.itemSelectionChanged.connect(self._refresh_metric_plot)
        self.btn_metrics_all.clicked.connect(self._select_all_metrics)
        self.btn_metrics_clear.clicked.connect(self._clear_metrics_selection)
        self.list_good.itemSelectionChanged.connect(lambda: self._on_unit_selection_changed(self.list_good))
        self.list_noise.itemSelectionChanged.connect(lambda: self._on_unit_selection_changed(self.list_noise))
        self.list_mua.itemSelectionChanged.connect(lambda: self._on_unit_selection_changed(self.list_mua))
        self.list_non_soma.itemSelectionChanged.connect(lambda: self._on_unit_selection_changed(self.list_non_soma))
        body_split.splitterMoved.connect(lambda _pos, _idx: self._persist_splitter_sizes())
        unit_split.splitterMoved.connect(lambda _pos, _idx: self._persist_splitter_sizes())
        bomb_subsections.currentChanged.connect(lambda _idx: self._persist_splitter_sizes())

    def show_phy_page(self) -> None:
        if hasattr(self, "_main_sections"):
            self._main_sections.setCurrentIndex(0)

    def _populate_pybomb_defaults_table(self, values: Dict[str, object]) -> None:
        if not hasattr(self, "tbl_pybomb_defaults"):
            return
        self._updating_pybomb_table = True
        self.tbl_pybomb_defaults.setRowCount(0)
        for key, default in PYBOMBCELL_SETTINGS_SCHEMA:
            row = self.tbl_pybomb_defaults.rowCount()
            self.tbl_pybomb_defaults.insertRow(row)
            key_item = QtWidgets.QTableWidgetItem(str(key))
            key_item.setFlags(key_item.flags() & ~QtCore.Qt.ItemIsEditable)
            type_item = QtWidgets.QTableWidgetItem(PyBombcellSettingsDialog._type_label(default))
            type_item.setFlags(type_item.flags() & ~QtCore.Qt.ItemIsEditable)
            value_item = QtWidgets.QTableWidgetItem(
                PyBombcellSettingsDialog._format_value(values.get(key, default))
            )
            self.tbl_pybomb_defaults.setItem(row, 0, key_item)
            self.tbl_pybomb_defaults.setItem(row, 1, value_item)
            self.tbl_pybomb_defaults.setItem(row, 2, type_item)
        self.tbl_pybomb_defaults.resizeColumnsToContents()
        self._updating_pybomb_table = False

    def _current_pybombcell_settings_from_table(self) -> Dict[str, object]:
        if not hasattr(self, "tbl_pybomb_defaults"):
            return dict(self._pybombcell_settings)
        values: Dict[str, object] = {}
        for row, (key, default) in enumerate(PYBOMBCELL_SETTINGS_SCHEMA):
            item = self.tbl_pybomb_defaults.item(row, 1)
            raw = item.text() if item is not None else ""
            values[key] = PyBombcellSettingsDialog._parse_value(raw, default)
        return normalize_pybombcell_settings(values)

    def _apply_pybombcell_defaults(self, *, announce: bool = True) -> bool:
        try:
            self._pybombcell_settings = self._current_pybombcell_settings_from_table()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid py_bombcell default", str(exc))
            return False
        self.settings.setValue("curation/pybombcell_settings_json", json.dumps(self._pybombcell_settings))
        self._refresh_folder_item_texts()
        if announce:
            self._log("Applied py_bombcell default parameters.")
        return True

    def _reset_pybombcell_defaults_table(self) -> None:
        self._pybombcell_settings = pybombcell_default_settings()
        self._populate_pybomb_defaults_table(self._pybombcell_settings)
        self.settings.setValue("curation/pybombcell_settings_json", json.dumps(self._pybombcell_settings))
        self._refresh_folder_item_texts()
        self._log("Restored bundled py_bombcell defaults.")

    def _persist_folder_list(self) -> None:
        self.settings.setValue("curation/ks_folders_json", json.dumps(self.ks_folders))
        current = self.ed_phy_folder.text().strip()
        self.settings.setValue("curation/phy_folder", current)
        self.settings.setValue("curation/bomb_folder", current)

    def _folder_summary(self, folder: str | Path) -> Dict[str, object]:
        return summarize_saved_pybombcell_results(folder, settings=self._pybombcell_settings)

    def _folder_run_name(self, folder: str | Path) -> str:
        try:
            name = infer_completed_run_name(folder)
        except Exception:
            name = ""
        return str(name or Path(folder).name)

    def _folder_item_text(self, folder: str | Path) -> str:
        path = Path(folder)
        run_name = self._folder_run_name(path)
        summary = self._folder_summary(path)
        counts = summary.get("counts", {})
        if counts:
            return (
                f"{run_name}\n{path.name}  |  good {counts.get('good', 0)}  "
                f"noise {counts.get('noise', 0)}  mua {counts.get('mua', 0)}  "
                f"non_soma {counts.get('non_soma', 0)}"
            )
        if summary.get("has_metrics"):
            return f"{run_name}\n{path.name}  |  metrics ready"
        return f"{run_name}\n{path.name}  |  no py_bombcell results"

    def _selected_ks_folders(self) -> List[str]:
        items = self.list_ks_folders.selectedItems()
        if not items:
            current = self.list_ks_folders.currentItem()
            if current is not None:
                items = [current]
        out: List[str] = []
        seen: set[str] = set()
        for item in items:
            folder = str(item.data(QtCore.Qt.UserRole) or "").strip()
            if not folder:
                continue
            key = folder.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(folder)
        return out

    def _selected_ks_folder(self) -> Optional[str]:
        item = self.list_ks_folders.currentItem()
        if item is None and self.list_ks_folders.count():
            item = self.list_ks_folders.item(0)
            self.list_ks_folders.setCurrentItem(item)
        if item is None:
            return None
        folder = str(item.data(QtCore.Qt.UserRole) or "").strip()
        return folder or None

    def _refresh_folder_list(self, select_folder: str | None = None) -> None:
        current = str(select_folder or self.ed_phy_folder.text().strip() or "")
        selected_folders = {folder.lower() for folder in self._selected_ks_folders()}
        self.list_ks_folders.blockSignals(True)
        self.list_ks_folders.clear()
        target_row = -1
        for row, folder in enumerate(self.ks_folders):
            item = QtWidgets.QListWidgetItem(self._folder_item_text(folder))
            item.setData(QtCore.Qt.UserRole, folder)
            item.setToolTip(str(folder))
            item.setSizeHint(QtCore.QSize(item.sizeHint().width(), 46))
            self.list_ks_folders.addItem(item)
            if folder.lower() in selected_folders:
                item.setSelected(True)
            if folder == current:
                target_row = row
        if self.list_ks_folders.count():
            self.list_ks_folders.setCurrentRow(target_row if target_row >= 0 else 0)
        else:
            self.ed_phy_folder.clear()
            self.ed_bomb_folder.clear()
            self._clear_metrics_state()
        self.list_ks_folders.blockSignals(False)
        self._sync_selected_folder_fields(clear_metrics=False)

    def _refresh_folder_item_texts(self) -> None:
        current = self._selected_ks_folder()
        self._refresh_folder_list(select_folder=current)

    def _coerce_ks_folder(self, folder: str | Path) -> Optional[str]:
        raw = str(folder).strip()
        if not raw:
            return None
        candidate = Path(raw).expanduser()
        if candidate.is_file():
            candidate = candidate.parent
        resolved = find_kilosort_output_dir(candidate, max_depth=4) or candidate
        if not resolved.exists():
            return None
        return str(resolved.resolve())

    def set_ks_folders(self, folders: List[str], *, preserve_existing: bool = False) -> None:
        base = list(self.ks_folders) if preserve_existing else []
        seen = {folder.lower() for folder in base}
        selected_folder: Optional[str] = None
        for folder in folders:
            normalized = self._coerce_ks_folder(folder)
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                selected_folder = normalized
                continue
            seen.add(key)
            base.append(normalized)
            selected_folder = normalized
        self.ks_folders = base
        self._refresh_folder_list(select_folder=selected_folder or (base[0] if base else None))

    def _add_ks_folder(self) -> None:
        start = self.settings.value("paths/last_folder", str(Path.cwd()))
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Kilosort folder", str(start))
        if not folder:
            return
        self.settings.setValue("paths/last_folder", folder)
        before = len(self.ks_folders)
        self.set_ks_folders([folder], preserve_existing=True)
        if len(self.ks_folders) != before:
            self.show_phy_page()

    def _remove_selected_ks_folders(self) -> None:
        selected = set(folder.lower() for folder in self._selected_ks_folders())
        if not selected:
            return
        rows = sorted(
            (idx for idx, folder in enumerate(self.ks_folders) if folder.lower() in selected),
            reverse=True,
        )
        if not rows:
            current = self.list_ks_folders.currentRow()
            if current >= 0:
                rows = [current]
        for row in rows:
            if 0 <= row < len(self.ks_folders):
                self.ks_folders.pop(row)
        self._refresh_folder_list()

    def _clear_ks_folders(self) -> None:
        self.ks_folders.clear()
        self._refresh_folder_list()

    def _open_selected_folder_in_explorer(self) -> None:
        folder = self._selected_ks_folder()
        if not folder:
            self._log("Select a Kilosort folder first.")
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(folder))

    def _open_selected_figures_in_explorer(self) -> None:
        folder = self._selected_ks_folder()
        if not folder:
            self._log("Select a Kilosort folder first.")
            return
        summary = self._folder_summary(folder)
        plots_dir = Path(str(summary.get("plots_dir") or ""))
        if not plots_dir.exists():
            self._log(f"No py_bombcell figures found for {folder}")
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(plots_dir)))

    def _open_selected_bombcell_gui(self) -> None:
        folder = self._selected_ks_folder()
        if not folder:
            self._log("Select a Kilosort folder first.")
            return
        try:
            payload = launch_pybombcell_gui(folder)
        except Exception as exc:
            self._log(f"BombCell GUI launch failed: {exc}")
            return
        self._log(
            f"Opened BombCell GUI notebook via {payload.get('launcher', 'Jupyter')} | "
            f"{payload.get('notebook_path', '')}"
        )

    def _run_pybombcell_on_selected_folders(self) -> None:
        folders = self._selected_ks_folders()
        if not folders:
            self._log("Select one or more Kilosort folders first.")
            return
        if not self._apply_pybombcell_defaults(announce=False):
            return
        if hasattr(self, "_main_sections"):
            self._main_sections.setCurrentIndex(0)
        self._busy_count += 1
        self._set_pybombcell_buttons_enabled(False)
        self._log(f"Running py_bombcell across {len(folders)} selected folder(s)...")
        worker = FunctionWorker(
            run_pybombcell_on_folders,
            folders,
            save_plots=True,
            force_recompute=False,
            settings=self._pybombcell_settings,
            extract_raw=self.ck_extract_raw.isChecked(),
        )
        worker.signals.error.connect(self._log)
        worker.signals.finished.connect(self._on_run_pybombcell_list_finished)
        self.pool.start(worker)

    @staticmethod
    def _split_folders_job(folders: List[str], event_search_roots: List[str]) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        for folder in folders:
            try:
                manifest = split_concatenated_sort(folder, event_search_roots=event_search_roots)
                results.append({"folder": folder, "ok": True, "manifest": manifest})
            except Exception as exc:
                results.append({"folder": folder, "ok": False, "error": str(exc)})
        return results

    def _split_selected_concatenated_sort(self) -> None:
        folders = self._selected_ks_folders()
        if not folders:
            self._log("Select one or more Kilosort folders first.")
            return
        concat_folders = [f for f in folders if find_concat_splitinfo_for_ks_folder(f) is not None]
        if not concat_folders:
            QtWidgets.QMessageBox.information(
                self,
                "Not a concatenated sort",
                "None of the selected folders look like a joint sort of a concatenated recording "
                "(no *.splitinfo.json was found). Splitting into sessions only applies to those.",
            )
            return
        message = (
            f"Split {len(concat_folders)} joint sort(s) into per-session phy folders?\n\n"
            "This cuts the spike trains only, as a read-only view on the sort. The binary is never "
            "re-cut, so the shared unit identities and templates are preserved.\n\n"
            "Each session folder reuses the same cluster IDs, holds only that session's spikes with "
            "session-local sample times, points params.py at the original recording, and gathers "
            "that session's TPrime-aligned NI event files. Output goes under <ks_folder>/sessions/."
        )
        if (
            QtWidgets.QMessageBox.question(
                self,
                "Split concatenated sort into sessions",
                message,
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes,
            )
            != QtWidgets.QMessageBox.Yes
        ):
            return
        if hasattr(self, "_main_sections"):
            self._main_sections.setCurrentIndex(0)
        self._busy_count += 1
        self.btn_split_sessions.setEnabled(False)
        self._log(f"Splitting {len(concat_folders)} concatenated sort(s) into sessions...")
        event_roots: List[str] = []
        processed_root = str(self.settings.value("preproc/output_root", "") or "").strip()
        if processed_root:
            event_roots.append(processed_root)
        worker = FunctionWorker(self._split_folders_job, list(concat_folders), event_roots)
        worker.signals.error.connect(self._log)
        worker.signals.finished.connect(self._on_split_sessions_finished)
        self.pool.start(worker)

    def _on_split_sessions_finished(self, result: Dict) -> None:
        self._busy_count = max(0, self._busy_count - 1)
        self.btn_split_sessions.setEnabled(True)
        if not result.get("ok"):
            self._log("Session split failed.")
            return
        results = result.get("result", []) or []
        new_folders: List[str] = []
        for entry in results:
            if not entry.get("ok"):
                self._log(f"Split failed for {entry.get('folder', '')}: {entry.get('error', '')}")
                continue
            manifest = entry.get("manifest", {}) or {}
            sessions = manifest.get("sessions", []) or []
            self._log(
                f"Split {entry.get('folder', '')} into {len(sessions)} session(s) under "
                f"{manifest.get('output_root', '')}"
            )
            for sess in sessions:
                out_dir = str(sess.get("output_dir") or "")
                events = sess.get("events_copied") or []
                self._log(
                    f"  {sess.get('run_name', '')}: {sess.get('n_spikes', 0)} spikes, "
                    f"{sess.get('n_clusters', 0)} clusters"
                    + (f", {len(events)} NI event file(s)" if events else ", no NI event files found")
                )
                if out_dir:
                    new_folders.append(out_dir)
        if new_folders:
            self.set_ks_folders(new_folders, preserve_existing=True)
            self._log(
                f"Added {len(new_folders)} per-session folder(s) to the list. Open any in Phy, or run "
                "py_bombcell per session for per-session QC."
            )

    def _sync_selected_folder_fields(self, *, clear_metrics: bool) -> None:
        folder = self._selected_ks_folder()
        self.ed_phy_folder.setText(folder or "")
        self.ed_bomb_folder.setText(folder or "")
        self._persist_folder_list()
        if clear_metrics:
            self._clear_metrics_state()

    def _on_folder_selection_changed(self) -> None:
        self._sync_selected_folder_fields(clear_metrics=True)

    def _resolve_folder(self, target: QtWidgets.QLineEdit) -> Path:
        folder = Path(target.text().strip())
        resolved = find_kilosort_output_dir(folder, max_depth=4)
        if resolved is not None and resolved != folder:
            self._log(f"Resolved KS folder to {resolved}")
            target.setText(str(resolved))
            return resolved
        return folder

    def _refresh_report_plot(self, counts: Dict[str, int]) -> None:
        self.report_plot.clear()
        labels = ["good", "noise", "mua", "non_soma"]
        values = [int(counts.get(label, 0)) for label in labels]
        x = np.arange(len(labels), dtype=float)
        colors = ["#5cb85c", "#d9534f", "#f0ad4e", "#5bc0de"]
        self.report_plot.addItem(
            pg.BarGraphItem(
                x=x,
                height=np.asarray(values, dtype=float),
                width=0.65,
                brushes=[pg.mkBrush(color) for color in colors],
            )
        )
        axis = self.report_plot.getAxis("bottom")
        axis.setTicks([list(zip(x, labels))])
        top = max(values) if values else 0
        self.report_plot.setYRange(0, max(1, top) * 1.2, padding=0.0)

    def _clear_metrics_state(self) -> None:
        self.metrics_df = pd.DataFrame()
        self.preview_labels = pd.DataFrame()
        self._selected_unit_id = None
        self.metrics_grid.clear()
        self._plot_lines.clear()
        self._metric_plots.clear()
        self._min_line = None
        self._max_line = None
        for label in [self.lbl_good, self.lbl_noise, self.lbl_mua, self.lbl_non_soma]:
            name = label.text().split(":")[0]
            label.setText(f"{name}: 0")
        for lst in [self.list_good, self.list_noise, self.list_mua, self.list_non_soma]:
            lst.clear()
        self.list_metrics.clear()
        self.tbl_unit_metrics.setRowCount(0)
        self.lbl_selected_unit.setText("Selected unit: -")
        self.lbl_selected_label.setText("Label: -")
        self._refresh_report_plot({})

    def _launch_phy(self) -> None:
        folder_text = self.ed_phy_folder.text().strip()
        if not folder_text:
            self._log("Select curated folder first.")
            return
        folder = self._resolve_folder(self.ed_phy_folder)

        params = folder / "params.py"
        if not params.exists():
            self._log(f"Missing params.py in {folder}")
            return
        repair_message = _repair_phy_params_path(folder)
        if repair_message:
            self._log(repair_message)
        try:
            plugin_status = ensure_phy_short_isi_plugin()
            if plugin_status.get("plugin_updated") or plugin_status.get("config_updated"):
                self._log("Phy plugin ready: added `Split short ISI` context-menu action.")
            if plugin_status.get("gamepad_plugin_updated"):
                self._log("Phy gamepad plugin ready: controller curation + gamification enabled.")
        except Exception as exc:
            self._log(f"Could not install NeuroPyGuiN Phy plugin: {exc}")
        sync_result = sync_phy_cluster_group(folder, force=False)
        if sync_result.get("updated"):
            self._log(
                f"Updated cluster_group.tsv from {sync_result.get('source', 'labels')} "
                f"({sync_result.get('n_units', 0)} units)"
            )

        if self.phy_process.state() != QtCore.QProcess.NotRunning:
            self._log("Phy is already running.")
            return

        program = "phy"
        args = ["template-gui", str(params)]

        self._log("Launching: " + " ".join([program] + args))
        self.phy_process.setWorkingDirectory(str(folder))
        self.phy_process.start(program, args)
        if not self.phy_process.waitForStarted(3000):
            self._log("Failed to start phy: " + self.phy_process.errorString())

    def set_ks_folder(self, folder: str) -> None:
        self.set_ks_folders([folder])
        if hasattr(self, "_main_sections"):
            self._main_sections.setCurrentIndex(1)

    def open_ks_folder(self, folder: str) -> None:
        self.set_ks_folder(folder)
        if hasattr(self, "_main_sections"):
            self._main_sections.setCurrentIndex(0)
        self._launch_phy()

    def _stop_phy(self) -> None:
        if self.phy_process.state() == QtCore.QProcess.NotRunning:
            self._log("Phy is not running.")
            return
        self.phy_process.terminate()
        if not self.phy_process.waitForFinished(2000):
            self.phy_process.kill()
        self._log("Phy stop requested.")

    def _on_phy_output(self) -> None:
        data = self.phy_process.readAllStandardOutput().data().decode(errors="ignore")
        if data.strip():
            for line in data.splitlines():
                self._log("[phy] " + line)

    def _on_phy_error(self, _err) -> None:
        self._log("Phy error: " + self.phy_process.errorString())

    def _reset_thresholds(self) -> None:
        defaults = bombcell_get_default_thresholds()
        self._updating_table = True
        self.tbl_thresh.setRowCount(0)
        for category, metrics in defaults.items():
            for metric, conf in metrics.items():
                row = self.tbl_thresh.rowCount()
                self.tbl_thresh.insertRow(row)
                self.tbl_thresh.setItem(row, 0, QtWidgets.QTableWidgetItem(str(category)))
                self.tbl_thresh.setItem(row, 1, QtWidgets.QTableWidgetItem(str(metric)))
                self.tbl_thresh.setItem(row, 2, QtWidgets.QTableWidgetItem("" if conf.get("min") is None else str(conf.get("min"))))
                self.tbl_thresh.setItem(row, 3, QtWidgets.QTableWidgetItem("" if conf.get("max") is None else str(conf.get("max"))))
                self.tbl_thresh.setItem(row, 4, QtWidgets.QTableWidgetItem("1" if conf.get("abs", False) else "0"))
        self.tbl_thresh.resizeColumnsToContents()
        self._updating_table = False
        self._refresh_metric_selector()
        self._recompute_preview()

    def _restore_settings(self) -> None:
        raw_settings = str(self.settings.value("curation/pybombcell_settings_json", "{}") or "{}")
        try:
            parsed_settings = json.loads(raw_settings)
        except Exception:
            parsed_settings = {}
        if isinstance(parsed_settings, dict):
            self._pybombcell_settings = normalize_pybombcell_settings(parsed_settings)
        self._populate_pybomb_defaults_table(self._pybombcell_settings)

        raw_folders = str(self.settings.value("curation/ks_folders_json", "[]") or "[]")
        try:
            parsed_folders = json.loads(raw_folders)
        except Exception:
            parsed_folders = []
        if isinstance(parsed_folders, list) and parsed_folders:
            self.set_ks_folders([str(folder) for folder in parsed_folders if str(folder).strip()])
        else:
            phy_folder = str(self.settings.value("curation/phy_folder", "") or "").strip()
            bomb_folder = str(self.settings.value("curation/bomb_folder", "") or "").strip()
            fallback_folder = phy_folder or bomb_folder
            if fallback_folder:
                self.set_ks_folders([fallback_folder])

        bomb_index = int(self.settings.value("curation/bomb_subsection_index", 0))
        if hasattr(self, "_bomb_subsections"):
            self._bomb_subsections.setCurrentIndex(bomb_index)

        body_sizes = self.settings.value("curation/body_split_sizes", [])
        parsed_body = self._parse_splitter_sizes(body_sizes)
        if parsed_body and self._body_split is not None:
            self._body_split.setSizes(parsed_body)

        unit_sizes = self.settings.value("curation/unit_split_sizes", [])
        parsed_unit = self._parse_splitter_sizes(unit_sizes)
        if parsed_unit and self._unit_split is not None:
            self._unit_split.setSizes(parsed_unit)

    def _parse_splitter_sizes(self, raw_value) -> List[int]:
        if raw_value is None:
            return []
        if isinstance(raw_value, str):
            values = [part for part in raw_value.split(",") if part.strip()]
        elif isinstance(raw_value, (list, tuple)):
            values = list(raw_value)
        else:
            return []
        out: List[int] = []
        for value in values:
            try:
                size = int(value)
            except Exception:
                continue
            if size > 0:
                out.append(size)
        return out if len(out) >= 2 else []

    def _persist_splitter_sizes(self) -> None:
        if self._body_split is not None:
            self.settings.setValue("curation/body_split_sizes", self._body_split.sizes())
        if self._unit_split is not None:
            self.settings.setValue("curation/unit_split_sizes", self._unit_split.sizes())
        if hasattr(self, "_bomb_subsections"):
            self.settings.setValue("curation/bomb_subsection_index", self._bomb_subsections.currentIndex())

    def _parse_optional_float(self, text: str):
        t = (text or "").strip()
        if not t:
            return None
        return float(t)

    def _thresholds_from_table(self) -> Dict:
        out: Dict = {"noise": {}, "mua": {}, "non-somatic": {}}
        for row in range(self.tbl_thresh.rowCount()):
            cat_item = self.tbl_thresh.item(row, 0)
            met_item = self.tbl_thresh.item(row, 1)
            min_item = self.tbl_thresh.item(row, 2)
            max_item = self.tbl_thresh.item(row, 3)
            abs_item = self.tbl_thresh.item(row, 4)
            if cat_item is None or met_item is None:
                continue
            cat = cat_item.text().strip()
            met = met_item.text().strip()
            if cat not in out or not met:
                continue
            conf = {
                "min": self._parse_optional_float(min_item.text() if min_item else ""),
                "max": self._parse_optional_float(max_item.text() if max_item else ""),
            }
            if abs_item is not None and abs_item.text().strip() in {"1", "true", "True", "yes", "YES"}:
                conf["abs"] = True
            out[cat][met] = conf
        return out

    def _load_metrics(self, allow_compute: bool = True) -> None:
        if hasattr(self, "_main_sections"):
            self._main_sections.setCurrentIndex(1)
        folder = Path(self.ed_bomb_folder.text().strip())
        if not folder.exists():
            self._log(f"Invalid folder: {folder}")
            return
        folder = self._resolve_folder(self.ed_bomb_folder)
        metrics_path = _preferred_metrics_file(folder)
        if metrics_path is None or not metrics_path.exists():
            if allow_compute:
                self._log(f"metrics.csv not found in {folder}; computing quality metrics...")
                self.btn_load_metrics.setEnabled(False)
                self._busy_count += 1
                json_root = str(self.settings.value("preproc/json_root", str(Path.cwd() / "NeuroPyGuiN_json")))
                worker = FunctionWorker(_recompute_quality_metrics, str(folder), json_root)
                worker.signals.error.connect(self._log)
                worker.signals.finished.connect(lambda result: self._on_compute_metrics_finished(result, folder))
                self.pool.start(worker)
            else:
                self._log(f"No metrics file found in {folder}")
            return

        df = pd.read_csv(metrics_path)
        if "cluster_id" in df.columns:
            df = df.set_index("cluster_id", drop=True)
        elif "unit_id" in df.columns:
            df = df.set_index("unit_id", drop=True)
        elif df.columns.size > 0 and str(df.columns[0]).lower().startswith("unnamed"):
            df = df.set_index(df.columns[0], drop=True)
        try:
            idx = pd.to_numeric(df.index, errors="coerce")
            ok = ~pd.isna(idx)
            if ok.any():
                df = df.loc[ok]
                df.index = idx[ok].astype(int)
        except Exception:
            pass

        df = self._augment_with_psd_metrics(df, folder)
        self.metrics_df = df
        if self._selected_unit_id is not None and self._selected_unit_id not in self.metrics_df.index:
            self._selected_unit_id = None
        source_name = "py_bombcell metrics" if "templates._bc_qMetrics.csv" in str(metrics_path) else "metrics.csv"
        self._log(f"Loaded {source_name}: {metrics_path} ({len(df)} units)")

        paths = self.watcher.files()
        if paths:
            self.watcher.removePaths(paths)
        self.watcher.addPath(str(metrics_path))

        self._refresh_metric_selector()
        self._recompute_preview()

    def _augment_with_psd_metrics(self, df: pd.DataFrame, folder: Path) -> pd.DataFrame:
        """Add per-unit PSD-derived metrics to loaded quality metrics."""
        needed = {
            "psd_peak_hz",
            "psd_peak_power",
            "psd_band_0_4",
            "psd_band_4_12",
            "psd_band_12_30",
            "psd_band_30_80",
        }
        if needed.issubset(set(map(str, df.columns))):
            return df

        key = str(folder.resolve()).lower()
        cached = self._psd_metrics_cache.get(key)
        if cached is None:
            cached = self._compute_psd_metrics(folder, list(map(int, df.index.tolist())))
            self._psd_metrics_cache[key] = cached
        if cached.empty:
            return df
        out = df.copy()
        for col in cached.columns:
            out[col] = cached[col]
        return out

    def _compute_psd_metrics(self, folder: Path, unit_ids: List[int]) -> pd.DataFrame:
        """
        Compute simple per-unit PSD descriptors from binned spike trains.
        These columns are then available in metric histograms and unit inspector.
        """
        st_path = folder / "spike_times.npy"
        sc_path = folder / "spike_clusters.npy"
        if not st_path.exists() or not sc_path.exists():
            return pd.DataFrame(index=pd.Index(unit_ids, name="cluster_id"))
        try:
            spike_times = np.load(st_path).ravel().astype(np.int64)
            spike_clusters = np.load(sc_path).ravel().astype(np.int64)
        except Exception as exc:
            self._log(f"PSD metrics: failed to load spike arrays ({exc})")
            return pd.DataFrame(index=pd.Index(unit_ids, name="cluster_id"))
        if spike_times.size == 0 or spike_clusters.size == 0:
            return pd.DataFrame(index=pd.Index(unit_ids, name="cluster_id"))

        fs_raw = 30000.0
        bin_s = 0.01  # 10 ms bins
        fs_bin = 1.0 / bin_s
        bin_samples = max(1, int(round(fs_raw * bin_s)))
        n_bins = int(np.ceil((float(np.max(spike_times)) + 1.0) / float(bin_samples)))
        n_bins = max(n_bins, 2)
        nperseg = min(1024, n_bins)

        rows: Dict[int, Dict[str, float]] = {}
        for u in unit_ids:
            st_u = spike_times[spike_clusters == int(u)]
            if st_u.size < 5:
                rows[int(u)] = {
                    "psd_peak_hz": np.nan,
                    "psd_peak_power": np.nan,
                    "psd_band_0_4": np.nan,
                    "psd_band_4_12": np.nan,
                    "psd_band_12_30": np.nan,
                    "psd_band_30_80": np.nan,
                }
                continue
            b = (st_u // bin_samples).astype(np.int64)
            b = b[(b >= 0) & (b < n_bins)]
            x = np.bincount(b, minlength=n_bins).astype(float)
            try:
                f, pxx = sps.welch(x, fs=fs_bin, nperseg=nperseg, detrend="constant", scaling="density")
            except Exception:
                rows[int(u)] = {
                    "psd_peak_hz": np.nan,
                    "psd_peak_power": np.nan,
                    "psd_band_0_4": np.nan,
                    "psd_band_4_12": np.nan,
                    "psd_band_12_30": np.nan,
                    "psd_band_30_80": np.nan,
                }
                continue
            mpos = f > 0
            if not np.any(mpos):
                rows[int(u)] = {
                    "psd_peak_hz": np.nan,
                    "psd_peak_power": np.nan,
                    "psd_band_0_4": np.nan,
                    "psd_band_4_12": np.nan,
                    "psd_band_12_30": np.nan,
                    "psd_band_30_80": np.nan,
                }
                continue
            fp = f[mpos]
            pp = pxx[mpos]
            ip = int(np.argmax(pp))
            def band_power(lo: float, hi: float) -> float:
                m = (fp >= lo) & (fp < hi)
                if not np.any(m):
                    return float("nan")
                return float(np.trapz(pp[m], fp[m]))
            rows[int(u)] = {
                "psd_peak_hz": float(fp[ip]),
                "psd_peak_power": float(pp[ip]),
                "psd_band_0_4": band_power(0.0, 4.0),
                "psd_band_4_12": band_power(4.0, 12.0),
                "psd_band_12_30": band_power(12.0, 30.0),
                "psd_band_30_80": band_power(30.0, 80.0),
            }

        psd_df = pd.DataFrame.from_dict(rows, orient="index")
        psd_df.index.name = "cluster_id"
        return psd_df

    def _set_pybombcell_buttons_enabled(self, enabled: bool) -> None:
        self.btn_load_metrics.setEnabled(enabled)
        self.btn_run_pybomb.setEnabled(enabled)
        self.btn_run_selected_pybomb.setEnabled(enabled)
        self.btn_apply_pybomb_defaults.setEnabled(enabled)
        self.btn_reset_pybomb_defaults.setEnabled(enabled)
        self.btn_open_bombcell_gui.setEnabled(enabled)

    def _run_pybombcell(self) -> None:
        if hasattr(self, "_main_sections"):
            self._main_sections.setCurrentIndex(1)
        folder = Path(self.ed_bomb_folder.text().strip())
        if not folder.exists():
            self._log(f"Invalid folder: {folder}")
            return
        if not self._apply_pybombcell_defaults(announce=False):
            return
        folder = self._resolve_folder(self.ed_bomb_folder)
        self._busy_count += 1
        self._set_pybombcell_buttons_enabled(False)
        self._log("Running py_bombcell (metrics + plots)...")
        worker = FunctionWorker(
            run_pybombcell_on_folder,
            str(folder),
            True,
            False,
            self._pybombcell_settings,
            extract_raw=self.ck_extract_raw.isChecked(),
        )
        worker.signals.error.connect(self._log)
        worker.signals.finished.connect(self._on_run_pybombcell_finished)
        self.pool.start(worker)

    def _on_run_pybombcell_finished(self, result: Dict) -> None:
        self._busy_count = max(0, self._busy_count - 1)
        self._set_pybombcell_buttons_enabled(True)
        if not result.get("ok"):
            self._log("py_bombcell run failed.")
            return
        payload = result.get("result", {})
        if payload.get("cached", False):
            self._log(
                f"py_bombcell skipped ({payload.get('cache_reason', 'cached')}) | units={payload.get('n_units', 'NA')} | "
                f"metrics={payload.get('metrics_csv', '')}"
            )
        elif payload.get("metrics_reused", False):
            self._log(
                f"py_bombcell refreshed from saved metrics | units={payload.get('n_units', 'NA')} | "
                f"plots={payload.get('plots_dir', '')}"
            )
        else:
            raw_note = " | raw waveforms + SNR" if payload.get("raw_extracted") else ""
            self._log(
                f"py_bombcell completed | units={payload.get('n_units', 'NA')}{raw_note} | "
                f"plots={payload.get('plots_dir', '')}"
            )
        if self.ck_extract_raw.isChecked() and not payload.get("raw_extracted"):
            self._log(
                "Note: raw extraction was requested but the raw .bin could not be resolved from "
                "params.py; metrics are template-only."
            )
        sync_result = payload.get("phy_group_sync", {})
        if isinstance(sync_result, dict) and sync_result.get("updated"):
            self._log(
                f"Updated cluster_group.tsv from {sync_result.get('source', 'labels')} "
                f"({sync_result.get('n_units', 0)} units)"
            )
        self._refresh_folder_item_texts()
        self._load_metrics(allow_compute=False)

    def _on_run_pybombcell_list_finished(self, result: Dict) -> None:
        self._busy_count = max(0, self._busy_count - 1)
        self._set_pybombcell_buttons_enabled(True)
        if not result.get("ok"):
            self._log("py_bombcell batch run failed.")
            return
        payload = result.get("result", {})
        for item in payload.get("results", []):
            folder = str(item.get("folder") or "")
            if item.get("ok"):
                run_result = item.get("result", {})
                counts = run_result.get("counts", {})
                if run_result.get("cached", False):
                    mode = "cached"
                elif run_result.get("metrics_reused", False):
                    mode = "reused_metrics"
                else:
                    mode = "reran"
                self._log(
                    f"[py_bombcell] {folder} | {mode} | good={counts.get('good', 0)} "
                    f"noise={counts.get('noise', 0)} mua={counts.get('mua', 0)} "
                    f"non_soma={counts.get('non_soma', 0)}"
                )
            else:
                self._log(f"[py_bombcell] {folder} | failed | {item.get('error', 'unknown error')}")
        summary = payload.get("summary", {})
        self._log(
            f"py_bombcell batch finished | total={summary.get('total', 0)} reran={summary.get('reran', 0)} "
            f"reused_metrics={summary.get('reused_metrics', 0)} cached={summary.get('cached', 0)} "
            f"failed={summary.get('failed', 0)}"
        )
        self._refresh_folder_item_texts()

    def _on_compute_metrics_finished(self, result: Dict, folder: Path) -> None:
        self._busy_count = max(0, self._busy_count - 1)
        self.btn_load_metrics.setEnabled(True)
        if result.get("ok"):
            msg = result.get("result", "")
            if msg:
                self._log(str(msg))
        else:
            self._log("Quality metrics recompute failed.")
        self._load_metrics(allow_compute=False)

    def _on_metrics_changed(self, path: str) -> None:
        self._log(f"Detected metrics file update: {path}")
        self._load_metrics(allow_compute=False)

    def _apply_settings(self) -> None:
        self._recompute_preview()
        self._log("Applied Bombcell settings.")

    def _on_threshold_changed(self, _item: QtWidgets.QTableWidgetItem) -> None:
        if self._updating_table:
            return
        self._refresh_metric_selector(keep_current=True)
        self._refresh_metric_plot()
        if self.ck_live_apply.isChecked():
            self._recompute_preview()

    def _on_threshold_row_selected(self, row: int, _current_col: int, _prev_row: int, _prev_col: int) -> None:
        if row < 0:
            return
        metric_item = self.tbl_thresh.item(row, 1)
        if metric_item is None:
            return
        metric = metric_item.text().strip()
        if not metric:
            return
        for i in range(self.list_metrics.count()):
            it = self.list_metrics.item(i)
            if it.text() == metric:
                it.setSelected(True)
                self.list_metrics.scrollToItem(it)
                break

    def _refresh_metric_selector(self, keep_current: bool = False) -> None:
        current = [it.text() for it in self.list_metrics.selectedItems()] if keep_current else []
        metric_names: List[str] = []
        for row in range(self.tbl_thresh.rowCount()):
            it = self.tbl_thresh.item(row, 1)
            if it is None:
                continue
            name = it.text().strip()
            if not name:
                continue
            if self.metrics_df.empty or name in self.metrics_df.columns:
                if name not in metric_names:
                    metric_names.append(name)
        # include all available numeric metrics to maximize visibility
        if not self.metrics_df.empty:
            for c in self.metrics_df.columns:
                try:
                    if np.issubdtype(self.metrics_df[c].dtype, np.number) and str(c) not in metric_names:
                        metric_names.append(str(c))
                except Exception:
                    continue
        self.list_metrics.blockSignals(True)
        self.list_metrics.clear()
        for m in metric_names:
            item = QtWidgets.QListWidgetItem(m)
            self.list_metrics.addItem(item)
            if m in current:
                item.setSelected(True)
        self.list_metrics.blockSignals(False)
        if self.list_metrics.count() > 0 and not self.list_metrics.selectedItems():
            self.list_metrics.item(0).setSelected(True)
        self._refresh_metric_plot()

    def _selected_metrics(self) -> List[str]:
        return [it.text().strip() for it in self.list_metrics.selectedItems() if it.text().strip()]

    def _select_all_metrics(self) -> None:
        for i in range(self.list_metrics.count()):
            self.list_metrics.item(i).setSelected(True)
        self._refresh_metric_plot()

    def _clear_metrics_selection(self) -> None:
        self.list_metrics.clearSelection()
        self._refresh_metric_plot()

    def _metric_threshold_conf(self, metric_name: str) -> Dict:
        for row in range(self.tbl_thresh.rowCount()):
            met_item = self.tbl_thresh.item(row, 1)
            if met_item is None or met_item.text().strip() != metric_name:
                continue
            min_item = self.tbl_thresh.item(row, 2)
            max_item = self.tbl_thresh.item(row, 3)
            abs_item = self.tbl_thresh.item(row, 4)
            return {
                "min": self._parse_optional_float(min_item.text() if min_item else ""),
                "max": self._parse_optional_float(max_item.text() if max_item else ""),
                "abs": abs_item is not None and abs_item.text().strip() in {"1", "true", "True", "yes", "YES"},
            }
        return {"min": None, "max": None, "abs": False}

    def _refresh_metric_plot(self) -> None:
        self.metrics_grid.clear()
        self._plot_lines.clear()
        self._metric_plots.clear()
        self._min_line = None
        self._max_line = None
        if self.metrics_df.empty:
            return
        metrics = [m for m in self._selected_metrics() if m in self.metrics_df.columns]
        if not metrics:
            return
        n = len(metrics)
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / max(cols, 1)))
        fg = "#e8eef7" if self._plot_theme == "Dark" else "#1a1f29"

        for i, metric in enumerate(metrics):
            r = i // cols
            c = i % cols
            plot = self.metrics_grid.addPlot(row=r, col=c, title=metric)
            plot.showGrid(x=self._show_grid, y=self._show_grid, alpha=0.25 if self._show_grid else 0.0)
            plot.getAxis("left").setTextPen(pg.mkPen(fg))
            plot.getAxis("bottom").setTextPen(pg.mkPen(fg))
            plot.getAxis("left").setPen(pg.mkPen(fg))
            plot.getAxis("bottom").setPen(pg.mkPen(fg))
            plot.setLabel("left", "units")
            plot.setLabel("bottom", "value")
            conf = self._metric_threshold_conf(metric)
            vals = pd.to_numeric(self.metrics_df[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if conf.get("abs", False):
                vals = np.abs(vals)
            if vals.size == 0:
                continue
            n_bins = min(80, max(15, int(np.sqrt(vals.size))))
            hist, edges = np.histogram(vals, bins=n_bins)
            curve = pg.BarGraphItem(
                x=edges[:-1],
                height=hist,
                width=np.diff(edges),
                brush=pg.intColor(i, hues=max(len(metrics), 6), alpha=90),
            )
            plot.addItem(curve)
            # Show threshold values directly on each metric distribution plot.
            min_v = conf.get("min")
            max_v = conf.get("max")
            if min_v is not None:
                plot.addItem(
                    pg.InfiniteLine(
                        pos=float(min_v),
                        angle=90,
                        movable=False,
                        pen=pg.mkPen((80, 220, 120), width=2),
                        label=f"min {float(min_v):.4g}",
                        labelOpts={"position": 0.92, "color": (80, 220, 120)},
                    )
                )
            if max_v is not None:
                plot.addItem(
                    pg.InfiniteLine(
                        pos=float(max_v),
                        angle=90,
                        movable=False,
                        pen=pg.mkPen((255, 110, 110), width=2),
                        label=f"max {float(max_v):.4g}",
                        labelOpts={"position": 0.82, "color": (255, 110, 110)},
                    )
                )
            # Overlay selected unit value on each distribution.
            if self._selected_unit_id is not None and self._selected_unit_id in self.metrics_df.index:
                try:
                    uv = float(pd.to_numeric(self.metrics_df.at[self._selected_unit_id, metric], errors="coerce"))
                    if conf.get("abs", False):
                        uv = abs(uv)
                    if np.isfinite(uv):
                        plot.addItem(
                            pg.InfiniteLine(
                                pos=float(uv),
                                angle=90,
                                movable=False,
                                pen=pg.mkPen((70, 150, 255), width=2),
                                label=f"u{self._selected_unit_id}: {uv:.4g}",
                                labelOpts={"position": 0.1, "color": (70, 150, 255)},
                            )
                        )
                except Exception:
                    pass
            self._plot_lines.append(pg.PlotDataItem())
            self._metric_plots.append(plot)

        # primary metric gets draggable threshold cursors
        primary = metrics[0]
        conf0 = self._metric_threshold_conf(primary)
        min_v = conf0.get("min")
        max_v = conf0.get("max")
        primary_plot = self._metric_plots[0] if self._metric_plots else None
        if primary_plot is None:
            return
        if min_v is not None:
            self._min_line = pg.InfiniteLine(
                pos=float(min_v),
                angle=90,
                movable=True,
                pen=pg.mkPen((80, 220, 120), width=2),
                label=f"min {float(min_v):.4g}",
                labelOpts={"position": 0.97, "color": (80, 220, 120)},
            )
            self._min_line.sigPositionChangeFinished.connect(lambda _=None: self._threshold_cursor_moved(primary, "min", self._min_line))
            primary_plot.addItem(self._min_line)
        else:
            self._min_line = None
        if max_v is not None:
            self._max_line = pg.InfiniteLine(
                pos=float(max_v),
                angle=90,
                movable=True,
                pen=pg.mkPen((255, 110, 110), width=2),
                label=f"max {float(max_v):.4g}",
                labelOpts={"position": 0.87, "color": (255, 110, 110)},
            )
            self._max_line.sigPositionChangeFinished.connect(lambda _=None: self._threshold_cursor_moved(primary, "max", self._max_line))
            primary_plot.addItem(self._max_line)
        else:
            self._max_line = None

    def _threshold_cursor_moved(self, metric: str, bound: str, line: Optional[pg.InfiniteLine]) -> None:
        if line is None:
            return
        val = float(line.value())
        col = 2 if bound == "min" else 3
        for row in range(self.tbl_thresh.rowCount()):
            met_item = self.tbl_thresh.item(row, 1)
            if met_item is None or met_item.text().strip() != metric:
                continue
            self._updating_table = True
            self.tbl_thresh.setItem(row, col, QtWidgets.QTableWidgetItem(f"{val:.6g}"))
            self._updating_table = False
            self._apply_settings()
            break

    def _recompute_preview(self) -> None:
        if self.metrics_df.empty:
            return
        try:
            labels = bombcell_label_units_from_metrics(self.metrics_df, thresholds=self._thresholds_from_table())
        except Exception as exc:
            self._log(f"Threshold preview error: {exc}")
            return

        self.preview_labels = labels
        counts = labels["bombcell_label"].value_counts().to_dict()
        self.lbl_good.setText(f"good: {counts.get('good', 0)}")
        self.lbl_noise.setText(f"noise: {counts.get('noise', 0)}")
        self.lbl_mua.setText(f"mua: {counts.get('mua', 0)}")
        self.lbl_non_soma.setText(f"non_soma: {counts.get('non_soma', 0)}")
        self._refresh_report_plot({str(key): int(value) for key, value in counts.items()})

        self._fill_list(self.list_good, labels, "good")
        self._fill_list(self.list_noise, labels, "noise")
        self._fill_list(self.list_mua, labels, "mua")
        self._fill_list(self.list_non_soma, labels, "non_soma")
        self._refresh_unit_inspector()
        self._refresh_metric_plot()

    def _fill_list(self, target: QtWidgets.QListWidget, labels_df: pd.DataFrame, label_name: str) -> None:
        target.clear()
        subset = labels_df.index[labels_df["bombcell_label"] == label_name]
        target.addItems([str(u) for u in subset.tolist()])

    def _on_unit_selection_changed(self, source: QtWidgets.QListWidget) -> None:
        items = source.selectedItems()
        if not items:
            return
        for lst in [self.list_good, self.list_noise, self.list_mua, self.list_non_soma]:
            if lst is source:
                continue
            lst.blockSignals(True)
            lst.clearSelection()
            lst.blockSignals(False)
        txt = items[0].text().strip()
        try:
            self._selected_unit_id = int(float(txt))
        except Exception:
            self._selected_unit_id = None
        self._refresh_unit_inspector()
        self._refresh_metric_plot()

    def _unit_label_for_id(self, unit_id: int) -> str:
        if self.preview_labels.empty or "bombcell_label" not in self.preview_labels.columns:
            return "unknown"
        if unit_id in self.preview_labels.index:
            return str(self.preview_labels.at[unit_id, "bombcell_label"])
        return "unknown"

    def _refresh_unit_inspector(self) -> None:
        unit_id = self._selected_unit_id
        if unit_id is None:
            self.lbl_selected_unit.setText("Selected unit: -")
            self.lbl_selected_label.setText("Label: -")
            self.tbl_unit_metrics.setRowCount(0)
            return
        self.lbl_selected_unit.setText(f"Selected unit: {unit_id}")
        self.lbl_selected_label.setText(f"Label: {self._unit_label_for_id(unit_id)}")
        if self.metrics_df.empty or unit_id not in self.metrics_df.index:
            self.tbl_unit_metrics.setRowCount(0)
            return
        row_values = self.metrics_df.loc[unit_id]
        metrics = [m for m in self._selected_metrics() if m in row_values.index]
        if not metrics:
            metrics = [str(c) for c in self.metrics_df.columns[:24]]
        self.tbl_unit_metrics.setRowCount(0)
        for m in metrics:
            row = self.tbl_unit_metrics.rowCount()
            self.tbl_unit_metrics.insertRow(row)
            self.tbl_unit_metrics.setItem(row, 0, QtWidgets.QTableWidgetItem(str(m)))
            raw = row_values.get(m, np.nan)
            try:
                txt = f"{float(raw):.6g}"
            except Exception:
                txt = str(raw)
            self.tbl_unit_metrics.setItem(row, 1, QtWidgets.QTableWidgetItem(txt))
        self.tbl_unit_metrics.resizeColumnsToContents()

    def _save_labels(self) -> None:
        folder = self.ed_bomb_folder.text().strip()
        if not folder:
            self._log("Select Kilosort folder first.")
            return
        if self.metrics_df.empty:
            self._load_metrics()
            if self.metrics_df.empty:
                return

        thresholds = self._thresholds_from_table()
        self.btn_save_labels.setEnabled(False)
        self._busy_count += 1
        worker = FunctionWorker(run_bombcell_on_folder_with_thresholds, folder, thresholds)
        worker.signals.error.connect(self._log)
        worker.signals.finished.connect(self._on_save_finished)
        self.pool.start(worker)

    def _on_save_finished(self, result: Dict) -> None:
        self._busy_count = max(0, self._busy_count - 1)
        self.btn_save_labels.setEnabled(True)
        if not result.get("ok"):
            self._log("Bombcell save failed.")
            return
        payload = result.get("result", {})
        counts = payload.get("counts", {})
        self._log(
            f"Saved bombcell_labels.csv | units={payload.get('n_units')} "
            f"good={counts.get('good', 0)} noise={counts.get('noise', 0)} "
            f"mua={counts.get('mua', 0)} non_soma={counts.get('non_soma', 0)}"
        )
        sync_result = payload.get("phy_group_sync", {})
        if isinstance(sync_result, dict) and sync_result.get("updated"):
            self._log(
                f"Updated cluster_group.tsv from {sync_result.get('source', 'labels')} "
                f"({sync_result.get('n_units', 0)} units)"
            )
        self._refresh_folder_item_texts()
        self._recompute_preview()

    def _log(self, line: str) -> None:
        self.log.appendPlainText(line)

    def _copy_log(self) -> None:
        QtWidgets.QApplication.clipboard().setText(self.log.toPlainText())
        self._log("Curation log copied to clipboard.")

    def set_plot_preferences(self, theme: str, show_grid: bool) -> None:
        self._plot_theme = "Dark" if str(theme).lower().startswith("dark") else "Light"
        self._show_grid = bool(show_grid)
        bg = "#0b0f14" if self._plot_theme == "Dark" else "#ffffff"
        self.metrics_grid.setBackground(bg)
        self.report_plot.setBackground(bg)
        self._refresh_metric_plot()
        if not self.preview_labels.empty:
            counts = self.preview_labels["bombcell_label"].value_counts().to_dict()
            self._refresh_report_plot({str(key): int(value) for key, value in counts.items()})

    def _export_plotted_data(self) -> None:
        if self.metrics_df.empty:
            self._log("Export: no metrics loaded.")
            return
        start = self.settings.value("paths/last_folder", str(Path.cwd()))
        fp, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export curation plotted data", str(start), "CSV files (*.csv)")
        if not fp:
            return
        base = Path(fp)
        metrics = [m for m in self._selected_metrics() if m in self.metrics_df.columns]
        if not metrics:
            self._log("Export: no selected metrics to export.")
            return
        rows = []
        for m in metrics:
            conf = self._metric_threshold_conf(m)
            vals = pd.to_numeric(self.metrics_df[m], errors="coerce")
            for unit_id, v in vals.items():
                if pd.isna(v):
                    continue
                vv = float(v)
                rows.append(
                    {
                        "cluster_id": int(unit_id) if str(unit_id).isdigit() else unit_id,
                        "metric": str(m),
                        "value": vv,
                        "abs_value": abs(vv) if conf.get("abs", False) else vv,
                        "threshold_min": conf.get("min"),
                        "threshold_max": conf.get("max"),
                        "threshold_abs": bool(conf.get("abs", False)),
                    }
                )
        out_df = pd.DataFrame(rows)
        out_df.to_csv(base, index=False)
        if not self.preview_labels.empty:
            self.preview_labels.reset_index().to_csv(base.with_name(base.stem + "_labels.csv"), index=False)
        self._log(f"Exported curation plotted data: {base}")

    def is_busy(self) -> bool:
        return self._busy_count > 0

    def _toggle_plot_detach(self, checked: bool) -> None:
        if checked:
            self._detach_plots()
        else:
            self._attach_plots()

    def _detach_plots(self) -> None:
        if self._plots_detached or self._right_metrics_l is None:
            return
        if self._body_split is not None:
            self._body_sizes_before_detach = self._body_split.sizes()
        self._right_metrics_l.removeWidget(self.metrics_grid)
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Curation plots")
        dlg.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        dlg.resize(1200, 760)
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(self.metrics_grid)
        dlg.finished.connect(lambda _=0: self._attach_plots())
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        self._plots_dialog = dlg
        self._plots_detached = True
        if self._right_metrics is not None:
            self._right_metrics.hide()
        if self._body_split is not None:
            self._body_split.setSizes([1, 0])
        self.btn_detach_plots.setText("Attach plots")

    def _attach_plots(self) -> None:
        if not self._plots_detached or self._right_metrics_l is None:
            self.btn_detach_plots.blockSignals(True)
            self.btn_detach_plots.setChecked(False)
            self.btn_detach_plots.blockSignals(False)
            self.btn_detach_plots.setText("Detach plots")
            return
        if self._plots_dialog is not None and self._plots_dialog.layout() is not None:
            self._plots_dialog.layout().removeWidget(self.metrics_grid)
        self._right_metrics_l.addWidget(self.metrics_grid, 1)
        if self._right_metrics is not None:
            self._right_metrics.show()
        if self._body_split is not None:
            if self._body_sizes_before_detach:
                self._body_split.setSizes(self._body_sizes_before_detach)
            else:
                self._body_split.setSizes([2, 5])
        if self._plots_dialog is not None and self._plots_dialog.isVisible():
            self._plots_dialog.blockSignals(True)
            self._plots_dialog.close()
            self._plots_dialog.blockSignals(False)
        self._plots_dialog = None
        self._plots_detached = False
        self.btn_detach_plots.blockSignals(True)
        self.btn_detach_plots.setChecked(False)
        self.btn_detach_plots.blockSignals(False)
        self.btn_detach_plots.setText("Detach plots")

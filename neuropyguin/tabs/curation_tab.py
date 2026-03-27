from __future__ import annotations

import ast
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import signal as sps
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from ..bombcell_core import (
    bombcell_get_default_thresholds,
    bombcell_label_units_from_metrics,
    run_bombcell_on_folder_with_thresholds,
)
from ..ecephys_runtime import ecephys_subprocess_env
from ..ks_output_resolver import find_kilosort_output_dir, find_metrics_file
from ..pybombcell_integration import run_pybombcell_on_folder
from ..workers import FunctionWorker


def _find_modules_input_json(json_root: Path, ks_folder: Path) -> Optional[Path]:
    if not json_root.exists():
        return None
    target_dir = find_kilosort_output_dir(ks_folder, max_depth=4) or ks_folder
    target = str(target_dir.resolve()).lower()
    for p in json_root.glob("*_modules-input.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            d = data.get("directories", {})
            kdir = str(d.get("kilosort_output_directory", "")).lower()
            resolved_kdir = find_kilosort_output_dir(kdir, max_depth=4) or Path(kdir)
            if kdir and resolved_kdir.resolve().as_posix().lower() == Path(target).resolve().as_posix().lower():
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
    proc = subprocess.run(cmd, cwd=str(folder), capture_output=True, text=True, env=ecephys_subprocess_env())
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


class CurationTab(QtWidgets.QWidget):
    def __init__(self, thread_pool: QtCore.QThreadPool) -> None:
        super().__init__()
        self.pool = thread_pool
        self.settings = QtCore.QSettings("NeuroPyGuiN", "NeuroPyGuiN")
        self.metrics_df = pd.DataFrame()
        self.preview_labels = pd.DataFrame()
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
        self._settings_dialog: Optional[QtWidgets.QDialog] = None
        self._body_split: Optional[QtWidgets.QSplitter] = None
        self._right_metrics: Optional[QtWidgets.QWidget] = None
        self._right_metrics_l: Optional[QtWidgets.QVBoxLayout] = None
        self._body_sizes_before_detach: List[int] = []

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
        self._plot_theme = "Light"
        self._show_grid = True

    def _build_ui(self) -> None:
        main = QtWidgets.QVBoxLayout(self)
        main.setContentsMargins(18, 16, 18, 18)
        main.setSpacing(14)

        grp_phy = QtWidgets.QGroupBox("Phy")
        grp_phy.setProperty("settingsSection", True)
        phy_row = QtWidgets.QHBoxLayout(grp_phy)
        phy_row.setSpacing(10)

        self.ed_phy_folder = QtWidgets.QLineEdit()
        btn_phy_folder = QtWidgets.QPushButton("Browse")
        self.btn_launch_phy = QtWidgets.QPushButton("Open in Phy")
        self.btn_stop_phy = QtWidgets.QPushButton("Stop")
        self.btn_launch_phy.setProperty("role", "primary")
        self.btn_stop_phy.setProperty("role", "ghost")
        phy_row.addWidget(QtWidgets.QLabel("Curated folder"))
        phy_row.addWidget(self.ed_phy_folder, 1)
        phy_row.addWidget(btn_phy_folder)
        phy_row.addWidget(self.btn_launch_phy)
        phy_row.addWidget(self.btn_stop_phy)

        grp_bomb = QtWidgets.QGroupBox("Bombcell: live QC")
        bomb_layout = QtWidgets.QVBoxLayout(grp_bomb)
        bomb_layout.setSpacing(10)

        bomb_top = QtWidgets.QHBoxLayout()
        self.ed_bomb_folder = QtWidgets.QLineEdit()
        btn_bomb_folder = QtWidgets.QPushButton("Browse")
        self.btn_load_metrics = QtWidgets.QPushButton("Load metrics.csv")
        self.btn_run_pybomb = QtWidgets.QPushButton("Run py_bombcell")
        self.btn_open_settings = QtWidgets.QPushButton("Threshold settings")
        self.btn_load_metrics.setProperty("role", "secondary")
        self.btn_run_pybomb.setProperty("role", "primary")
        self.btn_open_settings.setProperty("role", "secondary")
        bomb_top.addWidget(self.ed_bomb_folder, 1)
        bomb_top.addWidget(btn_bomb_folder)
        bomb_top.addWidget(self.btn_load_metrics)
        bomb_top.addWidget(self.btn_run_pybomb)
        bomb_top.addWidget(self.btn_open_settings)

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

        settings_panel = QtWidgets.QWidget()
        settings_panel_l = QtWidgets.QVBoxLayout(settings_panel)
        settings_panel_l.setContentsMargins(0, 0, 0, 0)
        settings_panel_l.setSpacing(10)
        threshold_box = QtWidgets.QGroupBox("Threshold settings")
        threshold_box.setProperty("settingsSection", True)
        threshold_l = QtWidgets.QVBoxLayout(threshold_box)
        threshold_l.addWidget(self.tbl_thresh, 1)
        metric_box = QtWidgets.QGroupBox("Metric selection")
        metric_box.setProperty("settingsSection", True)
        metric_box_l = QtWidgets.QVBoxLayout(metric_box)
        metric_box_l.addWidget(list_col, 1)
        settings_panel_l.addWidget(threshold_box, 2)
        settings_panel_l.addWidget(metric_box, 1)

        self._settings_dialog = QtWidgets.QDialog(self)
        self._settings_dialog.setWindowTitle("Bombcell settings")
        self._settings_dialog.setModal(False)
        self._settings_dialog.resize(1040, 760)
        self._settings_dialog.setMinimumSize(860, 620)
        settings_dialog_l = QtWidgets.QVBoxLayout(self._settings_dialog)
        settings_hint = QtWidgets.QLabel(
            "Threshold editing and metric selection live here so the curation tab can dedicate more space to plots and unit review."
        )
        settings_hint.setObjectName("SectionHint")
        settings_hint.setWordWrap(True)
        settings_dialog_l.addWidget(settings_hint)
        settings_dialog_l.addWidget(settings_panel, 1)
        settings_btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        settings_btns.rejected.connect(self._settings_dialog.close)
        settings_dialog_l.addWidget(settings_btns, 0)

        self.metrics_grid = pg.GraphicsLayoutWidget()
        self.metrics_grid.setMinimumHeight(420)
        plots_panel = QtWidgets.QWidget()
        plots_panel_l = QtWidgets.QVBoxLayout(plots_panel)
        plots_panel_l.setContentsMargins(0, 0, 0, 0)
        plots_panel_l.setSpacing(0)
        plots_panel_l.addWidget(self.metrics_grid, 1)
        self._right_metrics = plots_panel
        self._right_metrics_l = plots_panel_l

        action_row = QtWidgets.QHBoxLayout()
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
        for w in [self.lbl_good, self.lbl_noise, self.lbl_mua, self.lbl_non_soma]:
            action_row.addWidget(w)
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
        unit_split.addWidget(self.tabs_units)
        unit_split.addWidget(unit_inspector)
        unit_split.setStretchFactor(0, 2)
        unit_split.setStretchFactor(1, 3)
        unit_split.setSizes([300, 700])

        body_split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self._body_split = body_split
        body_split.addWidget(plots_panel)
        body_split.addWidget(unit_split)
        body_split.setStretchFactor(0, 5)
        body_split.setStretchFactor(1, 2)
        body_split.setSizes([680, 240])

        bomb_layout.addLayout(bomb_top)
        bomb_layout.addLayout(action_row)
        bomb_layout.addWidget(body_split, 1)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setProperty("logView", True)
        self.log.setPlaceholderText("Curation and Phy output will appear here.")
        self.log.setMaximumHeight(120)

        main.addWidget(grp_phy, 0)
        main.addWidget(grp_bomb, 1)
        main.addWidget(self.log, 0)

        btn_phy_folder.clicked.connect(lambda: self._pick_folder(self.ed_phy_folder))
        btn_bomb_folder.clicked.connect(lambda: self._pick_folder(self.ed_bomb_folder))
        self.btn_launch_phy.clicked.connect(self._launch_phy)
        self.btn_stop_phy.clicked.connect(self._stop_phy)

        self.btn_load_metrics.clicked.connect(self._load_metrics)
        self.btn_run_pybomb.clicked.connect(self._run_pybombcell)
        self.btn_open_settings.clicked.connect(self._open_settings_window)
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

    def _open_settings_window(self) -> None:
        if self._settings_dialog is None:
            return
        state = self._settings_dialog.windowState()
        if state & QtCore.Qt.WindowMinimized:
            self._settings_dialog.setWindowState(state & ~QtCore.Qt.WindowMinimized)
        self._settings_dialog.show()
        self._settings_dialog.raise_()
        self._settings_dialog.activateWindow()

    def _pick_folder(self, target: QtWidgets.QLineEdit) -> None:
        start = self.settings.value("paths/last_folder", str(Path.cwd()))
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder", str(start))
        if folder:
            target.setText(folder)
            self.settings.setValue("paths/last_folder", folder)
            if target is self.ed_phy_folder:
                self.settings.setValue("curation/phy_folder", folder)
            elif target is self.ed_bomb_folder:
                self.settings.setValue("curation/bomb_folder", folder)

    def _resolve_folder(self, target: QtWidgets.QLineEdit) -> Path:
        folder = Path(target.text().strip())
        resolved = find_kilosort_output_dir(folder, max_depth=4)
        if resolved is not None and resolved != folder:
            self._log(f"Resolved KS folder to {resolved}")
            target.setText(str(resolved))
            return resolved
        return folder

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
        self.ed_phy_folder.setText(folder)
        self.ed_bomb_folder.setText(folder)

    def open_ks_folder(self, folder: str) -> None:
        self.set_ks_folder(folder)
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
        phy_folder = self.settings.value("curation/phy_folder", "")
        bomb_folder = self.settings.value("curation/bomb_folder", "")
        if phy_folder:
            self.ed_phy_folder.setText(str(phy_folder))
        if bomb_folder:
            self.ed_bomb_folder.setText(str(bomb_folder))

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
        folder = Path(self.ed_bomb_folder.text().strip())
        if not folder.exists():
            self._log(f"Invalid folder: {folder}")
            return
        folder = self._resolve_folder(self.ed_bomb_folder)
        metrics_path = find_metrics_file(folder, max_depth=4)
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
                self._log(f"metrics.csv not found in {folder}")
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
        self._log(f"Loaded metrics: {metrics_path} ({len(df)} units)")

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

    def _run_pybombcell(self) -> None:
        folder = Path(self.ed_bomb_folder.text().strip())
        if not folder.exists():
            self._log(f"Invalid folder: {folder}")
            return
        folder = self._resolve_folder(self.ed_bomb_folder)
        self._busy_count += 1
        self.btn_run_pybomb.setEnabled(False)
        self._log("Running py_bombcell (metrics + plots)...")
        worker = FunctionWorker(run_pybombcell_on_folder, str(folder), True)
        worker.signals.error.connect(self._log)
        worker.signals.finished.connect(self._on_run_pybombcell_finished)
        self.pool.start(worker)

    def _on_run_pybombcell_finished(self, result: Dict) -> None:
        self._busy_count = max(0, self._busy_count - 1)
        self.btn_run_pybomb.setEnabled(True)
        if not result.get("ok"):
            self._log("py_bombcell run failed.")
            return
        payload = result.get("result", {})
        if payload.get("cached", False):
            self._log(
                f"py_bombcell skipped (cached metrics found) | units={payload.get('n_units', 'NA')} | "
                f"metrics={payload.get('metrics_csv', '')}"
            )
        else:
            self._log(f"py_bombcell completed | units={payload.get('n_units', 'NA')} | plots={payload.get('plots_dir', '')}")
        self._load_metrics(allow_compute=False)

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
        self._recompute_preview()

    def _log(self, line: str) -> None:
        self.log.appendPlainText(line)

    def set_plot_preferences(self, theme: str, show_grid: bool) -> None:
        self._plot_theme = "Dark" if str(theme).lower().startswith("dark") else "Light"
        self._show_grid = bool(show_grid)
        bg = "#0b0f14" if self._plot_theme == "Dark" else "#ffffff"
        self.metrics_grid.setBackground(bg)
        self._refresh_metric_plot()

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
            self._body_split.setSizes([0, 1])
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
                self._body_split.setSizes([5, 2])
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



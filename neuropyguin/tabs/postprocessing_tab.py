from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from ..postproc_events import inspect_event_csv, load_event_times
from ..postproc_engine import NeuropixelsDataset, cluster_synced_units, export_units_h5
from ..npyx_corr_bridge import PAIRWISE_ONLY_METHODS, method_metadata, method_options, run_method


def _is_bombcell_good_label(value: object) -> bool:
    text = str(value).strip().lower()
    if not text or text == "nan":
        return False
    normalized = text.replace("-", "_").replace(" ", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized in {"good", "non_soma", "non_soma_good", "nonsoma", "nonsomagood"}


def _sync_group_color(group_id: int, alpha: int = 220) -> tuple[int, int, int, int]:
    if int(group_id) <= 0:
        return (120, 132, 150, alpha)
    color = pg.intColor(int(group_id) - 1, hues=10, values=1, maxValue=235)
    return (int(color.red()), int(color.green()), int(color.blue()), int(alpha))


class PostProcessingTab(QtWidgets.QWidget):
    def __init__(self, thread_pool: QtCore.QThreadPool) -> None:
        super().__init__()
        self.pool = thread_pool
        self.settings = QtCore.QSettings('NeuroPyGuiN', 'NeuroPyGuiN')
        self.dataset: Optional[NeuropixelsDataset] = None
        self.metrics_df = pd.DataFrame()
        self.labels_df = pd.DataFrame()
        self.label_sources: Dict[str, pd.DataFrame] = {}
        self._all_units: list[int] = []
        self.results: Dict[str, object] = {}
        self._export_payloads: Dict[str, list[tuple[str, pd.DataFrame]]] = {}
        self._basic_cache: Dict[str, Dict[str, object]] = {}
        self._plot_theme = 'Light'
        self._show_grid = True
        self._busy = False
        self._plot_detached = False
        self._plot_dialog: Optional[QtWidgets.QDialog] = None
        self._right_panel_layout: Optional[QtWidgets.QVBoxLayout] = None
        self._body_splitter: Optional[QtWidgets.QSplitter] = None
        self._right_panel: Optional[QtWidgets.QWidget] = None
        self._basic_row2_layout: Optional[QtWidgets.QHBoxLayout] = None
        self._body_sizes_before_detach: list[int] = []
        self._build_ui()
        self._restore_settings()

    def _build_ui(self) -> None:
        main = QtWidgets.QVBoxLayout(self)
        main.setContentsMargins(18, 16, 18, 18)
        main.setSpacing(14)
        def with_help(widget: QtWidgets.QWidget, text: str) -> QtWidgets.QWidget:
            q = QtWidgets.QToolButton()
            q.setText("?")
            q.setAutoRaise(True)
            q.setToolTip(text)
            q.setProperty("helpButton", True)
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(widget, 1)
            row.addWidget(q, 0)
            host = QtWidgets.QWidget()
            host.setLayout(row)
            return host

        top = QtWidgets.QHBoxLayout()
        top.setSpacing(8)
        self.ed_folder = QtWidgets.QLineEdit()
        self.btn_browse = QtWidgets.QPushButton("Browse curated folder")
        self.btn_load = QtWidgets.QPushButton("Load dataset")
        self.btn_export = QtWidgets.QPushButton("Export plotted data")
        self.btn_export_units = QtWidgets.QPushButton("Export units to H5")
        self.btn_detach_plots = QtWidgets.QPushButton("Detach plots")
        self.btn_detach_plots.setCheckable(True)
        self.btn_browse.setProperty("role", "secondary")
        self.btn_load.setProperty("role", "primary")
        self.btn_export.setProperty("role", "ghost")
        self.btn_export_units.setProperty("role", "secondary")
        self.btn_detach_plots.setProperty("role", "ghost")
        top.addWidget(self.ed_folder, 1)
        top.addWidget(self.btn_browse)
        top.addWidget(self.btn_load)
        top.addWidget(self.btn_export)
        top.addWidget(self.btn_export_units)
        top.addWidget(self.btn_detach_plots)

        body = QtWidgets.QSplitter()
        self._body_splitter = body

        left_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        units_col = QtWidgets.QWidget()
        units_col_l = QtWidgets.QVBoxLayout(units_col)
        controls_col = QtWidgets.QWidget()
        controls_col_l = QtWidgets.QVBoxLayout(controls_col)

        grp_units = QtWidgets.QGroupBox("Units")
        u_l = QtWidgets.QVBoxLayout(grp_units)
        unit_filter_row = QtWidgets.QHBoxLayout()
        self.ed_unit_filter = QtWidgets.QLineEdit()
        self.ed_unit_filter.setPlaceholderText("Filter unit id")
        self.btn_good_only = QtWidgets.QPushButton("Show good units only")
        self.btn_good_only.setCheckable(True)
        self.cb_good_source = QtWidgets.QComboBox()
        self.cb_good_source.addItems(["Auto", "Bombcell", "Phy", "KSLabel"])
        unit_filter_row.addWidget(self.ed_unit_filter, 1)
        unit_filter_row.addWidget(self.btn_good_only)
        unit_filter_row.addWidget(QtWidgets.QLabel("Good source"))
        unit_filter_row.addWidget(self.cb_good_source)

        self.list_units = QtWidgets.QListWidget()
        self.list_units.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        self.tbl_unit_quality = QtWidgets.QTableWidget(0, 2)
        self.tbl_unit_quality.setAlternatingRowColors(True)
        self.tbl_unit_quality.setHorizontalHeaderLabels(["Metric", "Value"])
        self.tbl_unit_quality.horizontalHeader().setStretchLastSection(True)
        self.tbl_unit_quality.verticalHeader().setVisible(False)
        self.tbl_unit_quality.setMaximumHeight(220)

        u_l.addLayout(unit_filter_row)
        u_l.addWidget(self.list_units, 1)
        u_l.addWidget(self.tbl_unit_quality, 0)

        grp_menu = QtWidgets.QGroupBox("Analysis Pages")
        m_l = QtWidgets.QVBoxLayout(grp_menu)
        self.analysis_tabs = QtWidgets.QTabWidget()
        m_l.addWidget(self.analysis_tabs)

        t_basic = QtWidgets.QWidget()
        f_basic = QtWidgets.QFormLayout(t_basic)
        self.sp_basic_t0 = QtWidgets.QDoubleSpinBox()
        self.sp_basic_t0.setRange(0.0, 2e6)
        self.sp_basic_t0.setValue(0.0)
        self.sp_basic_t0.setSuffix(" s")
        self.sp_basic_dur = QtWidgets.QDoubleSpinBox()
        self.sp_basic_dur.setRange(0.05, 2e6)
        self.sp_basic_dur.setValue(1.0)
        self.sp_basic_dur.setSuffix(" s")
        self.sp_isi_max = QtWidgets.QDoubleSpinBox()
        self.sp_isi_max.setRange(10, 3000)
        self.sp_isi_max.setValue(200)
        self.sp_isi_max.setSuffix(" ms")
        self.sp_basic_acg_bin = QtWidgets.QDoubleSpinBox()
        self.sp_basic_acg_bin.setRange(0.1, 20.0)
        self.sp_basic_acg_bin.setValue(1.0)
        self.sp_basic_acg_bin.setSuffix(" ms")
        self.sp_basic_acg_win = QtWidgets.QDoubleSpinBox()
        self.sp_basic_acg_win.setRange(10.0, 1000.0)
        self.sp_basic_acg_win.setValue(100.0)
        self.sp_basic_acg_win.setSuffix(" ms")
        self.sp_basic_acg_ratio = QtWidgets.QSpinBox()
        self.sp_basic_acg_ratio.setRange(1, 12)
        self.sp_basic_acg_ratio.setValue(2)
        self.sp_basic_isi_ratio = QtWidgets.QSpinBox()
        self.sp_basic_isi_ratio.setRange(1, 12)
        self.sp_basic_isi_ratio.setValue(1)
        ratio_row = QtWidgets.QWidget()
        ratio_row_l = QtWidgets.QHBoxLayout(ratio_row)
        ratio_row_l.setContentsMargins(0, 0, 0, 0)
        ratio_row_l.addWidget(self.sp_basic_acg_ratio, 1)
        ratio_row_l.addWidget(QtWidgets.QLabel(":"))
        ratio_row_l.addWidget(self.sp_basic_isi_ratio, 1)
        self.ck_ifr = QtWidgets.QCheckBox("Overlay instantaneous firing rate")
        self.ck_ifr.setChecked(True)
        self.sp_ifr_smooth_ms = QtWidgets.QDoubleSpinBox()
        self.sp_ifr_smooth_ms.setRange(1.0, 500.0)
        self.sp_ifr_smooth_ms.setValue(30.0)
        self.sp_ifr_smooth_ms.setSuffix(" ms")
        f_basic.addRow("Window start", with_help(self.sp_basic_t0, "Start time (s) for unit-raster window."))
        f_basic.addRow("Window duration", with_help(self.sp_basic_dur, "Duration (s) of displayed unit-raster window."))
        f_basic.addRow("ISI max", with_help(self.sp_isi_max, "Maximum ISI bin range for histogram (ms)."))
        f_basic.addRow("ACG bin", with_help(self.sp_basic_acg_bin, "Auto-correlogram bin size (ms) for Unit Basics."))
        f_basic.addRow("ACG window", with_help(self.sp_basic_acg_win, "Auto-correlogram half-window (ms) for Unit Basics."))
        f_basic.addRow("ACG:ISI ratio", with_help(ratio_row, "Width ratio for the Unit Basics row (ACG : ISI, with waveform centered)."))
        f_basic.addRow("IFR smooth", with_help(self.sp_ifr_smooth_ms, "Bin/smoothing window for instantaneous firing rate (ms)."))
        f_basic.addRow(self.ck_ifr)
        self.analysis_tabs.addTab(t_basic, "Unit Basics")

        t_raw = QtWidgets.QWidget()
        f_raw = QtWidgets.QFormLayout(t_raw)
        self.sp_raw_t0 = QtWidgets.QDoubleSpinBox()
        self.sp_raw_t0.setRange(0.0, 2e6)
        self.sp_raw_t0.setValue(0.0)
        self.sp_raw_t0.setSuffix(" s")
        self.sp_raw_dur = QtWidgets.QDoubleSpinBox()
        self.sp_raw_dur.setRange(0.05, 60.0)
        self.sp_raw_dur.setValue(1.0)
        self.sp_raw_dur.setSuffix(" s")
        self.sp_raw_ch = QtWidgets.QSpinBox()
        self.sp_raw_ch.setRange(4, 256)
        self.sp_raw_ch.setValue(100)
        self.sp_raw_hp = QtWidgets.QDoubleSpinBox()
        self.sp_raw_hp.setRange(0.0, 10000.0)
        self.sp_raw_hp.setValue(300.0)
        self.sp_raw_hp.setSuffix(" Hz")
        self.sp_raw_lp = QtWidgets.QDoubleSpinBox()
        self.sp_raw_lp.setRange(0.0, 15000.0)
        self.sp_raw_lp.setValue(0.0)
        self.sp_raw_lp.setSuffix(" Hz")
        self.sp_raw_ds = QtWidgets.QSpinBox()
        self.sp_raw_ds.setRange(1, 50)
        self.sp_raw_ds.setValue(1)
        self.ck_raw_overlay = QtWidgets.QCheckBox("Overlay selected units")
        self.ck_raw_overlay.setChecked(True)
        self.cb_raw_y = QtWidgets.QComboBox()
        self.cb_raw_y.addItems(["Channel ID", "Depth (mm)"])
        f_raw.addRow("Start", with_help(self.sp_raw_t0, "Raw explorer start time (s)."))
        f_raw.addRow("Duration", with_help(self.sp_raw_dur, "Raw explorer window duration (s)."))
        f_raw.addRow("Channels around unit", with_help(self.sp_raw_ch, "Number of channels to display around the selected unit's best channel."))
        f_raw.addRow("High-pass", with_help(self.sp_raw_hp, "High-pass filter cutoff (Hz)."))
        f_raw.addRow("Low-pass (0=off)", with_help(self.sp_raw_lp, "Low-pass cutoff (Hz); 0 disables low-pass."))
        f_raw.addRow("Downsample", with_help(self.sp_raw_ds, "Downsampling factor for plotting speed."))
        f_raw.addRow("Y axis", with_help(self.cb_raw_y, "Y-axis mode: channel index or depth in mm."))
        f_raw.addRow(self.ck_raw_overlay)
        self.analysis_tabs.addTab(t_raw, "Raw Explorer")

        t_corr = QtWidgets.QWidget()
        f_corr = QtWidgets.QFormLayout(t_corr)
        self.cb_corr_mode = QtWidgets.QComboBox()
        self.cb_corr_mode.addItems(["ACG", "CCG"])
        self.sp_corr_bin = QtWidgets.QDoubleSpinBox()
        self.sp_corr_bin.setRange(0.1, 20.0)
        self.sp_corr_bin.setValue(1.0)
        self.sp_corr_bin.setSuffix(" ms")
        self.sp_corr_win = QtWidgets.QDoubleSpinBox()
        self.sp_corr_win.setRange(10.0, 1000.0)
        self.sp_corr_win.setValue(100.0)
        self.sp_corr_win.setSuffix(" ms")
        f_corr.addRow("Mode", with_help(self.cb_corr_mode, "ACG for auto-correlogram, CCG for cross-correlogram."))
        f_corr.addRow("Bin", with_help(self.sp_corr_bin, "Correlogram bin size (ms)."))
        f_corr.addRow("Window", with_help(self.sp_corr_win, "Half-window around zero lag (ms)."))
        self.analysis_tabs.addTab(t_corr, "Correlogram")
        t_psth = QtWidgets.QWidget()
        v_psth = QtWidgets.QVBoxLayout(t_psth)
        self.lbl_psth_hint = QtWidgets.QLabel(
            "Single unit: the heatmap shows one row per trial and the PSTH shows mean \u00b1 SEM across the selected trials. "
            "Multiple units: the heatmap shows one row per unit (trial-averaged) and the PSTH shows the mean across units."
        )
        self.lbl_psth_hint.setWordWrap(True)
        self.lbl_psth_hint.setObjectName("psthHintLabel")
        self.tbl_conditions = QtWidgets.QTableWidget(0, 3)
        self.tbl_conditions.setHorizontalHeaderLabels(["Condition", "Event label", "Events CSV"])
        self.tbl_conditions.setColumnWidth(0, 160)
        self.tbl_conditions.setColumnWidth(1, 180)
        self.tbl_conditions.horizontalHeader().setStretchLastSection(True)
        self.tbl_conditions.verticalHeader().setVisible(False)
        b_cond = QtWidgets.QHBoxLayout()
        self.btn_cond_add = QtWidgets.QPushButton("Add condition")
        self.btn_cond_remove = QtWidgets.QPushButton("Remove condition")
        self.btn_cond_browse = QtWidgets.QPushButton("Browse CSV for selected")
        b_cond.addWidget(self.btn_cond_add)
        b_cond.addWidget(self.btn_cond_remove)
        b_cond.addWidget(self.btn_cond_browse)
        b_cond.addStretch(1)

        f_psth = QtWidgets.QFormLayout()
        self.sp_psth_pre = QtWidgets.QDoubleSpinBox()
        self.sp_psth_pre.setRange(0.05, 20.0)
        self.sp_psth_pre.setValue(1.0)
        self.sp_psth_pre.setSuffix(" s")
        self.sp_psth_post = QtWidgets.QDoubleSpinBox()
        self.sp_psth_post.setRange(0.05, 20.0)
        self.sp_psth_post.setValue(2.0)
        self.sp_psth_post.setSuffix(" s")
        self.sp_psth_bin = QtWidgets.QDoubleSpinBox()
        self.sp_psth_bin.setRange(0.5, 50.0)
        self.sp_psth_bin.setValue(5.0)
        self.sp_psth_bin.setSuffix(" ms")
        self.sp_psth_trial_from = QtWidgets.QSpinBox()
        self.sp_psth_trial_from.setRange(1, 1_000_000)
        self.sp_psth_trial_from.setValue(1)
        self.sp_psth_trial_to = QtWidgets.QSpinBox()
        self.sp_psth_trial_to.setRange(0, 1_000_000)
        self.sp_psth_trial_to.setSpecialValueText("last")
        self.sp_psth_trial_to.setValue(0)
        self.btn_psth_all_trials = QtWidgets.QPushButton("All trials")
        self.btn_psth_all_trials.setProperty("role", "ghost")
        trial_row = QtWidgets.QWidget()
        trial_row_l = QtWidgets.QHBoxLayout(trial_row)
        trial_row_l.setContentsMargins(0, 0, 0, 0)
        trial_row_l.setSpacing(8)
        trial_row_l.addWidget(QtWidgets.QLabel("from"))
        trial_row_l.addWidget(self.sp_psth_trial_from)
        trial_row_l.addWidget(QtWidgets.QLabel("to"))
        trial_row_l.addWidget(self.sp_psth_trial_to)
        trial_row_l.addWidget(self.btn_psth_all_trials)
        trial_row_l.addStretch(1)
        self.lbl_psth_trial_status = QtWidgets.QLabel("Using all matching trials in each condition.")
        self.lbl_psth_trial_status.setWordWrap(True)
        self.lbl_psth_trial_status.setObjectName("psthMetaLabel")
        self.btn_psth_compute = QtWidgets.QPushButton("Compute")
        self.btn_psth_show = QtWidgets.QPushButton("Show")
        self.btn_psth_compute.setProperty("role", "primary")
        self.btn_psth_show.setProperty("role", "secondary")
        psth_btn_row = QtWidgets.QHBoxLayout()
        psth_btn_row.addWidget(self.btn_psth_compute)
        psth_btn_row.addWidget(self.btn_psth_show)
        f_psth.addRow("Pre window", with_help(self.sp_psth_pre, "Seconds before event for PSTH window."))
        f_psth.addRow("Post window", with_help(self.sp_psth_post, "Seconds after event for PSTH window."))
        f_psth.addRow("Bin", with_help(self.sp_psth_bin, "PSTH bin size (ms)."))
        f_psth.addRow("Trial range", with_help(trial_row, "1-based inclusive trial range within each selected event label. 'last' uses the final available trial."))
        f_psth.addRow(psth_btn_row)

        v_psth.addWidget(self.lbl_psth_hint)
        v_psth.addWidget(self.tbl_conditions)
        v_psth.addLayout(b_cond)
        v_psth.addLayout(f_psth)
        v_psth.addWidget(self.lbl_psth_trial_status)
        self.analysis_tabs.addTab(t_psth, "Condition PSTH")

        t_net = QtWidgets.QWidget()
        f_net = QtWidgets.QFormLayout(t_net)
        self.sp_net_bin = QtWidgets.QDoubleSpinBox()
        self.sp_net_bin.setRange(0.5, 20.0)
        self.sp_net_bin.setValue(1.0)
        self.sp_net_bin.setSuffix(" ms")
        self.sp_net_win = QtWidgets.QDoubleSpinBox()
        self.sp_net_win.setRange(10.0, 1000.0)
        self.sp_net_win.setValue(100.0)
        self.sp_net_win.setSuffix(" ms")
        self.sp_sync_bin = QtWidgets.QDoubleSpinBox()
        self.sp_sync_bin.setRange(1.0, 200.0)
        self.sp_sync_bin.setValue(10.0)
        self.sp_sync_bin.setSuffix(" ms")
        self.sp_sync_win = QtWidgets.QDoubleSpinBox()
        self.sp_sync_win.setRange(0.2, 60.0)
        self.sp_sync_win.setValue(2.0)
        self.sp_sync_win.setSuffix(" s")
        self.sp_sync_step = QtWidgets.QDoubleSpinBox()
        self.sp_sync_step.setRange(0.05, 20.0)
        self.sp_sync_step.setValue(0.5)
        self.sp_sync_step.setSuffix(" s")
        self.btn_net_compute = QtWidgets.QPushButton("Compute")
        self.btn_net_show = QtWidgets.QPushButton("Show")
        net_btn_row = QtWidgets.QHBoxLayout()
        net_btn_row.addWidget(self.btn_net_compute)
        net_btn_row.addWidget(self.btn_net_show)
        f_net.addRow("CCG bin", with_help(self.sp_net_bin, "Bin size (ms) for pairwise CCG matrix."))
        f_net.addRow("CCG window", with_help(self.sp_net_win, "Window (ms) for pairwise CCG matrix."))
        f_net.addRow("Synchrony bin", with_help(self.sp_sync_bin, "Bin size (ms) for synchrony index."))
        f_net.addRow("Synchrony window", with_help(self.sp_sync_win, "Window length (s) for synchrony index."))
        f_net.addRow("Synchrony step", with_help(self.sp_sync_step, "Step size (s) for synchrony index over time."))
        f_net.addRow(net_btn_row)
        self.analysis_tabs.addTab(t_net, "Network")

        t_npyx = QtWidgets.QWidget()
        f_npyx = QtWidgets.QFormLayout(t_npyx)
        self.cb_npyx_method = QtWidgets.QComboBox()
        self._npyx_methods = method_options()
        for key, label in self._npyx_methods:
            self.cb_npyx_method.addItem(label, userData=key)
        self.sp_npyx_bin = QtWidgets.QDoubleSpinBox()
        self.sp_npyx_bin.setRange(0.1, 20.0)
        self.sp_npyx_bin.setValue(0.5)
        self.sp_npyx_bin.setSuffix(" ms")
        self.sp_npyx_win = QtWidgets.QDoubleSpinBox()
        self.sp_npyx_win.setRange(10.0, 1000.0)
        self.sp_npyx_win.setValue(100.0)
        self.sp_npyx_win.setSuffix(" ms")
        self.tbl_npyx_params = QtWidgets.QTableWidget(0, 2)
        self.tbl_npyx_params.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.tbl_npyx_params.horizontalHeader().setStretchLastSection(True)
        self.tbl_npyx_params.verticalHeader().setVisible(False)
        self.tbl_npyx_params.setMinimumHeight(120)
        self.txt_npyx_desc = QtWidgets.QPlainTextEdit()
        self.txt_npyx_desc.setReadOnly(True)
        self.txt_npyx_desc.setMinimumHeight(90)
        self.txt_npyx_desc.setMaximumHeight(140)
        f_npyx.addRow("Method", with_help(self.cb_npyx_method, "Advanced correlation analysis methods (clear names)."))
        f_npyx.addRow("Bin", with_help(self.sp_npyx_bin, "Bin size (ms)."))
        f_npyx.addRow("Window", with_help(self.sp_npyx_win, "Window size (ms)."))
        f_npyx.addRow("Function parameters", self.tbl_npyx_params)
        f_npyx.addRow("Description", self.txt_npyx_desc)
        self.analysis_tabs.addTab(t_npyx, "Advanced Corr")

        self.page_progress = QtWidgets.QProgressBar()
        self.page_progress.setRange(0, 100)
        self.page_progress.setValue(0)

        units_col_l.addWidget(grp_units, 1)
        controls_col_l.addWidget(grp_menu, 1)
        controls_col_l.addWidget(self.page_progress, 0)
        left_split.addWidget(units_col)
        left_split.addWidget(controls_col)
        left_split.setStretchFactor(0, 3)
        left_split.setStretchFactor(1, 4)

        right = QtWidgets.QWidget()
        self._right_panel = right
        right_l = QtWidgets.QVBoxLayout(right)
        self.view_tabs = QtWidgets.QTabWidget()

        basics_container = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(basics_container)
        self.plot_basic_spikes = pg.PlotWidget(title="Unit raster + instantaneous firing rate")
        self.plot_basic_acg = pg.PlotWidget(title="Auto-correlogram")
        self.plot_basic_isi = pg.PlotWidget(title="ISI")
        self.gl_basic_wvf = pg.GraphicsLayoutWidget()
        basic_row2 = QtWidgets.QWidget()
        basic_row2_l = QtWidgets.QHBoxLayout(basic_row2)
        basic_row2_l.setContentsMargins(0, 0, 0, 0)
        self._basic_row2_layout = basic_row2_l
        basic_row2_l.addWidget(self.plot_basic_acg, 2)
        basic_row2_l.addWidget(self.gl_basic_wvf, 2)
        basic_row2_l.addWidget(self.plot_basic_isi, 1)
        bl.addWidget(self.plot_basic_spikes, 2)
        bl.addWidget(basic_row2, 2)

        self.plot_raw = pg.PlotWidget(title="Raw explorer")

        corr_view = QtWidgets.QWidget()
        corr_v = QtWidgets.QVBoxLayout(corr_view)
        self.gl_corr = pg.GraphicsLayoutWidget()
        corr_v.addWidget(self.gl_corr, 1)

        psth_view = QtWidgets.QWidget()
        psth_v = QtWidgets.QVBoxLayout(psth_view)
        self.psth_summary_card = QtWidgets.QFrame()
        self.psth_summary_card.setObjectName("psthSummaryCard")
        psth_summary_l = QtWidgets.QVBoxLayout(self.psth_summary_card)
        psth_summary_l.setContentsMargins(14, 12, 14, 12)
        psth_summary_l.setSpacing(4)
        self.lbl_psth_summary = QtWidgets.QLabel("Condition PSTH is ready after you compute it.")
        self.lbl_psth_summary.setWordWrap(True)
        self.lbl_psth_summary.setObjectName("psthSummaryTitle")
        self.lbl_psth_summary_meta = QtWidgets.QLabel(
            "Select units, choose an event label, then use Compute. Trial-range changes are applied on the displayed heatmap and averages."
        )
        self.lbl_psth_summary_meta.setWordWrap(True)
        self.lbl_psth_summary_meta.setObjectName("psthSummaryMeta")
        psth_summary_l.addWidget(self.lbl_psth_summary)
        psth_summary_l.addWidget(self.lbl_psth_summary_meta)
        self.plot_psth_lines = pg.PlotWidget(title="Condition PSTH lines")
        self.plot_psth_heat = pg.PlotWidget(title="Condition PSTH heatmap")
        self.psth_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.psth_splitter.addWidget(self.plot_psth_lines)
        self.psth_splitter.addWidget(self.plot_psth_heat)
        self.psth_splitter.setStretchFactor(0, 3)
        self.psth_splitter.setStretchFactor(1, 4)
        psth_v.addWidget(self.psth_summary_card, 0)
        psth_v.addWidget(self.psth_splitter, 1)

        net_view = QtWidgets.QWidget()
        net_v = QtWidgets.QVBoxLayout(net_view)
        self.plot_net_matrix = pg.PlotWidget(title="Pairwise CCG matrix")
        self.plot_net_sync = pg.PlotWidget(title="Synchrony index over time")
        net_v.addWidget(self.plot_net_matrix, 1)
        net_v.addWidget(self.plot_net_sync, 1)

        npyx_view = QtWidgets.QWidget()
        npyx_v = QtWidgets.QVBoxLayout(npyx_view)
        self.gl_npyx = pg.GraphicsLayoutWidget()
        npyx_v.addWidget(self.gl_npyx, 1)

        self.view_tabs.addTab(basics_container, "Unit Basics")
        self.view_tabs.addTab(self.plot_raw, "Raw Explorer")
        self.view_tabs.addTab(corr_view, "Correlogram")
        self.view_tabs.addTab(psth_view, "Condition PSTH")
        self.view_tabs.addTab(net_view, "Network")
        self.view_tabs.addTab(npyx_view, "Advanced Corr")

        right_l.addWidget(self.view_tabs, 1)
        self._right_panel_layout = right_l

        body.addWidget(left_split)
        body.addWidget(right)
        body.setStretchFactor(0, 3)
        body.setStretchFactor(1, 7)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setProperty("logView", True)
        self.log.setPlaceholderText("Dataset load and analysis output will appear here.")
        self.log.setMinimumHeight(90)
        self.log.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        main.addLayout(top)
        self.vertical_split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.vertical_split.addWidget(body)
        self.vertical_split.addWidget(self.log)
        self.vertical_split.setStretchFactor(0, 8)
        self.vertical_split.setStretchFactor(1, 2)
        main.addWidget(self.vertical_split, 1)

        self.btn_browse.clicked.connect(self._pick)
        self.btn_load.clicked.connect(self._load_dataset)
        self.btn_export.clicked.connect(self._export_plotted_data)
        self.btn_export_units.clicked.connect(self._export_units_file)
        self.btn_detach_plots.toggled.connect(self._toggle_plot_detach)
        self.list_units.itemSelectionChanged.connect(self._on_units_selection_changed)
        self.ed_unit_filter.textChanged.connect(self._refresh_units_list)
        self.btn_good_only.toggled.connect(self._refresh_units_list)
        self.cb_good_source.currentTextChanged.connect(self._on_good_source_changed)
        self.analysis_tabs.currentChanged.connect(self._on_analysis_page_changed)
        self.cb_corr_mode.currentTextChanged.connect(self._refresh_current_page)
        self.btn_cond_add.clicked.connect(self._add_condition_row)
        self.btn_cond_remove.clicked.connect(self._remove_condition_row)
        self.btn_cond_browse.clicked.connect(self._browse_condition_csv)
        self.sp_psth_trial_from.valueChanged.connect(self._on_psth_trial_range_changed)
        self.sp_psth_trial_to.valueChanged.connect(self._on_psth_trial_range_changed)
        self.btn_psth_all_trials.clicked.connect(self._reset_psth_trial_range)
        self.btn_psth_compute.clicked.connect(self._compute_psth)
        self.btn_psth_show.clicked.connect(self._show_psth)
        self.btn_net_compute.clicked.connect(self._compute_network)
        self.btn_net_show.clicked.connect(self._show_network)
        self.cb_npyx_method.currentTextChanged.connect(self._refresh_current_page)
        self.cb_npyx_method.currentIndexChanged.connect(self._update_npyx_method_ui)
        self.tbl_npyx_params.itemChanged.connect(self._on_npyx_params_changed)

        auto_widgets = [
            self.sp_basic_t0, self.sp_basic_dur, self.sp_isi_max, self.sp_basic_acg_bin, self.sp_basic_acg_win,
            self.sp_basic_acg_ratio, self.sp_basic_isi_ratio, self.ck_ifr, self.sp_ifr_smooth_ms,
            self.sp_raw_t0, self.sp_raw_dur, self.sp_raw_ch, self.sp_raw_hp, self.sp_raw_lp, self.sp_raw_ds,
            self.ck_raw_overlay, self.cb_raw_y, self.sp_corr_bin, self.sp_corr_win, self.sp_psth_pre,
            self.sp_psth_post, self.sp_psth_bin, self.sp_net_bin, self.sp_net_win, self.sp_sync_bin,
            self.sp_sync_win, self.sp_sync_step, self.sp_npyx_bin, self.sp_npyx_win,
        ]
        for w in auto_widgets:
            if hasattr(w, "valueChanged"):
                w.valueChanged.connect(self._refresh_current_page)
            elif hasattr(w, "currentTextChanged"):
                w.currentTextChanged.connect(self._refresh_current_page)
            elif hasattr(w, "toggled"):
                w.toggled.connect(self._refresh_current_page)
        self.sp_basic_acg_ratio.valueChanged.connect(self._update_basic_plot_ratio)
        self.sp_basic_isi_ratio.valueChanged.connect(self._update_basic_plot_ratio)

        self.tbl_conditions.itemChanged.connect(self._on_conditions_changed)
        self._apply_plot_style()
        self._update_basic_plot_ratio()
        self._update_psth_trial_status()
        self._update_npyx_method_ui()
    def set_plot_preferences(self, theme: str, show_grid: bool) -> None:
        self._plot_theme = "Dark" if str(theme).lower().startswith("dark") else "Light"
        self._show_grid = bool(show_grid)
        self._apply_plot_style()

    def _apply_plot_style(self) -> None:
        bg = "#0b0f14" if self._plot_theme == "Dark" else "#ffffff"
        fg = "#e8eef7" if self._plot_theme == "Dark" else "#1a1f29"
        grid_alpha = 0.25 if self._show_grid else 0.0
        card_bg = "rgba(90, 128, 255, 0.14)" if self._plot_theme == "Dark" else "rgba(67, 128, 255, 0.08)"
        card_border = "rgba(142, 170, 255, 0.34)" if self._plot_theme == "Dark" else "rgba(67, 128, 255, 0.22)"
        meta_fg = "#aab6ca" if self._plot_theme == "Dark" else "#5b6778"
        plots = [
            self.plot_basic_spikes, self.plot_basic_acg, self.plot_basic_isi, self.plot_raw,
            self.plot_psth_lines, self.plot_psth_heat,
            self.plot_net_matrix, self.plot_net_sync,
        ]
        for p in plots:
            p.setBackground(bg)
            p.getAxis("left").setTextPen(pg.mkPen(fg))
            p.getAxis("bottom").setTextPen(pg.mkPen(fg))
            p.getAxis("left").setPen(pg.mkPen(fg))
            p.getAxis("bottom").setPen(pg.mkPen(fg))
            p.showGrid(x=self._show_grid, y=self._show_grid, alpha=grid_alpha)
        self.gl_basic_wvf.setBackground(bg)
        self.gl_corr.setBackground(bg)
        self.gl_npyx.setBackground(bg)
        self.psth_summary_card.setStyleSheet(
            "QFrame#psthSummaryCard {"
            f"background: {card_bg};"
            f"border: 1px solid {card_border};"
            "border-radius: 14px;"
            "}"
            "QLabel#psthSummaryTitle {"
            f"color: {fg};"
            "font-size: 13px;"
            "font-weight: 700;"
            "}"
            "QLabel#psthSummaryMeta {"
            f"color: {meta_fg};"
            "font-size: 11px;"
            "}"
        )
        self.lbl_psth_hint.setStyleSheet(f"color: {meta_fg}; font-size: 11px;")
        self.lbl_psth_trial_status.setStyleSheet(f"color: {meta_fg}; font-size: 11px;")

    def _subplot_shape(self, n: int) -> tuple[int, int]:
        if n <= 1:
            return 1, 1
        cols = int(math.ceil(math.sqrt(float(n))))
        rows = int(math.ceil(float(n) / max(cols, 1)))
        return rows, cols

    def _style_plot_item(self, plot: pg.PlotItem, left: str = "", bottom: str = "") -> None:
        fg = "#e8eef7" if self._plot_theme == "Dark" else "#1a1f29"
        grid_alpha = 0.25 if self._show_grid else 0.0
        plot.showGrid(x=self._show_grid, y=self._show_grid, alpha=grid_alpha)
        plot.getAxis("left").setTextPen(pg.mkPen(fg))
        plot.getAxis("bottom").setTextPen(pg.mkPen(fg))
        plot.getAxis("left").setPen(pg.mkPen(fg))
        plot.getAxis("bottom").setPen(pg.mkPen(fg))
        if left:
            plot.setLabel("left", left)
        if bottom:
            plot.setLabel("bottom", bottom)

    def _psth_palette(self, count: int) -> list[tuple[int, int, int]]:
        if self._plot_theme == "Dark":
            base = [
                (107, 174, 255),
                (255, 122, 156),
                (104, 218, 193),
                (255, 210, 112),
                (182, 148, 255),
                (255, 164, 103),
            ]
        else:
            base = [
                (45, 128, 216),
                (222, 78, 121),
                (19, 165, 137),
                (220, 162, 47),
                (126, 95, 228),
                (230, 126, 34),
            ]
        if count <= len(base):
            return base[:count]
        return [base[i % len(base)] for i in range(count)]

    def _update_psth_trial_status(self) -> None:
        start = int(self.sp_psth_trial_from.value())
        stop = int(self.sp_psth_trial_to.value())
        if stop <= 0:
            text = f"Trial filter: using trials {start}\u2013last within each condition."
        else:
            lo, hi = sorted((start, stop))
            text = f"Trial filter: using trials {lo}\u2013{hi} within each condition."
        self.lbl_psth_trial_status.setText(text)

    def _reset_psth_trial_range(self) -> None:
        blockers = [QtCore.QSignalBlocker(self.sp_psth_trial_from), QtCore.QSignalBlocker(self.sp_psth_trial_to)]
        self.sp_psth_trial_from.setValue(1)
        self.sp_psth_trial_to.setValue(0)
        del blockers
        self._on_psth_trial_range_changed()

    def _on_psth_trial_range_changed(self) -> None:
        self._update_psth_trial_status()
        if self.analysis_tabs.currentIndex() == 3 and "psth" in self.results:
            self._show_psth()

    def _condition_trial_slice(self, total_trials: int) -> tuple[slice, dict]:
        total = max(0, int(total_trials))
        requested_start = max(1, int(self.sp_psth_trial_from.value()))
        requested_stop_raw = int(self.sp_psth_trial_to.value())
        if total == 0:
            return slice(0, 0), {
                "total_trials": 0,
                "requested_start": requested_start,
                "requested_stop": None if requested_stop_raw <= 0 else requested_stop_raw,
                "actual_start": 0,
                "actual_stop": 0,
                "used_trials": 0,
            }
        stop_value = total if requested_stop_raw <= 0 else max(1, requested_stop_raw)
        start_value = requested_start
        if stop_value < start_value:
            start_value, stop_value = stop_value, start_value
        start_value = min(start_value, total)
        stop_value = min(stop_value, total)
        used = max(0, stop_value - start_value + 1)
        return slice(start_value - 1, stop_value), {
            "total_trials": total,
            "requested_start": requested_start,
            "requested_stop": None if requested_stop_raw <= 0 else requested_stop_raw,
            "actual_start": start_value,
            "actual_stop": stop_value,
            "used_trials": used,
        }

    def _best_channel_index(self, unit: int, waveform: Optional[np.ndarray] = None) -> Optional[int]:
        if self.dataset is None:
            return None
        wvf = waveform if waveform is not None else self.dataset.mean_template_waveform(unit)
        if wvf is None or wvf.ndim != 2 or wvf.shape[1] == 0:
            return None
        peaks = np.nanmax(np.abs(wvf), axis=0)
        if peaks.size == 0 or not np.any(np.isfinite(peaks)):
            return None
        return int(np.nanargmax(peaks))

    def _waveform_support_indices(self, waveform: Optional[np.ndarray], limit: int = 24) -> np.ndarray:
        if waveform is None or waveform.ndim != 2 or waveform.shape[1] == 0:
            return np.array([], dtype=int)
        peaks = np.nanmax(np.abs(waveform), axis=0)
        peaks = np.nan_to_num(peaks, nan=0.0, posinf=0.0, neginf=0.0)
        if peaks.size == 0 or float(np.max(peaks)) <= 0.0:
            return np.array([], dtype=int)
        max_keep = min(max(1, int(limit)), int(peaks.size))
        min_keep = min(max_keep, max(10, min(24, int(peaks.size))))
        support = np.flatnonzero(peaks >= (0.1 * float(np.max(peaks))))
        if support.size < min_keep:
            support = np.argsort(peaks)[-min_keep:]
        if support.size > max_keep:
            strong = np.argsort(peaks[support])[::-1][:max_keep]
            support = support[strong]
        return np.sort(support.astype(int))

    def _nice_scale_value(self, value: float) -> float:
        value = float(abs(value))
        if value <= 0.0 or not np.isfinite(value):
            return 1.0
        exponent = math.floor(math.log10(value))
        fraction = value / (10 ** exponent)
        if fraction < 1.5:
            nice = 1.0
        elif fraction < 3.5:
            nice = 2.0
        elif fraction < 7.5:
            nice = 5.0
        else:
            nice = 10.0
        return nice * (10 ** exponent)

    def _add_scale_bar(
        self,
        plot: pg.PlotItem,
        x0: float,
        y0: float,
        dx: float,
        dy: float,
        x_label: str,
        y_label: str,
    ) -> None:
        fg = "#e8eef7" if self._plot_theme == "Dark" else "#1a1f29"
        pen = pg.mkPen(fg, width=2)
        plot.plot([x0, x0 + dx], [y0, y0], pen=pen)
        plot.plot([x0, x0], [y0, y0 + dy], pen=pen)
        x_margin = 0.10 * abs(dx if dx != 0 else 1.0)
        y_margin = 0.12 * abs(dy if dy != 0 else 1.0)
        x_text = pg.TextItem(text=x_label, color=fg, anchor=(0.5, 0.0))
        x_text.setPos(x0 + 0.5 * dx, y0 - y_margin)
        plot.addItem(x_text)
        y_text = pg.TextItem(text=y_label, color=fg, anchor=(0.0, 1.0))
        y_text.setPos(x0 + x_margin, y0 + dy + 0.35 * y_margin)
        plot.addItem(y_text)

    def _render_multichannel_waveform(self, unit: int, waveform: Optional[np.ndarray]) -> list[dict]:
        self.gl_basic_wvf.clear()
        if waveform is None or waveform.ndim != 2 or waveform.shape[1] == 0:
            return []
        fg = "#e8eef7" if self._plot_theme == "Dark" else "#1a1f29"
        self.gl_basic_wvf.ci.layout.setHorizontalSpacing(18)
        self.gl_basic_wvf.ci.layout.setVerticalSpacing(16)

        support = self._waveform_support_indices(waveform, limit=4)
        if support.size == 0:
            best = self._best_channel_index(unit, waveform)
            if best is None:
                return []
            support = np.array([best], dtype=int)
        peaks = np.nanmax(np.abs(waveform), axis=0)
        peaks = np.nan_to_num(peaks, nan=0.0, posinf=0.0, neginf=0.0)
        best = self._best_channel_index(unit, waveform)
        if best is None:
            best = int(support[np.argmax(peaks[support])])
        anchor_idx = int(np.nanargmax(np.abs(waveform[:, best])))
        t_ms = (np.arange(waveform.shape[0], dtype=float) - float(anchor_idx)) / float(self.dataset.sample_rate) * 1000.0
        amp_peak = max(float(np.nanmax(np.abs(waveform[:, support]))), 1.0)

        positions = None
        if self.dataset is not None and self.dataset.channel_positions is not None:
            pos = np.asarray(self.dataset.channel_positions, dtype=float)
            if pos.ndim == 2 and pos.shape[0] > int(np.max(support)) and pos.shape[1] >= 2:
                positions = pos

        if positions is not None:
            order = np.lexsort((positions[support, 0], positions[support, 1]))
            support = support[order]
        else:
            support = support[np.argsort(peaks[support])[::-1]]

        waveform_rows: list[dict] = []
        base_color = (91, 155, 255) if self._plot_theme == "Dark" else (67, 128, 255)
        glow_color = (91, 155, 255, 80) if self._plot_theme == "Dark" else (67, 128, 255, 70)
        y_lim = 1.18 * amp_peak
        panel_y_min = -1.34 * y_lim
        x_min = float(t_ms.min())
        x_max = float(t_ms.max())
        rows = 2 if support.size > 2 else 1
        cols = 2 if support.size > 1 else 1
        first_plot: Optional[pg.PlotItem] = None
        for idx, ch in enumerate(support):
            trace = np.asarray(waveform[:, int(ch)], dtype=float)
            channel_id = int(ch)
            if self.dataset is not None and self.dataset.channel_map is not None and self.dataset.channel_map.size > int(ch):
                channel_id = int(np.asarray(self.dataset.channel_map).squeeze()[int(ch)])
            is_best = int(ch) == int(best)
            plot = self.gl_basic_wvf.addPlot(row=idx // cols, col=idx % cols)
            if first_plot is None:
                first_plot = plot
            title = f"Unit {unit} | ch {channel_id}" if idx == 0 else f"ch {channel_id}"
            if idx == 0:
                plot.setTitle(f"Waveform | {title}", color=fg)
            else:
                plot.setTitle(str(channel_id), color=fg)
            plot.showGrid(x=False, y=False, alpha=0.0)
            plot.hideAxis("left")
            plot.hideAxis("bottom")
            plot.hideButtons()
            plot.setMouseEnabled(x=False, y=False)
            plot.setXRange(x_min, x_max, padding=0.0)
            plot.setYRange(panel_y_min, y_lim, padding=0.0)
            plot.plot([x_min, x_max], [0.0, 0.0], pen=pg.mkPen((160, 176, 198, 50), width=1))
            plot.plot(t_ms, trace, pen=pg.mkPen(glow_color, width=9 if is_best else 7))
            plot.plot(t_ms, trace, pen=pg.mkPen(base_color, width=2.8 if is_best else 2.0))
            waveform_rows.extend(
                [
                    {
                        "unit_id": int(unit),
                        "channel_order": int(ch),
                        "channel_id": channel_id,
                        "time_ms": float(tm),
                        "amplitude_uv": float(tv),
                        "subplot_row": int(idx // cols),
                        "subplot_col": int(idx % cols),
                    }
                    for tm, tv in zip(t_ms, trace)
                ]
            )

        bar_ms = self._nice_scale_value(max(0.25, 0.24 * (x_max - x_min)))
        bar_uv = self._nice_scale_value(max(20.0, 0.18 * amp_peak))
        scale_plot = self.gl_basic_wvf.ci.getItem(rows - 1, 0)
        if scale_plot is None:
            scale_plot = first_plot
        x0 = x_min + 0.06 * (x_max - x_min)
        y0 = panel_y_min + 0.08 * (y_lim - panel_y_min)
        if scale_plot is not None:
            self._add_scale_bar(
                scale_plot,
                x0=x0,
                y0=y0,
                dx=bar_ms,
                dy=bar_uv,
                x_label=f"{bar_ms:g} ms",
                y_label=f"{bar_uv:g} uV",
            )
        return waveform_rows

    def _annotate_image_values(self, plot: pg.PlotItem, mat: np.ndarray) -> None:
        if mat.ndim != 2 or mat.size == 0:
            return
        rows, cols = mat.shape
        if rows * cols > 2500:
            return
        fg = (235, 235, 235) if self._plot_theme == "Dark" else (25, 25, 25)
        for i in range(rows):
            for j in range(cols):
                v = mat[i, j]
                if not np.isfinite(v):
                    continue
                t = pg.TextItem(text=f"{v:.2f}", anchor=(0.5, 0.5), color=fg)
                t.setPos(float(j) + 0.5, float(i) + 0.5)
                plot.addItem(t)

    def _update_npyx_method_ui(self) -> None:
        key = self.cb_npyx_method.currentData()
        if not key:
            return
        meta = method_metadata(str(key))
        desc = str(meta.get("description", ""))
        if str(key) in PAIRWISE_ONLY_METHODS:
            desc = f"{desc}\n\nRequires at least two distinct selected units.".strip()
        self.txt_npyx_desc.setPlainText(desc)
        params = meta.get("params", {})
        if not isinstance(params, dict):
            params = {}
        self.tbl_npyx_params.blockSignals(True)
        self.tbl_npyx_params.setRowCount(0)
        for k, v in params.items():
            r = self.tbl_npyx_params.rowCount()
            self.tbl_npyx_params.insertRow(r)
            self.tbl_npyx_params.setItem(r, 0, QtWidgets.QTableWidgetItem(str(k)))
            self.tbl_npyx_params.setItem(r, 1, QtWidgets.QTableWidgetItem(str(v)))
        self.tbl_npyx_params.blockSignals(False)

    def _collect_npyx_params(self) -> Dict[str, object]:
        out: Dict[str, object] = {}
        for r in range(self.tbl_npyx_params.rowCount()):
            k_item = self.tbl_npyx_params.item(r, 0)
            v_item = self.tbl_npyx_params.item(r, 1)
            if k_item is None or v_item is None:
                continue
            key = str(k_item.text()).strip()
            if not key:
                continue
            raw = str(v_item.text()).strip()
            if raw.lower() in {"true", "false"}:
                out[key] = raw.lower() == "true"
                continue
            try:
                if any(c in raw for c in [".", "e", "E"]):
                    out[key] = float(raw)
                else:
                    out[key] = int(raw)
                continue
            except Exception:
                out[key] = raw
        return out

    def _on_npyx_params_changed(self, _item: QtWidgets.QTableWidgetItem) -> None:
        if self.analysis_tabs.currentIndex() == 5:
            self._refresh_current_page()

    def _update_basic_plot_ratio(self) -> None:
        if self._basic_row2_layout is None:
            return
        acg_r = max(1, int(self.sp_basic_acg_ratio.value()))
        isi_r = max(1, int(self.sp_basic_isi_ratio.value()))
        waveform_r = max(2, int(round((acg_r + isi_r) / 2.0)))
        self._basic_row2_layout.setStretch(0, acg_r)
        self._basic_row2_layout.setStretch(1, waveform_r)
        self._basic_row2_layout.setStretch(2, isi_r)

    def _recording_duration_s(self) -> float:
        if self.dataset is None:
            return 0.0
        d = 0.0
        try:
            if self.dataset.ap_bin_path is not None and self.dataset.ap_bin_path.exists():
                n_samples = (self.dataset.ap_bin_path.stat().st_size // 2) // max(int(self.dataset.n_channels), 1)
                d = max(d, float(n_samples) / float(self.dataset.sample_rate))
        except Exception:
            pass
        try:
            if self.dataset.spike_times.size:
                d = max(d, float(np.max(self.dataset.spike_times)) / float(self.dataset.sample_rate))
        except Exception:
            pass
        return max(d, 0.0)

    def _update_basic_time_bounds(self) -> None:
        dur = self._recording_duration_s()
        if dur <= 0.0:
            return
        t0_max = max(0.0, dur - 0.05)
        self.sp_basic_t0.setRange(0.0, t0_max)
        self.sp_basic_dur.setRange(0.05, dur)
        self.sp_basic_t0.setValue(min(float(self.sp_basic_t0.value()), t0_max))
        self.sp_basic_dur.setValue(min(float(self.sp_basic_dur.value()), dur))

    def _toggle_plot_detach(self, checked: bool) -> None:
        if checked:
            self._detach_plots()
        else:
            self._attach_plots()

    def _detach_plots(self) -> None:
        if self._plot_detached or self._right_panel_layout is None:
            return
        if self._body_splitter is not None:
            self._body_sizes_before_detach = self._body_splitter.sizes()
        self._right_panel_layout.removeWidget(self.view_tabs)
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Post Processing plots")
        dlg.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        dlg.resize(1200, 780)
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(self.view_tabs)
        dlg.finished.connect(lambda _=0: self._attach_plots())
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        self._plot_dialog = dlg
        self._plot_detached = True
        if self._right_panel is not None:
            self._right_panel.hide()
        if self._body_splitter is not None:
            self._body_splitter.setSizes([1, 0])
        self.btn_detach_plots.setText("Attach plots")

    def _attach_plots(self) -> None:
        if not self._plot_detached or self._right_panel_layout is None:
            self.btn_detach_plots.setChecked(False)
            self.btn_detach_plots.setText("Detach plots")
            return
        if self._plot_dialog is not None:
            self._plot_dialog.layout().removeWidget(self.view_tabs)
        self._right_panel_layout.addWidget(self.view_tabs, 1)
        if self._plot_dialog is not None and self._plot_dialog.isVisible():
            self._plot_dialog.blockSignals(True)
            self._plot_dialog.close()
            self._plot_dialog.blockSignals(False)
        self._plot_dialog = None
        self._plot_detached = False
        if self._right_panel is not None:
            self._right_panel.show()
        if self._body_splitter is not None:
            if self._body_sizes_before_detach:
                self._body_splitter.setSizes(self._body_sizes_before_detach)
            else:
                self._body_splitter.setSizes([3, 7])
        self.btn_detach_plots.blockSignals(True)
        self.btn_detach_plots.setChecked(False)
        self.btn_detach_plots.blockSignals(False)
        self.btn_detach_plots.setText("Detach plots")

    def set_ks_folder(self, folder: str) -> None:
        self.ed_folder.setText(folder)

    def open_ks_folder(self, folder: str) -> None:
        self.set_ks_folder(folder)
        self._load_dataset()

    def _pick(self) -> None:
        start = self.settings.value("paths/last_folder", str(Path.cwd()))
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select curated folder", str(start))
        if folder:
            self.ed_folder.setText(folder)
            self.settings.setValue("paths/last_folder", folder)
            self.settings.setValue("post/last_folder", folder)

    def _read_unit_labels(self, folder: Path) -> pd.DataFrame:
        out = pd.DataFrame()
        bpath = folder / "bombcell_labels.csv"
        if bpath.exists():
            try:
                df = self._normalize_label_df(pd.read_csv(bpath))
                out = df
            except Exception:
                pass
        cpath = folder / "cluster_group.tsv"
        if cpath.exists():
            try:
                cg = pd.read_csv(cpath, sep="\t")
                if "cluster_id" in cg.columns and "group" in cg.columns:
                    cg = cg.set_index("cluster_id", drop=True).rename(columns={"group": "cluster_group"})
                    out = cg if out.empty else out.join(cg[["cluster_group"]], how="outer")
            except Exception:
                pass
        return out

    @staticmethod
    def _normalize_label_df(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        cols = [str(c) for c in out.columns]
        # Prefer explicit id columns; otherwise accept first unnamed/first column as unit id.
        if "cluster_id" in cols:
            out = out.set_index("cluster_id", drop=True)
        elif "unit_id" in cols:
            out = out.set_index("unit_id", drop=True)
        elif cols:
            c0 = cols[0]
            if c0.lower().startswith("unnamed") or c0.lower() in {"id", "cluster", "unit"}:
                out = out.set_index(c0, drop=True)
        # Normalize index to int where possible.
        try:
            idx = pd.to_numeric(out.index, errors="coerce")
            valid = ~pd.isna(idx)
            if valid.any():
                out = out.loc[valid]
                out.index = idx[valid].astype(int)
        except Exception:
            pass
        return out

    def _read_label_sources(self, folder: Path) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        # Bombcell
        bpath = folder / "bombcell_labels.csv"
        if bpath.exists():
            try:
                out["Bombcell"] = self._normalize_label_df(pd.read_csv(bpath))
            except Exception:
                pass
        # Phy
        cpath = folder / "cluster_group.tsv"
        if cpath.exists():
            try:
                df = pd.read_csv(cpath, sep="\t")
                if "cluster_id" in df.columns and "group" in df.columns:
                    out["Phy"] = self._normalize_label_df(df)
            except Exception:
                pass
        # Kilosort label
        kpath = folder / "cluster_KSLabel.tsv"
        if kpath.exists():
            try:
                df = pd.read_csv(kpath, sep="\t")
                if "cluster_id" in df.columns and "KSLabel" in df.columns:
                    out["KSLabel"] = self._normalize_label_df(df)
            except Exception:
                pass
        return out

    @staticmethod
    def _row_for_unit(df: pd.DataFrame, unit: int):
        if df.empty:
            return None
        if unit in df.index:
            row = df.loc[unit]
            return row.iloc[0] if isinstance(row, pd.DataFrame) else row
        su = str(unit)
        if su in df.index:
            row = df.loc[su]
            return row.iloc[0] if isinstance(row, pd.DataFrame) else row
        return None

    def _load_dataset(self) -> None:
        folder = Path(self.ed_folder.text().strip())
        self.settings.setValue("post/last_folder", str(folder))
        if not folder.exists():
            self._log("Invalid folder")
            return
        try:
            self.dataset = NeuropixelsDataset.load(str(folder))
        except Exception as exc:
            self._log(f"Failed loading dataset: {exc}")
            return

        self.metrics_df = pd.DataFrame()
        mp = folder / "metrics.csv"
        if mp.exists():
            try:
                df = pd.read_csv(mp)
                if "cluster_id" in df.columns:
                    df = df.set_index("cluster_id", drop=True)
                elif "unit_id" in df.columns:
                    df = df.set_index("unit_id", drop=True)
                self.metrics_df = df
            except Exception as exc:
                self._log(f"metrics.csv read failed: {exc}")

        self.labels_df = self._read_unit_labels(folder)
        self.label_sources = self._read_label_sources(folder)
        self._all_units = [int(u) for u in self.dataset.units.tolist()]
        self._update_basic_time_bounds()
        self._refresh_units_list()
        self._basic_cache.clear()
        self._log(f"Loaded dataset with {len(self._all_units)} units")
        self._set_progress(0)
        self._refresh_current_page()

    def _unit_is_good(self, unit: int) -> bool:
        src = self.cb_good_source.currentText().strip()
        if src == "Auto":
            # Priority: Bombcell -> Phy -> KSLabel
            any_source = False
            for name in ["Bombcell", "Phy", "KSLabel"]:
                df = self.label_sources.get(name, pd.DataFrame())
                if df.empty:
                    continue
                any_source = True
                row = self._row_for_unit(df, unit)
                if row is None:
                    continue
                return self._unit_is_good_from_row(name, row)
            return False if any_source else True
        df = self.label_sources.get(src, pd.DataFrame())
        if df.empty:
            return True
        row = self._row_for_unit(df, unit)
        if row is None:
            return False
        return self._unit_is_good_from_row(src, row)

    def _unit_is_good_from_row(self, src: str, row) -> bool:
        # accept both canonical and fallback column names
        if src == "Bombcell":
            for key in ["bombcell_label", "label", "group", "kslabel"]:
                if key in row.index:
                    return _is_bombcell_good_label(row[key])
            return False
        if src == "Phy":
            return "group" in row and str(row["group"]).lower() in {"good", "single", "singleunit"}
        if src == "KSLabel":
            return "KSLabel" in row and str(row["KSLabel"]).lower() == "good"
        return False

    def _on_good_source_changed(self) -> None:
        self._refresh_units_list()
        self._update_unit_quality_table()

    def _refresh_units_list(self) -> None:
        prev = {int(i.text()) for i in self.list_units.selectedItems()}
        filt = self.ed_unit_filter.text().strip().lower()
        good_only = self.btn_good_only.isChecked()
        self.list_units.clear()
        for u in self._all_units:
            if filt and filt not in str(u):
                continue
            if good_only and not self._unit_is_good(u):
                continue
            self.list_units.addItem(str(u))
            if u in prev:
                self.list_units.item(self.list_units.count() - 1).setSelected(True)

    def _selected_units(self) -> list[int]:
        return [int(i.text()) for i in self.list_units.selectedItems()]

    def _on_units_selection_changed(self) -> None:
        self._update_unit_quality_table()
        self._refresh_current_page()

    def _update_unit_quality_table(self) -> None:
        self.tbl_unit_quality.setRowCount(0)
        units = self._selected_units()
        if not units:
            return
        u = units[0]
        entries: list[tuple[str, str]] = [("unit_id", str(u))]
        if not self.labels_df.empty and u in self.labels_df.index:
            row = self.labels_df.loc[u]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            if "bombcell_label" in row:
                entries.append(("bombcell_label", str(row["bombcell_label"])))
            if "cluster_group" in row:
                entries.append(("cluster_group", str(row["cluster_group"])))
        if not self.metrics_df.empty and u in self.metrics_df.index:
            row = self.metrics_df.loc[u]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            for key in ["isi_viol", "rp_contamination", "amplitude_cutoff", "presence_ratio", "snr", "amplitude_median"]:
                if key in row.index:
                    val = row[key]
                    try:
                        entries.append((key, f"{float(val):.4g}"))
                    except Exception:
                        entries.append((key, str(val)))
            for key in ["best_channel", "peak_channel", "maxChannels", "max_channel", "channel"]:
                if key in row.index:
                    try:
                        entries.append(("best_channel", str(int(float(row[key])))))
                    except Exception:
                        entries.append(("best_channel", str(row[key])))
                    break
        if not any(k == "best_channel" for k, _ in entries) and self.dataset is not None:
            try:
                wvf = self.dataset.mean_template_waveform(u)
                if wvf is not None and wvf.ndim == 2 and wvf.shape[1] > 0:
                    best_idx = int(np.nanargmax(np.max(np.abs(wvf), axis=0)))
                    if self.dataset.channel_map is not None and self.dataset.channel_map.size > best_idx:
                        best_ch = int(np.asarray(self.dataset.channel_map).squeeze()[best_idx])
                    else:
                        best_ch = best_idx
                    entries.append(("best_channel", str(best_ch)))
            except Exception:
                pass
        self.tbl_unit_quality.setRowCount(len(entries))
        for r, (k, v) in enumerate(entries):
            self.tbl_unit_quality.setItem(r, 0, QtWidgets.QTableWidgetItem(k))
            self.tbl_unit_quality.setItem(r, 1, QtWidgets.QTableWidgetItem(v))
        self.tbl_unit_quality.resizeColumnsToContents()
    def _on_analysis_page_changed(self, idx: int) -> None:
        self.view_tabs.setCurrentIndex(idx)
        self._refresh_current_page()

    def _refresh_current_page(self) -> None:
        if self.dataset is None:
            return
        idx = self.analysis_tabs.currentIndex()
        try:
            if idx == 0:
                self._visualize_basic()
            elif idx == 1:
                self._visualize_raw()
            elif idx == 2:
                self._show_corr()
            elif idx == 3:
                self._show_psth()
            elif idx == 4:
                self._show_network()
            elif idx == 5:
                self._show_npyx_corr()
        except Exception as exc:
            self._log(f"Page render error: {exc}")

    def _set_progress(self, value: int) -> None:
        self.page_progress.setValue(int(max(0, min(100, value))))
        QtWidgets.QApplication.processEvents()

    def _corr_items_from_selection(self) -> list[dict]:
        if self.dataset is None:
            return []
        units = self._selected_units()
        if not units:
            return []
        bin_ms = float(self.sp_corr_bin.value())
        win_ms = float(self.sp_corr_win.value())
        mode = self.cb_corr_mode.currentText()
        items: list[dict] = []
        if mode == "ACG":
            for u in units:
                centers, counts = self.dataset.correlogram(u, u, bin_ms=bin_ms, win_ms=win_ms, remove_zero=True)
                items.append({"u1": u, "u2": u, "centers": centers, "counts": counts})
        else:
            if len(units) < 2:
                return []
            for ua, ub in combinations(units, 2):
                centers, counts = self.dataset.correlogram(ua, ub, bin_ms=bin_ms, win_ms=win_ms, remove_zero=False)
                items.append({"u1": ua, "u2": ub, "centers": centers, "counts": counts})
        return items

    def _show_corr(self) -> None:
        items = self._corr_items_from_selection()
        if not items:
            self.gl_corr.clear()
            self._export_payloads["corr"] = []
            return
        self._set_progress(100)
        self._visualize_corr({"mode": self.cb_corr_mode.currentText(), "items": items})

    def _compute_psth(self) -> None:
        units = self._selected_units()
        if not units:
            self._log("PSTH compute: select one or more units first.")
            return
        self._busy = True
        self._log("Computing condition PSTH...")
        self._set_progress(10)
        try:
            condition_entries: List[dict] = []
            t_ref = np.array([], dtype=float)
            for r in range(self.tbl_conditions.rowCount()):
                name = self._condition_name_for_row(r)
                fpath = self._condition_path_for_row(r)
                selected_label = self._condition_selected_label(r)
                if not fpath:
                    continue
                p = Path(fpath)
                if not p.exists():
                    self._log(f"PSTH compute: missing CSV for row {r + 1}: {fpath}")
                    continue
                ev = self._load_event_csv(fpath, selected_label=selected_label)
                if ev.size == 0:
                    label_desc = f" [{selected_label}]" if selected_label else ""
                    self._log(f"PSTH compute: no valid events in {p.name}{label_desc}.")
                    continue
                condition_unit_ids: List[int] = []
                condition_trial_mats: List[np.ndarray] = []
                for unit in units:
                    t_ms, trial_mat = self.dataset.psth_trials(
                        int(unit),
                        ev,
                        float(self.sp_psth_pre.value()),
                        float(self.sp_psth_post.value()),
                        float(self.sp_psth_bin.value()),
                    )
                    if t_ms.size == 0 or trial_mat.size == 0:
                        continue
                    if t_ref.size == 0:
                        t_ref = np.asarray(t_ms, dtype=float)
                    condition_unit_ids.append(int(unit))
                    condition_trial_mats.append(np.asarray(trial_mat, dtype=float))
                if t_ref.size == 0 or not condition_trial_mats:
                    label_desc = f" [{selected_label}]" if selected_label else ""
                    self._log(f"PSTH compute: unable to build PSTH for {name}{label_desc}.")
                    continue
                condition_entries.append(
                    {
                        "condition": str(name),
                        "selected_label": str(selected_label),
                        "source_csv": str(p),
                        "unit_ids": list(condition_unit_ids),
                        "unit_trial_mats": list(condition_trial_mats),
                        "trial_count": int(ev.size),
                    }
                )
            if not condition_entries or t_ref.size == 0:
                self._log("PSTH compute: no valid conditions.")
                self._set_progress(0)
                return
            self.results["psth"] = {
                "t_ms": np.asarray(t_ref, dtype=float),
                "conditions": list(condition_entries),
                "units": [int(u) for u in units],
                "pre_s": float(self.sp_psth_pre.value()),
                "post_s": float(self.sp_psth_post.value()),
                "bin_ms": float(self.sp_psth_bin.value()),
            }
            self._set_progress(100)
            self._log("Condition PSTH computed.")
            self._show_psth()
        finally:
            self._busy = False

    def _show_psth(self) -> None:
        r = self.results.get("psth")
        if not r:
            return
        self._visualize_condition_psth(r)

    def _compute_network(self) -> None:
        units = self._selected_units()
        if len(units) < 2:
            self._log("Network compute: select at least 2 units.")
            return
        self._busy = True
        self._log("Computing network metrics...")
        self._set_progress(10)
        try:
            mat = self.dataset.ccg_matrix(units, bin_ms=float(self.sp_net_bin.value()), win_ms=float(self.sp_net_win.value()))
            grouping = cluster_synced_units(units, mat)
            self._set_progress(60)
            t_s, idx = self.dataset.synchrony_over_time(
                bin_ms=float(self.sp_sync_bin.value()), window_s=float(self.sp_sync_win.value()), step_s=float(self.sp_sync_step.value())
            )
            self.results["network"] = {
                "mat": mat,
                "t_s": t_s,
                "idx": idx,
                "units": np.asarray(units, dtype=int),
                "order": grouping["order"],
                "sorted_units": grouping["sorted_units"],
                "group_labels": grouping["group_labels"],
                "sync_group_threshold": grouping["threshold"],
            }
            self._set_progress(100)
            self._log("Network metrics computed.")
            self._show_network()
        finally:
            self._busy = False

    def _show_network(self) -> None:
        r = self.results.get("network")
        if not r:
            return
        self._visualize_network(r)

    def _visualize_basic(self) -> None:
        units = self._selected_units()
        if not units:
            return
        u0 = int(units[0])
        t0 = float(self.sp_basic_t0.value())
        rec_dur = self._recording_duration_s()
        t1 = t0 + float(self.sp_basic_dur.value())
        if rec_dur > 0.0:
            t1 = min(t1, rec_dur)
        self.plot_basic_spikes.clear()
        self.plot_basic_acg.clear()
        self.plot_basic_isi.clear()
        self.gl_basic_wvf.clear()
        payload = {
            "units": [int(u) for u in units],
            "t0": t0,
            "dur": float(self.sp_basic_dur.value()),
            "isi_max": float(self.sp_isi_max.value()),
            "ifr": bool(self.ck_ifr.isChecked()),
            "ifr_smooth_ms": float(self.sp_ifr_smooth_ms.value()),
            "acg_bin_ms": float(self.sp_basic_acg_bin.value()),
            "acg_win_ms": float(self.sp_basic_acg_win.value()),
        }
        cache_key = json.dumps(payload, sort_keys=True)
        basic = self._basic_cache.get(cache_key)
        if basic is None:
            spike_items: list[dict] = []
            ifr_items: list[dict] = []
            for i, u in enumerate(units):
                spikes = self.dataset.unit_spike_times_s(u)
                st = spikes[(spikes >= t0) & (spikes <= t1)]
                spike_items.append({"unit": int(u), "row": int(i + 1), "st": st - t0})
                if self.ck_ifr.isChecked() and st.size:
                    bin_s = max(float(self.sp_ifr_smooth_ms.value()) / 1000.0, 0.005)
                    edges = np.arange(t0, t1 + bin_s, bin_s)
                    c, _ = np.histogram(st, bins=edges)
                    rate = c / bin_s
                    centers = 0.5 * (edges[:-1] + edges[1:]) - t0
                    ifr_items.append({"unit": int(u), "row": int(i + 1), "centers": centers, "rate": rate})
            isi_edges, isi_hist = self.dataset.isi_hist(u0, max_ms=float(self.sp_isi_max.value()), bins=80)
            acg_centers, acg_counts = self.dataset.correlogram(
                u0,
                u0,
                bin_ms=float(self.sp_basic_acg_bin.value()),
                win_ms=float(self.sp_basic_acg_win.value()),
                remove_zero=True,
            )
            waveform = self.dataset.mean_template_waveform(u0)
            basic = {
                "spike_items": spike_items,
                "ifr_items": ifr_items,
                "waveform_unit": u0,
                "waveform": waveform,
                "isi_edges": isi_edges,
                "isi_hist": isi_hist,
                "acg_centers": acg_centers,
                "acg_counts": acg_counts,
            }
            self._basic_cache[cache_key] = basic

        spike_rows: list[dict] = []
        ifr_rows: list[dict] = []
        wvf_rows: list[dict] = []
        acg_rows: list[dict] = []
        isi_rows: list[dict] = []

        for i, item in enumerate(basic.get("spike_items", [])):
            u = int(item["unit"])
            st_rel = np.asarray(item["st"], dtype=float)
            row = float(item["row"])
            if st_rel.size:
                y = np.full_like(st_rel, row, dtype=float)
                self.plot_basic_spikes.plot(st_rel, y, pen=None, symbol="o", symbolSize=3, symbolBrush=pg.intColor(i, hues=max(len(units), 4)))
                spike_rows.extend([{"unit_id": u, "time_s": float(v)} for v in st_rel])

        for i, item in enumerate(basic.get("ifr_items", [])):
            centers = np.asarray(item["centers"], dtype=float)
            rate = np.asarray(item["rate"], dtype=float)
            if centers.size == 0 or rate.size == 0:
                continue
            row = float(item["row"])
            scale = 0.35 / max(float(np.max(rate)), 1e-9)
            self.plot_basic_spikes.plot(centers, row + rate * scale, pen=pg.mkPen(pg.intColor(i), width=1.5))
            ifr_rows.extend([{"unit_id": int(item["unit"]), "time_s": float(tc), "ifr_hz": float(rv)} for tc, rv in zip(centers, rate)])
        self.plot_basic_spikes.setYRange(0.5, len(units) + 1.2)
        self.plot_basic_spikes.setLabel("left", "Selected unit index")
        self.plot_basic_spikes.setLabel("bottom", "Time (s)")

        acg_centers = np.asarray(basic.get("acg_centers", []), dtype=float)
        acg_counts = np.asarray(basic.get("acg_counts", []), dtype=float)
        if acg_centers.size and acg_counts.size:
            acg_width = float(np.median(np.diff(acg_centers))) if acg_centers.size > 1 else float(self.sp_corr_bin.value())
            bar_rgb = (108, 168, 255) if self._plot_theme == "Dark" else (74, 144, 226)
            bar_alpha = 170 if self._plot_theme == "Dark" else 185
            bar_width = abs(acg_width) * 0.90
            for cx, cc in zip(acg_centers, acg_counts):
                self.plot_basic_acg.addItem(
                    pg.BarGraphItem(
                        x=[float(cx)],
                        height=[float(cc)],
                        width=bar_width,
                        brush=bar_rgb + (bar_alpha,),
                        pen=pg.mkPen(bar_rgb + (min(bar_alpha + 20, 255),), width=0.7),
                    )
                )
            refractory_ms = min(2.0, max(0.5, 2.0 * abs(acg_width)))
            self.plot_basic_acg.addItem(
                pg.LinearRegionItem(
                    values=(-refractory_ms, refractory_ms),
                    orientation="vertical",
                    brush=pg.mkBrush(92, 154, 255, 22),
                    pen=pg.mkPen((92, 154, 255, 0)),
                    movable=False,
                )
            )
            self.plot_basic_acg.setLabel("left", "count")
            self.plot_basic_acg.setLabel("bottom", "Lag (ms)")
            self.plot_basic_acg.addItem(pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen((150, 150, 150), width=1, style=QtCore.Qt.DashLine)))
            self.plot_basic_acg.setXRange(float(acg_centers.min()) - abs(acg_width), float(acg_centers.max()) + abs(acg_width), padding=0.0)
            self.plot_basic_acg.setYRange(0.0, float(np.max(acg_counts)) * 1.18 if np.max(acg_counts) > 0 else 1.0, padding=0.0)
            acg_rows.extend([{"unit_id": int(units[0]), "lag_ms": float(cx), "count": float(cc)} for cx, cc in zip(acg_centers, acg_counts)])

        isi_edges = np.asarray(basic.get("isi_edges", []), dtype=float)
        isi_hist = np.asarray(basic.get("isi_hist", []), dtype=float)
        if isi_edges.size and isi_hist.size:
            widths = np.diff(isi_edges)
            self.plot_basic_isi.addItem(
                pg.BarGraphItem(
                    x=isi_edges[:-1],
                    height=isi_hist,
                    width=widths * 0.92,
                    brush=(245, 171, 66, 210),
                    pen=pg.mkPen((125, 82, 18), width=0.8),
                )
            )
            self.plot_basic_isi.setXRange(float(isi_edges[0]), float(isi_edges[-1]), padding=0.0)
            self.plot_basic_isi.setYRange(0.0, float(np.max(isi_hist)) * 1.18 if np.max(isi_hist) > 0 else 1.0, padding=0.0)
            self.plot_basic_isi.setLabel("left", "count")
            self.plot_basic_isi.setLabel("bottom", "ISI (ms)")
            isi_rows.extend([{"unit_id": int(units[0]), "isi_left_ms": float(l), "isi_right_ms": float(r), "count": float(h)} for l, r, h in zip(isi_edges[:-1], isi_edges[1:], isi_hist)])

        waveform = basic.get("waveform")
        if isinstance(waveform, np.ndarray):
            wvf_rows = self._render_multichannel_waveform(int(basic.get("waveform_unit", u0)), waveform)
        self._export_payloads["basic"] = [
            ("unit_basics_raster.csv", pd.DataFrame(spike_rows)),
            ("unit_basics_ifr.csv", pd.DataFrame(ifr_rows)),
            ("unit_basics_autocorrelogram.csv", pd.DataFrame(acg_rows)),
            ("unit_basics_isi.csv", pd.DataFrame(isi_rows)),
            ("unit_basics_waveform.csv", pd.DataFrame(wvf_rows)),
        ]
        self.view_tabs.setCurrentIndex(0)

    def _visualize_raw(self) -> None:
        self.plot_raw.clear()
        units = self._selected_units()
        focus_unit = int(units[0]) if units else None
        focus_waveform = self.dataset.mean_template_waveform(focus_unit) if focus_unit is not None else None
        focus_center = self._best_channel_index(focus_unit, focus_waveform) if focus_unit is not None else None
        t, x, channel_ids, channel_order = self.dataset.raw_explorer_chunk(
            t0_s=float(self.sp_raw_t0.value()),
            dur_s=float(self.sp_raw_dur.value()),
            max_channels=int(self.sp_raw_ch.value()),
            hp_hz=float(self.sp_raw_hp.value()),
            lp_hz=float(self.sp_raw_lp.value()),
            downsample=int(self.sp_raw_ds.value()),
            center_channel=focus_center,
        )
        if x.size == 0:
            return
        n_ch = int(x.shape[1])
        channel_ids = np.asarray(channel_ids, dtype=int)
        channel_order = np.asarray(channel_order, dtype=int)
        axis_labels: list[str]
        if (
            self.cb_raw_y.currentText().startswith("Depth")
            and self.dataset.channel_positions is not None
            and np.max(channel_order) < int(self.dataset.channel_positions.shape[0])
        ):
            y_values = np.asarray(self.dataset.channel_positions, dtype=float)[channel_order, 1] / 1000.0
            axis_labels = [f"{float(v):.3f}" for v in y_values]
            y_label = "Depth (mm)"
        else:
            y_values = channel_ids.astype(float)
            axis_labels = [str(int(v)) for v in channel_ids]
            y_label = "Channel ID"
        y_base = np.arange(n_ch, dtype=float)
        x_centered = x - np.nanmedian(x, axis=0, keepdims=True)
        amp = max(float(np.nanpercentile(np.abs(x_centered), 99.8)), 1.0)
        spacing = 1.0
        scale = 0.32 / amp
        self.plot_raw.setLabel("left", y_label)
        left_axis = self.plot_raw.getAxis("left")
        tick_step = max(1, int(math.ceil(float(n_ch) / 10.0)))
        tick_pairs = [(float(y_base[i]), axis_labels[i]) for i in range(0, n_ch, tick_step)]
        if tick_pairs[-1][0] != float(y_base[-1]):
            tick_pairs.append((float(y_base[-1]), axis_labels[-1]))
        left_axis.setTicks([tick_pairs])
        raw_pen = pg.mkPen((220, 228, 240, 150), width=0.8) if self._plot_theme == "Dark" else pg.mkPen((28, 36, 48, 145), width=0.8)
        raw_focus_pen = pg.mkPen((255, 255, 255, 205), width=1.05) if self._plot_theme == "Dark" else pg.mkPen((20, 24, 30, 205), width=1.05)
        for c in range(n_ch):
            pen = raw_pen
            if focus_center is not None and int(channel_order[c]) == int(focus_center):
                pen = raw_focus_pen
            self.plot_raw.plot(t, y_base[c] + x_centered[:, c] * scale, pen=pen)

        if focus_waveform is not None:
            focus_support = self._waveform_support_indices(focus_waveform, limit=min(32, n_ch))
            display_mask = np.isin(channel_order, focus_support)
            if np.any(display_mask):
                support_y = y_base[display_mask]
                pad = 0.36 * spacing
                band = pg.LinearRegionItem(
                    values=(float(np.min(support_y)) - pad, float(np.max(support_y)) + pad),
                    orientation="horizontal",
                    brush=pg.mkBrush(67, 128, 255, 12),
                    pen=pg.mkPen((67, 128, 255, 35), width=0.8),
                    movable=False,
                )
                self.plot_raw.addItem(band)

        if self.ck_raw_overlay.isChecked():
            for i, u in enumerate(units):
                st = self.dataset.unit_spike_times_s(u)
                st = st[(st >= t[0]) & (st <= t[-1])]
                if st.size == 0:
                    continue
                wvf = self.dataset.mean_template_waveform(u)
                support = self._waveform_support_indices(wvf, limit=min(28, n_ch))
                if support.size == 0:
                    best = self._best_channel_index(u, wvf)
                    support = np.array([best], dtype=int) if best is not None else np.array([], dtype=int)
                mask = np.isin(channel_order, support)
                if not np.any(mask):
                    continue
                if st.size > 36:
                    stride = int(math.ceil(float(st.size) / 36.0))
                    st = st[::stride]
                color = pg.intColor(i, hues=max(len(units), 4))
                overlay_pen = pg.mkPen(color, width=1.0)
                for ch_local in np.flatnonzero(mask):
                    y0 = float(y_base[ch_local] - 0.28 * spacing)
                    y1 = float(y_base[ch_local] + 0.28 * spacing)
                    xs = np.empty(st.size * 3, dtype=float)
                    ys = np.empty(st.size * 3, dtype=float)
                    xs[0::3] = st
                    xs[1::3] = st
                    xs[2::3] = np.nan
                    ys[0::3] = y0
                    ys[1::3] = y1
                    ys[2::3] = np.nan
                    self.plot_raw.plot(xs, ys, pen=overlay_pen)
        self.plot_raw.setLabel("bottom", "Time (s)")
        self.plot_raw.setXRange(float(t[0]), float(t[-1]), padding=0.0)
        self.plot_raw.setYRange(float(np.min(y_base) - spacing), float(np.max(y_base) + spacing), padding=0.0)
        raw_rows: list[dict] = []
        for c in range(n_ch):
            raw_rows.extend(
                [
                    {
                        "time_s": float(tv),
                        "channel_index": int(channel_order[c]),
                        "channel_id": int(channel_ids[c]),
                        "axis_value": float(y_base[c]),
                        "signal_scaled": float(y_base[c] + xv * scale),
                    }
                    for tv, xv in zip(t, x_centered[:, c])
                ]
            )
        self._export_payloads["raw"] = [("raw_explorer.csv", pd.DataFrame(raw_rows))]
        self.view_tabs.setCurrentIndex(1)

    def _visualize_corr(self, r: Dict[str, object]) -> None:
        self.gl_corr.clear()
        items = r.get("items", [])
        if not isinstance(items, list) or not items:
            centers = np.asarray(r.get("centers", []), dtype=float)
            counts = np.asarray(r.get("counts", []), dtype=float)
            if centers.size and counts.size:
                items = [{"u1": int(r.get("u1", 0)), "u2": int(r.get("u2", 0)), "centers": centers, "counts": counts}]
            else:
                return
        rows, cols = self._subplot_shape(len(items))
        corr_rows: list[dict] = []
        for i, item in enumerate(items):
            centers = np.asarray(item.get("centers", []), dtype=float)
            counts = np.asarray(item.get("counts", []), dtype=float)
            if centers.size == 0 or counts.size == 0:
                continue
            u1 = int(item.get("u1", -1))
            u2 = int(item.get("u2", -1))
            plot = self.gl_corr.addPlot(row=i // cols, col=i % cols, title=f"{u1} vs {u2}")
            self._style_plot_item(plot, left="Count", bottom="Lag (ms)")
            width = float(np.median(np.diff(centers))) if centers.size > 1 else float(self.sp_corr_bin.value())
            plot.addItem(pg.BarGraphItem(x=centers, height=counts, width=abs(width), brush=(255, 180, 120, 110)))
            y = pd.Series(counts).rolling(window=max(3, int(5.0 / max(abs(width), 1e-6))), center=True, min_periods=1).mean().to_numpy()
            plot.plot(centers, y, pen=pg.mkPen((80, 150, 255), width=2))
            plot.addItem(pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen((150, 150, 150), width=1, style=QtCore.Qt.DashLine)))
            corr_rows.extend(
                [
                    {"unit_a": u1, "unit_b": u2, "lag_ms": float(cx), "count": float(cc), "smoothed_count": float(cs)}
                    for cx, cc, cs in zip(centers, counts, y)
                ]
            )
        self._export_payloads["corr"] = [("correlogram.csv", pd.DataFrame(corr_rows))]
        self.view_tabs.setCurrentIndex(2)

    def _add_condition_row(self) -> None:
        r = self.tbl_conditions.rowCount()
        self.tbl_conditions.insertRow(r)
        self.tbl_conditions.setItem(r, 0, QtWidgets.QTableWidgetItem(f"cond_{r+1}"))
        self._set_condition_label_options(r, [], "")
        self.tbl_conditions.setItem(r, 2, QtWidgets.QTableWidgetItem(""))

    def _remove_condition_row(self) -> None:
        rows = sorted({i.row() for i in self.tbl_conditions.selectedIndexes()}, reverse=True)
        for r in rows:
            self.tbl_conditions.removeRow(r)
        self._refresh_current_page()

    def _browse_condition_csv(self) -> None:
        if self.tbl_conditions.rowCount() == 0:
            self._add_condition_row()
        row = self.tbl_conditions.currentRow()
        if row < 0:
            row = 0
        start = self.settings.value("paths/last_folder", str(Path.cwd()))
        fp, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select condition events CSV", str(start), "CSV files (*.csv)")
        if fp:
            self.settings.setValue("paths/last_folder", str(Path(fp).parent))
            self._apply_condition_csv_to_row(row, fp)
            self._refresh_current_page()

    def _on_conditions_changed(self, _item: QtWidgets.QTableWidgetItem) -> None:
        if _item is not None and _item.column() == 2:
            path = self._condition_path_for_row(_item.row())
            if path:
                self._apply_condition_csv_to_row(_item.row(), path, preserve_name=True, announce=False)
        if self.analysis_tabs.currentIndex() == 3:
            self._refresh_current_page()

    def _condition_label_combo(self, row: int) -> Optional[QtWidgets.QComboBox]:
        widget = self.tbl_conditions.cellWidget(row, 1)
        return widget if isinstance(widget, QtWidgets.QComboBox) else None

    def _set_condition_label_options(self, row: int, labels: List[str], selected_label: str = "") -> None:
        combo = self._condition_label_combo(row)
        if combo is None:
            combo = QtWidgets.QComboBox(self.tbl_conditions)
            combo.currentIndexChanged.connect(self._on_condition_label_selection_changed)
            self.tbl_conditions.setCellWidget(row, 1, combo)
        current = str(selected_label or combo.currentData() or "").strip()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("All events", "")
        for label in labels:
            combo.addItem(str(label), str(label))
        if current:
            idx = combo.findData(current)
            if idx < 0:
                idx = combo.findText(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            elif combo.count() > 1:
                combo.setCurrentIndex(1)
            else:
                combo.setCurrentIndex(0)
        else:
            combo.setCurrentIndex(1 if combo.count() > 1 else 0)
        combo.setEnabled(combo.count() > 1)
        combo.blockSignals(False)

    def _condition_selected_label(self, row: int) -> str:
        combo = self._condition_label_combo(row)
        if combo is None:
            return ""
        value = combo.currentData()
        if value is None:
            value = combo.currentText()
        return str(value or "").strip()

    def _condition_name_for_row(self, row: int) -> str:
        name_item = self.tbl_conditions.item(row, 0)
        name = name_item.text().strip() if name_item else ""
        if name:
            return name
        label = self._condition_selected_label(row)
        if label:
            return label
        path = self._condition_path_for_row(row)
        return Path(path).stem if path else f"cond_{row + 1}"

    def _condition_path_for_row(self, row: int) -> str:
        file_item = self.tbl_conditions.item(row, 2)
        return file_item.text().strip() if file_item else ""

    def _apply_condition_csv_to_row(
        self,
        row: int,
        path: str,
        *,
        preserve_name: bool = False,
        announce: bool = True,
    ) -> None:
        csv_path = str(path).strip()
        if not csv_path:
            self._set_condition_label_options(row, [], "")
            return
        try:
            info = inspect_event_csv(csv_path)
        except Exception as exc:
            self._log(f"Events CSV read failed: {exc}")
            return
        labels = [str(v) for v in info.get("labels", [])]
        current_selected_label = self._condition_selected_label(row)
        if current_selected_label and current_selected_label in labels:
            selected_label = current_selected_label
        else:
            selected_label = labels[0] if labels else ""
        blocker = QtCore.QSignalBlocker(self.tbl_conditions)
        self.tbl_conditions.setItem(row, 2, QtWidgets.QTableWidgetItem(csv_path))
        name_item = self.tbl_conditions.item(row, 0)
        current_name = name_item.text().strip() if name_item else ""
        auto_name = selected_label or Path(csv_path).stem
        if (not preserve_name) or (not current_name) or current_name.startswith("cond_"):
            self.tbl_conditions.setItem(row, 0, QtWidgets.QTableWidgetItem(auto_name))
        del blocker
        self._set_condition_label_options(row, labels, selected_label)
        time_column = str(info.get("time_column") or "")
        label_column = str(info.get("label_column") or "")
        if announce:
            if time_column and label_column:
                self._log(
                    f"Events CSV loaded: {Path(csv_path).name} | time='{time_column}' | "
                    f"label='{label_column}' ({len(labels)} labels)"
                )
            elif time_column:
                self._log(f"Events CSV loaded: {Path(csv_path).name} | time='{time_column}' | all rows")
            else:
                self._log(f"Events CSV loaded but no numeric event-time column was detected: {Path(csv_path).name}")

    def _on_condition_label_selection_changed(self, _index: int) -> None:
        combo = self.sender()
        if isinstance(combo, QtWidgets.QComboBox):
            for row in range(self.tbl_conditions.rowCount()):
                if self.tbl_conditions.cellWidget(row, 1) is combo:
                    name_item = self.tbl_conditions.item(row, 0)
                    current_name = name_item.text().strip() if name_item else ""
                    if (not current_name) or current_name.startswith("cond_"):
                        auto_name = self._condition_selected_label(row) or Path(self._condition_path_for_row(row) or "").stem or f"cond_{row + 1}"
                        blocker = QtCore.QSignalBlocker(self.tbl_conditions)
                        self.tbl_conditions.setItem(row, 0, QtWidgets.QTableWidgetItem(auto_name))
                        del blocker
                    break
        if self.analysis_tabs.currentIndex() == 3:
            self._refresh_current_page()

    def _load_event_csv(self, path: str, selected_label: str = "") -> np.ndarray:
        return load_event_times(path, selected_label=selected_label).to_numpy(dtype=float)

    def _visualize_condition_psth(self, r: Dict[str, object]) -> None:
        self.plot_psth_lines.clear()
        self.plot_psth_heat.clear()
        t_ms = np.asarray(r.get("t_ms", []), dtype=float)
        conditions = list(r.get("conditions", []))
        units = [int(v) for v in r.get("units", [])]
        if t_ms.size == 0 or not conditions:
            self.lbl_psth_summary.setText("Condition PSTH is ready after you compute it.")
            self.lbl_psth_summary_meta.setText(
                "Select units, choose an event label, then use Compute. Trial-range changes are applied on the displayed heatmap and averages."
            )
            return

        single_unit = len(units) == 1
        fg = "#e8eef7" if self._plot_theme == "Dark" else "#1a1f29"
        title_color = fg
        zero_pen = pg.mkPen((120, 120, 120), width=1, style=QtCore.Qt.DashLine)
        pre_ms = float(r.get("pre_s", self.sp_psth_pre.value())) * 1000.0
        pre_region_brush = pg.mkBrush(91, 155, 255, 20) if self._plot_theme == "Dark" else pg.mkBrush(67, 128, 255, 16)
        palette = self._psth_palette(len(conditions))

        plot_item = self.plot_psth_lines.getPlotItem()
        if plot_item.legend is not None:
            plot_item.legend.scene().removeItem(plot_item.legend)
            plot_item.legend = None
        plot_item.addLegend(offset=(12, 12))
        self._style_plot_item(plot_item, left="Rate (Hz)", bottom="Time (ms) relative to event")
        self.plot_psth_lines.setTitle(
            "Condition PSTH | mean \u00b1 SEM across trials" if single_unit else "Condition PSTH | mean across units",
            color=title_color,
            size="11pt",
        )
        if pre_ms > 0:
            self.plot_psth_lines.addItem(
                pg.LinearRegionItem(
                    values=(-pre_ms, 0.0),
                    orientation="vertical",
                    brush=pre_region_brush,
                    pen=pg.mkPen((0, 0, 0, 0)),
                    movable=False,
                )
            )
        self.plot_psth_lines.addItem(pg.InfiniteLine(pos=0.0, angle=90, movable=False, pen=zero_pen))

        heat_rows: list[np.ndarray] = []
        heat_labels: list[str] = []
        heat_meta: list[dict] = []
        heat_boundaries: list[int] = []
        mean_rows: list[dict] = []
        summary_parts: list[str] = []
        valid_conditions = 0

        for i, entry in enumerate(conditions):
            condition_name = str(entry.get("condition", f"cond_{i + 1}"))
            unit_ids = [int(v) for v in entry.get("unit_ids", units)]
            unit_trial_mats = [np.asarray(v, dtype=float) for v in entry.get("unit_trial_mats", [])]
            if not unit_trial_mats:
                continue
            trial_slice, trial_info = self._condition_trial_slice(int(entry.get("trial_count", 0)))
            used_trials = int(trial_info.get("used_trials", 0))
            if used_trials <= 0:
                continue

            color = palette[valid_conditions % len(palette)]
            valid_conditions += 1
            if single_unit:
                trial_mat = unit_trial_mats[0][trial_slice]
                if trial_mat.size == 0:
                    continue
                line_mean = np.nanmean(trial_mat, axis=0)
                line_sem = (
                    np.nanstd(trial_mat, axis=0, ddof=1) / math.sqrt(float(trial_mat.shape[0]))
                    if trial_mat.shape[0] > 1
                    else np.zeros(trial_mat.shape[1], dtype=float)
                )
                upper = pg.PlotCurveItem(t_ms, line_mean + line_sem, pen=pg.mkPen(color[0], color[1], color[2], 0))
                lower = pg.PlotCurveItem(
                    t_ms,
                    np.maximum(0.0, line_mean - line_sem),
                    pen=pg.mkPen(color[0], color[1], color[2], 0),
                )
                plot_item.addItem(upper)
                plot_item.addItem(lower)
                plot_item.addItem(
                    pg.FillBetweenItem(
                        upper,
                        lower,
                        brush=pg.mkBrush(color[0], color[1], color[2], 48 if self._plot_theme == "Dark" else 36),
                    )
                )
                legend_name = f"{condition_name} (n={trial_mat.shape[0]} trials)"
                for offset, row_values in enumerate(trial_mat):
                    actual_trial = int(trial_info["actual_start"]) + offset
                    display_label = f"T{actual_trial}" if len(conditions) == 1 else f"{condition_name} \u00b7 T{actual_trial}"
                    heat_rows.append(np.asarray(row_values, dtype=float))
                    heat_labels.append(display_label)
                    heat_meta.append(
                        {
                            "condition": condition_name,
                            "row_kind": "trial",
                            "unit_id": int(unit_ids[0]),
                            "trial_index": actual_trial,
                            "n_trials_averaged": 1,
                            "display_label": display_label,
                        }
                    )
                aggregate_count = int(trial_mat.shape[0])
                aggregate_mode = "trial_mean_sem"
            else:
                unit_rows: list[np.ndarray] = []
                for unit_id, unit_trial_mat in zip(unit_ids, unit_trial_mats):
                    visible_trials = np.asarray(unit_trial_mat[trial_slice], dtype=float)
                    if visible_trials.size == 0:
                        continue
                    unit_mean = np.nanmean(visible_trials, axis=0)
                    unit_rows.append(unit_mean)
                    display_label = str(int(unit_id)) if len(conditions) == 1 else f"{condition_name} \u00b7 {int(unit_id)}"
                    heat_rows.append(np.asarray(unit_mean, dtype=float))
                    heat_labels.append(display_label)
                    heat_meta.append(
                        {
                            "condition": condition_name,
                            "row_kind": "unit_average",
                            "unit_id": int(unit_id),
                            "trial_index": -1,
                            "n_trials_averaged": int(visible_trials.shape[0]),
                            "display_label": display_label,
                        }
                    )
                if not unit_rows:
                    continue
                unit_mat = np.vstack(unit_rows)
                line_mean = np.nanmean(unit_mat, axis=0)
                line_sem = None
                legend_name = f"{condition_name} (n={unit_mat.shape[0]} units)"
                aggregate_count = int(unit_mat.shape[0])
                aggregate_mode = "unit_mean"

            self.plot_psth_lines.plot(
                t_ms,
                line_mean,
                pen=pg.mkPen(color, width=2.4),
                name=legend_name,
            )
            mean_rows.extend(
                [
                    {
                        "condition": condition_name,
                        "time_ms": float(tv),
                        "mean_rate_hz": float(rv),
                        "sem_rate_hz": float(line_sem[idx]) if line_sem is not None else np.nan,
                        "aggregation": aggregate_mode,
                        "n_rows": aggregate_count,
                        "trial_start": int(trial_info["actual_start"]),
                        "trial_stop": int(trial_info["actual_stop"]),
                        "total_trials": int(trial_info["total_trials"]),
                    }
                    for idx, (tv, rv) in enumerate(zip(t_ms, line_mean))
                ]
            )
            heat_boundaries.append(len(heat_rows))
            summary_parts.append(
                f"{condition_name}: {int(trial_info['used_trials'])}/{int(trial_info['total_trials'])} trials"
            )

        if valid_conditions == 0:
            self.lbl_psth_summary.setText("Condition PSTH is ready after you compute it.")
            self.lbl_psth_summary_meta.setText("No conditions matched the selected trial range.")
            self._export_payloads["psth"] = []
            return

        heat_mat = np.vstack(heat_rows) if heat_rows else np.zeros((0, t_ms.size), dtype=float)
        heat_plot_item = self.plot_psth_heat.getPlotItem()
        self._style_plot_item(
            heat_plot_item,
            left="Trial" if single_unit else "Unit average",
            bottom="Time (ms) relative to event",
        )
        self.plot_psth_heat.setTitle(
            "Condition PSTH heatmap | trial rows" if single_unit else "Condition PSTH heatmap | unit-average rows",
            color=title_color,
            size="11pt",
        )
        if pre_ms > 0:
            self.plot_psth_heat.addItem(
                pg.LinearRegionItem(
                    values=(-pre_ms, 0.0),
                    orientation="vertical",
                    brush=pre_region_brush,
                    pen=pg.mkPen((0, 0, 0, 0)),
                    movable=False,
                )
            )
        if heat_mat.size:
            img = pg.ImageItem(heat_mat)
            cm = (
                pg.colormap.get("CET-L17")
                or pg.colormap.get("CET-L9" if self._plot_theme == "Dark" else "CET-L4")
            )
            if cm is not None:
                img.setLookupTable(cm.getLookupTable(nPts=256))
            vmax = float(np.nanpercentile(heat_mat, 99.0)) if np.isfinite(np.nanmax(heat_mat)) else 1.0
            img.setLevels((0.0, max(vmax, 1.0)))
            dt = float(np.median(np.diff(t_ms))) if t_ms.size > 1 else 1.0
            x0 = float(t_ms[0] - 0.5 * dt)
            img.setRect(QtCore.QRectF(x0, 0.0, float(dt * heat_mat.shape[1]), float(heat_mat.shape[0])))
            self.plot_psth_heat.addItem(img)
            self.plot_psth_heat.addItem(pg.InfiniteLine(pos=0.0, angle=90, movable=False, pen=zero_pen))
            for boundary in heat_boundaries[:-1]:
                self.plot_psth_heat.addItem(pg.InfiniteLine(pos=float(boundary), angle=0, movable=False, pen=zero_pen))
            self.plot_psth_heat.getViewBox().invertY(True)
            if heat_labels:
                stride = max(1, int(math.ceil(len(heat_labels) / 18.0)))
                ticks = [(float(i) + 0.5, heat_labels[i]) for i in range(0, len(heat_labels), stride)]
                self.plot_psth_heat.getAxis("left").setTicks([ticks])
        heat_export_rows: list[dict] = []
        for row_index, (row_meta, row_values) in enumerate(zip(heat_meta, heat_mat), start=1):
            heat_export_rows.extend(
                [
                    {
                        "row_index": int(row_index),
                        "condition": str(row_meta.get("condition", "")),
                        "row_kind": str(row_meta.get("row_kind", "")),
                        "display_label": str(row_meta.get("display_label", "")),
                        "unit_id": int(row_meta.get("unit_id", -1)),
                        "trial_index": int(row_meta.get("trial_index", -1)),
                        "n_trials_averaged": int(row_meta.get("n_trials_averaged", 0)),
                        "time_ms": float(tv),
                        "rate_hz": float(rv),
                    }
                    for tv, rv in zip(t_ms, row_values)
                ]
            )
        trial_mode_text = (
            "Heatmap rows show trials; lines show mean \u00b1 SEM across the selected trials."
            if single_unit
            else "Heatmap rows show unit averages across the selected trials; lines show the mean across units."
        )
        requested_range_text = self.lbl_psth_trial_status.text().replace("Trial filter: ", "").strip()
        self.lbl_psth_summary.setText(
            f"Condition PSTH \u00b7 {len(units)} selected unit{'s' if len(units) != 1 else ''} \u00b7 {requested_range_text}"
        )
        self.lbl_psth_summary_meta.setText(
            f"{trial_mode_text}  {'  |  '.join(summary_parts)}"
        )
        self._export_payloads["psth"] = [
            ("condition_psth_average.csv", pd.DataFrame(mean_rows)),
            ("condition_psth_heatmap.csv", pd.DataFrame(heat_export_rows)),
        ]
        self.view_tabs.setCurrentIndex(3)

    def _visualize_network(self, r: Dict[str, object]) -> None:
        self.plot_net_matrix.clear()
        self.plot_net_sync.clear()
        mat = np.asarray(r.get("mat", []), dtype=float)
        order = np.asarray(r.get("order", []), dtype=int)
        sorted_units = np.asarray(r.get("sorted_units", []), dtype=int)
        group_labels = np.asarray(r.get("group_labels", []), dtype=int)
        t_s = np.asarray(r.get("t_s", []), dtype=float)
        idx = np.asarray(r.get("idx", []), dtype=float)
        if mat.size:
            if (
                order.size == mat.shape[0]
                and sorted_units.size == mat.shape[0]
                and group_labels.size == mat.shape[0]
            ):
                display_mat = mat[np.ix_(order, order)]
            else:
                display_mat = mat
                sorted_units = np.asarray(self._selected_units(), dtype=int)
                group_labels = np.zeros(display_mat.shape[0], dtype=int)
            self.plot_net_matrix.setTitle("Pairwise CCG matrix | synced units clustered row-wise")
            img = pg.ImageItem(display_mat)
            img.setRect(QtCore.QRectF(0, 0, display_mat.shape[1], display_mat.shape[0]))
            cm = pg.colormap.get("CET-L9" if self._plot_theme == "Dark" else "CET-L4")
            if cm is not None:
                img.setLookupTable(cm.getLookupTable(nPts=256))
            self.plot_net_matrix.addItem(img)
            n_rows = int(display_mat.shape[0])
            n_cols = int(display_mat.shape[1])
            for row, group_id in enumerate(group_labels):
                gid = int(group_id)
                if gid <= 0:
                    continue
                stripe = QtWidgets.QGraphicsRectItem(0, row, n_cols, 1.0)
                stripe.setBrush(pg.mkBrush(_sync_group_color(gid, alpha=38)))
                stripe.setPen(pg.mkPen((0, 0, 0, 0)))
                stripe.setZValue(5)
                self.plot_net_matrix.addItem(stripe)
                bar = QtWidgets.QGraphicsRectItem(-0.55, row, 0.35, 1.0)
                bar.setBrush(pg.mkBrush(_sync_group_color(gid, alpha=235)))
                bar.setPen(pg.mkPen((0, 0, 0, 0)))
                bar.setZValue(6)
                self.plot_net_matrix.addItem(bar)
            for gid in sorted(int(v) for v in np.unique(group_labels) if int(v) > 0):
                rows = np.flatnonzero(group_labels == gid)
                if rows.size == 0:
                    continue
                label = pg.TextItem(f"G{gid}", color=_sync_group_color(gid, alpha=255), anchor=(1.0, 0.5))
                label.setPos(-0.64, float(rows.mean()) + 0.5)
                label.setZValue(7)
                self.plot_net_matrix.addItem(label)
            unit_ticks = [(i + 0.5, str(int(unit))) for i, unit in enumerate(sorted_units)]
            if len(unit_ticks) <= 60:
                self.plot_net_matrix.getAxis("left").setTicks([unit_ticks])
                self.plot_net_matrix.getAxis("bottom").setTicks([unit_ticks])
            else:
                self.plot_net_matrix.getAxis("left").setTicks([])
                self.plot_net_matrix.getAxis("bottom").setTicks([])
            self.plot_net_matrix.setLabel("left", "Unit (clustered)")
            self.plot_net_matrix.setLabel("bottom", "Unit (clustered)")
            self.plot_net_matrix.setXRange(-0.75, n_cols, padding=0.02)
            self.plot_net_matrix.setYRange(0, n_rows, padding=0.02)
            self.plot_net_matrix.getViewBox().invertY(True)
        if t_s.size:
            self.plot_net_sync.plot(t_s, idx, pen=pg.mkPen((180, 120, 255), width=1.7))
        net_rows = []
        if mat.size:
            if order.size == mat.shape[0] and sorted_units.size == mat.shape[0]:
                export_order = order
                export_units = sorted_units
                export_groups = group_labels
                export_mat = mat[np.ix_(export_order, export_order)]
            else:
                export_order = np.arange(mat.shape[0], dtype=int)
                export_units = np.asarray(self._selected_units(), dtype=int)
                export_groups = np.zeros(mat.shape[0], dtype=int)
                export_mat = mat
            for i in range(export_mat.shape[0]):
                for j in range(export_mat.shape[1]):
                    net_rows.append(
                        {
                            "row": int(i),
                            "col": int(j),
                            "unit_a": int(export_units[i]) if i < export_units.size else int(i),
                            "unit_b": int(export_units[j]) if j < export_units.size else int(j),
                            "group_a": int(export_groups[i]) if i < export_groups.size else 0,
                            "group_b": int(export_groups[j]) if j < export_groups.size else 0,
                            "value": float(export_mat[i, j]),
                        }
                    )
            group_rows = [
                {
                    "sorted_row": int(i),
                    "source_row": int(export_order[i]) if i < export_order.size else int(i),
                    "unit_id": int(export_units[i]) if i < export_units.size else int(i),
                    "sync_group": int(export_groups[i]) if i < export_groups.size else 0,
                    "sync_group_threshold": float(r.get("sync_group_threshold", np.nan)),
                }
                for i in range(export_units.size)
            ]
        else:
            group_rows = []
        sync_rows = [{"time_s": float(tv), "synchrony_index": float(iv)} for tv, iv in zip(t_s, idx)] if t_s.size else []
        self._export_payloads["network"] = [
            ("network_matrix.csv", pd.DataFrame(net_rows)),
            ("network_sync_groups.csv", pd.DataFrame(group_rows)),
            ("network_synchrony.csv", pd.DataFrame(sync_rows)),
        ]
        self.view_tabs.setCurrentIndex(4)

    def _show_npyx_corr(self) -> None:
        self.gl_npyx.clear()
        units = self._selected_units()
        if not units:
            return
        key = self.cb_npyx_method.currentData()
        if not key:
            return
        if self.dataset is None:
            return
        dp = str(self.dataset.ks_folder)
        try:
            self._busy = True
            self._set_progress(10)
            res = run_method(
                str(key),
                dp,
                units,
                bin_ms=float(self.sp_npyx_bin.value()),
                win_ms=float(self.sp_npyx_win.value()),
                params=self._collect_npyx_params(),
            )
            self._set_progress(100)
        except Exception as exc:
            self._log(f"Advanced corr error ({key}): {exc}")
            self._busy = False
            return
        finally:
            self._busy = False

        requested_dp = str(res.get("requested_dp", ""))
        resolved_dp = str(res.get("resolved_dp", ""))
        if requested_dp and resolved_dp and requested_dp != resolved_dp:
            self._log(f"Advanced corr datapath fallback: {requested_dp} -> {resolved_dp}")

        kind = str(res.get("kind", "text"))
        title = str(res.get("title", str(key)))
        cmap_name = "CET-L9" if self._plot_theme == "Dark" else "CET-L4"
        if kind in {"line", "hist"}:
            plot = self.gl_npyx.addPlot(row=0, col=0, title=title)
            self._style_plot_item(plot, left="value", bottom="x")
            x = np.asarray(res.get("x", []), dtype=float)
            y = np.asarray(res.get("y", []), dtype=float)
            if x.size and y.size:
                if kind == "hist":
                    w = np.asarray(res.get("w", np.ones_like(x)), dtype=float)
                    plot.addItem(pg.BarGraphItem(x=x, height=y, width=w, brush=(120, 190, 255, 160)))
                else:
                    plot.plot(x, y, pen=pg.mkPen((80, 150, 255), width=2))
            self._export_payloads["npyx"] = [
                ("npyx_corr_line.csv", pd.DataFrame({"x": x, "y": y, "method": [key] * len(x)}))
            ]
        elif kind == "multi_line":
            plot = self.gl_npyx.addPlot(row=0, col=0, title=title)
            self._style_plot_item(plot, left="value", bottom="x")
            x = np.asarray(res.get("x", []), dtype=float)
            series = res.get("series", [])
            out_rows: list[dict] = []
            if isinstance(series, list):
                for i, s in enumerate(series):
                    if not isinstance(s, dict):
                        continue
                    name = str(s.get("name", f"series_{i+1}"))
                    y = np.asarray(s.get("y", []), dtype=float)
                    if x.size and y.size:
                        n = min(x.size, y.size)
                        xi = x[:n]
                        yi = y[:n]
                        plot.plot(xi, yi, pen=pg.mkPen(pg.intColor(i, hues=max(len(series), 4)), width=2), name=name)
                        out_rows.extend([{"x": float(xx), "y": float(yy), "series": name} for xx, yy in zip(xi, yi)])
            self._export_payloads["npyx"] = [("npyx_corr_multi_line.csv", pd.DataFrame(out_rows))]
        elif kind == "image":
            plot = self.gl_npyx.addPlot(row=0, col=0, title=title)
            self._style_plot_item(plot, left="row", bottom="col")
            mat = np.asarray(res.get("mat", []), dtype=float)
            if mat.size:
                img = pg.ImageItem(mat)
                if mat.ndim == 2:
                    img.setRect(QtCore.QRectF(0, 0, float(mat.shape[1]), float(mat.shape[0])))
                cm = pg.colormap.get(cmap_name)
                if cm is not None:
                    img.setLookupTable(cm.getLookupTable(nPts=256))
                plot.addItem(img)
                plot.getViewBox().invertY(True)
                self._annotate_image_values(plot, mat)
            self._export_payloads["npyx"] = [("npyx_corr_matrix.csv", pd.DataFrame(mat))]
        elif kind == "corr_pairs":
            items = res.get("items", [])
            if not isinstance(items, list) or not items:
                return
            rows, cols = self._subplot_shape(len(items))
            out_rows: list[dict] = []
            for i, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                u1 = int(item.get("u1", -1))
                u2 = int(item.get("u2", -1))
                x = np.asarray(item.get("x", []), dtype=float)
                y = np.asarray(item.get("y", []), dtype=float)
                if x.size == 0 or y.size == 0:
                    continue
                n = min(x.size, y.size)
                x = x[:n]
                y = y[:n]
                sig = bool(item.get("significant", False))
                score = item.get("score", None)
                score_txt = f" | {float(score):.2f}" if score is not None and np.isfinite(float(score)) else ""
                p = self.gl_npyx.addPlot(row=i // cols, col=i % cols, title=f"{u1} vs {u2}{' *' if sig else ''}{score_txt}")
                self._style_plot_item(p, left="value", bottom="lag (ms)")
                p.plot(x, y, pen=pg.mkPen((255, 120, 120) if sig else (80, 150, 255), width=2))
                p.addItem(pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen((150, 150, 150), width=1, style=QtCore.Qt.DashLine)))
                out_rows.extend([{"unit_a": u1, "unit_b": u2, "lag_ms": float(xx), "value": float(yy), "significant": int(sig)} for xx, yy in zip(x, y)])
            self._export_payloads["npyx"] = [("npyx_corr_pairs.csv", pd.DataFrame(out_rows))]
        elif kind == "pair_bars":
            labels = list(res.get("labels", []))
            vals = np.asarray(res.get("values", []), dtype=float)
            traces = res.get("traces", [])
            if vals.size:
                top = self.gl_npyx.addPlot(row=0, col=0, title=f"{title} (pairs)")
                self._style_plot_item(top, left="value", bottom="pair index")
                xs = np.arange(vals.size, dtype=float)
                top.addItem(pg.BarGraphItem(x=xs, height=vals, width=0.7, brush=(100, 180, 255, 180)))
                if vals.size > 1:
                    m = float(np.nanmean(vals))
                    sem = float(np.nanstd(vals, ddof=1) / np.sqrt(vals.size))
                    top.addItem(pg.InfiniteLine(pos=m, angle=0, pen=pg.mkPen((255, 140, 80), width=2)))
                    top.addItem(pg.InfiniteLine(pos=m + sem, angle=0, pen=pg.mkPen((255, 140, 80, 120), width=1)))
                    top.addItem(pg.InfiniteLine(pos=m - sem, angle=0, pen=pg.mkPen((255, 140, 80, 120), width=1)))
                for i, lbl in enumerate(labels):
                    txt = pg.TextItem(text=str(lbl), anchor=(0.5, 0), color=(230, 230, 230) if self._plot_theme == "Dark" else (20, 20, 20))
                    txt.setPos(float(i), float(vals[i]))
                    top.addItem(txt)
            if isinstance(traces, list) and traces:
                rows, cols = self._subplot_shape(min(len(traces), 9))
                for i, tr in enumerate(traces[: rows * cols]):
                    if not isinstance(tr, dict):
                        continue
                    x = np.asarray(tr.get("x", []), dtype=float)
                    y = np.asarray(tr.get("y", []), dtype=float)
                    if x.size == 0 or y.size == 0:
                        continue
                    n = min(x.size, y.size)
                    p = self.gl_npyx.addPlot(row=1 + i // cols, col=i % cols, title=str(tr.get("name", f"pair_{i+1}")))
                    self._style_plot_item(p, left="CCG", bottom="lag (ms)")
                    p.plot(x[:n], y[:n], pen=pg.mkPen(pg.intColor(i, hues=max(len(traces), 4)), width=1.5))
                    p.addItem(pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen((150, 150, 150), width=1, style=QtCore.Qt.DashLine)))
            self._export_payloads["npyx"] = [("npyx_pair_values.csv", pd.DataFrame({"pair": labels, "value": vals}))]
        elif kind == "scalar":
            plot = self.gl_npyx.addPlot(row=0, col=0, title=title)
            self._style_plot_item(plot, left="value", bottom="")
            y = float(res.get("value", np.nan))
            plot.addItem(pg.BarGraphItem(x=[0.0], height=[y], width=0.6, brush=(120, 190, 255, 160)))
            self._export_payloads["npyx"] = [("npyx_corr_scalar.csv", pd.DataFrame({"method": [key], "value": [y]}))]
        else:
            plot = self.gl_npyx.addPlot(row=0, col=0, title=title)
            self._style_plot_item(plot)
            txt = str(res.get("text", "No output"))
            label = pg.TextItem(text=txt, anchor=(0, 1), color=(200, 200, 200) if self._plot_theme == "Dark" else (30, 30, 30))
            label.setPos(0, 1)
            plot.addItem(label)
            self._export_payloads["npyx"] = [("npyx_corr_text.csv", pd.DataFrame({"method": [key], "text": [txt]}))]
        self.view_tabs.setCurrentIndex(5)

    def _restore_settings(self) -> None:
        folder = self.settings.value("post/last_folder", "")
        if folder:
            self.ed_folder.setText(str(folder))

    def _log(self, msg: str) -> None:
        self.log.appendPlainText(str(msg))

    def _export_plotted_data(self) -> None:
        idx = self.view_tabs.currentIndex()
        key = ["basic", "raw", "corr", "psth", "network", "npyx"][idx] if 0 <= idx < 6 else ""
        payloads = self._export_payloads.get(key, [])
        if not payloads:
            self._log("Export: no plotted data available for current page.")
            return
        start = self.settings.value("paths/last_folder", str(Path.cwd()))
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select export folder", str(start))
        if not folder:
            return
        root = Path(folder)
        for name, df in payloads:
            if df is None or df.empty:
                continue
            df.to_csv(root / name, index=False)
        self._log(f"Exported plotted data for '{key}' to {root}")

    def _prompt_unit_export_scope(self) -> Optional[bool]:
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Export units to H5")
        box.setText("Choose which units to export.")
        box.setInformativeText(
            f"'Good units only' uses the current good-unit source: {self.cb_good_source.currentText().strip()}."
        )
        all_button = box.addButton("All units", QtWidgets.QMessageBox.AcceptRole)
        good_button = box.addButton("Good units only", QtWidgets.QMessageBox.AcceptRole)
        box.addButton(QtWidgets.QMessageBox.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is all_button:
            return False
        if clicked is good_button:
            return True
        return None

    def _good_unit_ids(self) -> set[int]:
        return {int(u) for u in self._all_units if self._unit_is_good(int(u))}

    def _export_units_file(self) -> None:
        if self.dataset is None:
            self._log("Unit export: load a dataset first.")
            return

        good_only = self._prompt_unit_export_scope()
        if good_only is None:
            return

        good_unit_ids = self._good_unit_ids()
        units = [int(u) for u in self._all_units if (u in good_unit_ids or not good_only)]
        if not units:
            self._log("Unit export: no units matched the requested scope.")
            return

        if good_only and not any(not df.empty for df in self.label_sources.values()):
            self._log("Unit export: no unit labels were found, so 'good units only' resolves to all units.")

        start = Path(str(self.settings.value("post/last_folder", str(self.dataset.ks_folder))))
        suffix = "good_units" if good_only else "all_units"
        default_path = start / f"{self.dataset.ks_folder.name}_{suffix}.h5"
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export units to H5",
            str(default_path),
            "HDF5 files (*.h5 *.hdf5)",
        )
        if not file_path:
            return

        out_path = Path(file_path)
        if out_path.suffix.lower() not in {".h5", ".hdf5"}:
            out_path = out_path.with_suffix(".h5")

        export_mode = "good_only" if good_only else "all"
        good_source = self.cb_good_source.currentText().strip()
        self._busy = True
        self._set_progress(0)
        self._log(
            f"Unit export: writing {len(units)} units to {out_path} "
            f"(mode={export_mode}, good_source={good_source})."
        )
        try:
            export_units_h5(
                dataset=self.dataset,
                output_path=out_path,
                units=units,
                labels_df=self.labels_df,
                metrics_df=self.metrics_df,
                label_sources=self.label_sources,
                good_units=good_unit_ids,
                good_source=good_source,
                export_mode=export_mode,
                progress_callback=self._set_progress,
            )
            self.settings.setValue("paths/last_folder", str(out_path.parent))
            self.settings.setValue("post/last_folder", str(out_path.parent))
            self._set_progress(100)
            self._log(f"Unit export: wrote {len(units)} unit groups to {out_path}")
        except Exception as exc:
            self._log(f"Unit export failed: {exc}")
        finally:
            self._busy = False

    def is_busy(self) -> bool:
        return bool(self._busy)



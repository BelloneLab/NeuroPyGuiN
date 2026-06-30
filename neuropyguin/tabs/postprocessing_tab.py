"""Post-processing tab for the NeuroPyGuiN GUI.

Provides the "Post Processing" page where a curated Kilosort/SpikeGLX dataset is
loaded and explored: unit basics (raster, ACG, ISI, waveform), a raw-signal
explorer, correlograms, condition PSTHs, network synchrony, and the advanced
correlation methods exposed through the npyx bridge. Each analysis page also
records the plotted values so they can be exported to CSV.
"""

from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from ..postproc_events import (
    ALIGNMENT_OPTIONS as EVENT_ALIGNMENT_OPTIONS,
    inspect_event_csv,
    load_event_times,
)
from ..postproc_engine import NeuropixelsDataset, export_units_h5
from ..npyx_corr_bridge import PAIRWISE_ONLY_METHODS, method_metadata, method_options, run_method
from ..npyx_figures import NpyxFigureView, acg_grid_figure, ccg_grid_figure, waveform_figure


def _is_bombcell_good_label(value: object) -> bool:
    """Return True if a Bombcell label string denotes a usable ("good") unit.

    Handles the various spellings produced by Bombcell (for example "good",
    "non-soma", "non_soma_good") by lower-casing and collapsing separators.
    """
    text = str(value).strip().lower()
    if not text or text == "nan":
        return False
    normalized = text.replace("-", "_").replace(" ", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized in {"good", "non_soma", "non_soma_good", "nonsoma", "nonsomagood"}


def _export_good_unit_figures(dataset, good_units, out_dir, dark=False, progress_cb=None) -> dict:
    """Export per-unit waveform + ACG figures for every good unit (PNG + one PDF).

    For each unit in ``good_units`` this builds the npyx-style two-panel card via
    :func:`neuropyguin.unit_figures.unit_waveform_acg_figure`, writes it as
    ``out_dir/unit_<id>_waveform_acg.png`` (dpi 150) and appends the same figure
    as a page of a single multi-page PDF ``out_dir/good_units_waveform_acg.pdf``.
    Each figure is closed immediately after use so memory stays bounded across
    the (potentially ~77-unit) batch.

    Designed to run off the GUI thread (it reads the AP binary per unit for the
    +/-SEM waveform, so it takes minutes). ``progress_cb(i, n)`` is invoked
    periodically with the number of units processed so far and the total.

    Returns a dict ``{"n", "pdf", "png_dir", "error"}``. Per-unit render/save
    errors are caught and logged into the result's ``error`` only when they are
    fatal (e.g. the figure module is missing); individual failing units are
    skipped so the batch keeps going.
    """
    # Lazy imports so a momentarily-missing dependency yields a clean error dict
    # rather than crashing the worker / import of this module.
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        from .. import unit_figures
    except Exception as exc:  # noqa: BLE001
        return {"n": 0, "pdf": None, "png_dir": str(out_dir), "error": f"export dependencies unavailable: {exc}"}

    out_path = Path(out_dir)
    units = [int(u) for u in good_units]
    n = len(units)
    if n == 0:
        return {"n": 0, "pdf": None, "png_dir": str(out_path), "error": "No good units to export."}

    try:
        out_path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        return {"n": 0, "pdf": None, "png_dir": str(out_path), "error": f"Cannot create output folder: {exc}"}

    pdf_path = out_path / "good_units_waveform_acg.pdf"
    written = 0
    try:
        with PdfPages(str(pdf_path)) as pdf:
            for i, unit in enumerate(units):
                try:
                    fig = unit_figures.unit_waveform_acg_figure(dataset, unit, dark=bool(dark))
                except Exception:
                    # A single bad unit must not abort the whole batch.
                    if progress_cb is not None:
                        try:
                            progress_cb(i + 1, n)
                        except Exception:
                            pass
                    continue
                try:
                    fig.savefig(str(out_path / f"unit_{int(unit)}_waveform_acg.png"), dpi=150)
                    pdf.savefig(fig)
                    written += 1
                except Exception:
                    pass
                finally:
                    plt.close(fig)
                if progress_cb is not None:
                    try:
                        progress_cb(i + 1, n)
                    except Exception:
                        pass
    except Exception as exc:  # noqa: BLE001
        return {"n": written, "pdf": str(pdf_path), "png_dir": str(out_path), "error": f"{type(exc).__name__}: {exc}"}

    return {"n": written, "pdf": str(pdf_path), "png_dir": str(out_path), "error": None}


class PostProcessingTab(QtWidgets.QWidget):
    """Qt widget that hosts the post-processing analysis pages.

    Owns the loaded dataset plus the per-page controls and plot widgets, and
    coordinates loading, plotting, and CSV/H5 export. Analysis pages are kept in
    sync with their plot views via the analysis-tab index.
    """

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
        self._plot_theme = 'Light'
        self._show_grid = True
        self._busy = False
        self._c4_running = False
        self._exporting_waveforms = False
        self._plot_detached = False
        self._plot_dialog: Optional[QtWidgets.QDialog] = None
        self._right_panel_layout: Optional[QtWidgets.QVBoxLayout] = None
        self._analysis_area_layout: Optional[QtWidgets.QVBoxLayout] = None
        self._figure_row_layout: Optional[QtWidgets.QHBoxLayout] = None
        self._analysis_area: Optional[QtWidgets.QWidget] = None
        self._body_splitter: Optional[QtWidgets.QSplitter] = None
        self._right_panel: Optional[QtWidgets.QWidget] = None
        self._settings_visible: bool = True
        self._body_sizes_before_detach: list[int] = []
        self._build_ui()
        self._restore_settings()

    def _build_ui(self) -> None:
        main = QtWidgets.QVBoxLayout(self)
        main.setContentsMargins(10, 10, 10, 10)
        main.setSpacing(8)
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
        self.btn_export_waveforms = QtWidgets.QPushButton("Export waveforms")
        self.btn_export_waveforms.setToolTip(
            "Export every good unit's waveform + ACG figure to PNGs and one multi-page PDF "
            "(runs off the GUI thread; can take a few minutes)."
        )
        self.btn_detach_plots = QtWidgets.QPushButton("Detach plots")
        self.btn_detach_plots.setCheckable(True)
        self.btn_browse.setProperty("role", "secondary")
        self.btn_load.setProperty("role", "primary")
        self.btn_export.setProperty("role", "ghost")
        self.btn_export_units.setProperty("role", "secondary")
        self.btn_export_waveforms.setProperty("role", "secondary")
        self.btn_detach_plots.setProperty("role", "ghost")
        top.addWidget(self.ed_folder, 1)
        top.addWidget(self.btn_browse)
        top.addWidget(self.btn_load)
        top.addWidget(self.btn_export)
        top.addWidget(self.btn_export_units)
        top.addWidget(self.btn_export_waveforms)
        top.addWidget(self.btn_detach_plots)

        body = QtWidgets.QSplitter()
        self._body_splitter = body

        # LEFT: a persistent, tall Units inspector. This is its own splitter
        # child (no longer stacked above the controls), so the unit list keeps
        # the full sidebar height and many units stay visible at once.
        grp_units = QtWidgets.QGroupBox("Units")
        u_l = QtWidgets.QVBoxLayout(grp_units)
        u_l.setSpacing(8)
        unit_filter_row = QtWidgets.QHBoxLayout()
        unit_filter_row.setSpacing(6)
        self.ed_unit_filter = QtWidgets.QLineEdit()
        self.ed_unit_filter.setPlaceholderText("Filter unit id")
        self.btn_good_only = QtWidgets.QPushButton("Good only")
        self.btn_good_only.setToolTip("Show only units labelled good by the selected source.")
        self.btn_good_only.setCheckable(True)
        self.cb_good_source = QtWidgets.QComboBox()
        self.cb_good_source.addItems(["Auto", "Bombcell", "Phy", "KSLabel"])
        unit_filter_row.addWidget(self.ed_unit_filter, 1)
        unit_filter_row.addWidget(self.btn_good_only)
        good_source_row = QtWidgets.QHBoxLayout()
        good_source_row.setSpacing(6)
        good_source_row.addWidget(QtWidgets.QLabel("Good source"))
        good_source_row.addWidget(self.cb_good_source, 1)

        self.list_units = QtWidgets.QListWidget()
        self.list_units.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.list_units.setUniformItemSizes(True)
        # A tall list is the whole point of the redesign: let it claim the
        # sidebar height and never collapse below a generous minimum.
        self.list_units.setMinimumHeight(360)
        self.list_units.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        # Compact rows so many units are visible at once (the app-wide list style
        # pads rows heavily for nav lists; the unit list wants density instead).
        self.list_units.setStyleSheet(
            "QListWidget::item { padding: 2px 8px; margin: 0px; border-radius: 6px; }"
        )

        self.lbl_units_count = QtWidgets.QLabel("No dataset loaded.")
        self.lbl_units_count.setObjectName("psthMetaLabel")

        self.tbl_unit_quality = QtWidgets.QTableWidget(0, 2)
        self.tbl_unit_quality.setAlternatingRowColors(True)
        self.tbl_unit_quality.setHorizontalHeaderLabels(["Metric", "Value"])
        self.tbl_unit_quality.horizontalHeader().setStretchLastSection(True)
        self.tbl_unit_quality.verticalHeader().setVisible(False)
        self.tbl_unit_quality.setMaximumHeight(180)

        u_l.addLayout(unit_filter_row)
        u_l.addLayout(good_source_row)
        u_l.addWidget(self.lbl_units_count, 0)
        u_l.addWidget(self.list_units, 1)
        u_l.addWidget(QtWidgets.QLabel("Selected unit metrics"), 0)
        u_l.addWidget(self.tbl_unit_quality, 0)
        # Keep the Units sidebar narrow so the figures stay dominant.
        grp_units.setMinimumWidth(280)
        grp_units.setMaximumWidth(440)

        # RIGHT: the analysis area separates NAVIGATION from SETTINGS.
        #   analysis_tabs : slim, always-visible nav tab bar (empty pages). It is
        #                   the authoritative "current analysis" index and drives
        #                   both the figure stack and the settings stack.
        #   view_tabs     : the stacked figure pages (tab bar hidden).
        #   settings_stack: the per-analysis control forms, docked in a
        #                   collapsible panel on the right of the figure.
        analysis_area = QtWidgets.QWidget()
        analysis_area_l = QtWidgets.QVBoxLayout(analysis_area)
        analysis_area_l.setContentsMargins(0, 0, 0, 0)
        analysis_area_l.setSpacing(8)

        self.analysis_tabs = QtWidgets.QTabWidget()
        self.analysis_tabs.setDocumentMode(True)
        # Nav only: pages are empty, the bar just selects the analysis. Cap the
        # height so the empty pane never steals room from the figures.
        self.analysis_tabs.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.analysis_tabs.setMaximumHeight(52)

        # The settings forms live here, one page per analysis, kept in sync with
        # the nav index.
        self.settings_stack = QtWidgets.QStackedWidget()

        def _compact_form(form: QtWidgets.QFormLayout) -> QtWidgets.QFormLayout:
            """Tighten a controls form so the settings panel stays compact."""
            form.setContentsMargins(6, 6, 6, 6)
            form.setVerticalSpacing(6)
            form.setHorizontalSpacing(8)
            form.setLabelAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
            return form

        def _scroll(widget: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
            area = QtWidgets.QScrollArea()
            area.setWidgetResizable(True)
            area.setFrameShape(QtWidgets.QFrame.NoFrame)
            area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            area.setWidget(widget)
            return area

        def _add_settings_page(widget: QtWidgets.QWidget, title: str) -> None:
            """Add an empty nav tab + the matching settings page (kept in sync)."""
            self.analysis_tabs.addTab(QtWidgets.QWidget(), title)
            self.settings_stack.addWidget(_scroll(widget))

        t_basic = QtWidgets.QWidget()
        f_basic = _compact_form(QtWidgets.QFormLayout(t_basic))
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
        f_basic.addRow("IFR smooth", with_help(self.sp_ifr_smooth_ms, "Bin/smoothing window for instantaneous firing rate (ms)."))
        f_basic.addRow(self.ck_ifr)
        _add_settings_page(t_basic, "Unit Basics")

        t_raw = QtWidgets.QWidget()
        f_raw = _compact_form(QtWidgets.QFormLayout(t_raw))
        self.sp_raw_t0 = QtWidgets.QDoubleSpinBox()
        self.sp_raw_t0.setRange(0.0, 2e6)
        self.sp_raw_t0.setValue(0.0)
        self.sp_raw_t0.setSuffix(" s")
        self.sp_raw_dur = QtWidgets.QDoubleSpinBox()
        self.sp_raw_dur.setRange(0.05, 60.0)
        self.sp_raw_dur.setValue(0.15)
        self.sp_raw_dur.setSuffix(" s")
        self.sp_raw_ch = QtWidgets.QSpinBox()
        self.sp_raw_ch.setRange(4, 256)
        self.sp_raw_ch.setValue(32)
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
        _add_settings_page(t_raw, "Raw Explorer")

        t_corr = QtWidgets.QWidget()
        f_corr = _compact_form(QtWidgets.QFormLayout(t_corr))
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
        self.cb_corr_norm = QtWidgets.QComboBox()
        self.cb_corr_norm.addItems(["Hertz", "Counts", "Pearson", "zscore"])
        self.cb_corr_style = QtWidgets.QComboBox()
        self.cb_corr_style.addItems(["bar", "line"])
        f_corr.addRow("Mode", with_help(self.cb_corr_mode, "ACG: per-unit autocorrelogram grid. CCG: NeuroPyxels grid with ACGs on the diagonal and cross-correlograms off-diagonal."))
        f_corr.addRow("Normalize", with_help(self.cb_corr_norm, "y-axis unit (NeuroPyxels convention): Hertz (firing rate), Counts, Pearson, or zscore. The CCG grid uses Hertz ACGs + z-scored CCGs ('mixte') when Hertz is chosen."))
        f_corr.addRow("Style", with_help(self.cb_corr_style, "Histogram (bar) or line correlograms, as in NeuroPyxels."))
        f_corr.addRow("Bin", with_help(self.sp_corr_bin, "Correlogram bin size (ms)."))
        f_corr.addRow("Window", with_help(self.sp_corr_win, "Full window around zero lag (ms)."))
        self.lbl_corr_note = QtWidgets.QLabel("Rendered with NeuroPyxels (npyx.plot). First render caches into the dataset folder.")
        self.lbl_corr_note.setObjectName("psthMetaLabel")
        self.lbl_corr_note.setWordWrap(True)
        f_corr.addRow(self.lbl_corr_note)
        _add_settings_page(t_corr, "Correlogram")
        t_psth = QtWidgets.QWidget()
        v_psth = QtWidgets.QVBoxLayout(t_psth)
        v_psth.setContentsMargins(6, 4, 6, 4)
        v_psth.setSpacing(6)
        self.lbl_psth_hint = QtWidgets.QLabel(
            "Single unit: per-trial rows + mean \u00b1 SEM across trials. "
            "Multiple units: 'Average across units' shows one mean trace; 'Per-unit panels' shows one panel per unit."
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
        self.btn_cond_add_all = QtWidgets.QPushButton("Add all behaviors")
        self.btn_cond_add_all.setToolTip(
            "Pick a binary behavior-matrix CSV and create one PSTH condition per behavior column "
            "(behaviors with no events are skipped)."
        )
        b_cond.addWidget(self.btn_cond_add)
        b_cond.addWidget(self.btn_cond_remove)
        b_cond.addWidget(self.btn_cond_browse)
        b_cond.addWidget(self.btn_cond_add_all)
        b_cond.addStretch(1)

        f_psth = _compact_form(QtWidgets.QFormLayout())
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
        self.cb_psth_align = QtWidgets.QComboBox()
        self.cb_psth_align.addItems(list(EVENT_ALIGNMENT_OPTIONS))
        self.sp_psth_fps = QtWidgets.QDoubleSpinBox()
        self.sp_psth_fps.setRange(1.0, 10000.0)
        self.sp_psth_fps.setDecimals(2)
        self.sp_psth_fps.setValue(30.0)
        self.sp_psth_fps.setSuffix(" fps")
        self.ck_psth_baseline = QtWidgets.QCheckBox("Baseline-subtract (pre-window mean)")
        self.cb_psth_mode = QtWidgets.QComboBox()
        self.cb_psth_mode.addItems(["Average across units", "Per-unit panels"])
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
        f_psth.addRow("Event alignment", with_help(self.cb_psth_align, "For a binary behavior matrix: align to bout onset (rising 0->1), offset (falling 1->0), or bout midpoint."))
        f_psth.addRow("Frame rate", with_help(self.sp_psth_fps, "Frame rate (fps) used to convert behavior frames to seconds when the CSV has no time column. Ignored when a time column is present."))
        f_psth.addRow("Display mode", with_help(self.cb_psth_mode, "With multiple units selected: 'Average across units' shows one mean trace; 'Per-unit panels' shows one panel per unit. Single-unit selections show the per-trial view either way."))
        f_psth.addRow(self.ck_psth_baseline)
        f_psth.addRow("Trial range", with_help(trial_row, "1-based inclusive trial range within each selected event label. 'last' uses the final available trial."))
        f_psth.addRow(psth_btn_row)

        v_psth.addWidget(self.lbl_psth_hint)
        v_psth.addWidget(self.tbl_conditions)
        v_psth.addLayout(b_cond)
        v_psth.addLayout(f_psth)
        v_psth.addWidget(self.lbl_psth_trial_status)
        _add_settings_page(t_psth, "Condition PSTH")

        t_net = QtWidgets.QWidget()
        f_net = _compact_form(QtWidgets.QFormLayout(t_net))
        # sp_net_bin is repurposed as the spike-count correlation bin (ms).
        self.sp_net_bin = QtWidgets.QDoubleSpinBox()
        self.sp_net_bin.setRange(1.0, 500.0)
        self.sp_net_bin.setValue(25.0)
        self.sp_net_bin.setSuffix(" ms")
        self.sp_net_z = QtWidgets.QDoubleSpinBox()
        self.sp_net_z.setRange(1.0, 20.0)
        self.sp_net_z.setSingleStep(0.5)
        self.sp_net_z.setValue(5.0)
        self.btn_net_compute = QtWidgets.QPushButton("Compute")
        self.btn_net_show = QtWidgets.QPushButton("Show")
        net_btn_row = QtWidgets.QHBoxLayout()
        net_btn_row.addWidget(self.btn_net_compute)
        net_btn_row.addWidget(self.btn_net_show)
        f_net.addRow("Correlation bin", with_help(self.sp_net_bin, "Bin size (ms) for the pairwise spike-count (noise) correlation matrix."))
        f_net.addRow("Connection z-threshold", with_help(self.sp_net_z, "Z-score threshold for flagging putative short-latency CCG connections."))
        f_net.addRow(net_btn_row)
        _add_settings_page(t_net, "Network")

        t_npyx = QtWidgets.QWidget()
        f_npyx = _compact_form(QtWidgets.QFormLayout(t_npyx))
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
        _add_settings_page(t_npyx, "Advanced")

        # Cell Types: a button-driven cell-type classifier. Two methods are
        # offered. "C4" is the NeuroPyxels cerebellar CNN ensemble (isolated env,
        # ~1 min). "Bombcell" is a threshold-based region-specific classifier that
        # runs in this env (~1 min to compute ephys properties). Both run off the
        # GUI thread.
        t_c4 = QtWidgets.QWidget()
        f_c4 = _compact_form(QtWidgets.QFormLayout(t_c4))
        self.cb_celltype_method = QtWidgets.QComboBox()
        self.cb_celltype_method.addItem("C4 (cerebellar)", userData="c4")
        self.cb_celltype_method.addItem("Bombcell (cortex/striatum)", userData="bombcell")
        self.cb_celltype_region = QtWidgets.QComboBox()
        self.cb_celltype_region.addItem("Cortex", userData="cortex")
        self.cb_celltype_region.addItem("Striatum", userData="striatum")
        self.lbl_c4_desc = QtWidgets.QLabel()
        self.lbl_c4_desc.setWordWrap(True)
        self.lbl_c4_desc.setObjectName("psthMetaLabel")
        self.sp_c4_threshold = QtWidgets.QDoubleSpinBox()
        self.sp_c4_threshold.setRange(0.0, 50.0)
        self.sp_c4_threshold.setSingleStep(0.5)
        self.sp_c4_threshold.setValue(2.0)
        self.btn_c4_run = QtWidgets.QPushButton("Run classifier")
        self.btn_c4_run.setProperty("role", "primary")
        f_c4.addRow("Method", with_help(self.cb_celltype_method, "C4: NeuroPyxels cerebellar CNN ensemble (isolated env). Bombcell: threshold-based region-specific classes (runs in this env)."))
        f_c4.addRow("Brain region", with_help(self.cb_celltype_region, "Region-specific Bombcell classes (cortex or striatum). Only used by the Bombcell method."))
        f_c4.addRow(self.lbl_c4_desc)
        f_c4.addRow("Confidence threshold", with_help(self.sp_c4_threshold, "C4 only: minimum confidence ratio for a confident call; lower values keep more predictions."))
        f_c4.addRow(self.btn_c4_run)
        _add_settings_page(t_c4, "Cell Types")

        self.page_progress = QtWidgets.QProgressBar()
        self.page_progress.setRange(0, 100)
        self.page_progress.setValue(0)
        # A quiet footer-style bar (thin, no chrome) so it never competes with
        # the figures for attention.
        self.page_progress.setProperty("footerProgress", True)
        self.page_progress.setTextVisible(False)

        self.view_tabs = QtWidgets.QTabWidget()
        self.view_tabs.setDocumentMode(True)
        # The view tabs are now a pure stacked page container: hide their tab bar
        # so the user sees only ONE navigation strip (analysis_tabs). Index sync
        # is preserved in _on_analysis_page_changed.
        self.view_tabs.tabBar().hide()

        # Unit Basics is now a single matplotlib canvas (npyx-style composite),
        # rendered through the shared NpyxFigureView.
        self.basics_view = NpyxFigureView()
        basics_container = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(basics_container)
        bl.setContentsMargins(4, 4, 4, 4)
        bl.setSpacing(0)
        bl.addWidget(self.basics_view, 1)

        # Raw Explorer is likewise a matplotlib canvas.
        self.raw_view = NpyxFigureView()
        raw_container = QtWidgets.QWidget()
        raw_l = QtWidgets.QVBoxLayout(raw_container)
        raw_l.setContentsMargins(4, 4, 4, 4)
        raw_l.setSpacing(0)
        raw_l.addWidget(self.raw_view, 1)

        corr_view = QtWidgets.QWidget()
        corr_v = QtWidgets.QVBoxLayout(corr_view)
        corr_v.setContentsMargins(4, 4, 4, 4)
        self.corr_npyx = NpyxFigureView()
        corr_v.addWidget(self.corr_npyx, 1)

        psth_view = QtWidgets.QWidget()
        psth_v = QtWidgets.QVBoxLayout(psth_view)
        psth_v.setContentsMargins(4, 4, 4, 4)
        psth_v.setSpacing(8)
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
        # Condition PSTH is now a single matplotlib canvas (npyx-style),
        # rendered through the shared NpyxFigureView.
        self.psth_view = NpyxFigureView()
        psth_v.addWidget(self.psth_summary_card, 0)
        psth_v.addWidget(self.psth_view, 1)

        # Network is now a single matplotlib canvas (3-panel network figure).
        self.net_view = NpyxFigureView()
        net_view = QtWidgets.QWidget()
        net_v = QtWidgets.QVBoxLayout(net_view)
        net_v.setContentsMargins(4, 4, 4, 4)
        net_v.setSpacing(0)
        net_v.addWidget(self.net_view, 1)

        npyx_view = QtWidgets.QWidget()
        npyx_v = QtWidgets.QVBoxLayout(npyx_view)
        self.gl_npyx = pg.GraphicsLayoutWidget()
        npyx_v.addWidget(self.gl_npyx, 1)

        # Cell Types (C4): a matplotlib canvas driven by the Run button.
        self.c4_view = NpyxFigureView()
        c4_view = QtWidgets.QWidget()
        c4_v = QtWidgets.QVBoxLayout(c4_view)
        c4_v.setContentsMargins(4, 4, 4, 4)
        c4_v.setSpacing(0)
        c4_v.addWidget(self.c4_view, 1)

        self.view_tabs.addTab(basics_container, "Unit Basics")
        self.view_tabs.addTab(raw_container, "Raw Explorer")
        self.view_tabs.addTab(corr_view, "Correlogram")
        self.view_tabs.addTab(psth_view, "Condition PSTH")
        self.view_tabs.addTab(net_view, "Network")
        self.view_tabs.addTab(npyx_view, "Advanced")
        self.view_tabs.addTab(c4_view, "Cell Types")

        # The settings panel: the per-analysis form stack with a small header,
        # docked on the right of the figure and collapsible via a chevron.
        self.settings_panel = QtWidgets.QWidget()
        self.settings_panel.setObjectName("settingsPanel")
        settings_panel_l = QtWidgets.QVBoxLayout(self.settings_panel)
        settings_panel_l.setContentsMargins(8, 6, 4, 6)
        settings_panel_l.setSpacing(6)
        self.lbl_settings_title = QtWidgets.QLabel("Settings")
        self.lbl_settings_title.setObjectName("settingsPanelTitle")
        settings_panel_l.addWidget(self.lbl_settings_title, 0)
        settings_panel_l.addWidget(self.settings_stack, 1)
        self.settings_panel.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        self.settings_panel.setMinimumWidth(300)
        self.settings_panel.setMaximumWidth(340)

        # The discrete chevron toggle, on the boundary between figure and panel.
        self.btn_settings_toggle = QtWidgets.QToolButton()
        self.btn_settings_toggle.setObjectName("settingsToggle")
        self.btn_settings_toggle.setAutoRaise(True)
        self.btn_settings_toggle.setCheckable(False)
        self.btn_settings_toggle.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_settings_toggle.setToolTip("Hide settings")
        self.btn_settings_toggle.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        self.btn_settings_toggle.setText("›")  # right chevron; flipped on toggle
        self.btn_settings_toggle.setFixedWidth(22)

        # Figure row: [ figures (expanding) ][ chevron ][ settings panel ].
        figure_row = QtWidgets.QWidget()
        figure_row_l = QtWidgets.QHBoxLayout(figure_row)
        figure_row_l.setContentsMargins(0, 0, 0, 0)
        figure_row_l.setSpacing(0)
        figure_row_l.addWidget(self.view_tabs, 1)
        figure_row_l.addWidget(self.btn_settings_toggle, 0)
        figure_row_l.addWidget(self.settings_panel, 0)

        # Assemble the right (analysis) panel: slim nav tab bar on top, the
        # figure+settings row below, then the per-page progress bar.
        analysis_area_l.addWidget(self.analysis_tabs, 0)
        analysis_area_l.addWidget(figure_row, 1)
        analysis_area_l.addWidget(self.page_progress, 0)
        self._analysis_area_layout = analysis_area_l
        self._figure_row_layout = figure_row_l
        self._settings_visible = True

        right = QtWidgets.QWidget()
        self._right_panel = right
        right_l = QtWidgets.QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)
        right_l.addWidget(analysis_area, 1)
        self._right_panel_layout = right_l
        self._analysis_area = analysis_area

        body.addWidget(grp_units)
        body.addWidget(right)
        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)
        body.setCollapsible(0, False)
        body.setSizes([330, 1370])

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setProperty("logView", True)
        self.log.setPlaceholderText("Dataset load and analysis output will appear here.")
        # Keep the log a thin strip so the figures own the vertical space.
        self.log.setMinimumHeight(70)
        self.log.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)

        main.addLayout(top)
        self.vertical_split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.vertical_split.addWidget(body)
        self.vertical_split.addWidget(self.log)
        self.vertical_split.setStretchFactor(0, 9)
        self.vertical_split.setStretchFactor(1, 1)
        self.vertical_split.setCollapsible(1, True)
        self.vertical_split.setSizes([900, 120])
        main.addWidget(self.vertical_split, 1)

        self.btn_browse.clicked.connect(self._pick)
        self.btn_load.clicked.connect(self._load_dataset)
        self.btn_export.clicked.connect(self._export_plotted_data)
        self.btn_export_units.clicked.connect(self._export_units_file)
        self.btn_export_waveforms.clicked.connect(self._export_waveform_figures)
        self.btn_detach_plots.toggled.connect(self._toggle_plot_detach)
        self.btn_settings_toggle.clicked.connect(self._toggle_settings_panel)
        self.list_units.itemSelectionChanged.connect(self._on_units_selection_changed)
        self.ed_unit_filter.textChanged.connect(self._refresh_units_list)
        self.btn_good_only.toggled.connect(self._refresh_units_list)
        self.cb_good_source.currentTextChanged.connect(self._on_good_source_changed)
        self.analysis_tabs.currentChanged.connect(self._on_analysis_page_changed)
        self.cb_corr_mode.currentTextChanged.connect(self._refresh_current_page)
        self.cb_corr_norm.currentTextChanged.connect(self._refresh_current_page)
        self.cb_corr_style.currentTextChanged.connect(self._refresh_current_page)
        self.btn_cond_add.clicked.connect(self._add_condition_row)
        self.btn_cond_remove.clicked.connect(self._remove_condition_row)
        self.btn_cond_browse.clicked.connect(self._browse_condition_csv)
        self.btn_cond_add_all.clicked.connect(self._add_all_behaviors)
        self.cb_psth_align.currentTextChanged.connect(self._on_psth_event_options_changed)
        self.sp_psth_fps.valueChanged.connect(self._on_psth_event_options_changed)
        # Baseline, display mode and trial range only change how the cached PSTH
        # results are drawn, so they re-render rather than recompute.
        self.ck_psth_baseline.toggled.connect(self._on_psth_display_options_changed)
        self.cb_psth_mode.currentIndexChanged.connect(self._on_psth_display_options_changed)
        self.sp_psth_trial_from.valueChanged.connect(self._on_psth_trial_range_changed)
        self.sp_psth_trial_to.valueChanged.connect(self._on_psth_trial_range_changed)
        self.btn_psth_all_trials.clicked.connect(self._reset_psth_trial_range)
        self.btn_psth_compute.clicked.connect(self._compute_psth)
        self.btn_psth_show.clicked.connect(self._show_psth)
        self.btn_net_compute.clicked.connect(self._compute_network)
        self.btn_net_show.clicked.connect(self._show_network)
        self.btn_c4_run.clicked.connect(self._run_celltypes)
        self.cb_celltype_method.currentIndexChanged.connect(self._update_celltype_method_ui)
        self.cb_npyx_method.currentTextChanged.connect(self._refresh_current_page)
        self.cb_npyx_method.currentIndexChanged.connect(self._update_npyx_method_ui)
        self.tbl_npyx_params.itemChanged.connect(self._on_npyx_params_changed)

        # Network and C4 are button-driven (potentially slow), so their controls
        # are intentionally NOT in the auto-refresh list.
        auto_widgets = [
            self.sp_basic_t0, self.sp_basic_dur, self.sp_isi_max, self.sp_basic_acg_bin, self.sp_basic_acg_win,
            self.ck_ifr, self.sp_ifr_smooth_ms,
            self.sp_raw_t0, self.sp_raw_dur, self.sp_raw_ch, self.sp_raw_hp, self.sp_raw_lp, self.sp_raw_ds,
            self.ck_raw_overlay, self.cb_raw_y, self.sp_corr_bin, self.sp_corr_win, self.sp_psth_pre,
            self.sp_psth_post, self.sp_psth_bin, self.sp_npyx_bin, self.sp_npyx_win,
        ]
        for w in auto_widgets:
            if hasattr(w, "valueChanged"):
                w.valueChanged.connect(self._refresh_current_page)
            elif hasattr(w, "currentTextChanged"):
                w.currentTextChanged.connect(self._refresh_current_page)
            elif hasattr(w, "toggled"):
                w.toggled.connect(self._refresh_current_page)

        self.tbl_conditions.itemChanged.connect(self._on_conditions_changed)
        self._apply_plot_style()
        self._update_psth_trial_status()
        self._update_npyx_method_ui()
        self._update_celltype_method_ui()
        # Sync the settings stack to the initial analysis and restore the
        # last-used settings-panel visibility (default = shown).
        self._sync_settings_page(self.analysis_tabs.currentIndex())
        show_settings = self.settings.value("post/settings_visible", True)
        if isinstance(show_settings, str):
            show_settings = show_settings.lower() not in {"false", "0", "no"}
        self._set_settings_visible(bool(show_settings))

    def _sync_settings_page(self, idx: int) -> None:
        """Keep the right-docked settings stack and its title aligned to the nav."""
        if 0 <= idx < self.settings_stack.count():
            self.settings_stack.setCurrentIndex(idx)
        title = self.analysis_tabs.tabText(idx) if 0 <= idx < self.analysis_tabs.count() else "Settings"
        self.lbl_settings_title.setText(f"{title} settings")

    def _set_settings_visible(self, visible: bool) -> None:
        """Show/hide the right settings panel; the chevron always stays reachable."""
        self._settings_visible = bool(visible)
        self.settings_panel.setVisible(self._settings_visible)
        # Chevron points toward the action: '›' hides (panel open), '‹' reveals.
        self.btn_settings_toggle.setText("›" if self._settings_visible else "‹")
        self.btn_settings_toggle.setToolTip("Hide settings" if self._settings_visible else "Show settings")
        self.settings.setValue("post/settings_visible", self._settings_visible)

    def _toggle_settings_panel(self) -> None:
        self._set_settings_visible(not self._settings_visible)

    def set_plot_preferences(self, theme: str, show_grid: bool) -> None:
        """Apply the global plot theme (Light/Dark) and grid visibility."""
        self._plot_theme = "Dark" if str(theme).lower().startswith("dark") else "Light"
        self._show_grid = bool(show_grid)
        self._apply_plot_style()
        # The matplotlib canvases bake the theme in at render time, so re-render
        # the current page to pick up the new dark/light figures.
        self._refresh_current_page()

    def _apply_plot_style(self) -> None:
        bg = "#0b0f14" if self._plot_theme == "Dark" else "#ffffff"
        fg = "#e8eef7" if self._plot_theme == "Dark" else "#1a1f29"
        card_bg = "rgba(90, 128, 255, 0.14)" if self._plot_theme == "Dark" else "rgba(67, 128, 255, 0.08)"
        card_border = "rgba(142, 170, 255, 0.34)" if self._plot_theme == "Dark" else "rgba(67, 128, 255, 0.22)"
        meta_fg = "#aab6ca" if self._plot_theme == "Dark" else "#5b6778"
        # Unit Basics, Raw Explorer, Condition PSTH, Network and Cell Types are now
        # matplotlib canvases (themed via the figure functions' dark arg), so the
        # only remaining pyqtgraph surface is the Advanced graphics layout. A theme
        # switch re-renders the canvases via _refresh_current_page.
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
        self.lbl_units_count.setStyleSheet(f"color: {meta_fg}; font-size: 11px; font-weight: 600;")
        self.lbl_c4_desc.setStyleSheet(f"color: {meta_fg}; font-size: 11px;")

        # Settings dock + chevron: a quiet, card-like panel with a discrete
        # boundary toggle (no heavy chrome).
        panel_bg = "rgba(20, 28, 40, 0.55)" if self._plot_theme == "Dark" else "rgba(244, 247, 251, 0.85)"
        panel_border = "rgba(120, 150, 210, 0.30)" if self._plot_theme == "Dark" else "rgba(67, 128, 255, 0.18)"
        chevron_fg = "#9fb3d0" if self._plot_theme == "Dark" else "#6a778d"
        chevron_hover_bg = "rgba(120, 150, 210, 0.22)" if self._plot_theme == "Dark" else "rgba(67, 128, 255, 0.12)"
        self.settings_panel.setStyleSheet(
            "QWidget#settingsPanel {"
            f"background: {panel_bg};"
            f"border: 1px solid {panel_border};"
            "border-radius: 12px;"
            "}"
            "QLabel#settingsPanelTitle {"
            f"color: {fg};"
            "font-size: 12px;"
            "font-weight: 700;"
            "padding: 2px 2px 4px 2px;"
            "}"
        )
        chevron_rest_bg = "rgba(120, 150, 210, 0.10)" if self._plot_theme == "Dark" else "rgba(67, 128, 255, 0.06)"
        self.btn_settings_toggle.setStyleSheet(
            "QToolButton#settingsToggle {"
            f"color: {chevron_fg};"
            f"border: 1px solid {panel_border};"
            f"background: {chevron_rest_bg};"
            "border-radius: 6px;"
            "margin: 40px 0px;"  # a centered, pill-like grip rather than a full-height bar
            "font-size: 16px;"
            "font-weight: 700;"
            "}"
            "QToolButton#settingsToggle:hover {"
            f"background: {chevron_hover_bg};"
            f"color: {fg};"
            "}"
        )

    def _subplot_shape(self, n: int) -> tuple[int, int]:
        if n <= 1:
            return 1, 1
        cols = int(math.ceil(math.sqrt(float(n))))
        rows = int(math.ceil(float(n) / max(cols, 1)))
        return rows, cols

    def _style_plot_item(self, plot: pg.PlotItem, left: str = "", bottom: str = "") -> None:
        fg = "#e8eef7" if self._plot_theme == "Dark" else "#1a1f29"
        grid_alpha = 0.25 if self._show_grid else 0.0
        tick_font = QtGui.QFont("Segoe UI", 9)
        plot.showGrid(x=self._show_grid, y=self._show_grid, alpha=grid_alpha)
        for ax_name in ("left", "bottom"):
            ax = plot.getAxis(ax_name)
            ax.setTextPen(pg.mkPen(fg))
            ax.setPen(pg.mkPen(fg, width=1.2))
            try:
                ax.setStyle(tickFont=tick_font, tickTextOffset=6)
            except Exception:
                pass
        plot.hideAxis("top")
        plot.hideAxis("right")
        label_css = {"color": fg, "font-size": "11pt"}
        if left:
            plot.setLabel("left", left, **label_css)
        if bottom:
            plot.setLabel("bottom", bottom, **label_css)

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

    def _on_psth_display_options_changed(self, *args) -> None:
        """Baseline / display-mode are render-time options: re-render, don't recompute."""
        if self.analysis_tabs.currentIndex() == 3 and "psth" in self.results:
            self._show_psth()

    def _condition_trial_slice(self, total_trials: int) -> tuple[slice, dict]:
        """Resolve the user's 1-based trial-range spinboxes into a 0-based slice.

        Returns the slice to apply to a trial matrix plus an info dict describing
        what was actually used (clamped to the available trial count). A stop
        value of 0 or below means "through the last trial".
        """
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
        """Read the editable npyx parameter table into a typed dict.

        Each cell value is coerced to bool, then int or float, falling back to
        the raw string when it cannot be parsed as a number.
        """
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
        if self._plot_detached or self._figure_row_layout is None:
            return
        if self._body_splitter is not None:
            self._body_sizes_before_detach = self._body_splitter.sizes()
        self._figure_row_layout.removeWidget(self.view_tabs)
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
        if not self._plot_detached or self._figure_row_layout is None:
            self.btn_detach_plots.setChecked(False)
            self.btn_detach_plots.setText("Detach plots")
            return
        if self._plot_dialog is not None:
            self._plot_dialog.layout().removeWidget(self.view_tabs)
        # Re-insert the figure stack at the left of the figure row (before the
        # chevron + settings panel) so the layout returns to its original order.
        self._figure_row_layout.insertWidget(0, self.view_tabs, 1)
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
                self._body_splitter.setSizes([330, 1370])
        self.btn_detach_plots.blockSignals(True)
        self.btn_detach_plots.setChecked(False)
        self.btn_detach_plots.blockSignals(False)
        self.btn_detach_plots.setText("Detach plots")

    def set_ks_folder(self, folder: str) -> None:
        """Populate the curated-folder field without loading the dataset."""
        self.ed_folder.setText(folder)

    def open_ks_folder(self, folder: str) -> None:
        """Set the curated folder and immediately load it as the dataset."""
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
        self._log(f"Loaded dataset with {len(self._all_units)} units")
        self._set_progress(0)
        self._refresh_current_page()

    def _unit_is_good(self, unit: int) -> bool:
        """Decide whether a unit is "good" under the selected label source.

        In "Auto" mode the first available source (Bombcell, then Phy, then
        KSLabel) that has a row for this unit decides the verdict; if no source
        is present at all the unit is treated as good.
        """
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
        shown = 0
        for u in self._all_units:
            if filt and filt not in str(u):
                continue
            if good_only and not self._unit_is_good(u):
                continue
            self.list_units.addItem(str(u))
            shown += 1
            if u in prev:
                self.list_units.item(self.list_units.count() - 1).setSelected(True)
        total = len(self._all_units)
        if total == 0:
            self.lbl_units_count.setText("No dataset loaded.")
        elif shown == total:
            self.lbl_units_count.setText(f"{total} units")
        else:
            self.lbl_units_count.setText(f"{shown} of {total} units shown")

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
        self._sync_settings_page(idx)
        self._refresh_current_page()

    def _refresh_current_page(self) -> None:
        """Re-render whichever analysis page is currently selected.

        Render errors are caught and logged so a single bad page cannot break
        the rest of the GUI.
        """
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
            elif idx == 6:
                # Cell Types (C4) is expensive and runs only on the Run button.
                # Leave the last result shown; otherwise prompt to run.
                self._show_c4_idle()
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
        """Render the Correlogram view with real NeuroPyxels figures (npyx.plot)."""
        if self.dataset is None:
            return
        units = self._selected_units()
        mode = self.cb_corr_mode.currentText()
        if not units:
            self.corr_npyx.show_message("Select unit(s) in the Units list to render the NeuroPyxels correlogram.")
            self._export_payloads["corr"] = []
            return
        if mode == "CCG" and len(units) < 2:
            self.corr_npyx.show_message("Select at least two units for a cross-correlogram (CCG) grid.")
            self._export_payloads["corr"] = []
            return

        dp = str(self.dataset.ks_folder)
        fs = float(self.dataset.sample_rate)
        cbin = float(self.sp_corr_bin.value())
        cwin = float(self.sp_corr_win.value())
        normalize = self.cb_corr_norm.currentText()
        style = self.cb_corr_style.currentText()
        dark = self._plot_theme == "Dark"
        # ACG is one panel per unit, so show all selected (generous safety cap).
        # CCG is an N x N grid (quadratic), so keep a smaller cap to stay legible.
        cap = 64 if mode == "ACG" else 8
        plot_units = units[:cap]

        self._set_progress(20)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            if mode == "ACG":
                fig = acg_grid_figure(dp, plot_units, cbin=cbin, cwin=cwin, fs=fs, normalize=normalize, dark=dark)
            else:
                fig = ccg_grid_figure(dp, plot_units, cbin=cbin, cwin=cwin, fs=fs, normalize=normalize, style=style, dark=dark)
            self.corr_npyx.show_figure(fig)
            if len(units) > len(plot_units):
                self._log(f"Correlogram ({mode}): showing {len(plot_units)} of {len(units)} selected units (cap {cap}).")
        except Exception as exc:
            self.corr_npyx.show_message(f"NeuroPyxels correlogram failed: {exc}")
            self._log(f"Correlogram render error: {exc}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self._set_progress(100)

        try:
            self._export_payloads["corr"] = self._corr_export_rows()
        except Exception:
            self._export_payloads["corr"] = []

    def _corr_export_rows(self) -> list[tuple[str, "pd.DataFrame"]]:
        """Build the CSV export rows for the current correlogram selection (engine arrays)."""
        items = self._corr_items_from_selection()
        flat: list[dict] = []
        for it in items:
            centers = np.asarray(it.get("centers", []), dtype=float)
            counts = np.asarray(it.get("counts", []), dtype=float)
            for c, v in zip(centers, counts):
                flat.append({"unit_a": int(it["u1"]), "unit_b": int(it["u2"]), "lag_ms": float(c), "value": float(v)})
        return [("correlogram.csv", pd.DataFrame(flat))]

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
            self.net_view.show_message("Select at least two units to compute network metrics.")
            return
        self._busy = True
        self._log("Computing network analysis...")
        self._set_progress(10)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.results["network"] = self.dataset.network_analysis(
                units,
                bin_ms=float(self.sp_net_bin.value()),
                conn_z=float(self.sp_net_z.value()),
            )
            self._set_progress(100)
            n_sig = int(self.results["network"].get("n_significant", 0))
            self._log(f"Network analysis computed ({len(units)} units, {n_sig} significant connections).")
            self._show_network()
        except Exception as exc:  # noqa: BLE001
            self.net_view.show_message(f"Network analysis failed: {exc}")
            self._log(f"Network compute error: {exc}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self._busy = False

    def _show_network(self) -> None:
        r = self.results.get("network")
        if not r:
            self.net_view.show_message("Select units and click Compute to run the network analysis.")
            return
        self._visualize_network(r)

    def _celltype_method(self) -> str:
        """Return the selected cell-type method key ('c4' or 'bombcell')."""
        data = self.cb_celltype_method.currentData()
        return str(data) if data else "c4"

    def _celltype_region(self) -> str:
        """Return the selected Bombcell brain region key ('cortex' or 'striatum')."""
        data = self.cb_celltype_region.currentData()
        return str(data) if data else "cortex"

    def _update_celltype_method_ui(self) -> None:
        """Sync the Cell Types description and which controls are relevant per method.

        The region combo is greyed for C4 (it only applies to Bombcell); the
        confidence-threshold spinbox is greyed for Bombcell (C4-only). The
        description label is rewritten to match the chosen method.
        """
        method = self._celltype_method()
        is_bombcell = method == "bombcell"
        self.cb_celltype_region.setEnabled(is_bombcell)
        self.sp_c4_threshold.setEnabled(not is_bombcell)
        if is_bombcell:
            self.lbl_c4_desc.setText(
                "Bombcell threshold-based, region-specific cell types. Cortex: "
                "Wide vs Narrow (FS) from waveform duration. Striatum: MSN / FSI / TAN / "
                "UIN from waveform duration, post-spike suppression and long-ISI proportion. "
                "Runs in this environment (~1 min to compute ephys properties). "
                "The confidence threshold is ignored for this method."
            )
        else:
            self.lbl_c4_desc.setText(
                "NeuroPyxels C4 cerebellar cell-type classifier "
                "(GoC / MLI / MFB / PkC_ss / PkC_cs), a CNN ensemble run in an isolated "
                "environment (~1 min). Cerebellum-trained, so labels are indicative on "
                "other regions."
            )

    def _show_c4_idle(self) -> None:
        """Cell Types page render: show the last result if present, else an instruction."""
        if self._c4_running:
            return
        result = self.results.get("celltypes")
        if result:
            self._render_celltypes(result)
        else:
            self.c4_view.show_message(
                "Cell-type classifier.\n"
                "Choose a Method (C4 or Bombcell), select units, then click 'Run classifier' "
                "(this can take ~1 min)."
            )

    def _render_celltypes(self, result: Dict[str, object]) -> None:
        """Render a cell-type result dict into the canvas (graceful on errors).

        Branches on the result's ``method`` so a stored Bombcell result renders
        via ``bombcell_celltype_figure`` and a C4 result via ``c4_figure``.
        """
        try:
            from .. import unit_figures  # lazy import
        except Exception as exc:  # noqa: BLE001
            self.c4_view.show_message(f"Cell-type figure module unavailable: {exc}")
            return
        method = str(result.get("method", "")) if isinstance(result, dict) else ""
        try:
            if method == "bombcell":
                fig = unit_figures.bombcell_celltype_figure(result, dark=(self._plot_theme == "Dark"))
            else:
                fig = unit_figures.c4_figure(result, dark=(self._plot_theme == "Dark"))
            self.c4_view.show_figure(fig)
        except Exception as exc:  # noqa: BLE001
            self.c4_view.show_message(f"Cell-type figure failed: {exc}")
            self._log(f"Cell-type render error: {exc}")

    def _run_celltypes(self) -> None:
        """Dispatch the Run-classifier button to the selected method (off-thread)."""
        if self._c4_running:
            return
        if self.dataset is None:
            self._log("Cell types: load a dataset first.")
            return
        units = self._selected_units()
        if not units:
            self._log("Cell types: select one or more units to classify.")
            self.c4_view.show_message("Select one or more units, then click 'Run classifier'.")
            return
        if self._celltype_method() == "bombcell":
            self._run_bombcell_celltypes(units)
        else:
            self._run_c4(units)

    def _run_c4(self, units: list[int]) -> None:
        """Run the C4 classifier off the GUI thread via FunctionWorker."""
        try:
            from ..workers import FunctionWorker
            from ..c4_runner import run_c4_classifier
        except Exception as exc:  # noqa: BLE001
            self.c4_view.show_message(f"C4 runner unavailable: {exc}")
            self._log(f"C4 unavailable: {exc}")
            return

        self.view_tabs.setCurrentIndex(6)
        self._c4_running = True
        self.btn_c4_run.setEnabled(False)
        self.btn_c4_run.setText("Running C4...")
        self._log(f"Running C4 on {len(units)} unit(s) (this can take ~1 min)...")
        self.c4_view.show_message("Running the C4 classifier in an isolated environment...\nThis can take about a minute.")
        self._set_progress(10)
        worker = FunctionWorker(
            run_c4_classifier,
            str(self.dataset.ks_folder),
            list(units),
            threshold=float(self.sp_c4_threshold.value()),
        )
        worker.signals.finished.connect(self._on_c4_finished)
        self.pool.start(worker)

    def _run_bombcell_celltypes(self, units: list[int]) -> None:
        """Run the Bombcell region-specific cell-type classifier off the GUI thread."""
        try:
            from ..workers import FunctionWorker
            from ..bombcell_classify import run_bombcell_classifier
        except Exception as exc:  # noqa: BLE001
            self.c4_view.show_message(f"Bombcell runner unavailable: {exc}")
            self._log(f"Bombcell unavailable: {exc}")
            return

        region = self._celltype_region()
        self.view_tabs.setCurrentIndex(6)
        self._c4_running = True
        self.btn_c4_run.setEnabled(False)
        self.btn_c4_run.setText("Running Bombcell...")
        self._log(f"Running bombcell cell-type classification ({region}, ~1 min)...")
        self.c4_view.show_message(
            f"Running the Bombcell {region} cell-type classifier...\n"
            "Computing ephys properties can take about a minute."
        )
        self._set_progress(10)
        worker = FunctionWorker(
            run_bombcell_classifier,
            str(self.dataset.ks_folder),
            list(units),
            region=region,
        )
        worker.signals.finished.connect(self._on_bombcell_finished)
        self.pool.start(worker)

    @QtCore.Slot(dict)
    def _on_c4_finished(self, payload: Dict[str, object]) -> None:
        """GUI-thread handler for the C4 worker result."""
        self._c4_running = False
        self.btn_c4_run.setEnabled(True)
        self.btn_c4_run.setText("Run classifier")
        self._set_progress(100)
        result = payload.get("result") if isinstance(payload, dict) else None
        if not payload.get("ok") or not isinstance(result, dict):
            msg = "C4 classification failed (see log)."
            self.c4_view.show_message(msg)
            self._log(msg)
            return
        result["method"] = "c4"
        self.results["celltypes"] = result
        err = result.get("error")
        if err:
            self._log(f"C4: {err}")
        else:
            n_units = len(result.get("units", []))
            n_skipped = len(result.get("skipped_units", []))
            model = str(result.get("model_type", "C4"))
            self._log(f"C4 classified {n_units} unit(s) with {model}; {n_skipped} skipped.")
        self._render_celltypes(result)
        # Build a non-crashing CSV export from the result.
        try:
            units_out = [int(u) for u in result.get("units", [])]
            predicted = [str(x) for x in result.get("predicted_type", [])]
            confidence = np.asarray(result.get("confidence", []), dtype=float)
            ratio = np.asarray(result.get("confidence_ratio", []), dtype=float)
            votes = np.asarray(result.get("model_votes", []), dtype=float)
            rows = []
            for i, u in enumerate(units_out):
                rows.append(
                    {
                        "cluster_id": int(u),
                        "predicted_cell_type": predicted[i] if i < len(predicted) else "",
                        "confidence": float(confidence[i]) if i < confidence.size else np.nan,
                        "confidence_ratio": float(ratio[i]) if i < ratio.size else np.nan,
                        "model_votes": float(votes[i]) if i < votes.size else np.nan,
                    }
                )
            df = pd.DataFrame(rows)
            self._export_payloads["c4"] = [("cell_types_c4.csv", df)]
            if not err:
                self._write_celltype_files(df, "c4")
        except Exception:
            self._export_payloads["c4"] = [("cell_types_c4.csv", pd.DataFrame())]

    @QtCore.Slot(dict)
    def _on_bombcell_finished(self, payload: Dict[str, object]) -> None:
        """GUI-thread handler for the Bombcell worker result."""
        self._c4_running = False
        self.btn_c4_run.setEnabled(True)
        self.btn_c4_run.setText("Run classifier")
        self._set_progress(100)
        result = payload.get("result") if isinstance(payload, dict) else None
        if not payload.get("ok") or not isinstance(result, dict):
            msg = "Bombcell classification failed (see log)."
            self.c4_view.show_message(msg)
            self._log(msg)
            return
        result.setdefault("method", "bombcell")
        self.results["celltypes"] = result
        region = str(result.get("region", self._celltype_region()))
        err = result.get("error")
        if err:
            self.c4_view.show_message(f"Bombcell: {err}")
            self._log(f"Bombcell: {err}")
            return
        n_units = len(result.get("units", []))
        n_skipped = len(result.get("skipped_units", []))
        self._log(f"Bombcell classified {n_units} unit(s) ({region}); {n_skipped} skipped.")
        self._render_celltypes(result)
        # Build the CSV export + write the result files to the dataset folder.
        try:
            units_out = [int(u) for u in result.get("units", [])]
            predicted = [str(x) for x in result.get("predicted_type", [])]
            metrics = result.get("metrics", {}) if isinstance(result.get("metrics"), dict) else {}
            wf = np.asarray(metrics.get("waveform_duration_us", []), dtype=float)
            pss = np.asarray(metrics.get("post_spike_suppression_ms", []), dtype=float)
            pli = np.asarray(metrics.get("prop_long_isi", []), dtype=float)
            fr = np.asarray(metrics.get("firing_rate_hz", []), dtype=float)
            rows = []
            for i, u in enumerate(units_out):
                rows.append(
                    {
                        "cluster_id": int(u),
                        "predicted_cell_type": predicted[i] if i < len(predicted) else "",
                        "region": region,
                        "waveform_duration_us": float(wf[i]) if i < wf.size else np.nan,
                        "post_spike_suppression_ms": float(pss[i]) if i < pss.size else np.nan,
                        "prop_long_isi": float(pli[i]) if i < pli.size else np.nan,
                        "firing_rate_hz": float(fr[i]) if i < fr.size else np.nan,
                    }
                )
            df = pd.DataFrame(rows)
            method_tag = f"bombcell_{region}"
            self._export_payloads["c4"] = [(f"cell_types_{method_tag}.csv", df)]
            self._write_celltype_files(df, method_tag)
        except Exception as exc:  # noqa: BLE001
            self._export_payloads["c4"] = [("cell_types_bombcell.csv", pd.DataFrame())]
            self._log(f"Bombcell CSV/TSV build error: {exc}")

    def _write_celltype_files(self, df: "pd.DataFrame", method: str) -> None:
        """Write a CSV + phy-compatible TSV of cell-type predictions to the dataset folder.

        ``df`` must carry ``cluster_id`` and ``predicted_cell_type`` columns plus
        any method-specific extras (already present for both C4 and Bombcell). The
        CSV keeps every column; the TSV is the two-column phy-friendly subset
        named ``cluster_<method>_cell_type.tsv``. Failures are logged, never
        raised, so a write error cannot crash the GUI.
        """
        if self.dataset is None or df is None or df.empty:
            return
        try:
            folder = Path(self.dataset.ks_folder)
            csv_path = folder / f"cell_types_{method}.csv"
            tsv_path = folder / f"cluster_{method}_cell_type.tsv"
            df.to_csv(csv_path, index=False)
            tsv_df = df[["cluster_id", "predicted_cell_type"]].copy()
            tsv_df.to_csv(tsv_path, sep="\t", index=False)
            self._log(f"Cell types: wrote {csv_path}")
            self._log(f"Cell types: wrote {tsv_path}")
        except Exception as exc:  # noqa: BLE001
            self._log(f"Cell types: failed to write result files: {exc}")

    def _visualize_basic(self) -> None:
        """Render the npyx-style Unit Basics composite as a matplotlib canvas."""
        self.view_tabs.setCurrentIndex(0)
        units = self._selected_units()
        if not units:
            self.basics_view.show_message("Select unit(s) in the Units list to render the Unit Basics figure.")
            self._export_payloads["basic"] = []
            return
        try:
            from .. import unit_figures  # lazy import: tolerate a momentarily absent module
        except Exception as exc:  # noqa: BLE001
            self.basics_view.show_message(f"Unit Basics figure module unavailable: {exc}")
            self._export_payloads["basic"] = []
            return

        self._set_progress(20)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            fig = unit_figures.unit_basics_figure(
                self.dataset,
                self._selected_units(),
                window_start_s=float(self.sp_basic_t0.value()),
                window_s=float(self.sp_basic_dur.value()),
                ifr_bin_ms=float(self.sp_ifr_smooth_ms.value()),
                acg_bin_ms=float(self.sp_basic_acg_bin.value()),
                acg_win_ms=float(self.sp_basic_acg_win.value()),
                isi_max_ms=float(self.sp_isi_max.value()),
                show_ifr=bool(self.ck_ifr.isChecked()),
                dark=(self._plot_theme == "Dark"),
            )
            self.basics_view.show_figure(fig)
        except Exception as exc:  # noqa: BLE001
            self.basics_view.show_message(f"Unit Basics figure failed: {exc}")
            self._log(f"Unit Basics render error: {exc}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self._set_progress(100)

        # A simple, non-crashing export summary of the current selection/window.
        try:
            self._export_payloads["basic"] = [
                (
                    "unit_basics_summary.csv",
                    pd.DataFrame(
                        [
                            {
                                "units": ",".join(str(int(u)) for u in units),
                                "window_start_s": float(self.sp_basic_t0.value()),
                                "window_s": float(self.sp_basic_dur.value()),
                                "ifr_bin_ms": float(self.sp_ifr_smooth_ms.value()),
                                "acg_bin_ms": float(self.sp_basic_acg_bin.value()),
                                "acg_win_ms": float(self.sp_basic_acg_win.value()),
                                "isi_max_ms": float(self.sp_isi_max.value()),
                                "show_ifr": bool(self.ck_ifr.isChecked()),
                            }
                        ]
                    ),
                )
            ]
        except Exception:
            self._export_payloads["basic"] = []

    def _visualize_raw(self) -> None:
        """Render the Raw Explorer as a matplotlib canvas (npyx-style)."""
        self.view_tabs.setCurrentIndex(1)
        try:
            from .. import unit_figures  # lazy import: tolerate a momentarily absent module
        except Exception as exc:  # noqa: BLE001
            self.raw_view.show_message(f"Raw Explorer figure module unavailable: {exc}")
            self._export_payloads["raw"] = []
            return

        units = self._selected_units()
        focus_unit = int(units[0]) if units else None
        focus_center = None
        if focus_unit is not None:
            focus_waveform = self.dataset.mean_template_waveform(focus_unit)
            focus_center = self._best_channel_index(focus_unit, focus_waveform)
        overlay_units = tuple(units) if self.ck_raw_overlay.isChecked() else ()
        y_mode = "depth" if self.cb_raw_y.currentText().startswith("Depth") else "channel"

        self._set_progress(20)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            fig = unit_figures.raw_explorer_figure(
                self.dataset,
                t0_s=float(self.sp_raw_t0.value()),
                dur_s=float(self.sp_raw_dur.value()),
                n_channels=int(self.sp_raw_ch.value()),
                hp_hz=float(self.sp_raw_hp.value()),
                lp_hz=float(self.sp_raw_lp.value()),
                center_channel=focus_center,
                overlay_units=overlay_units,
                y_mode=y_mode,
                dark=(self._plot_theme == "Dark"),
            )
            self.raw_view.show_figure(fig)
        except Exception as exc:  # noqa: BLE001
            self.raw_view.show_message(f"Raw Explorer figure failed: {exc}")
            self._log(f"Raw Explorer render error: {exc}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self._set_progress(100)

        try:
            self._export_payloads["raw"] = [
                (
                    "raw_explorer_summary.csv",
                    pd.DataFrame(
                        [
                            {
                                "t0_s": float(self.sp_raw_t0.value()),
                                "dur_s": float(self.sp_raw_dur.value()),
                                "n_channels": int(self.sp_raw_ch.value()),
                                "hp_hz": float(self.sp_raw_hp.value()),
                                "lp_hz": float(self.sp_raw_lp.value()),
                                "center_channel": (-1 if focus_center is None else int(focus_center)),
                                "y_mode": y_mode,
                                "overlay_units": ",".join(str(int(u)) for u in overlay_units),
                            }
                        ]
                    ),
                )
            ]
        except Exception:
            self._export_payloads["raw"] = []

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

    def _set_condition_label_options(
        self, row: int, labels: List[str], selected_label: str = "", all_label: str = "All events"
    ) -> None:
        combo = self._condition_label_combo(row)
        if combo is None:
            combo = QtWidgets.QComboBox(self.tbl_conditions)
            combo.currentIndexChanged.connect(self._on_condition_label_selection_changed)
            self.tbl_conditions.setCellWidget(row, 1, combo)
        current = str(selected_label or combo.currentData() or "").strip()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(all_label, "")
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
        is_matrix = info.get("kind") == "behavior_matrix"
        all_label = "All behaviors (pooled)" if is_matrix else "All events"
        self._set_condition_label_options(row, labels, selected_label, all_label=all_label)
        time_column = str(info.get("time_column") or "")
        label_column = str(info.get("label_column") or "")
        if announce:
            if is_matrix:
                fps = float(info.get("frame_rate") or 30.0)
                tnote = f"time='{time_column}'" if time_column else f"no time column (using {fps:g} fps)"
                self._log(
                    f"Behavior matrix loaded: {Path(csv_path).name} | {len(labels)} behaviors | "
                    f"{tnote}. Pick a behavior in the Event label column (use 'Add all behaviors' for all)."
                )
            elif time_column and label_column:
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
        return load_event_times(
            path,
            selected_label=selected_label,
            frame_rate=float(self.sp_psth_fps.value()),
            alignment=self.cb_psth_align.currentText(),
        ).to_numpy(dtype=float)

    def _on_psth_event_options_changed(self, *args) -> None:
        """Recompute the PSTH when event-derivation options (alignment, fps, baseline) change."""
        if self.analysis_tabs.currentIndex() == 3 and self.tbl_conditions.rowCount() > 0:
            self._compute_psth()

    def _add_all_behaviors(self) -> None:
        """Create one PSTH condition per behavior column of a chosen binary-matrix CSV."""
        path = ""
        for r in range(self.tbl_conditions.rowCount()):
            path = self._condition_path_for_row(r)
            if path:
                break
        if not path:
            start = self.settings.value("paths/last_folder", str(Path.cwd()))
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select behavior-matrix CSV", str(start), "CSV files (*.csv)"
            )
            if not path:
                return
            self.settings.setValue("paths/last_folder", str(Path(path).parent))
        try:
            info = inspect_event_csv(path)
        except Exception as exc:
            self._log(f"Add all behaviors: failed to read CSV: {exc}")
            return
        if info.get("kind") != "behavior_matrix":
            self._log("Add all behaviors: this CSV is not a binary behavior matrix.")
            return
        behaviors = [str(b) for b in info.get("labels", [])]
        existing = {self._condition_name_for_row(r) for r in range(self.tbl_conditions.rowCount())}
        added = 0
        for b in behaviors:
            ev = self._load_event_csv(path, selected_label=b)
            if ev.size == 0 or b in existing:
                continue
            r = self.tbl_conditions.rowCount()
            blocker = QtCore.QSignalBlocker(self.tbl_conditions)
            self.tbl_conditions.insertRow(r)
            self.tbl_conditions.setItem(r, 0, QtWidgets.QTableWidgetItem(b))
            self.tbl_conditions.setItem(r, 2, QtWidgets.QTableWidgetItem(str(path)))
            del blocker
            self._set_condition_label_options(r, behaviors, b, all_label="All behaviors (pooled)")
            added += 1
        self._log(f"Add all behaviors: created {added} condition(s) from {Path(path).name}.")
        if added:
            self._compute_psth()

    def _visualize_condition_psth(self, r: Dict[str, object]) -> None:
        """Render the Condition PSTH as a matplotlib canvas (npyx-style)."""
        self.view_tabs.setCurrentIndex(3)
        t_ms = np.asarray(r.get("t_ms", []), dtype=float)
        conditions = list(r.get("conditions", []))
        units = [int(v) for v in r.get("units", [])]
        mode = "per_unit" if self.cb_psth_mode.currentText().startswith("Per-unit") else "average"
        baseline = bool(self.ck_psth_baseline.isChecked())
        trial_from = int(self.sp_psth_trial_from.value())
        trial_to = int(self.sp_psth_trial_to.value())

        if t_ms.size == 0 or not conditions:
            self.lbl_psth_summary.setText("Condition PSTH is ready after you compute it.")
            self.lbl_psth_summary_meta.setText(
                "Select units, choose an event label, then use Compute. Trial-range and display-mode changes apply on the displayed figure."
            )
            self.psth_view.show_message("Add a condition CSV, select units, then Compute to render the Condition PSTH.")
            self._export_payloads["psth"] = self._psth_export_rows(r)
            return

        try:
            from .. import unit_figures  # lazy import: tolerate a momentarily absent module
        except Exception as exc:  # noqa: BLE001
            self.psth_view.show_message(f"Condition PSTH figure module unavailable: {exc}")
            self._export_payloads["psth"] = self._psth_export_rows(r)
            return

        self._set_progress(20)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            fig = unit_figures.condition_psth_figure(
                self.results.get("psth"),
                mode=mode,
                baseline=baseline,
                trial_from=trial_from,
                trial_to=trial_to,
                dark=(self._plot_theme == "Dark"),
            )
            self.psth_view.show_figure(fig)
        except Exception as exc:  # noqa: BLE001
            self.psth_view.show_message(f"Condition PSTH figure failed: {exc}")
            self._log(f"Condition PSTH render error: {exc}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self._set_progress(100)

        # Update the summary banner from the results + current trial range.
        single_unit = len(units) == 1
        summary_parts: list[str] = []
        for i, entry in enumerate(conditions):
            condition_name = str(entry.get("condition", f"cond_{i + 1}"))
            _slice, trial_info = self._condition_trial_slice(int(entry.get("trial_count", 0)))
            summary_parts.append(
                f"{condition_name}: {int(trial_info['used_trials'])}/{int(trial_info['total_trials'])} trials"
            )
        if single_unit:
            mode_text = "Single unit: per-trial rows + mean ± SEM across the selected trials."
        elif mode == "per_unit":
            mode_text = "Per-unit panels: one panel per selected unit (mean across the selected trials)."
        else:
            mode_text = "Average across units: one mean trace per condition across the selected units."
        requested_range_text = self.lbl_psth_trial_status.text().replace("Trial filter: ", "").strip()
        self.lbl_psth_summary.setText(
            f"Condition PSTH · {len(units)} selected unit{'s' if len(units) != 1 else ''} · {requested_range_text}"
        )
        self.lbl_psth_summary_meta.setText(f"{mode_text}  {'  |  '.join(summary_parts)}")
        self._export_payloads["psth"] = self._psth_export_rows(r)

    def _psth_export_rows(self, r: Dict[str, object]) -> list[tuple[str, "pd.DataFrame"]]:
        """Build a non-crashing PSTH CSV export straight from the results dict.

        Produces a per-(condition, unit, time) mean-rate table over the currently
        selected trial range, honoring the baseline-subtract option. Independent
        of the matplotlib figure artists so export never depends on rendering.
        """
        try:
            t_ms = np.asarray(r.get("t_ms", []), dtype=float)
            conditions = list(r.get("conditions", []))
            if t_ms.size == 0 or not conditions:
                return [("condition_psth.csv", pd.DataFrame())]
            baseline = bool(self.ck_psth_baseline.isChecked())
            pre_mask = t_ms < 0.0
            rows: list[dict] = []
            for ci, entry in enumerate(conditions):
                condition_name = str(entry.get("condition", f"cond_{ci + 1}"))
                unit_ids = [int(v) for v in entry.get("unit_ids", r.get("units", []))]
                unit_trial_mats = [np.asarray(v, dtype=float) for v in entry.get("unit_trial_mats", [])]
                trial_slice, trial_info = self._condition_trial_slice(int(entry.get("trial_count", 0)))
                if int(trial_info.get("used_trials", 0)) <= 0:
                    continue
                for unit_id, trial_mat in zip(unit_ids, unit_trial_mats):
                    visible = np.asarray(trial_mat[trial_slice], dtype=float)
                    if visible.size == 0:
                        continue
                    unit_mean = np.nanmean(visible, axis=0)
                    if baseline and np.any(pre_mask):
                        unit_mean = unit_mean - float(np.nanmean(unit_mean[pre_mask]))
                    n_trials = int(visible.shape[0])
                    rows.extend(
                        {
                            "condition": condition_name,
                            "unit_id": int(unit_id),
                            "time_ms": float(tv),
                            "mean_rate_hz": float(rv),
                            "n_trials": n_trials,
                            "trial_start": int(trial_info["actual_start"]),
                            "trial_stop": int(trial_info["actual_stop"]),
                            "total_trials": int(trial_info["total_trials"]),
                            "baseline_subtracted": bool(baseline),
                        }
                        for tv, rv in zip(t_ms, unit_mean)
                    )
            return [("condition_psth.csv", pd.DataFrame(rows))]
        except Exception:
            return [("condition_psth.csv", pd.DataFrame())]

    def _visualize_network(self, r: Dict[str, object]) -> None:
        """Render the population network analysis as a matplotlib canvas."""
        self.view_tabs.setCurrentIndex(4)
        try:
            from .. import unit_figures  # lazy import: tolerate a momentarily absent module
        except Exception as exc:  # noqa: BLE001
            self.net_view.show_message(f"Network figure module unavailable: {exc}")
            self._export_payloads["network"] = self._network_export_rows(r)
            return
        try:
            fig = unit_figures.network_figure(r, dark=(self._plot_theme == "Dark"))
            self.net_view.show_figure(fig)
        except Exception as exc:  # noqa: BLE001
            self.net_view.show_message(f"Network figure failed: {exc}")
            self._log(f"Network render error: {exc}")
        self._export_payloads["network"] = self._network_export_rows(r)

    def _network_export_rows(self, r: Dict[str, object]) -> list[tuple[str, "pd.DataFrame"]]:
        """Build non-crashing CSV exports straight from the network results dict."""
        try:
            labels = [str(x) for x in r.get("labels", r.get("units", []))]
            n = len(labels)
            corr = np.asarray(r.get("corr_matrix", []), dtype=float)
            coupling = np.asarray(r.get("population_coupling", []), dtype=float)
            depths = r.get("depths_um")
            conn = r.get("connections")
            matrix_rows: list[dict] = []
            if corr.ndim == 2 and corr.shape == (n, n):
                conn_arr = np.asarray(conn, dtype=float) if conn is not None else None
                for i in range(n):
                    for j in range(n):
                        row = {
                            "unit_a": labels[i],
                            "unit_b": labels[j],
                            "corr": float(corr[i, j]),
                        }
                        if conn_arr is not None and conn_arr.shape == (n, n):
                            row["connection_z"] = float(conn_arr[i, j])
                        matrix_rows.append(row)
            depths_arr = np.asarray(depths, dtype=float) if depths is not None else None
            unit_rows = [
                {
                    "unit_id": labels[i],
                    "population_coupling": float(coupling[i]) if i < coupling.size else np.nan,
                    "depth_um": (float(depths_arr[i]) if depths_arr is not None and i < depths_arr.size else np.nan),
                }
                for i in range(n)
            ]
            return [
                ("network_correlation_matrix.csv", pd.DataFrame(matrix_rows)),
                ("network_units.csv", pd.DataFrame(unit_rows)),
            ]
        except Exception:
            return [("network.csv", pd.DataFrame())]

    def _show_npyx_corr(self) -> None:
        # Guard the pyqtgraph rendering so a transient re-entrant repaint (e.g. a
        # queued refresh firing mid scene-rebuild) is attributed clearly and does
        # not surface as a generic page-render error.
        try:
            self._render_npyx_corr()
        except Exception as exc:  # noqa: BLE001
            self._log(f"Advanced render error: {exc}")

    def _render_npyx_corr(self) -> None:
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
                fs=float(self.dataset.sample_rate),
            )
            self._set_progress(100)
        except Exception as exc:
            self._log(f"Advanced error ({key}): {exc}")
            self._busy = False
            return
        finally:
            self._busy = False

        requested_dp = str(res.get("requested_dp", ""))
        resolved_dp = str(res.get("resolved_dp", ""))
        if requested_dp and resolved_dp and requested_dp != resolved_dp:
            self._log(f"Advanced datapath fallback: {requested_dp} -> {resolved_dp}")

        kind = str(res.get("kind", "text"))
        title = str(res.get("title", str(key)))
        cmap_name = "CET-L9" if self._plot_theme == "Dark" else "CET-L4"
        x_label = str(res.get("x_label", "x"))
        y_label = str(res.get("y_label", "value"))
        if kind in {"line", "hist"}:
            plot = self.gl_npyx.addPlot(row=0, col=0, title=title)
            self._style_plot_item(plot, left=y_label, bottom=x_label)
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
            self._style_plot_item(plot, left=y_label, bottom=x_label)
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
            self._style_plot_item(plot, left=str(res.get("y_label", "row")), bottom=str(res.get("x_label", "col")))
            mat = np.asarray(res.get("mat", []), dtype=float)
            if mat.size:
                img = pg.ImageItem(mat)
                xv = np.asarray(res.get("x", []), dtype=float)
                yv = np.asarray(res.get("y", []), dtype=float)
                has_coords = xv.size >= 2 and yv.size >= 2 and mat.ndim == 2
                if has_coords:
                    x0, x1 = float(xv[0]), float(xv[-1])
                    y0, y1 = float(yv[0]), float(yv[-1])
                    img.setRect(QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))
                elif mat.ndim == 2:
                    img.setRect(QtCore.QRectF(0, 0, float(mat.shape[1]), float(mat.shape[0])))
                cm = pg.colormap.get(cmap_name)
                if cm is not None:
                    img.setLookupTable(cm.getLookupTable(nPts=256))
                plot.addItem(img)
                # Index matrices read top-to-bottom; coordinate images keep natural y.
                if not has_coords:
                    plot.getViewBox().invertY(True)
                    # Per-cell value labels only make sense for small index matrices,
                    # not coordinate images (acg_3D/ccg_3D); and guard the pyqtgraph
                    # auto-range hiccup that adding many TextItems can trigger.
                    try:
                        self._annotate_image_values(plot, mat)
                    except Exception:
                        pass
                try:
                    levels = img.getLevels()
                    cbar = pg.ColorBarItem(values=levels, colorMap=cm, label=str(res.get("cbar_label", "")), width=12)
                    cbar.setImageItem(img, insert_in=plot)
                except Exception:
                    pass
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
        """Write the current page's recorded plot data to CSV files in a folder."""
        idx = self.view_tabs.currentIndex()
        keys = ["basic", "raw", "corr", "psth", "network", "npyx", "c4"]
        key = keys[idx] if 0 <= idx < len(keys) else ""
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

    def _export_waveform_figures(self) -> None:
        """Export every good unit's waveform + ACG figure off the GUI thread.

        Builds one PNG per good unit plus a single multi-page PDF, all written to
        a user-chosen folder. The work reads the AP binary for each unit's +/-SEM
        waveform (~minutes for the full good-unit set), so it runs on the thread
        pool with periodic progress logging; the button is disabled meanwhile.
        """
        if self._exporting_waveforms:
            return
        if self.dataset is None:
            self._log("Export waveforms: load a dataset first.")
            return
        good_units = sorted(self._good_unit_ids())
        if not good_units:
            self._log("Export waveforms: no good units under the current good-unit source.")
            return
        try:
            from ..workers import FunctionWorker
        except Exception as exc:  # noqa: BLE001
            self._log(f"Export waveforms: worker unavailable: {exc}")
            return

        start = self.settings.value("paths/last_folder", str(self.dataset.ks_folder))
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select folder for waveform + ACG figures", str(start)
        )
        if not out_dir:
            return
        self.settings.setValue("paths/last_folder", str(out_dir))

        n = len(good_units)
        self._exporting_waveforms = True
        self.btn_export_waveforms.setEnabled(False)
        self.btn_export_waveforms.setText("Exporting...")
        self._set_progress(0)
        self._log(
            f"Export waveforms: rendering {n} good-unit figure(s) to {out_dir} "
            "(reads the AP binary per unit; this can take a few minutes)..."
        )

        # progress_cb runs on the worker thread, so it must only emit a Qt signal
        # (delivered queued to the GUI thread) and never touch widgets directly.
        worker = FunctionWorker(
            _export_good_unit_figures,
            self.dataset,
            good_units,
            str(out_dir),
            dark=(self._plot_theme == "Dark"),
        )

        progress_signal = worker.signals.progress

        def _progress_cb(i: int, total: int) -> None:
            pct = int(100 * float(i) / float(max(total, 1)))
            try:
                progress_signal.emit(max(0, min(100, pct)))
            except RuntimeError:
                # Receiver/source may already be gone during shutdown.
                pass

        worker.kwargs["progress_cb"] = _progress_cb
        worker.signals.progress.connect(self._on_waveform_export_progress)
        worker.signals.finished.connect(self._on_waveform_export_finished)
        self.pool.start(worker)

    @QtCore.Slot(int)
    def _on_waveform_export_progress(self, pct: int) -> None:
        """GUI-thread progress handler for the waveform export worker."""
        self._set_progress(int(pct))

    @QtCore.Slot(dict)
    def _on_waveform_export_finished(self, payload: Dict[str, object]) -> None:
        """GUI-thread handler for the waveform export worker result."""
        self._exporting_waveforms = False
        self.btn_export_waveforms.setEnabled(True)
        self.btn_export_waveforms.setText("Export waveforms")
        self._set_progress(100)
        result = payload.get("result") if isinstance(payload, dict) else None
        if not payload.get("ok") or not isinstance(result, dict):
            self._log("Export waveforms: failed (see log).")
            return
        err = result.get("error")
        if err:
            self._log(f"Export waveforms: {err}")
            if not result.get("n"):
                return
        n = int(result.get("n", 0))
        pdf = result.get("pdf")
        self._log(f"Exported {n} good-unit figures to {pdf}")

    def is_busy(self) -> bool:
        """Return True while a compute/export task is running on this tab."""
        return bool(self._busy or self._c4_running or self._exporting_waveforms)



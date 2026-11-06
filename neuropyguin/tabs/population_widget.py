"""Population and behavior workspace with isolated computation and paper exports."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from ..plot_toolbar import PaperNavigationToolbar

from ..population_analysis import PopulationConfig, load_result, result_tables


# Central figure styling: conservative print sizes and colorblind-friendly ink.
FIGURE_STYLE = {"ink": "#243649", "blue": "#0072B2", "orange": "#D55E00",
                "teal": "#009E73", "muted": "#6A7885", "font": 9, "line": 1.0}


def population_figure(result):
    """Build a four-panel, publication-sized summary from computed arrays only."""
    fig = Figure(figsize=(10.2, 7.2), constrained_layout=True, facecolor="white")
    ax_heat, ax_embed, ax_trace, ax_behavior = fig.subplots(2, 2).ravel()
    axes = [ax_heat, ax_embed, ax_trace, ax_behavior]
    for ax in axes:
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#AAB4BE")
        ax.tick_params(labelsize=8, width=0.6, length=3, colors=FIGURE_STYLE["ink"])
        ax.xaxis.label.set_size(9)
        ax.yaxis.label.set_size(9)
    t, rates, Z = result["times"], result["rates"], result["embedding"]
    cfg = result["metadata"]["config"]
    width = cfg["bin_s"]
    normalized = (rates - rates.mean(axis=0)) / np.maximum(rates.std(axis=0), 1e-8)
    # Render at most 2500 columns; all original bins remain in exported tables.
    stride = max(1, int(np.ceil(len(t) / 2500)))
    # Pool within display blocks instead of aliasing the heatmap by subsampling.
    block_means = np.stack([normalized[i:i + stride].mean(axis=0)
                            for i in range(0, len(t), stride)])
    im = ax_heat.imshow(block_means.T, aspect="auto", origin="lower", cmap="RdBu_r",
                        vmin=-3, vmax=3, interpolation="nearest", rasterized=True,
                        extent=(t[0] - width / 2, t[-1] + width / 2, -0.5, rates.shape[1] - 0.5))
    ax_heat.set(title="a  Population activity", xlabel="Recording time (s)", ylabel="Unit (selection order)")
    ticks = np.unique(np.linspace(0, len(result["units"]) - 1, min(6, len(result["units"]))).astype(int))
    ax_heat.set_yticks(ticks, result["units"][ticks])
    fig.colorbar(im, ax=ax_heat, shrink=0.75, pad=0.025, label="Rate (z score)").ax.tick_params(labelsize=7)
    color = result.get("behavior", t)
    color_name = cfg["behavior_column"] if "behavior" in result else "Time (s)"
    # Geometry is a view of the entire interval, never used as cross-validation input.
    idx = np.linspace(0, len(t) - 1, min(len(t), 8000)).astype(int)
    valid = np.isfinite(color[idx])
    if not valid.all():
        ax_embed.scatter(Z[idx[~valid], 0], Z[idx[~valid], 1], s=3, color="#CCD2D8", rasterized=True)
    points = ax_embed.scatter(Z[idx[valid], 0], Z[idx[valid], 1], c=color[idx[valid]], cmap="viridis",
                              s=5, linewidths=0, alpha=0.8, rasterized=True, picker=5)
    points._population_indices = idx[valid]
    ax_embed.set(title=f"b  {cfg['method'].upper()} geometry", xlabel="Component 1", ylabel="Component 2")
    fig.colorbar(points, ax=ax_embed, shrink=0.75, pad=0.025, label=color_name).ax.tick_params(labelsize=7)
    palette = [FIGURE_STYLE["blue"], FIGURE_STYLE["orange"], FIGURE_STYLE["teal"]]
    for component in range(min(3, Z.shape[1])):
        trace = Z[:, component]
        trace = (trace - trace.mean()) / max(trace.std(), 1e-8) + component * 4
        ax_trace.plot(t, trace, lw=0.75, color=palette[component], label=f"C{component + 1}", rasterized=len(t) > 10000)
    ax_trace.set(title="c  Latent dynamics", xlabel="Recording time (s)", ylabel="Standardized component + offset")
    ax_trace.legend(frameon=False, fontsize=8, loc="upper right", ncol=3)
    low, high = ax_trace.get_ylim()
    ax_trace.set_ylim(low, high + 0.18 * (high - low))
    ax_trace.sharex(ax_heat)
    ax_trace.set_xlim(t[0] - width / 2, t[-1] + width / 2)
    if "decoding" in result:
        dec = result["decoding"]
        ax_behavior.plot(t, result["behavior"], color="#8D98A4", lw=0.9, label="Observed")
        ax_behavior.plot(t, dec["predictions"], color=FIGURE_STYLE["blue"], lw=0.8, label="Held-out prediction")
        ax_behavior.plot(t, dec["baseline"], color=FIGURE_STYLE["orange"], lw=0.7, ls=":", label="Training mean")
        p = f"; shift p={dec['p_shift']:.3g}" if dec["p_shift"] is not None else "; no null test"
        ax_behavior.set(title=f"d  Raw-rate decoding\nR²={dec['r2']:.2f}{p}",
                        xlabel="Recording time (s)", ylabel=cfg["behavior_column"])
        ax_behavior.legend(frameon=False, fontsize=7, loc="upper right")
        low, high = ax_behavior.get_ylim()
        ax_behavior.set_ylim(low, high + 0.35 * (high - low))
    elif "behavior" in result:
        ax_behavior.plot(t, result["behavior"], color=FIGURE_STYLE["teal"], lw=1)
        ax_behavior.set(title="d  Aligned continuous behavior", xlabel="Recording time (s)", ylabel=cfg["behavior_column"])
    elif result["explained_variance"].size:
        variance = 100 * result["explained_variance"]
        ax_behavior.bar(np.arange(len(variance)) + 1, variance, width=0.45, color=FIGURE_STYLE["blue"])
        ax_behavior.set(title="d  Variance explained", xlabel="Principal component", ylabel="Explained variance (%)")
        ax_behavior.set_xticks(np.arange(len(variance)) + 1)
        for i, value in enumerate(variance):
            ax_behavior.annotate(f"{value:.1f}%", (i + 1, value), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
        ax_behavior.set_ylim(0, max(variance) * 1.2)
    else:
        ax_behavior.hist(rates.mean(axis=0), bins=min(20, len(result["units"])), color=FIGURE_STYLE["teal"], rwidth=0.75)
        ax_behavior.set(title="d  Unit firing rates", xlabel="Mean firing rate (Hz)", ylabel="Units")
    fig._population_time_axes = [ax_heat, ax_trace]
    if "behavior" in result:
        ax_behavior.sharex(ax_heat)
        fig._population_time_axes.append(ax_behavior)
    for ax in axes:
        ax.title.set_fontsize(10)
        ax.title.set_fontweight("semibold")
        ax.title.set_color(FIGURE_STYLE["ink"])
    return fig


class PopulationWidget(QtWidgets.QWidget):
    """A self-contained population workspace; parent supplies dataset and selection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataset = None
        self.units = []
        self.result = None
        self._temporary = None
        self._process = None
        self._log = ""
        self.settings = QtCore.QSettings("NeuroPyGuiN", "Population")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.configuration = QtWidgets.QDialog(self)
        self.configuration.setWindowTitle("Population analysis settings")
        self.configuration.resize(900, 560)
        dialog_layout = QtWidgets.QVBoxLayout(self.configuration)
        config_scroll = QtWidgets.QScrollArea()
        config_scroll.setWidgetResizable(True)
        config_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        config_content = QtWidgets.QWidget()
        config_layout = QtWidgets.QVBoxLayout(config_content)
        config_scroll.setWidget(config_content)
        dialog_layout.addWidget(config_scroll)
        close_settings = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        close_settings.rejected.connect(self.configuration.hide)
        dialog_layout.addWidget(close_settings)
        title = QtWidgets.QLabel("Population + behavior")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        subtitle = QtWidgets.QLabel("Explore collective activity, compare representations, and test continuous behavior decoding on held-out time.")
        subtitle.setWordWrap(True)
        config_layout.addWidget(title)
        config_layout.addWidget(subtitle)
        controls = QtWidgets.QGridLayout()
        self.scope = QtWidgets.QComboBox()
        self.scope.addItems(["All loaded units", "Selected units"])
        self.method = QtWidgets.QComboBox()
        for label, key in [("PCA · transparent baseline", "pca"), ("Factor analysis · shared variability", "fa"),
                           ("NEMBA · temporal convolution", "nemba:tcn"), ("NEMBA · PCA baseline", "nemba:pca"),
                           ("NEMBA · factor analysis", "nemba:fa"), ("NEMBA · GPFA", "nemba:gpfa")]:
            self.method.addItem(label, key)
        for combo in (self.scope, self.method):
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)
            combo.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        self.method.setToolTip("PCA baseline; factor analysis; optional NEMBA TCN, PCA, FA or GPFA")
        self.start = self._number(0, 1e9, 0, 2, " s")
        self.stop = self._number(0.01, 1e9, 60, 2, " s")
        self.bin_width = self._number(1, 60000, 100, 1, " ms")
        controls.addWidget(QtWidgets.QLabel("Unit scope"), 0, 0)
        controls.addWidget(self.scope, 1, 0)
        controls.addWidget(QtWidgets.QLabel("Representation"), 0, 1)
        controls.addWidget(self.method, 1, 1)
        for column, (name, field) in enumerate([("Start", self.start), ("End", self.stop), ("Time bin", self.bin_width)], 2):
            controls.addWidget(QtWidgets.QLabel(name), 0, column)
            controls.addWidget(field, 1, column)
        config_layout.addLayout(controls)
        behavior_row = QtWidgets.QHBoxLayout()
        self.behavior_file = QtWidgets.QLineEdit()
        self.behavior_file.setPlaceholderText("Optional continuous behavior CSV: time_s, speed, position, ...")
        self.behavior_file.setToolTip("Times must be seconds on the spike clock. This workspace does not infer clock synchronization. Angles and categorical labels require their dedicated models; use continuous targets here.")
        browse = QtWidgets.QPushButton("Behavior CSV…")
        browse.clicked.connect(self._browse_behavior)
        self.target = QtWidgets.QComboBox()
        self.target.setMinimumWidth(120)
        self.target.setEditable(True)
        self.target.setPlaceholderText("Target column")
        behavior_row.addWidget(browse)
        behavior_row.addWidget(self.behavior_file, 1)
        behavior_row.addWidget(self.target)
        config_layout.addLayout(behavior_row)
        self.advanced_toggle = QtWidgets.QToolButton()
        self.advanced_toggle.setText("Advanced settings and NEMBA setup")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.advanced_toggle.setArrowType(QtCore.Qt.RightArrow)
        config_layout.addWidget(self.advanced_toggle)
        self.advanced = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(self.advanced)
        self.gap = self._number(0, 1000, 1, 2, " s")
        self.max_gap = self._number(0.001, 1000, 1, 3, " s")
        self.permutations = QtWidgets.QSpinBox()
        self.permutations.setRange(0, 999)
        self.permutations.setValue(99)
        self.components = QtWidgets.QSpinBox()
        self.components.setRange(2, 16)
        self.components.setValue(3)
        self.standardize = QtWidgets.QCheckBox("Give each variable unit equal variance")
        self.standardize.setChecked(True)
        self.nemba_python = QtWidgets.QLineEdit(str(self.settings.value("python", os.environ.get("NEMBA_PYTHON", ""))))
        self.nemba_python.setPlaceholderText("Optional absolute Python path of an existing NEMBA environment")
        self.nemba_source = QtWidgets.QLineEdit(str(self.settings.value("source", os.environ.get("NEMBA_PATH", ""))))
        self.nemba_source.setPlaceholderText("Optional NEMBA source checkout containing embedders_backbones/")
        form.addRow("Latent dimensions", self.components)
        form.addRow("Scaling", self.standardize)
        form.addRow("CV purge gap", self.gap)
        form.addRow("Maximum interpolation gap", self.max_gap)
        form.addRow("Circular-shift null draws", self.permutations)
        form.addRow("NEMBA Python", self.nemba_python)
        form.addRow("NEMBA source", self.nemba_source)
        note = QtWidgets.QLabel("Continuous decoding uses raw rates and five time blocks. Increase the purge gap beyond relevant autocorrelation times. Shift-null inference assumes approximate stationarity. Embeddings are descriptive, not independent validation.")
        note.setWordWrap(True)
        form.addRow(note)
        self.advanced.setVisible(False)
        self.advanced_scroll = QtWidgets.QScrollArea()
        self.advanced_scroll.setWidgetResizable(True)
        self.advanced_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.advanced_scroll.setMaximumHeight(220)
        self.advanced_scroll.setWidget(self.advanced)
        self.advanced_scroll.hide()
        config_layout.addWidget(self.advanced_scroll)
        config_layout.addStretch(1)
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        row = QtWidgets.QHBoxLayout()
        self.settings_button = QtWidgets.QPushButton("Analysis settings…")
        self.settings_button.clicked.connect(self._show_settings)
        row.addWidget(self.settings_button)
        self.run_button = QtWidgets.QPushButton("Analyze population")
        self.run_button.setProperty("role", "primary")
        self.run_button.setEnabled(False)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.export_button = QtWidgets.QPushButton("Export figure + data")
        self.export_button.setEnabled(False)
        self.status = QtWidgets.QLabel("Load a curated dataset to begin.")
        self.status.setWordWrap(True)
        row.addWidget(self.run_button)
        row.addWidget(self.cancel_button)
        row.addWidget(self.export_button)
        row.addStretch(1)
        layout.addLayout(row)
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        layout.addWidget(self.status)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setMaximumHeight(5)
        self.progress.setTextVisible(False)
        self.progress.hide()
        layout.addWidget(self.progress)
        self.figure = Figure(figsize=(10.2, 7.2), facecolor="white")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.mpl_connect("pick_event", self._pick_embedding)
        self._cursor_lines = []
        self.toolbar = self._make_toolbar(self.canvas)
        self.canvas.setMinimumHeight(520)
        self.paper_scroll = QtWidgets.QScrollArea()
        self.paper_scroll.setWidgetResizable(True)
        self.paper_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.paper_scroll.setMinimumHeight(240)
        self.paper_scroll.setWidget(self.canvas)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.paper_scroll, 1)
        self._empty_figure()
        self.run_button.clicked.connect(self._run)
        self.cancel_button.clicked.connect(self._cancel)
        self.export_button.clicked.connect(self._export)
        self.method.currentIndexChanged.connect(self._method_changed)
        self.behavior_file.editingFinished.connect(self._read_columns)
        self._settings_controls = [self.scope, self.method, self.start, self.stop, self.bin_width,
                                   browse, self.behavior_file, self.target, self.gap, self.max_gap,
                                   self.permutations, self.components, self.standardize,
                                   self.nemba_python, self.nemba_source]
        for field in [self.start, self.stop, self.bin_width, self.gap, self.max_gap, self.permutations, self.components]:
            field.valueChanged.connect(self._mark_stale)
        self.scope.currentIndexChanged.connect(self._mark_stale)
        self.standardize.toggled.connect(self._mark_stale)
        self.behavior_file.textChanged.connect(self._mark_stale)
        self.target.currentTextChanged.connect(self._mark_stale)
        self._update_summary()

    def _make_toolbar(self, canvas):
        """Keep paper-navigation icons readable and compact in either app theme."""
        return PaperNavigationToolbar(canvas, self)

    def _show_settings(self):
        """Expose all settings in one keyboard-accessible dialog, preserving plot area."""
        self.configuration.show()
        self.configuration.raise_()
        self.configuration.activateWindow()

    def _update_summary(self):
        """Keep the current analysis scope visible while controls are tucked away."""
        if not hasattr(self, "summary"):
            return
        unit_count = len(self.dataset.units) if self.dataset is not None and self.scope.currentIndex() == 0 else len(self.units)
        self.summary.setText(f"{str(self.method.currentData()).upper()} · {self.scope.currentText().lower()} ({unit_count}) · "
                             f"{self.start.value():g}–{self.stop.value():g} s · {self.bin_width.value():g} ms bins")

    @staticmethod
    def _number(low, high, value, decimals, suffix):
        """Create a compact, consistently formatted numeric control."""
        control = QtWidgets.QDoubleSpinBox()
        control.setDecimals(decimals)
        control.setRange(low, high)
        control.setValue(value)
        control.setSuffix(suffix)
        return control

    @property
    def busy(self):
        """Allow the parent to guard application shutdown while work is running."""
        return self._process is not None and self._process.state() != QtCore.QProcess.NotRunning

    def set_dataset(self, dataset, units=None):
        """Reset results when the source changes, cancelling stale computations."""
        self.shutdown()
        self.dataset = dataset
        self.units = list(units) if units is not None else []
        self.result = None
        self.export_button.setEnabled(False)
        self.run_button.setEnabled(dataset is not None)
        self._empty_figure()
        if dataset is not None:
            # Last spike is a lower bound; expose the interval so trailing silence is explicit.
            if len(dataset.spike_times):
                duration = float(np.max(dataset.spike_times)) / dataset.sample_rate
                self.stop.setValue(max(self.bin_width.value() / 100, min(60, duration)))
            self.status.setText(f"{len(dataset.units)} units loaded. Choose a time interval, then analyze.")
        else:
            self.status.setText("Load a curated dataset to begin.")
        self._update_summary()

    def set_units(self, units):
        """Update selection without discarding a completed analysis."""
        self.units = list(units)
        self._update_summary()
        if self.scope.currentIndex() == 1:
            self._mark_stale()

    def _empty_figure(self):
        """Offer an informative, quiet landing state instead of an empty canvas."""
        self.figure.clear()
        self.figure.text(0.08, 0.78, "Explore population structure", fontsize=16, color="#243649", weight="semibold")
        self.figure.text(0.08, 0.54, "01   Open Analysis settings to choose units and an interval\n02   Explore PCA, factor analysis, or NEMBA representations\n03   Align behavior and test decoding on held-out time", fontsize=10, color="#607185", va="top")
        self.figure.text(0.08, 0.14, "Every export includes vector figures, plotted data and analysis settings.", fontsize=10, color="#607185")
        self.canvas.draw_idle()

    def _toggle_advanced(self, visible):
        """Reveal specialist controls only when needed."""
        self.advanced.setVisible(visible)
        self.advanced_scroll.setVisible(visible)
        self.advanced_toggle.setArrowType(QtCore.Qt.DownArrow if visible else QtCore.Qt.RightArrow)

    def _method_changed(self):
        """NEMBA standardizes input explicitly, so reflect that in the controls."""
        nemba = str(self.method.currentData()).startswith("nemba:")
        if nemba:
            self.standardize.setChecked(True)
        self.standardize.setEnabled(not nemba)
        self._mark_stale()

    def _mark_stale(self, *_):
        """Keep the provenance of the last result distinct from pending settings."""
        self._update_summary()
        if self.result is not None and not self.busy:
            self.status.setText("Showing the last completed run. Analyze again to apply changed settings.")

    def _browse_behavior(self):
        """Choose a CSV without writing into the source folder."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Continuous behavior on the spike clock", "", "CSV (*.csv)")
        if path:
            self.behavior_file.setText(path)
            self._read_columns()

    def _read_columns(self):
        """Populate target names using only a CSV header read."""
        path = self.behavior_file.text().strip()
        if not path:
            self.target.clear()
            return
        try:
            names = pd.read_csv(path, nrows=0).columns
            self.target.clear()
            self.target.addItems([str(x) for x in names if x not in {"time", "time_s"}])
        except Exception as exc:
            self.status.setText(f"Cannot read behavior CSV: {exc}")

    def _run(self):
        """Start a fresh process; heavy fits and decoding never block Qt."""
        if self.dataset is None or self.busy:
            return
        units = list(map(int, self.dataset.units)) if self.scope.currentIndex() == 0 else list(map(int, self.units))
        if len(units) < 2:
            self.status.setText("Select at least two units, or choose All loaded units.")
            return
        config = PopulationConfig(start_s=self.start.value(), stop_s=self.stop.value(),
                                  bin_s=self.bin_width.value() / 1000, method=self.method.currentData(),
                                  components=self.components.value(), standardize=self.standardize.isChecked(),
                                  behavior_path=self.behavior_file.text().strip(), behavior_column=self.target.currentText(),
                                  cv_gap_s=self.gap.value(), max_behavior_gap_s=self.max_gap.value(),
                                  permutations=self.permutations.value(), nemba_path=self.nemba_source.text().strip())
        self.settings.setValue("python", self.nemba_python.text().strip())
        self.settings.setValue("source", config.nemba_path)
        self.configuration.hide()
        self._temporary = tempfile.TemporaryDirectory(prefix="neuropyguin-population-")
        root = Path(self._temporary.name)
        request = {"dataset": str(self.dataset.ks_folder), "sample_rate": float(self.dataset.sample_rate),
                   "units": units, "config": asdict(config)}
        ap_binary = getattr(self.dataset, "ap_bin_path", None)
        request["metadata_files"] = [str(Path(ap_binary).with_suffix(".meta"))] if ap_binary else []
        (root / "request.json").write_text(json.dumps(request), encoding="utf-8")
        process = QtCore.QProcess(self)
        process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        environment = QtCore.QProcessEnvironment.systemEnvironment()
        package_root = str(Path(__file__).resolve().parents[2])
        environment.insert("PYTHONPATH", package_root + os.pathsep + environment.value("PYTHONPATH"))
        environment.insert("PYTHONUNBUFFERED", "1")
        environment.insert("OMP_NUM_THREADS", "2")
        process.setProcessEnvironment(environment)
        interpreter = self.nemba_python.text().strip() if config.method.startswith("nemba:") else ""
        process.setProgram(interpreter or sys.executable)
        process.setArguments(["-m", "neuropyguin.population_runner", str(root / "request.json"), str(root / "result.npz")])
        process.readyReadStandardOutput.connect(self._read_log)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(self._process_error)
        self._process = process
        self._log = ""
        self._run_units = list(units)
        for control in self._settings_controls:
            control.setEnabled(False)
        self._cancelled = False
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.export_button.setEnabled(False)
        self.progress.show()
        self.status.setText("Analyzing population in a background process…")
        process.start()

    def _read_log(self):
        """Keep bounded diagnostics available while showing one concise progress line."""
        if self._process is None:
            return
        text = bytes(self._process.readAllStandardOutput()).decode("utf-8", "replace")
        self._log = (self._log + text)[-20000:]
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            self.status.setText(lines[-1][-200:])

    def _process_error(self, error):
        """Handle missing interpreters, which may not emit QProcess.finished."""
        if error == QtCore.QProcess.FailedToStart:
            self._log += "\nCould not start the selected Python interpreter. Check its absolute path."
            self._finished(-1, QtCore.QProcess.CrashExit)

    def _finished(self, exit_code, _status):
        """Accept only a complete successful result and release process resources."""
        if self._process is None:
            return
        self._read_log()
        process, self._process = self._process, None
        self.run_button.setEnabled(self.dataset is not None)
        for control in self._settings_controls:
            control.setEnabled(True)
        self.standardize.setEnabled(not str(self.method.currentData()).startswith("nemba:"))
        self.cancel_button.setEnabled(False)
        self.progress.hide()
        try:
            result_file = Path(self._temporary.name) / "result.npz"
            if self._cancelled:
                self.status.setText("Analysis cancelled.")
            elif exit_code == 0 and result_file.is_file():
                self.result = load_result(result_file)
                self._install_figure(population_figure(self.result))
                coverage = self.result["metadata"].get("behavior_coverage")
                detail = f" · behavior coverage {coverage:.0%}" if coverage is not None else ""
                if self.scope.currentIndex() == 1 and list(map(int, self.units)) != self._run_units:
                    detail += " · selection changed; showing units from the completed run"
                self.status.setText(f"{len(self.result['units'])} units · {len(self.result['times'])} bins{detail}. " + self.result["metadata"].get("decoding_unavailable", "Click a geometry point to inspect its time."))
            else:
                self.status.setText("Analysis failed. Review the input or NEMBA setup.")
                dialog = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, "Population analysis", self._log.splitlines()[-1] if self._log else "Analysis process failed.", parent=self)
                dialog.setDetailedText(self._log)
                dialog.exec()
        except Exception as exc:
            self.status.setText(f"Could not render result: {exc}")
        finally:
            process.deleteLater()
            self.export_button.setEnabled(self.result is not None)
            if self._temporary is not None:
                self._temporary.cleanup()
                self._temporary = None

    def _cancel(self):
        """Terminate the isolated computation without leaving a worker thread alive."""
        if self.busy:
            self._cancelled = True
            self._process.kill()
            self.status.setText("Cancelling…")

    def shutdown(self):
        """Stop the worker before a source change or application shutdown."""
        if self.busy:
            self._cancelled = True
            self._process.kill()
            self._process.waitForFinished(3000)

    def closeEvent(self, event):
        """Ensure standalone widget windows do not orphan computation processes."""
        self.shutdown()
        super().closeEvent(event)

    def _install_figure(self, figure):
        """Replace canvas and toolbar together so callbacks belong to this figure.

        Matplotlib event registries live on Figure. Merely assigning
        canvas.figure leaves toolbar callbacks attached to the previous figure,
        silently breaking pan/zoom after a run.
        """
        old_canvas, old_toolbar = self.canvas, self.toolbar
        self.figure = figure
        self.canvas = FigureCanvasQTAgg(figure)
        self.canvas.mpl_connect("pick_event", self._pick_embedding)
        self.toolbar = self._make_toolbar(self.canvas)
        self.canvas.setMinimumHeight(520)
        self.layout().replaceWidget(old_toolbar, self.toolbar)
        self.paper_scroll.takeWidget()
        self.paper_scroll.setWidget(self.canvas)
        self.toolbar.show()
        self.canvas.show()
        old_toolbar.hide()
        old_canvas.hide()
        old_toolbar.deleteLater()
        old_canvas.deleteLater()
        self._cursor_lines = []
        self.canvas.draw_idle()

    def _pick_embedding(self, event):
        """Link a geometry point back to its exact recording time on the other panels."""
        indices = getattr(event.artist, "_population_indices", None)
        if indices is None or self.result is None or not len(event.ind):
            return
        index = int(indices[event.ind[0]])
        moment = float(self.result["times"][index])
        for line in self._cursor_lines:
            line.remove()
        self._cursor_lines = [ax.axvline(moment, color=FIGURE_STYLE["orange"], lw=0.8, ls="--")
                              for ax in getattr(self.figure, "_population_time_axes", [])]
        detail = ""
        if "behavior" in self.result and np.isfinite(self.result["behavior"][index]):
            detail = f" · {self.result['metadata']['config']['behavior_column']} = {self.result['behavior'][index]:.4g}"
        self.status.setText(f"Selected time: {moment:.3f} s{detail}. Time axes share pan and zoom.")
        self.canvas.draw_idle()

    def _export(self):
        """Write vector/raster figures, raw plotted values and a provenance manifest."""
        if self.result is None:
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Export population figure and data")
        if not folder:
            return
        try:
            from ..publication import export_figure_bundle
            metadata = dict(self.result["metadata"])
            if "decoding" in self.result:
                metadata["decoding_summary"] = {k: v for k, v in self.result["decoding"].items() if not isinstance(v, np.ndarray)}
            # Inspection cursors are transient navigation aids, not results.
            for line in self._cursor_lines:
                line.set_visible(False)
            try:
                files = export_figure_bundle(self.figure, folder, "population_behavior", tables=result_tables(self.result), metadata=metadata)
            finally:
                for line in self._cursor_lines:
                    line.set_visible(True)
                self.canvas.draw_idle()
            self.status.setText(f"Exported {len(files)} files: PDF, SVG, PNG, data tables and settings.")
        except Exception as exc:
            self.status.setText(f"Export failed: {exc}")

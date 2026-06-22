from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from ..ecephys_runtime import ecephys_subprocess_env
from ..ks_output_resolver import find_kilosort_output_dir, find_metrics_file
from ..processes import tracked_run
from ..pybombcell_integration import run_pybombcell_on_folder
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
        return "No matching modules-input.json found. Reloaded existing metrics if present."

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


class QualityMetricsTab(QtWidgets.QWidget):
    def __init__(self, thread_pool: QtCore.QThreadPool) -> None:
        super().__init__()
        self.pool = thread_pool
        self.qsettings = QtCore.QSettings("NeuroPyGuiN", "NeuroPyGuiN")
        self.df = pd.DataFrame()
        self._busy = False
        self._plots_detached = False
        self._plots_dialog: Optional[QtWidgets.QDialog] = None
        self._splitter: Optional[QtWidgets.QSplitter] = None
        self._right_panel: Optional[QtWidgets.QWidget] = None
        self._split_sizes_before_detach: list[int] = []
        self._build_ui()
        self._restore_settings()
        self._plot_theme = "Light"
        self._show_grid = True

    def _build_ui(self) -> None:
        main = QtWidgets.QVBoxLayout(self)
        main.setContentsMargins(18, 16, 18, 18)
        main.setSpacing(14)

        top = QtWidgets.QHBoxLayout()
        top.setSpacing(8)
        self.ed_folder = QtWidgets.QLineEdit()
        self.btn_browse = QtWidgets.QPushButton("Browse")
        self.btn_run = QtWidgets.QPushButton("Run/Reprocess + Load Quality Metrics")
        self.btn_run_pybomb = QtWidgets.QPushButton("Run py_bombcell")
        self.btn_export = QtWidgets.QPushButton("Export plotted data")
        self.btn_detach_plots = QtWidgets.QPushButton("Detach plots")
        self.btn_detach_plots.setCheckable(True)
        self.btn_browse.setProperty("role", "secondary")
        self.btn_run.setProperty("role", "primary")
        self.btn_run_pybomb.setProperty("role", "secondary")
        self.btn_export.setProperty("role", "ghost")
        self.btn_detach_plots.setProperty("role", "ghost")
        self.ed_filter = QtWidgets.QLineEdit()
        self.ed_filter.setPlaceholderText("Filter rows (contains)")
        top.addWidget(self.ed_folder, 1)
        top.addWidget(self.btn_browse)
        top.addWidget(self.btn_run)
        top.addWidget(self.btn_run_pybomb)
        top.addWidget(self.btn_export)
        top.addWidget(self.btn_detach_plots)
        top.addWidget(self.ed_filter, 1)

        splitter = QtWidgets.QSplitter()
        self._splitter = splitter

        left = QtWidgets.QWidget()
        left_l = QtWidgets.QVBoxLayout(left)
        self.table = QtWidgets.QTableWidget()
        self.table.setAlternatingRowColors(True)
        left_l.addWidget(self.table)

        right = QtWidgets.QWidget()
        self._right_panel = right
        right_l = QtWidgets.QVBoxLayout(right)
        self.metric_combo = QtWidgets.QComboBox()
        self.plot = pg.PlotWidget(title="Metric distribution")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        right_l.addWidget(self.metric_combo)
        right_l.addWidget(self.plot, 1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setProperty("logView", True)
        self.log.setPlaceholderText("Quality-metrics processing output will appear here.")

        main.addLayout(top)
        main.addWidget(splitter, 1)
        main.addWidget(self.log, 1)

        self.btn_browse.clicked.connect(self._pick)
        self.btn_run.clicked.connect(self._run_reprocess_and_load)
        self.btn_run_pybomb.clicked.connect(self._run_pybombcell)
        self.btn_export.clicked.connect(self._export_plotted_data)
        self.btn_detach_plots.toggled.connect(self._toggle_plot_detach)
        self.ed_filter.textChanged.connect(self._refresh_table)
        self.metric_combo.currentTextChanged.connect(self._refresh_plot)

    def set_ks_folder(self, folder: str) -> None:
        self.ed_folder.setText(folder)

    def open_ks_folder(self, folder: str) -> None:
        self.set_ks_folder(folder)
        self._run_reprocess_and_load()

    def _pick(self) -> None:
        start = self.qsettings.value("paths/last_folder", str(Path.cwd()))
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select curated folder", str(start))
        if folder:
            self.ed_folder.setText(folder)
            self.qsettings.setValue("paths/last_folder", folder)
            self.qsettings.setValue("quality/last_folder", folder)

    def _resolve_folder(self, folder: Path) -> Path:
        resolved = find_kilosort_output_dir(folder, max_depth=4)
        if resolved is not None and resolved != folder:
            self._log(f"Resolved KS folder to {resolved}")
            self.ed_folder.setText(str(resolved))
            return resolved
        return folder

    def _run_reprocess_and_load(self) -> None:
        folder = Path(self.ed_folder.text().strip())
        if not folder.exists():
            self._log(f"Invalid folder: {folder}")
            return
        folder = self._resolve_folder(folder)
        self.qsettings.setValue("quality/last_folder", str(folder))

        self._busy = True
        self.btn_run.setEnabled(False)
        json_root = str(self.qsettings.value("preproc/json_root", str(Path.cwd() / "NeuroPyGuiN_json")))
        worker = FunctionWorker(_recompute_quality_metrics, str(folder), json_root)
        worker.signals.error.connect(self._log)
        worker.signals.finished.connect(lambda result: self._on_reprocess_done(result, folder))
        self.pool.start(worker)

    def _on_reprocess_done(self, result: dict, folder: Path) -> None:
        self._busy = False
        self.btn_run.setEnabled(True)
        if result.get("ok"):
            msg = result.get("result", "")
            if msg:
                self._log(str(msg))
        metrics_path = find_metrics_file(folder, max_depth=4)
        if metrics_path is None or not metrics_path.exists():
            self._log(f"metrics.csv not found: {metrics_path}")
            return

        self.df = pd.read_csv(metrics_path)
        if self.df.columns.size > 0 and str(self.df.columns[0]).lower().startswith("unnamed"):
            self.df = self.df.drop(columns=[self.df.columns[0]])
        self.metric_combo.clear()
        numeric_cols = [c for c in self.df.columns if np.issubdtype(self.df[c].dtype, np.number)]
        self.metric_combo.addItems(numeric_cols)
        self._refresh_table()
        self._refresh_plot()
        self._log(f"Loaded {metrics_path}")

    def _run_pybombcell(self) -> None:
        folder = Path(self.ed_folder.text().strip())
        if not folder.exists():
            self._log(f"Invalid folder: {folder}")
            return
        folder = self._resolve_folder(folder)
        self._busy = True
        self.btn_run_pybomb.setEnabled(False)
        worker = FunctionWorker(run_pybombcell_on_folder, str(folder), True)
        worker.signals.error.connect(self._log)
        worker.signals.finished.connect(lambda result: self._on_pybomb_done(result, folder))
        self.pool.start(worker)

    def _on_pybomb_done(self, result: dict, folder: Path) -> None:
        self._busy = False
        self.btn_run_pybomb.setEnabled(True)
        if not result.get("ok"):
            self._log("py_bombcell run failed.")
            return
        payload = result.get("result", {})
        if payload.get("cached", False):
            self._log(f"py_bombcell skipped (cached metrics found): {payload.get('metrics_csv', '')}")
        metrics_path = Path(str(payload.get("metrics_csv", "")))
        if not metrics_path.exists():
            metrics_path = folder / "bombcell" / "templates._bc_qMetrics.csv"
        if not metrics_path.exists():
            self._log("py_bombcell finished but metrics file was not found.")
            return
        self.df = pd.read_csv(metrics_path)
        if self.df.columns.size > 0 and str(self.df.columns[0]).lower().startswith("unnamed"):
            self.df = self.df.drop(columns=[self.df.columns[0]])
        self.metric_combo.clear()
        numeric_cols = [c for c in self.df.columns if np.issubdtype(self.df[c].dtype, np.number)]
        self.metric_combo.addItems(numeric_cols)
        self._refresh_table()
        self._refresh_plot()
        self._log(f"Loaded py_bombcell metrics: {metrics_path}")

    def _filtered_df(self) -> pd.DataFrame:
        if self.df.empty:
            return self.df
        text = self.ed_filter.text().strip().lower()
        if not text:
            return self.df
        mask = self.df.astype(str).apply(lambda c: c.str.lower().str.contains(text, na=False))
        return self.df[mask.any(axis=1)]

    def _refresh_table(self) -> None:
        sdf = self._filtered_df()
        self.table.clear()
        self.table.setRowCount(len(sdf))
        self.table.setColumnCount(len(sdf.columns))
        self.table.setHorizontalHeaderLabels([str(c) for c in sdf.columns])
        for r, (_, row) in enumerate(sdf.iterrows()):
            for c, col in enumerate(sdf.columns):
                self.table.setItem(r, c, QtWidgets.QTableWidgetItem(str(row[col])))
        self.table.resizeColumnsToContents()

    def _refresh_plot(self) -> None:
        self.plot.clear()
        if self.df.empty:
            return
        metric = self.metric_combo.currentText().strip()
        if not metric:
            return
        y = pd.to_numeric(self._filtered_df()[metric], errors="coerce").dropna().to_numpy()
        if y.size == 0:
            return
        hist, edges = np.histogram(y, bins=min(70, max(10, int(np.sqrt(y.size)))))
        bar = pg.BarGraphItem(x=edges[:-1], height=hist, width=np.diff(edges), brush=(90, 165, 255, 160))
        self.plot.addItem(bar)
        self.plot.setLabel("bottom", metric)
        self.plot.setLabel("left", "count")

    def _restore_settings(self) -> None:
        folder = self.qsettings.value("quality/last_folder", "")
        if folder:
            self.ed_folder.setText(str(folder))

    def set_plot_preferences(self, theme: str, show_grid: bool) -> None:
        self._plot_theme = "Dark" if str(theme).lower().startswith("dark") else "Light"
        self._show_grid = bool(show_grid)
        bg = "#0b0f14" if self._plot_theme == "Dark" else "#ffffff"
        fg = "#e8eef7" if self._plot_theme == "Dark" else "#1a1f29"
        self.plot.setBackground(bg)
        self.plot.getAxis("left").setTextPen(pg.mkPen(fg))
        self.plot.getAxis("bottom").setTextPen(pg.mkPen(fg))
        self.plot.getAxis("left").setPen(pg.mkPen(fg))
        self.plot.getAxis("bottom").setPen(pg.mkPen(fg))
        self.plot.showGrid(x=self._show_grid, y=self._show_grid, alpha=0.25 if self._show_grid else 0.0)

    def _log(self, msg: str) -> None:
        self.log.appendPlainText(str(msg))

    def is_busy(self) -> bool:
        return bool(self._busy)

    def _export_plotted_data(self) -> None:
        if self.df.empty:
            self._log("Export: no metrics loaded.")
            return
        metric = self.metric_combo.currentText().strip()
        if not metric:
            self._log("Export: no metric selected.")
            return
        start = self.qsettings.value("paths/last_folder", str(Path.cwd()))
        fp, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export quality metric plotted data", str(start), "CSV files (*.csv)")
        if not fp:
            return
        base = Path(fp)
        sdf = self._filtered_df().copy()
        sdf.to_csv(base.with_name(base.stem + "_filtered_table.csv"), index=False)
        y = pd.to_numeric(sdf[metric], errors="coerce").dropna().to_numpy(dtype=float)
        if y.size:
            hist, edges = np.histogram(y, bins=min(70, max(10, int(np.sqrt(y.size)))))
            hist_df = pd.DataFrame(
                {
                    "metric": metric,
                    "bin_left": edges[:-1],
                    "bin_right": edges[1:],
                    "count": hist,
                }
            )
            hist_df.to_csv(base, index=False)
        self._log(f"Exported quality plotted data: {base}")

    def _toggle_plot_detach(self, checked: bool) -> None:
        if checked:
            self._detach_plots()
        else:
            self._attach_plots()

    def _detach_plots(self) -> None:
        if self._plots_detached or self._splitter is None or self._right_panel is None:
            return
        self._split_sizes_before_detach = self._splitter.sizes()
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Quality Metrics plot")
        dlg.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        dlg.resize(1050, 740)
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.setContentsMargins(4, 4, 4, 4)
        self._right_panel.setParent(None)
        lay.addWidget(self._right_panel, 1)
        dlg.finished.connect(lambda _=0: self._attach_plots())
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        self._plots_dialog = dlg
        self._plots_detached = True
        self._splitter.setSizes([1, 0])
        self.btn_detach_plots.setText("Attach plots")

    def _attach_plots(self) -> None:
        if not self._plots_detached or self._splitter is None or self._right_panel is None:
            self.btn_detach_plots.blockSignals(True)
            self.btn_detach_plots.setChecked(False)
            self.btn_detach_plots.blockSignals(False)
            self.btn_detach_plots.setText("Detach plots")
            return
        if self._plots_dialog is not None and self._plots_dialog.layout() is not None:
            self._plots_dialog.layout().removeWidget(self._right_panel)
        self._splitter.addWidget(self._right_panel)
        self._splitter.setStretchFactor(0, 3)
        self._splitter.setStretchFactor(1, 2)
        if self._split_sizes_before_detach:
            self._splitter.setSizes(self._split_sizes_before_detach)
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



"""Preprocessing tab for the NeuroPyGuiN GUI.

Builds the queue-based SpikeGLX/Kilosort preprocessing UI: a drop-and-scan run
queue, the per-step configuration form, optional multi-session concatenation,
and a completed-runs history. Pipeline execution itself is delegated to the
worker classes in ``..workers``; this module only wires the widgets, persists
settings via ``QSettings``, and reflects worker progress in the step panel.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import json
import math

from PySide6 import QtCore, QtGui, QtWidgets

from ..preprocessing import (
    build_concat_run_name,
    completed_run_target_folders,
    default_concat_run_layout,
    default_kilosort_output_name,
    discover_completed_runs,
    discover_bin_files,
    find_meta_for_bin,
    is_concatenated_run_bin,
    mirrored_concat_base_dir,
    parse_spikeglx_bin_name,
    validate_concat_inputs,
    validate_spikeglx_ap_bin,
)
from ..side_nav import SideNavStack
from ..string_builders import (
    BitFieldBuilderDialog,
    CatGTStringBuilderDialog,
    TPrimeStringBuilderDialog,
    catgt_command_bf_extractors,
    catgt_command_extractors,
    merge_bitfields_into_catgt_command,
    merge_extractors_into_catgt_command,
)
from ..workers import (
    ConcatenationConfig,
    ConcatenationWorker,
    EcephysPipelineConfig,
    EcephysPipelineWorker,
)


class BinDropList(QtWidgets.QListWidget):
    """List widget that accepts file drops and re-emits them as local paths.

    Dropped URLs are filtered to local files and forwarded via ``filesDropped``
    so the owning tab can validate and queue SpikeGLX AP bins.
    """

    filesDropped = QtCore.Signal(list)

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DropOnly)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        if event.mimeData().hasUrls():
            files = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
            self.filesDropped.emit(files)
            event.acceptProposedAction()
            return
        super().dropEvent(event)


class StepStatusItem(QtWidgets.QWidget):
    """One pipeline step row in the live status panel.

    Shows a title, an icon (animated while running), a percent label, and a
    progress bar. The ``set_*`` methods switch between the pending, running,
    progressing, finished, and failed visual states.
    """

    _ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets"
    _OK_ICON = _ASSET_ROOT / "ok-icon.png"
    _LOADING_ICON = _ASSET_ROOT / "loading-icon.gif"
    _FAILED_ICON = _ASSET_ROOT / "failed.gif"

    def __init__(self, title: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._movie: QtGui.QMovie | None = None
        self._title = title
        self.setProperty("stepStatusItem", True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        self.icon_label = QtWidgets.QLabel()
        self.icon_label.setFixedSize(18, 18)
        self.icon_label.setScaledContents(True)
        self.icon_label.setAlignment(QtCore.Qt.AlignCenter)
        self.title_label = QtWidgets.QLabel(title)
        self.title_label.setObjectName("StepStatusTitle")
        self.percent_label = QtWidgets.QLabel("Pending")
        self.percent_label.setObjectName("StepStatusPercent")
        header.addWidget(self.icon_label, 0)
        header.addWidget(self.title_label, 1)
        header.addWidget(self.percent_label, 0)
        layout.addLayout(header)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setProperty("stepProgress", True)
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress, 0)
        self.set_pending()

    def _clear_movie(self) -> None:
        if self._movie is not None:
            self._movie.stop()
        self._movie = None
        self.icon_label.clear()

    def _set_icon_pixmap(self, path: Path) -> None:
        self._clear_movie()
        pixmap = QtGui.QPixmap(str(path))
        if not pixmap.isNull():
            self.icon_label.setPixmap(pixmap.scaled(18, 18, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def _set_icon_movie(self, path: Path) -> None:
        self._clear_movie()
        movie = QtGui.QMovie(str(path))
        if not movie.isValid():
            return
        movie.setScaledSize(QtCore.QSize(18, 18))
        self.icon_label.setMovie(movie)
        movie.start()
        self._movie = movie

    def set_pending(self) -> None:
        self._clear_movie()
        self.percent_label.setText("Pending")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

    def set_running(self) -> None:
        self._set_icon_movie(self._LOADING_ICON)
        self.percent_label.setText("Running")
        self.progress.setRange(0, 0)

    def set_progress(self, percent: int) -> None:
        percent = max(0, min(100, int(percent)))
        if self.progress.maximum() == 0:
            self.progress.setRange(0, 100)
        self.progress.setValue(percent)
        self.percent_label.setText(f"{percent}%")
        if percent < 100 and self._movie is None:
            self._set_icon_movie(self._LOADING_ICON)

    def set_finished(self) -> None:
        self._set_icon_pixmap(self._OK_ICON)
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.percent_label.setText("100%")

    def set_failed(self) -> None:
        self._set_icon_movie(self._FAILED_ICON)
        if self.progress.maximum() == 0:
            self.progress.setRange(0, 100)
        self.percent_label.setText("Failed")


class Ks4AdvancedDialog(QtWidgets.QDialog):
    """Form dialog for the advanced Kilosort 4 parameters.

    Builds one editor per entry in ``PARAM_SPECS`` (grouped by ``PARAM_GROUPS``)
    and exposes the edited values via ``values()``. Blank float fields default to
    the spec default, with NaN defaults mapped to ``None`` to request auto-estimation.
    """

    PARAM_SPECS = [
        ("n_chan_bin", int, 385, "Total channels in the binary file (including non-ephys channels)."),
        ("batch_size", int, 60000, "Number of samples per sorting batch."),
        ("Th_universal", float, 9.0, "Universal spike detection threshold. Lower to detect more units."),
        ("Th_learned", float, 8.0, "Learned-template threshold. Lower if neurons disappear/reappear over time."),
        ("Th_single_ch", float, 8.0, "Single-channel threshold in PCA projection space."),
        ("nblocks", int, 5, "Drift correction blocks: 0=off, 1=rigid, >1=non-rigid."),
        ("tmin", float, 0.0, "Start time (s) included in sorting."),
        ("tmax", float, -1.0, "End time (s). Use -1 for full session (equivalent to inf)."),
        ("nt", int, 61, "Number of time samples in waveform representation."),
        ("dmin", float, float("nan"), "Vertical spacing for universal templates (um); use NaN to auto-estimate."),
        ("dminx", float, 32.0, "Lateral spacing for universal templates (um)."),
        ("min_template_size", float, 10.0, "Minimum Gaussian template size (um)."),
        ("nearest_chans", int, 10, "Nearest channels for local maxima during detection."),
        ("nearest_templates", int, 150, "Nearest templates for local maxima during detection."),
        ("template_sizes", int, 7, "Number of template sizes tested."),
        ("templates_from_data", bool, True, "If true, initialize templates from data."),
        ("whitening_range", int, 32, "Channels used for whitening each channel."),
        ("sig_interp", float, 20.0, "Sigma (um) for drift interpolation."),
        ("ccg_threshold", float, 0.1, "CCG refractory-violation threshold for split/merge."),
        ("acg_threshold", float, 0.25, "ACG refractory-violation threshold for unit quality."),
        ("template_seed", int, 1, "Seed for template initialization."),
        ("cluster_seed", int, 1, "Seed for clustering."),
    ]
    PARAM_GROUPS = [
        (
            "Input and timing",
            "Input and timing",
            "Core data-shape and time-window values for the sorter.",
            ["n_chan_bin", "batch_size", "tmin", "tmax", "nt"],
        ),
        (
            "Detection",
            "Detection",
            "Thresholds and nearest-neighbor settings that control spike detection sensitivity.",
            ["Th_universal", "Th_learned", "Th_single_ch", "nearest_chans", "nearest_templates"],
        ),
        (
            "Templates and drift",
            "Templates and drift",
            "Template geometry and drift-correction parameters.",
            ["nblocks", "dmin", "dminx", "min_template_size", "template_sizes", "templates_from_data", "whitening_range", "sig_interp"],
        ),
        (
            "Quality and seeds",
            "Quality and seeds",
            "Split/merge quality thresholds and reproducibility seeds.",
            ["ccg_threshold", "acg_threshold", "template_seed", "cluster_seed"],
        ),
    ]

    def __init__(self, values: Dict[str, object], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("compactDialog", True)
        self.setWindowTitle("Advanced KS4 Parameters")
        self.resize(960, 700)
        main = QtWidgets.QVBoxLayout(self)
        self._editors: Dict[str, QtWidgets.QWidget] = {}

        note = QtWidgets.QLabel("Only parameters supported by current ks4_helper schema are shown here.")
        note.setObjectName("SectionHint")
        note.setWordWrap(True)
        main.addWidget(note)

        sections = SideNavStack(
            "Sections",
            "Edit one KS4 parameter group at a time instead of scrolling through a long form.",
        )
        main.addWidget(sections, 1)

        spec_by_key = {key: (key, typ, default, desc) for key, typ, default, desc in self.PARAM_SPECS}

        def _editor_for_spec(key: str, typ, default, desc: str) -> QtWidgets.QWidget:
            val = values.get(key, default)
            editor: QtWidgets.QWidget
            if typ is bool:
                w = QtWidgets.QCheckBox()
                w.setChecked(bool(val))
                editor = w
            else:
                w = QtWidgets.QLineEdit(str(val))
                editor = w
            help_btn = QtWidgets.QToolButton()
            help_btn.setText("?")
            help_btn.setToolTip(desc)
            help_btn.setAutoRaise(True)
            help_btn.setProperty("helpButton", True)
            row = QtWidgets.QHBoxLayout()
            row.addWidget(editor, 1)
            row.addWidget(help_btn, 0)
            wrap = QtWidgets.QWidget()
            wrap.setLayout(row)
            self._editors[key] = editor
            return wrap

        for label, title, subtitle, keys in self.PARAM_GROUPS:
            page = QtWidgets.QWidget()
            page_layout = QtWidgets.QVBoxLayout(page)
            page_layout.setContentsMargins(0, 0, 0, 0)
            page_layout.setSpacing(12)
            box = QtWidgets.QGroupBox(title)
            box.setProperty("settingsSection", True)
            box_layout = QtWidgets.QVBoxLayout(box)
            box_layout.setSpacing(10)
            hint = QtWidgets.QLabel(subtitle)
            hint.setObjectName("SectionHint")
            hint.setWordWrap(True)
            box_layout.addWidget(hint)
            form = QtWidgets.QFormLayout()
            form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
            form.setFormAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
            form.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            form.setHorizontalSpacing(14)
            form.setVerticalSpacing(10)
            for key in keys:
                _key, typ, default, desc = spec_by_key[key]
                form.addRow(key, _editor_for_spec(key, typ, default, desc))
            box_layout.addLayout(form)
            page_layout.addWidget(box)
            page_layout.addStretch(1)
            sections.add_page(label, page)

        sections.setCurrentIndex(0)
        self.sections = sections

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        main.addWidget(btns)

    def values(self) -> Dict[str, object]:
        """Collect the edited parameters, coercing each field to its spec type.

        Empty fields fall back to the spec default (NaN float defaults become
        ``None``). For floats, ``inf`` maps to ``-1.0`` (full session) and
        ``nan``/``none``/``auto`` map to ``None``. Unparsable input keeps the default.
        """
        out: Dict[str, object] = {}
        for key, typ, default, _desc in self.PARAM_SPECS:
            ed = self._editors[key]
            if typ is bool and isinstance(ed, QtWidgets.QCheckBox):
                out[key] = bool(ed.isChecked())
                continue
            txt = ed.text().strip() if isinstance(ed, QtWidgets.QLineEdit) else ""
            if txt == "":
                if isinstance(default, float) and math.isnan(default):
                    out[key] = None
                else:
                    out[key] = default
                continue
            try:
                if typ is int:
                    out[key] = int(float(txt))
                elif typ is float:
                    if txt.lower() in {"inf", "np.inf"}:
                        out[key] = -1.0
                    elif txt.lower() in {"nan", "none", "auto"}:
                        out[key] = None
                    else:
                        out[key] = float(txt)
                else:
                    out[key] = txt
            except Exception:
                out[key] = default
        return out


class ConcatenateDialog(QtWidgets.QDialog):
    """Configure a multi-session concatenation for joint spike sorting."""

    def __init__(
        self,
        source_runs: List[str],
        default_output_dir: str,
        default_run_name: str,
        defaults: Dict[str, object],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setProperty("compactDialog", True)
        self.setWindowTitle("Concatenate recordings for joint spike sorting")
        self.resize(680, 540)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        intro = QtWidgets.QLabel(
            "Fuse the selected AP recordings into one binary so Kilosort sorts them together and "
            "assigns the same unit identities across sessions. The original recordings are left "
            "untouched, and a split-info map is saved so sorted spikes can be separated back into "
            "each session afterward. Concatenation follows the order listed below (queue order)."
        )
        intro.setObjectName("SectionHint")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        order_label = QtWidgets.QLabel("Concatenation order")
        order_label.setObjectName("FieldTitle")
        layout.addWidget(order_label)
        order_list = QtWidgets.QListWidget()
        order_list.setAlternatingRowColors(True)
        order_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        order_list.setMaximumHeight(150)
        for idx, run in enumerate(source_runs, start=1):
            order_list.addItem(f"{idx}. {run}")
        layout.addWidget(order_list)

        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)

        self.ed_run = QtWidgets.QLineEdit(default_run_name)
        self.ed_dir = QtWidgets.QLineEdit(default_output_dir)
        btn_browse = QtWidgets.QPushButton("Browse")
        btn_browse.setProperty("role", "ghost")
        dir_row = QtWidgets.QHBoxLayout()
        dir_row.setContentsMargins(0, 0, 0, 0)
        dir_row.addWidget(self.ed_dir, 1)
        dir_row.addWidget(btn_browse, 0)
        dir_wrap = QtWidgets.QWidget()
        dir_wrap.setLayout(dir_row)

        self.ck_svd = QtWidgets.QCheckBox("Remove shared SVD components (artifact cleaning)")
        self.ck_svd.setChecked(bool(defaults.get("svd_clean", True)))
        self.ck_svd.setToolTip(
            "Removes the leading shared spatial components of the AP channel block batch by batch, "
            "matching the legacy MATLAB concatenateAndCleanBinaries denoising. Turn off for a plain "
            "raw concatenation when CatGT filtering will run afterward."
        )
        self.sp_comp = QtWidgets.QSpinBox()
        self.sp_comp.setRange(0, 50)
        self.sp_comp.setValue(int(defaults.get("n_svd_components", 5)))
        self.sp_batch = QtWidgets.QDoubleSpinBox()
        self.sp_batch.setRange(0.05, 10.0)
        self.sp_batch.setDecimals(2)
        self.sp_batch.setSingleStep(0.1)
        self.sp_batch.setValue(float(defaults.get("batch_seconds", 0.5)))

        self.ck_extract_ni = QtWidgets.QCheckBox(
            "Also queue NI event extraction (CatGT extract-only) for each source session"
        )
        self.ck_extract_ni.setChecked(bool(defaults.get("extract_ni", True)))
        self.ck_extract_ni.setToolTip(
            "Queues each original session for a CatGT extract-only pass so its NI digital/analog "
            "event files are produced. The joint sort is done on the fused file, while events stay "
            "per session; the curation 'Split concat → sessions' step then attaches each session's "
            "events to its split spike trains. The source sessions are switched to events-only in "
            "the queue so they are not sorted again individually."
        )

        form.addRow("Combined run name", self.ed_run)
        form.addRow("Output folder", dir_wrap)
        form.addRow("", self.ck_svd)
        form.addRow("SVD components to remove", self.sp_comp)
        form.addRow("SVD batch length (s)", self.sp_batch)
        form.addRow("", self.ck_extract_ni)
        layout.addLayout(form)

        note = QtWidgets.QLabel(
            "A new SpikeGLX-style run folder is created under the output folder, so every "
            "downstream preprocessing step (CatGT, Kilosort, metrics) treats the fused file like an "
            "ordinary recording."
        )
        note.setObjectName("SectionHint")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btns.button(QtWidgets.QDialogButtonBox.Ok).setText("Concatenate")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        btn_browse.clicked.connect(self._pick_output_dir)
        self.ck_svd.toggled.connect(self.sp_comp.setEnabled)
        self.ck_svd.toggled.connect(self.sp_batch.setEnabled)
        self.sp_comp.setEnabled(self.ck_svd.isChecked())
        self.sp_batch.setEnabled(self.ck_svd.isChecked())

    def _pick_output_dir(self) -> None:
        start = self.ed_dir.text().strip() or str(Path.cwd())
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder", start)
        if folder:
            self.ed_dir.setText(folder)

    def values(self) -> Dict[str, object]:
        return {
            "run_name": self.ed_run.text().strip(),
            "output_dir": self.ed_dir.text().strip(),
            "svd_clean": self.ck_svd.isChecked(),
            "n_svd_components": int(self.sp_comp.value()),
            "batch_seconds": float(self.sp_batch.value()),
            "extract_ni": self.ck_extract_ni.isChecked(),
        }


class PreprocessingTab(QtWidgets.QWidget):
    """Main preprocessing tab: queue, settings, run control, and history.

    Owns the run queue and raw-run catalog, the per-step configuration widgets,
    and the completed-runs list. Submits jobs to the shared thread pool through
    ``EcephysPipelineWorker`` (and ``ConcatenationWorker`` for fused runs), and
    emits the ``open*Requested`` signals so the host window can switch tabs.
    """

    openCurationRequested = QtCore.Signal(list)
    openPostProcessingRequested = QtCore.Signal(str)
    openHistologyRequested = QtCore.Signal(str)
    saveSettingsFileRequested = QtCore.Signal()
    loadSettingsFileRequested = QtCore.Signal()
    _OK_ICON = Path(__file__).resolve().parents[1] / "assets" / "ok-icon.png"

    def __init__(self, thread_pool: QtCore.QThreadPool) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self.pool = thread_pool
        self.settings = QtCore.QSettings("NeuroPyGuiN", "NeuroPyGuiN")
        self._settings_sync_timer = QtCore.QTimer(self)
        self._settings_sync_timer.setSingleShot(True)
        self._settings_sync_timer.setInterval(200)
        self._settings_sync_timer.timeout.connect(self.settings.sync)
        self._restoring_settings = False
        self.jobs: List[Dict[str, str]] = []
        self._raw_run_catalog: Dict[str, Dict[str, str]] = {}
        self.completed_runs: List[Dict[str, object]] = []
        self._queue: List[Dict[str, str]] = []
        self._running = False
        self._concatenating = False
        self._pending_concat_ni_extract: List[str] = []
        self._ks4_adv_params: Dict[str, object] = {}
        self._active_run_context: Dict[str, object] | None = None

        self._build_ui()
        self._restore_settings()

    def _build_ui(self) -> None:
        main = QtWidgets.QVBoxLayout(self)
        main.setContentsMargins(18, 16, 18, 18)
        main.setSpacing(14)
        def help_button(text: str) -> QtWidgets.QToolButton:
            q = QtWidgets.QToolButton()
            q.setText("?")
            q.setAutoRaise(True)
            q.setToolTip(text)
            q.setProperty("helpButton", True)
            return q

        def with_help(widget: QtWidgets.QWidget, text: str) -> QtWidgets.QWidget:
            q = help_button(text)
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(widget, 1)
            row.addWidget(q, 0)
            host = QtWidgets.QWidget()
            host.setLayout(row)
            return host

        def make_field(title: str, widget: QtWidgets.QWidget, text: str) -> QtWidgets.QWidget:
            title_label = QtWidgets.QLabel(title)
            title_label.setObjectName("FieldTitle")
            title_label.setWordWrap(True)
            header = QtWidgets.QHBoxLayout()
            header.setContentsMargins(0, 0, 0, 0)
            header.setSpacing(6)
            header.addWidget(title_label, 1)
            header.addWidget(help_button(text), 0)
            layout = QtWidgets.QVBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)
            layout.addLayout(header)
            layout.addWidget(widget, 0)
            host = QtWidgets.QWidget()
            host.setLayout(layout)
            return host

        def make_section(title: str, subtitle: str) -> tuple[QtWidgets.QGroupBox, QtWidgets.QGridLayout]:
            box = QtWidgets.QGroupBox(title)
            box.setProperty("settingsSection", True)
            layout = QtWidgets.QVBoxLayout(box)
            layout.setSpacing(10)
            hint = QtWidgets.QLabel(subtitle)
            hint.setObjectName("SectionHint")
            hint.setWordWrap(True)
            layout.addWidget(hint)
            grid = QtWidgets.QGridLayout()
            grid.setHorizontalSpacing(14)
            grid.setVerticalSpacing(10)
            grid.setColumnStretch(0, 1)
            grid.setColumnStretch(1, 1)
            layout.addLayout(grid)
            return box, grid

        top = QtWidgets.QHBoxLayout()
        top.setSpacing(8)
        self.btn_add_files = QtWidgets.QPushButton("Add AP .bin files")
        self.btn_add_folder = QtWidgets.QPushButton("Add folder")
        self.btn_scan_raw_root = QtWidgets.QPushButton("Scan raw root")
        self.btn_recent_files = QtWidgets.QPushButton("Open recent file")
        self.btn_recent_folders = QtWidgets.QPushButton("Open recent folder")
        self.cb_queue_filter = QtWidgets.QComboBox()
        self.cb_queue_filter.addItem("Non-processed", "non_processed")
        self.cb_queue_filter.addItem("All runs", "all")
        self.btn_concat = QtWidgets.QPushButton("Concatenate selected")
        self.btn_remove = QtWidgets.QPushButton("Remove selected")
        self.btn_clear = QtWidgets.QPushButton("Clear")
        self.btn_run = QtWidgets.QPushButton("Run queue")
        self.btn_add_files.setProperty("role", "secondary")
        self.btn_add_folder.setProperty("role", "secondary")
        self.btn_scan_raw_root.setProperty("role", "secondary")
        self.btn_recent_files.setProperty("role", "ghost")
        self.btn_recent_folders.setProperty("role", "ghost")
        self.btn_concat.setProperty("role", "secondary")
        self.btn_concat.setEnabled(False)
        self.btn_concat.setToolTip(
            "Fuse 2+ selected AP recordings into a single binary so Kilosort sorts them jointly and "
            "tracks the same units across sessions. Produces a new queued run plus a split-info map "
            "for separating spikes per session afterward."
        )
        self.btn_remove.setProperty("role", "ghost")
        self.btn_clear.setProperty("role", "ghost")
        self.btn_run.setProperty("role", "primary")
        top.addWidget(self.btn_add_files)
        top.addWidget(self.btn_add_folder)
        top.addWidget(self.btn_scan_raw_root)
        top.addWidget(self.btn_recent_files)
        top.addWidget(self.btn_recent_folders)
        top.addStretch(1)
        top.addWidget(QtWidgets.QLabel("Show"))
        top.addWidget(self.cb_queue_filter)
        top.addWidget(self.btn_concat)
        top.addWidget(self.btn_remove)
        top.addWidget(self.btn_clear)
        top.addWidget(self.btn_run)

        self.list_jobs = BinDropList()
        self.list_jobs.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.list_jobs.setToolTip("Drop SpikeGLX AP files here: *.imecX.ap.bin")
        self.list_jobs.setProperty("dropZone", True)
        self.list_jobs.setAlternatingRowColors(True)
        self.list_jobs.setSpacing(4)
        self.list_jobs.setMinimumHeight(220)
        self.list_jobs.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.list_jobs.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)

        config = QtWidgets.QGroupBox("Integrated ecephys spike sorting preprocessing")
        config.setProperty("settingsSection", True)
        config.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        config_layout = QtWidgets.QVBoxLayout(config)
        config_layout.setSpacing(12)

        self.ck_catgt = QtWidgets.QCheckBox("CatGT")
        self.ck_catgt_extract_only = QtWidgets.QCheckBox("CatGT extract-only")
        self.ck_save_catgt_ap_bin = QtWidgets.QCheckBox("Save CatGT AP bin")
        self.ck_tprime = QtWidgets.QCheckBox("TPrime")
        self.ck_ks = QtWidgets.QCheckBox("Kilosort")
        self.ck_post = QtWidgets.QCheckBox("Kilosort Postprocessing")
        self.ck_noise = QtWidgets.QCheckBox("Noise Templates")
        self.ck_wvf = QtWidgets.QCheckBox("Mean Waveforms")
        self.ck_qm = QtWidgets.QCheckBox("Quality Metrics")
        self.ck_pybomb = QtWidgets.QCheckBox("py_bombcell")

        self.ck_catgt.setChecked(True)
        self.ck_ks.setChecked(True)
        self.ck_post.setChecked(True)
        self.ck_wvf.setChecked(True)
        self.ck_qm.setChecked(True)
        self.ck_pybomb.setChecked(True)
        self.ck_catgt_extract_only.setToolTip(
            "Run a CatGT extract-only pass to regenerate XA/XD/XIA/XID event text files without repeating filtering "
            "or gfix. Queue the raw *.imecX.ap.bin input for NI extraction or TPrime alignment because the NI "
            "stream lives at the run root; existing *_tcat inputs are only suitable for probe/AP-only reruns."
        )
        self.ck_save_catgt_ap_bin.setToolTip(
            "When enabled, raw-input CatGT extract-only reruns keep a real *_tcat.imecX.ap.bin in the CatGT folder "
            "by running the full CatGT AP-processing path. Disable this to regenerate only the extractor text files."
        )

        modules_row = QtWidgets.QVBoxLayout()
        for w in [self.ck_catgt, self.ck_catgt_extract_only, self.ck_ks, self.ck_post, self.ck_noise, self.ck_wvf, self.ck_qm, self.ck_pybomb, self.ck_tprime]:
            modules_row.addWidget(w)

        self.cb_ks_ver = QtWidgets.QComboBox()
        self.cb_ks_ver.addItems(["4", "3.0", "2.5", "2.0"])
        self.btn_adv_ks4 = QtWidgets.QPushButton("Advanced sorting parameters")
        self.btn_adv_ks4.setProperty("role", "secondary")

        self.ed_gate = QtWidgets.QLineEdit("0")
        self.ed_trigger = QtWidgets.QLineEdit("0,0")
        self.ed_probe = QtWidgets.QLineEdit("0")
        self.ed_region = QtWidgets.QLineEdit("default")
        self.ed_ks_th = QtWidgets.QLineEdit("[8,9]")
        self.ed_qm_isi = QtWidgets.QDoubleSpinBox()
        self.ed_qm_isi.setRange(0.0001, 0.01)
        self.ed_qm_isi.setDecimals(4)
        self.ed_qm_isi.setValue(0.002)
        self.ed_sync_period = QtWidgets.QDoubleSpinBox()
        self.ed_sync_period.setRange(0.1, 10.0)
        self.ed_sync_period.setValue(1.0)
        self.ed_sync_period.setDecimals(3)

        self.ed_ni_extract = QtWidgets.QLineEdit("-xd=0,0,8,7,0 -xd=0,0,8,5,0 -xd=0,0,8,6,0 -xd=0,0,8,3,0")
        self.ed_tostream = QtWidgets.QLineEdit("imec0")
        self.ed_catgt_cmd = QtWidgets.QLineEdit("-prb_fld -out_prb_fld -apfilter=butter,12,300,10000 -gfix=0.4,0.10,0.02")
        self.btn_build_tprime = QtWidgets.QPushButton("Build")
        self.btn_build_catgt = QtWidgets.QPushButton("Build")
        self.btn_build_bitfield = QtWidgets.QPushButton("Bit-field")
        self.btn_build_tprime.setProperty("role", "secondary")
        self.btn_build_catgt.setProperty("role", "secondary")
        self.btn_build_bitfield.setProperty("role", "ghost")
        self.btn_build_tprime.setToolTip(
            "Open the TPrime/CatGT extractor builder. Includes a guided preset for aligning NI analog channels "
            "onto an imec stream such as imec0."
        )
        self.cb_catgt_car_mode = QtWidgets.QComboBox()
        self.cb_catgt_car_mode.addItems(["gbldmx", "loccar", "none"])
        self.sp_loccar_min = QtWidgets.QDoubleSpinBox()
        self.sp_loccar_min.setRange(1.0, 500.0)
        self.sp_loccar_min.setValue(40.0)
        self.sp_loccar_max = QtWidgets.QDoubleSpinBox()
        self.sp_loccar_max.setRange(1.0, 1000.0)
        self.sp_loccar_max.setValue(160.0)
        self.sp_ks4_dup_ms = QtWidgets.QDoubleSpinBox()
        self.sp_ks4_dup_ms.setDecimals(3)
        self.sp_ks4_dup_ms.setRange(0.01, 2.0)
        self.sp_ks4_dup_ms.setValue(0.25)
        self.sp_ks4_min_template = QtWidgets.QDoubleSpinBox()
        self.sp_ks4_min_template.setRange(1.0, 100.0)
        self.sp_ks4_min_template.setValue(10.0)
        self.sp_cwaves_um = QtWidgets.QDoubleSpinBox()
        self.sp_cwaves_um.setRange(10.0, 400.0)
        self.sp_cwaves_um.setValue(160.0)

        self.ed_catgt_path = QtWidgets.QLineEdit(str((Path.cwd() / "tools" / "CatGT-win").resolve()))
        self.ed_tprime_path = QtWidgets.QLineEdit(str((Path.cwd() / "tools" / "TPrime-win").resolve()))
        self.ed_cwaves_path = QtWidgets.QLineEdit(str((Path.cwd() / "tools" / "C_Waves-win").resolve()))
        self.ed_ks4_repo = QtWidgets.QLineEdit(str((Path.cwd() / "tools" / "Kilosort" / "kilosort").resolve()))
        self.ed_ks_tmp = QtWidgets.QLineEdit(str((Path.cwd() / "kilosort_datatemp").resolve()))

        btn_catgt_path = QtWidgets.QPushButton("Browse")
        btn_tprime_path = QtWidgets.QPushButton("Browse")
        btn_cwaves_path = QtWidgets.QPushButton("Browse")
        btn_ks4_repo = QtWidgets.QPushButton("Browse")
        btn_ks_tmp = QtWidgets.QPushButton("Browse")
        for btn in [btn_catgt_path, btn_tprime_path, btn_cwaves_path, btn_ks4_repo, btn_ks_tmp]:
            btn.setProperty("role", "ghost")

        row_catgt_path = QtWidgets.QHBoxLayout(); row_catgt_path.setContentsMargins(0, 0, 0, 0)
        row_tprime_path = QtWidgets.QHBoxLayout(); row_tprime_path.setContentsMargins(0, 0, 0, 0)
        row_cwaves_path = QtWidgets.QHBoxLayout(); row_cwaves_path.setContentsMargins(0, 0, 0, 0)
        row_ks4_repo = QtWidgets.QHBoxLayout(); row_ks4_repo.setContentsMargins(0, 0, 0, 0)
        row_ks_tmp = QtWidgets.QHBoxLayout(); row_ks_tmp.setContentsMargins(0, 0, 0, 0)
        for row, ed, btn in [
            (row_catgt_path, self.ed_catgt_path, btn_catgt_path),
            (row_tprime_path, self.ed_tprime_path, btn_tprime_path),
            (row_cwaves_path, self.ed_cwaves_path, btn_cwaves_path),
            (row_ks4_repo, self.ed_ks4_repo, btn_ks4_repo),
            (row_ks_tmp, self.ed_ks_tmp, btn_ks_tmp),
        ]:
            row.addWidget(ed, 1)
            row.addWidget(btn, 0)
        wrap_catgt_path = QtWidgets.QWidget(); wrap_catgt_path.setLayout(row_catgt_path)
        wrap_tprime_path = QtWidgets.QWidget(); wrap_tprime_path.setLayout(row_tprime_path)
        wrap_cwaves_path = QtWidgets.QWidget(); wrap_cwaves_path.setLayout(row_cwaves_path)
        wrap_ks4_repo = QtWidgets.QWidget(); wrap_ks4_repo.setLayout(row_ks4_repo)
        wrap_ks_tmp = QtWidgets.QWidget(); wrap_ks_tmp.setLayout(row_ks_tmp)

        row_tostream = QtWidgets.QHBoxLayout()
        row_tostream.setContentsMargins(0, 0, 0, 0)
        row_tostream.addWidget(self.ed_tostream, 1)
        row_tostream.addWidget(self.btn_build_tprime, 0)
        wrap_tostream = QtWidgets.QWidget()
        wrap_tostream.setLayout(row_tostream)

        row_catgt_cmd = QtWidgets.QHBoxLayout()
        row_catgt_cmd.setContentsMargins(0, 0, 0, 0)
        row_catgt_cmd.addWidget(self.ed_catgt_cmd, 1)
        row_catgt_cmd.addWidget(self.btn_build_catgt, 0)
        row_catgt_cmd.addWidget(self.btn_build_bitfield, 0)
        wrap_catgt_cmd = QtWidgets.QWidget()
        wrap_catgt_cmd.setLayout(row_catgt_cmd)

        self.ed_output = QtWidgets.QLineEdit(str((Path.cwd() / "NeuroPyGuiN_output").resolve()))
        self.ck_mirror_raw_hierarchy_output = QtWidgets.QCheckBox(
            "Mirror rawData hierarchy into output root and append spike_sorting"
        )
        self.ck_mirror_raw_hierarchy_output.setChecked(True)
        btn_output = QtWidgets.QPushButton("Browse")
        btn_output.setProperty("role", "ghost")
        out_row = QtWidgets.QHBoxLayout()
        out_row.setContentsMargins(0, 0, 0, 0)
        out_row.addWidget(self.ed_output)
        out_row.addWidget(btn_output)
        out_wrap = QtWidgets.QWidget()
        out_wrap.setLayout(out_row)

        self.ed_json = QtWidgets.QLineEdit(str((Path.cwd() / "NeuroPyGuiN_json").resolve()))
        btn_json = QtWidgets.QPushButton("Browse")
        btn_json.setProperty("role", "ghost")
        json_row = QtWidgets.QHBoxLayout()
        json_row.setContentsMargins(0, 0, 0, 0)
        json_row.addWidget(self.ed_json)
        json_row.addWidget(btn_json)
        json_wrap = QtWidgets.QWidget()
        json_wrap.setLayout(json_row)

        steps_box = QtWidgets.QGroupBox("Pipeline steps")
        steps_box.setProperty("settingsSection", True)
        steps_layout = QtWidgets.QVBoxLayout(steps_box)
        steps_hint = QtWidgets.QLabel("Choose the preprocessing modules that should run for every queued recording.")
        steps_hint.setObjectName("SectionHint")
        steps_hint.setWordWrap(True)
        steps_layout.addWidget(steps_hint)
        steps_layout.addLayout(modules_row)
        config_layout.addWidget(steps_box)

        settings_columns = QtWidgets.QHBoxLayout()
        settings_columns.setSpacing(14)
        left_column = QtWidgets.QVBoxLayout()
        left_column.setSpacing(12)
        right_column = QtWidgets.QVBoxLayout()
        right_column.setSpacing(12)

        acquisition_box, acquisition_grid = make_section(
            "Run naming and sorter",
            "Session identifiers and main sorter settings used to generate the pipeline inputs.",
        )
        ks_ver_row = QtWidgets.QHBoxLayout()
        ks_ver_row.setContentsMargins(0, 0, 0, 0)
        ks_ver_row.addWidget(self.cb_ks_ver, 1)
        ks_ver_row.addWidget(self.btn_adv_ks4, 0)
        ks_ver_wrap = QtWidgets.QWidget()
        ks_ver_wrap.setLayout(ks_ver_row)
        acquisition_grid.addWidget(
            make_field("Kilosort version", ks_ver_wrap, "Select sorter backend version. Use 4 for Kilosort4 helper."),
            0,
            0,
        )
        acquisition_grid.addWidget(
            make_field("Gate string", self.ed_gate, "Gate id used by CatGT naming convention (g#)."),
            0,
            1,
        )
        acquisition_grid.addWidget(
            make_field("Trigger string", self.ed_trigger, "Trigger index (t#). Accepts single value or pair like 0,0."),
            1,
            0,
        )
        acquisition_grid.addWidget(
            make_field("Probe string", self.ed_probe, "Probe id in SpikeGLX naming convention (imec#)."),
            1,
            1,
        )
        acquisition_grid.addWidget(
            make_field("Region", self.ed_region, "Optional region label used in metadata outputs."),
            2,
            0,
            1,
            2,
        )

        metrics_box, metrics_grid = make_section(
            "Sorting and metrics",
            "Thresholds and downstream quality settings that shape unit detection and validation.",
        )
        metrics_grid.addWidget(
            make_field(
                "KS threshold [universal, learned]",
                self.ed_ks_th,
                "KS4 thresholds [Th_universal,Th_learned]. Lower values detect more spikes and more units.",
            ),
            0,
            0,
            1,
            2,
        )
        metrics_grid.addWidget(
            make_field(
                "Quality metrics ISI thresh (s)",
                self.ed_qm_isi,
                "Refractory-violation threshold used in the quality metrics module.",
            ),
            1,
            0,
        )
        metrics_grid.addWidget(
            make_field(
                "KS4 duplicate spike ms",
                self.sp_ks4_dup_ms,
                "Remove same-unit spikes within this interval as likely duplicates.",
            ),
            1,
            1,
        )
        metrics_grid.addWidget(
            make_field(
                "KS4 min template size (um)",
                self.sp_ks4_min_template,
                "Smallest Gaussian spatial envelope width for templates.",
            ),
            2,
            0,
        )
        metrics_grid.addWidget(
            make_field(
                "C_Waves SNR radius (um)",
                self.sp_cwaves_um,
                "Radius used for waveform SNR calculations.",
            ),
            2,
            1,
        )

        sync_box, sync_grid = make_section(
            "Sync and CatGT",
            "Synchronization inputs and CatGT-specific preprocessing values.",
        )
        sync_grid.addWidget(
            make_field("TPrime sync period", self.ed_sync_period, "Period (s) for synchronization pulses in TPrime."),
            0,
            0,
        )
        sync_grid.addWidget(
            make_field(
                "TPrime toStream",
                wrap_tostream,
                "Reference stream in TPrime nomenclature, for example ni, imec0, or obx0.",
            ),
            0,
            1,
        )
        sync_grid.addWidget(
            make_field("CatGT CAR mode", self.cb_catgt_car_mode, "CAR mode for CatGT: gbldmx, loccar, or none."),
            1,
            0,
        )
        sync_grid.addWidget(
            make_field("CatGT command string", wrap_catgt_cmd, "Additional raw CatGT flags appended to command line."),
            1,
            1,
        )
        sync_grid.addWidget(
            make_field("CatGT loccar min (um)", self.sp_loccar_min, "Inner radius for loccar mode in microns."),
            2,
            0,
        )
        sync_grid.addWidget(
            make_field("CatGT loccar max (um)", self.sp_loccar_max, "Outer radius for loccar mode in microns."),
            2,
            1,
        )
        sync_grid.addWidget(
            make_field(
                "TPrime/CatGT extractors",
                self.ed_ni_extract,
                "CatGT extractor flags used to generate TPrime-alignable event files. Supports NI, imec, and obx "
                "streams with rising/falling digital edges (xd/xid) and rising/falling analog edges (xa/xia). "
                "Use Build for a guided NI-analog-to-imec preset.",
            ),
            3,
            0,
            1,
            2,
        )

        paths_box, paths_grid = make_section(
            "Tool and output paths",
            "Executable folders and output roots used by the preprocessing pipeline.",
        )
        paths_grid.addWidget(
            make_field(
                "CatGT executable dir",
                wrap_catgt_path,
                "Path to the CatGT folder containing the CatGT executable.",
            ),
            0,
            0,
        )
        paths_grid.addWidget(
            make_field(
                "TPrime executable dir",
                wrap_tprime_path,
                "Path to the TPrime folder containing the TPrime executable.",
            ),
            0,
            1,
        )
        paths_grid.addWidget(
            make_field(
                "C_Waves executable dir",
                wrap_cwaves_path,
                "Path to the C_Waves folder containing the C_Waves executable.",
            ),
            1,
            0,
        )
        paths_grid.addWidget(
            make_field(
                "KS4 repository dir",
                wrap_ks4_repo,
                "Path to the Kilosort repository used by the helper metadata.",
            ),
            1,
            1,
        )
        paths_grid.addWidget(
            make_field(
                "Kilosort temp dir",
                wrap_ks_tmp,
                "Temporary fast directory used by sorting helpers.",
            ),
            2,
            0,
            1,
            2,
        )
        paths_grid.addWidget(
            make_field("Output root", out_wrap, "Root folder where per-run outputs are saved."),
            3,
            0,
            1,
            2,
        )
        paths_grid.addWidget(
            make_field(
                "Output layout",
                self.ck_mirror_raw_hierarchy_output,
                "When enabled, raw SpikeGLX inputs under a .../rawData/.../<session>/<run>/<probe>/ layout are "
                "written under <Output root>/.../<session>/spike_sorting/. The CatGT run folder then lives inside "
                "that spike_sorting folder.",
            ),
            4,
            0,
        )
        paths_grid.addWidget(
            make_field(
                "CatGT extract-only AP output",
                self.ck_save_catgt_ap_bin,
                "Keep a *_tcat.imecX.ap.bin and matching .meta in the CatGT probe folder when CatGT extract-only "
                "is used on raw AP input. This reruns the full CatGT AP processing instead of generating only text "
                "extractor outputs.",
            ),
            4,
            1,
            1,
            1,
        )
        paths_grid.addWidget(
            make_field("JSON root", json_wrap, "Folder where generated pipeline JSON files are stored."),
            5,
            0,
            1,
            2,
        )

        left_column.addWidget(acquisition_box)
        left_column.addWidget(metrics_box)
        left_column.addStretch(1)
        right_column.addWidget(sync_box)
        right_column.addWidget(paths_box)
        right_column.addStretch(1)

        settings_columns.addLayout(left_column, 1)
        settings_columns.addLayout(right_column, 1)
        config_layout.addLayout(settings_columns)

        queue_box = QtWidgets.QGroupBox("Queue")
        queue_box.setProperty("heroCard", True)
        queue_layout = QtWidgets.QVBoxLayout(queue_box)
        queue_layout.setSpacing(10)
        queue_hint = QtWidgets.QLabel(
            "Build a queue from SpikeGLX AP recordings. Drag AP .bin files into the area below or add folders explicitly."
        )
        queue_hint.setObjectName("SectionHint")
        queue_hint.setWordWrap(True)
        self.lbl_queue_summary = QtWidgets.QLabel()
        self.lbl_queue_summary.setObjectName("QueueSummary")
        queue_layout.addWidget(queue_hint)
        queue_layout.addLayout(top)
        queue_layout.addWidget(self.list_jobs, 1)
        queue_layout.addWidget(self.lbl_queue_summary)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        self.progress.setProperty("footerProgress", True)
        self.progress.setToolTip("Progress reflects the currently active preprocessing task.")

        self.btn_copy_log = QtWidgets.QPushButton("Copy log")
        self.btn_copy_log.setProperty("role", "secondary")

        done_box = QtWidgets.QGroupBox("Completed runs")
        done_box.setProperty("settingsSection", True)
        done_layout = QtWidgets.QVBoxLayout(done_box)
        done_layout.setSpacing(8)
        done_hint = QtWidgets.QLabel(
            "Finished runs stay in history until Clear History. You can also scan an existing processed-data root to import completed runs. Double-click opens curation, and right-click exposes more actions."
        )
        done_hint.setObjectName("SectionHint")
        done_hint.setWordWrap(True)
        done_layout.addWidget(done_hint)
        self.list_completed = QtWidgets.QListWidget()
        self.list_completed.setAlternatingRowColors(True)
        self.list_completed.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.list_completed.setSpacing(4)
        self.list_completed.setMinimumHeight(220)
        self.list_completed.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.list_completed.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        done_layout.addWidget(self.list_completed, 1)
        done_row = QtWidgets.QHBoxLayout()
        self.btn_scan_completed_root = QtWidgets.QPushButton("Scan processed root")
        self.btn_scan_completed_root.setProperty("role", "ghost")
        self.btn_to_curation = QtWidgets.QPushButton("To Curation")
        self.btn_to_curation.setProperty("role", "secondary")
        self.btn_to_histology = QtWidgets.QPushButton("To Histology")
        self.btn_to_histology.setProperty("role", "secondary")
        self.btn_to_histology.setToolTip(
            "Set up a histology session from the selected sorted run "
            "(auto-fills the Kilosort, ephys, and histology folders)."
        )
        done_row.addWidget(self.btn_scan_completed_root, 0)
        done_row.addStretch(1)
        done_row.addWidget(self.btn_to_curation)
        done_row.addWidget(self.btn_to_histology)
        done_layout.addLayout(done_row)

        log_box = QtWidgets.QGroupBox("Pipeline log")
        log_box.setProperty("settingsSection", True)
        log_layout = QtWidgets.QVBoxLayout(log_box)
        log_layout.setSpacing(8)
        log_header = QtWidgets.QHBoxLayout()
        log_hint = QtWidgets.QLabel("Live output from the preprocessing worker.")
        log_hint.setObjectName("SectionHint")
        log_hint.setWordWrap(True)
        log_header.addWidget(log_hint, 1)
        log_header.addWidget(self.btn_copy_log, 0)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setProperty("logView", True)
        self.log.setPlaceholderText("Pipeline output will appear here.")
        self.log.setMinimumHeight(220)
        self.log.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        log_body = QtWidgets.QHBoxLayout()
        log_body.setSpacing(12)
        log_body.addWidget(self.log, 3)

        step_status_panel = QtWidgets.QWidget()
        step_status_panel.setProperty("stepStatusCard", True)
        step_status_panel.setMinimumWidth(320)
        step_status_panel.setMaximumWidth(420)
        step_status_layout = QtWidgets.QVBoxLayout(step_status_panel)
        step_status_layout.setContentsMargins(14, 14, 14, 14)
        step_status_layout.setSpacing(10)
        step_status_title = QtWidgets.QLabel("Current run")
        step_status_title.setObjectName("FieldTitle")
        self.lbl_active_run_name = QtWidgets.QLabel("No active run")
        self.lbl_active_run_name.setObjectName("StepStatusRunName")
        self.lbl_active_run_name.setWordWrap(True)
        step_status_hint = QtWidgets.QLabel(
            "Each enabled preprocessing step is tracked here while the worker writes detailed output on the left."
        )
        step_status_hint.setObjectName("SectionHint")
        step_status_hint.setWordWrap(True)
        step_status_layout.addWidget(step_status_title, 0)
        step_status_layout.addWidget(self.lbl_active_run_name, 0)
        step_status_layout.addWidget(step_status_hint, 0)

        self.step_status_scroll = QtWidgets.QScrollArea()
        self.step_status_scroll.setWidgetResizable(True)
        self.step_status_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.step_status_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.step_status_host = QtWidgets.QWidget()
        self.step_status_host_layout = QtWidgets.QVBoxLayout(self.step_status_host)
        self.step_status_host_layout.setContentsMargins(0, 0, 0, 0)
        self.step_status_host_layout.setSpacing(8)
        self.step_status_scroll.setWidget(self.step_status_host)
        step_status_layout.addWidget(self.step_status_scroll, 1)
        log_body.addWidget(step_status_panel, 1)
        self._step_widgets: Dict[str, StepStatusItem] = {}
        self._clear_step_status_panel()
        log_layout.addLayout(log_header)
        log_layout.addLayout(log_body, 1)

        for box in [queue_box, done_box, log_box]:
            box.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        settings_page = QtWidgets.QWidget()
        settings_page_layout = QtWidgets.QVBoxLayout(settings_page)
        settings_page_layout.setContentsMargins(0, 0, 0, 0)
        settings_page_layout.setSpacing(12)
        settings_hint = QtWidgets.QLabel(
            "Only the selected settings section is shown so the central panel can stay dense and readable. "
            "Save As Default updates the next-launch defaults, while the file buttons export or import a full app preset."
        )
        settings_hint.setObjectName("SectionHint")
        settings_hint.setWordWrap(True)
        settings_page_layout.addWidget(settings_hint)

        params_sections = SideNavStack(
            "Sections",
            "Switch between pipeline steps, sorter setup, synchronization, and output paths.",
        )
        settings_page_layout.addWidget(params_sections, 1)

        def _new_params_page(box: QtWidgets.QGroupBox) -> QtWidgets.QWidget:
            page = QtWidgets.QWidget()
            page_layout = QtWidgets.QVBoxLayout(page)
            page_layout.setContentsMargins(0, 0, 0, 0)
            page_layout.setSpacing(12)
            page_layout.addWidget(box)
            page_layout.addStretch(1)
            return page

        params_sections.add_page("Pipeline steps", _new_params_page(steps_box))
        params_sections.add_page("Run naming", _new_params_page(acquisition_box))
        params_sections.add_page("Sorting and metrics", _new_params_page(metrics_box))
        params_sections.add_page("Sync and CatGT", _new_params_page(sync_box))
        params_sections.add_page("Tool and outputs", _new_params_page(paths_box))
        params_sections.setCurrentIndex(0)
        self.params_sections = params_sections

        settings_actions = QtWidgets.QHBoxLayout()
        self.btn_load_settings_file = QtWidgets.QPushButton("Load Settings File")
        self.btn_save_settings_file = QtWidgets.QPushButton("Save Settings File")
        self.btn_load_settings_file.setProperty("role", "ghost")
        self.btn_save_settings_file.setProperty("role", "ghost")
        settings_actions.addWidget(self.btn_load_settings_file, 0)
        settings_actions.addWidget(self.btn_save_settings_file, 0)
        settings_actions.addStretch(1)
        self.btn_save_params = QtWidgets.QPushButton("Save As Default")
        self.btn_save_params.setProperty("role", "secondary")
        self.btn_save_params.clicked.connect(self.save_settings)
        settings_actions.addWidget(self.btn_save_params, 0)
        settings_page_layout.addLayout(settings_actions)

        work_sections = SideNavStack(vertical_labels=True, compact_rail=True)
        work_sections.add_page("Queue", queue_box)
        self.settings_section_index = work_sections.add_page("Settings", settings_page)
        self.log_section_index = work_sections.add_page("Log", log_box)
        self.completed_section_index = work_sections.add_page("Completed", done_box)
        work_sections.setCurrentIndex(0)
        self.work_sections = work_sections
        main.addWidget(work_sections, 1)
        main.addWidget(self.progress, 0)
        self._refresh_queue_summary()

        self.btn_add_files.clicked.connect(self._add_files)
        self.btn_add_folder.clicked.connect(self._add_folder)
        self.btn_scan_raw_root.clicked.connect(self._scan_raw_root)
        self.btn_recent_files.clicked.connect(self._open_recent_file_menu)
        self.btn_recent_folders.clicked.connect(self._open_recent_folder_menu)
        self.cb_queue_filter.currentIndexChanged.connect(self._on_queue_filter_changed)
        self.btn_concat.clicked.connect(self._concatenate_selected)
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_clear.clicked.connect(self._clear)
        self.btn_run.clicked.connect(self._run_queue)
        self.list_jobs.itemSelectionChanged.connect(self._update_concat_button_state)
        self.btn_copy_log.clicked.connect(self._copy_log)
        self.btn_scan_completed_root.clicked.connect(self._scan_completed_root)
        self.list_jobs.filesDropped.connect(self._consume_drop)
        self.list_jobs.customContextMenuRequested.connect(self._open_job_context_menu)
        self.list_completed.itemDoubleClicked.connect(self._open_selected_curation)
        self.list_completed.customContextMenuRequested.connect(self._open_completed_context_menu)
        btn_output.clicked.connect(lambda: self._pick_folder(self.ed_output))
        btn_json.clicked.connect(lambda: self._pick_folder(self.ed_json))
        btn_catgt_path.clicked.connect(lambda: self._pick_folder(self.ed_catgt_path))
        btn_tprime_path.clicked.connect(lambda: self._pick_folder(self.ed_tprime_path))
        btn_cwaves_path.clicked.connect(lambda: self._pick_folder(self.ed_cwaves_path))
        btn_ks4_repo.clicked.connect(lambda: self._pick_folder(self.ed_ks4_repo))
        btn_ks_tmp.clicked.connect(lambda: self._pick_folder(self.ed_ks_tmp))
        self.ed_output.editingFinished.connect(self._persist_settings)
        self.ck_mirror_raw_hierarchy_output.toggled.connect(lambda _checked: self._persist_settings())
        self.cb_queue_filter.currentIndexChanged.connect(lambda _idx: self._persist_settings())
        self.ed_json.editingFinished.connect(self._persist_settings)
        for checkbox in [
            self.ck_catgt,
            self.ck_catgt_extract_only,
            self.ck_save_catgt_ap_bin,
            self.ck_tprime,
            self.ck_ks,
            self.ck_post,
            self.ck_noise,
            self.ck_wvf,
            self.ck_qm,
            self.ck_pybomb,
        ]:
            checkbox.toggled.connect(lambda _checked=False: self._persist_settings())
        self.cb_ks_ver.currentTextChanged.connect(self._persist_settings)
        self.ed_gate.editingFinished.connect(self._persist_settings)
        self.ed_trigger.editingFinished.connect(self._persist_settings)
        self.ed_probe.editingFinished.connect(self._persist_settings)
        self.ed_region.editingFinished.connect(self._persist_settings)
        self.ed_ks_th.editingFinished.connect(self._persist_settings)
        self.ed_ni_extract.editingFinished.connect(self._persist_settings)
        self.ed_catgt_cmd.editingFinished.connect(self._persist_settings)
        self.ed_tostream.editingFinished.connect(self._persist_settings)
        self.ed_qm_isi.valueChanged.connect(lambda _value: self._persist_settings())
        self.ed_sync_period.valueChanged.connect(lambda _value: self._persist_settings())
        self.cb_catgt_car_mode.currentTextChanged.connect(self._persist_settings)
        self.sp_loccar_min.valueChanged.connect(lambda _value: self._persist_settings())
        self.sp_loccar_max.valueChanged.connect(lambda _value: self._persist_settings())
        self.sp_ks4_dup_ms.valueChanged.connect(lambda _value: self._persist_settings())
        self.sp_ks4_min_template.valueChanged.connect(lambda _value: self._persist_settings())
        self.sp_cwaves_um.valueChanged.connect(lambda _value: self._persist_settings())
        self.ed_catgt_path.editingFinished.connect(self._persist_settings)
        self.ed_tprime_path.editingFinished.connect(self._persist_settings)
        self.ed_cwaves_path.editingFinished.connect(self._persist_settings)
        self.ed_ks4_repo.editingFinished.connect(self._persist_settings)
        self.ed_ks_tmp.editingFinished.connect(self._persist_settings)
        self.btn_build_catgt.clicked.connect(self._open_catgt_builder)
        self.btn_build_bitfield.clicked.connect(self._open_bitfield_builder)
        self.btn_build_tprime.clicked.connect(self._open_tprime_builder)
        self.btn_to_curation.clicked.connect(self._open_selected_curation)
        self.btn_to_histology.clicked.connect(self._open_selected_histology)
        self.btn_adv_ks4.clicked.connect(self._open_ks4_advanced)
        self.btn_save_settings_file.clicked.connect(self.saveSettingsFileRequested.emit)
        self.btn_load_settings_file.clicked.connect(self.loadSettingsFileRequested.emit)

    def _open_parameters_window(self) -> None:
        if hasattr(self, "work_sections") and hasattr(self, "settings_section_index"):
            self.work_sections.setCurrentIndex(int(self.settings_section_index))

    def _pick_folder(self, target: QtWidgets.QLineEdit) -> None:
        start = self.settings.value("paths/last_folder", str(Path.cwd()))
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder", start)
        if folder:
            target.setText(folder)
            self._set_last_folder(folder)
            self._add_recent("recent_folders", folder)
            self._persist_settings()

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        if event.mimeData().hasUrls():
            dropped = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
            self._consume_drop(dropped)
            event.acceptProposedAction()
            return
        super().dropEvent(event)

    def _add_files(self) -> None:
        start = self.settings.value("paths/last_file_dir", self.settings.value("paths/last_folder", str(Path.cwd())))
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select AP .bin files",
            str(start),
            "SpikeGLX AP BIN (*.ap.bin);;BIN files (*.bin)",
        )
        if files:
            self._set_last_file_dir(files[0])
            self._add_recent("recent_files", files[0])
        self._add_paths(files)

    def _add_folder(self) -> None:
        start = self.settings.value("paths/last_folder", str(Path.cwd()))
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder", start)
        if folder:
            self._set_last_folder(folder)
            self._add_recent("recent_folders", folder)
            self._add_paths([folder])

    def _consume_drop(self, dropped: List[str]) -> None:
        self._add_paths(dropped)

    @staticmethod
    def _normalized_path(path: str) -> str:
        try:
            return str(Path(path).resolve())
        except Exception:
            return str(Path(path))

    def _queue_filter_mode(self) -> str:
        return str(self.cb_queue_filter.currentData() or "non_processed")

    def _completed_entry_for_job(self, job: Dict[str, str]) -> Dict[str, object] | None:
        """Find the completed-history entry matching a queue job, or None.

        Matches first on normalized bin path, then falls back to matching the
        run name plus gate and probe strings so completed runs scanned from a
        different root still pair with their queued counterpart.
        """
        job_bin = self._normalized_path(str(job.get("bin_file") or ""))
        job_run = str(job.get("name") or "")
        job_gate = str(job.get("gate_string") or "")
        job_probe = str(job.get("probe_string") or "")
        for entry in self.completed_runs:
            entry_bin = str(entry.get("bin_file") or "")
            if entry_bin and self._normalized_path(entry_bin) == job_bin:
                return self._normalize_completed_entry(entry)
            if entry_bin:
                parsed = parse_spikeglx_bin_name(entry_bin)
                entry_run = str(parsed.get("run_name") or entry.get("run_name") or "")
                entry_gate = str(parsed.get("gate_string") or "")
                entry_probe = str(parsed.get("probe_string") or "")
            else:
                entry_run = str(entry.get("run_name") or "")
                entry_gate = str((entry.get("job_snapshot") or {}).get("gate_string") or "")
                entry_probe = str((entry.get("job_snapshot") or {}).get("probe_string") or "")
            if entry_run == job_run and entry_gate == job_gate and entry_probe == job_probe:
                return self._normalize_completed_entry(entry)
        return None

    def _queue_item_text(self, job: Dict[str, str], completed_entry: Dict[str, object] | None = None) -> str:
        base = (
            f"{job['name']}  |  g{job['gate_string']} t{job['trigger_string']} p{job['probe_string']}  |  {job['bin_file']}"
        )
        overrides = job.get("cfg_overrides")
        if isinstance(overrides, dict) and overrides.get("run_catgt_extract_only") and not overrides.get("run_kilosort", True):
            base = f"{base}  |  NI events only"
        if completed_entry is not None:
            return f"{base}  |  completed"
        return base

    def _refresh_job_list_view(self) -> None:
        if not hasattr(self, "list_jobs"):
            return
        self.list_jobs.clear()
        if self._queue_filter_mode() == "all":
            jobs_to_show = list(self._raw_run_catalog.values())
        else:
            jobs_to_show = list(self.jobs)
        queue_bin_paths = {self._normalized_path(str(job.get("bin_file") or "")) for job in self.jobs}
        ok_icon = QtGui.QIcon(str(self._OK_ICON)) if self._OK_ICON.exists() else QtGui.QIcon()
        for job in jobs_to_show:
            completed_entry = self._completed_entry_for_job(job)
            item = QtWidgets.QListWidgetItem(self._queue_item_text(job, completed_entry))
            payload = {
                "job": dict(job),
                "bin_file": str(job.get("bin_file") or ""),
                "in_queue": self._normalized_path(str(job.get("bin_file") or "")) in queue_bin_paths,
                "completed_entry": completed_entry,
            }
            item.setData(QtCore.Qt.UserRole, payload)
            if completed_entry is not None and not ok_icon.isNull():
                item.setIcon(ok_icon)
                tooltip = [
                    "Completed run",
                    *self._completed_tooltip_lines(completed_entry),
                ]
                item.setToolTip("\n".join(tooltip))
            else:
                item.setToolTip(str(job.get("bin_file") or ""))
            self.list_jobs.addItem(item)
        self._update_concat_button_state()

    def _ingest_bins(self, bins: List[str], *, queue_completed: bool) -> tuple[int, int, int]:
        """Validate AP bins and register them in the catalog and queue.

        Every valid bin is recorded in the raw-run catalog; it is also appended
        to the queue unless it is already queued, or it is already completed and
        ``queue_completed`` is False. Returns ``(added_to_catalog, queued, completed)``.
        """
        added_catalog = 0
        queued = 0
        completed = 0
        existing_queue = {self._normalized_path(j["bin_file"]) for j in self.jobs}
        for b in bins:
            ok, reason = validate_spikeglx_ap_bin(b)
            if not ok:
                self._append_log(f"Skipping {b}: {reason}")
                continue
            parsed = parse_spikeglx_bin_name(b)
            job = {
                "name": parsed["run_name"],
                "bin_file": b,
                "workdir": str(Path(b).parent),
                "gate_string": parsed["gate_string"],
                "trigger_string": parsed["trigger_string"],
                "probe_string": parsed["probe_string"],
            }
            norm_bin = self._normalized_path(b)
            if norm_bin not in self._raw_run_catalog:
                added_catalog += 1
            self._raw_run_catalog[norm_bin] = job
            is_completed = self._completed_entry_for_job(job) is not None
            if is_completed:
                completed += 1
            if is_completed and not queue_completed:
                continue
            if norm_bin in existing_queue:
                continue
            self.jobs.append(job)
            existing_queue.add(norm_bin)
            queued += 1
        return added_catalog, queued, completed

    def _add_paths(self, paths: List[str]) -> None:
        bins = discover_bin_files(paths)
        if not bins and paths:
            self._append_log("No valid AP files found. Expected names like *.imec0.ap.bin")
        _added_catalog, queued, completed = self._ingest_bins(bins, queue_completed=True)
        if paths:
            first = paths[0]
            p = Path(first)
            if p.exists():
                if p.is_file():
                    self._set_last_file_dir(str(p.parent))
                    self._add_recent("recent_files", str(p))
                else:
                    self._set_last_folder(str(p))
                    self._add_recent("recent_folders", str(p))
                self._persist_settings()
        if bins:
            self._append_log(
                f"Added {queued} run(s) to queue from selection."
                + (f" {completed} already completed run(s) are still shown with the completed icon in All runs." if completed else "")
            )
        self._refresh_job_list_view()
        self._refresh_queue_summary()

    def _refresh_queue_summary(self) -> None:
        if not hasattr(self, "lbl_queue_summary"):
            return
        n_jobs = len(self.jobs)
        total_known = len(self._raw_run_catalog)
        total_completed = sum(1 for job in self._raw_run_catalog.values() if self._completed_entry_for_job(job) is not None)
        if self._concatenating:
            self.lbl_queue_summary.setText(
                "Concatenating selected recordings into a joint binary. The fused run will be added "
                "to the queue when finished."
            )
            return
        if self._running:
            remaining = len(self._queue)
            msg = f"{n_jobs} recording(s) loaded. Queue running with {remaining} remaining after the active job."
        elif n_jobs == 0:
            msg = "Queue is empty. Add AP .bin files or folders to begin."
        elif n_jobs == 1:
            msg = "1 recording queued and ready to run."
        else:
            msg = f"{n_jobs} recordings queued and ready to run."
        if total_known and self._queue_filter_mode() == "all":
            msg += f" Showing {total_known} known raw run(s), {total_completed} completed."
        elif total_known:
            pending_known = max(total_known - total_completed, 0)
            msg += f" {pending_known} known non-processed run(s) in the current raw-run catalog."
        if self.completed_runs:
            msg += f" {len(self.completed_runs)} completed run(s) available in Completed."
        self.lbl_queue_summary.setText(msg)

    def _on_queue_filter_changed(self, _index: int) -> None:
        self._refresh_job_list_view()
        self._refresh_queue_summary()

    def _remove_selected(self) -> None:
        selected_items = list(self.list_jobs.selectedItems())
        if not selected_items:
            return
        remove_bins: List[str] = []
        ignored = 0
        for item in selected_items:
            payload = item.data(QtCore.Qt.UserRole)
            if not isinstance(payload, dict):
                continue
            if bool(payload.get("in_queue")):
                remove_bins.append(self._normalized_path(str(payload.get("bin_file") or "")))
            else:
                ignored += 1
        if remove_bins:
            remove_set = set(remove_bins)
            self.jobs = [job for job in self.jobs if self._normalized_path(job["bin_file"]) not in remove_set]
        if ignored:
            self._append_log("Completed-only runs shown in All runs were not removed from history or catalog.")
        self._refresh_job_list_view()
        self._refresh_queue_summary()

    def _clear(self) -> None:
        self.list_jobs.clear()
        self.jobs.clear()
        self._raw_run_catalog.clear()
        self._refresh_queue_summary()

    def _selected_inqueue_jobs(self) -> List[Dict[str, str]]:
        """Return queued job dicts for the current selection, in visual (top-down) order."""
        out: List[Dict[str, str]] = []
        for row in range(self.list_jobs.count()):
            item = self.list_jobs.item(row)
            if item is None or not item.isSelected():
                continue
            payload = item.data(QtCore.Qt.UserRole)
            if not isinstance(payload, dict) or not bool(payload.get("in_queue")):
                continue
            job = payload.get("job")
            if isinstance(job, dict) and job.get("bin_file"):
                out.append(dict(job))
        return out

    def _update_concat_button_state(self) -> None:
        if not hasattr(self, "btn_concat"):
            return
        busy = self._running or self._concatenating
        self.btn_concat.setEnabled(len(self._selected_inqueue_jobs()) >= 2 and not busy)

    def _prepare_concat_status_panel(self, run_name: str, n_files: int) -> None:
        self._clear_step_status_panel()
        self.lbl_active_run_name.setText(f"Concatenating {n_files} runs into {run_name}")
        widget = StepStatusItem("Concatenate binaries", self.step_status_host)
        self._step_widgets[ConcatenationWorker.STEP_KEY] = widget
        self.step_status_host_layout.insertWidget(self.step_status_host_layout.count() - 1, widget)

    def _concatenate_selected(self) -> None:
        if self._running or self._concatenating:
            self._append_log("Cannot concatenate while a job is running.")
            return
        jobs = self._selected_inqueue_jobs()
        if len(jobs) < 2:
            self._append_log("Select at least two queued AP recordings to concatenate.")
            return

        bin_files = [str(job["bin_file"]) for job in jobs]
        meta_files = [str(find_meta_for_bin(b)) for b in bin_files]
        missing = [b for b, m in zip(bin_files, meta_files) if not Path(m).exists()]
        if missing:
            QtWidgets.QMessageBox.warning(
                self,
                "Missing meta files",
                "Cannot concatenate; no .meta found next to:\n" + "\n".join(missing),
            )
            return

        ok, reason, _info = validate_concat_inputs(meta_files)
        if not ok:
            QtWidgets.QMessageBox.warning(self, "Incompatible recordings", reason)
            return

        run_names = [
            str(job.get("name") or parse_spikeglx_bin_name(b).get("run_name") or Path(b).stem)
            for job, b in zip(jobs, bin_files)
        ]
        combined_default = build_concat_run_name(run_names)
        first = Path(bin_files[0])
        # Default the fused run into the mirrored processedData hierarchy (a new
        # session folder under the output root), not into rawData. User can still
        # override the destination in the dialog.
        default_dir = mirrored_concat_base_dir(
            first,
            self.ed_output.text().strip(),
            combined_default,
            mirror_raw_hierarchy=self.ck_mirror_raw_hierarchy_output.isChecked(),
        )

        defaults = {
            "svd_clean": self.settings.value("preproc/concat_svd_clean", True, type=bool),
            "n_svd_components": int(self.settings.value("preproc/concat_n_components", 5)),
            "batch_seconds": float(self.settings.value("preproc/concat_batch_seconds", 0.5)),
            "extract_ni": self.settings.value("preproc/concat_extract_ni", True, type=bool),
        }
        dlg = ConcatenateDialog(run_names, str(default_dir), combined_default, defaults, self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        vals = dlg.values()
        combined = str(vals.get("run_name") or combined_default).strip() or combined_default
        out_dir = str(vals.get("output_dir") or default_dir).strip() or str(default_dir)
        probe = str(jobs[0].get("probe_string") or "0")
        layout = default_concat_run_layout(out_dir, combined, probe)
        target_bin = layout["bin"]

        if target_bin.exists():
            answer = QtWidgets.QMessageBox.question(
                self,
                "Overwrite existing file?",
                f"{target_bin}\nalready exists. Overwrite it?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if answer != QtWidgets.QMessageBox.Yes:
                return

        try:
            layout["probe_folder"].mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot create folder", str(exc))
            return

        self.settings.setValue("preproc/concat_svd_clean", bool(vals["svd_clean"]))
        self.settings.setValue("preproc/concat_n_components", int(vals["n_svd_components"]))
        self.settings.setValue("preproc/concat_batch_seconds", float(vals["batch_seconds"]))
        self.settings.setValue("preproc/concat_extract_ni", bool(vals.get("extract_ni", True)))
        self._settings_sync_timer.start()

        # Remember which source sessions to switch to NI events-only once the
        # fused run has been written (applied in _on_concat_finished).
        self._pending_concat_ni_extract = (
            [self._normalized_path(b) for b in bin_files]
            if bool(vals.get("extract_ni", True))
            else []
        )

        cfg = ConcatenationConfig(
            bin_files=bin_files,
            meta_files=meta_files,
            target_bin=str(target_bin),
            run_name=combined,
            svd_clean=bool(vals["svd_clean"]),
            n_svd_components=int(vals["n_svd_components"]),
            batch_seconds=float(vals["batch_seconds"]),
        )

        self._concatenating = True
        self.btn_concat.setEnabled(False)
        self.btn_run.setEnabled(False)
        self.progress.setValue(0)
        self._prepare_concat_status_panel(combined, len(bin_files))
        if hasattr(self, "work_sections") and hasattr(self, "log_section_index"):
            self.work_sections.setCurrentIndex(int(self.log_section_index))
        self._append_log(
            f"Starting concatenation of {len(bin_files)} recording(s) into {target_bin}"
        )

        worker = ConcatenationWorker(cfg)
        worker.signals.log.connect(self._append_log)
        worker.signals.progress.connect(self.progress.setValue)
        worker.signals.error.connect(self._append_log)
        worker.signals.stepStarted.connect(self._on_worker_step_started)
        worker.signals.stepProgress.connect(self._on_worker_step_progress)
        worker.signals.stepFinished.connect(self._on_worker_step_finished)
        worker.signals.finished.connect(self._on_concat_finished)
        self.pool.start(worker)
        self._refresh_queue_summary()

    def _on_concat_finished(self, result: Dict) -> None:
        self._concatenating = False
        self.btn_run.setEnabled(True)
        ok = bool(result.get("ok"))
        run_name = str(result.get("job") or result.get("run_name") or "concat")
        if ok:
            target_bin = str(result.get("target_bin") or "")
            self.lbl_active_run_name.setText(f"{run_name} (concatenated)")
            self._append_log(f"Concatenation complete: {target_bin}")
            split = str(result.get("splitinfo_path") or "")
            if split:
                self._append_log(
                    f"Per-session split map saved to {split}. "
                    "Use it to separate sorted spikes back into each session."
                )
            if target_bin and Path(target_bin).exists():
                self._add_paths([target_bin])
                self._select_job_in_queue(target_bin)
                if hasattr(self, "work_sections"):
                    self.work_sections.setCurrentIndex(0)
                self._append_log(
                    f"Added concatenated run '{run_name}' to the queue. "
                    "Configure the pipeline steps and Run queue to sort all sessions together."
                )
                if self._pending_concat_ni_extract:
                    self._apply_ni_extract_overrides(self._pending_concat_ni_extract)
        self._pending_concat_ni_extract = []
        if not ok:
            self.lbl_active_run_name.setText(f"{run_name} (concatenation failed)")
            widget = self._step_widgets.get(ConcatenationWorker.STEP_KEY)
            if widget is not None and widget.percent_label.text() != "Failed":
                widget.set_failed()
        if ok:
            self.progress.setValue(100)
        self._update_concat_button_state()
        self._refresh_queue_summary()

    def _apply_ni_extract_overrides(self, source_bins: List[str]) -> None:
        """Switch the source sessions to a CatGT extract-only (NI events) pass.

        They are kept in the queue so a single Run produces both the joint sort
        (fused run) and each session's NI event files, but they are no longer
        sorted individually. TPrime alignment is included only if the user has
        the TPrime step enabled (it needs a per-session sort to resolve).
        """
        overrides = {
            "run_catgt": False,
            "run_catgt_extract_only": True,
            "save_catgt_ap_bin": False,
            "run_tprime": self.ck_tprime.isChecked(),
            "run_kilosort": False,
            "run_kilosort_postproc": False,
            "run_noise_templates": False,
            "run_mean_waveforms": False,
            "run_quality_metrics": False,
            "run_pybombcell": False,
        }
        targets = {self._normalized_path(b) for b in source_bins}
        matched: set[str] = set()
        for job in self.jobs:
            norm = self._normalized_path(str(job.get("bin_file") or ""))
            if norm in targets:
                job["cfg_overrides"] = dict(overrides)
                matched.add(norm)
        for raw_bin in source_bins:
            norm = self._normalized_path(raw_bin)
            if norm in matched:
                continue
            ok_bin, _reason = validate_spikeglx_ap_bin(raw_bin)
            if not ok_bin:
                continue
            parsed = parse_spikeglx_bin_name(raw_bin)
            job = {
                "name": parsed["run_name"],
                "bin_file": raw_bin,
                "workdir": str(Path(raw_bin).parent),
                "gate_string": parsed["gate_string"],
                "trigger_string": parsed["trigger_string"],
                "probe_string": parsed["probe_string"],
                "cfg_overrides": dict(overrides),
            }
            self.jobs.append(job)
            self._raw_run_catalog[norm] = {k: v for k, v in job.items() if k != "cfg_overrides"}
            matched.add(norm)
        if matched:
            self._append_log(
                f"Switched {len(matched)} source session(s) to CatGT extract-only (NI events)"
                + (" with TPrime alignment" if self.ck_tprime.isChecked() else "")
                + ". They stay in the queue but are not sorted individually."
            )
        self._refresh_job_list_view()
        self._refresh_queue_summary()

    def _collect_cfg(self) -> EcephysPipelineConfig:
        return EcephysPipelineConfig(
            output_root=self.ed_output.text().strip(),
            json_root=self.ed_json.text().strip(),
            mirror_raw_hierarchy_output=self.ck_mirror_raw_hierarchy_output.isChecked(),
            save_catgt_ap_bin=self.ck_save_catgt_ap_bin.isChecked(),
            run_catgt=self.ck_catgt.isChecked(),
            run_catgt_extract_only=self.ck_catgt_extract_only.isChecked(),
            run_tprime=self.ck_tprime.isChecked(),
            run_kilosort=self.ck_ks.isChecked(),
            run_kilosort_postproc=self.ck_post.isChecked(),
            run_noise_templates=self.ck_noise.isChecked(),
            run_mean_waveforms=self.ck_wvf.isChecked(),
            run_quality_metrics=self.ck_qm.isChecked(),
            run_pybombcell=self.ck_pybomb.isChecked(),
            ks_ver=self.cb_ks_ver.currentText(),
            gate_string=self.ed_gate.text().strip(),
            trigger_string=self.ed_trigger.text().strip(),
            probe_string=self.ed_probe.text().strip(),
            region_name=self.ed_region.text().strip(),
            ni_extract_string=self.ed_ni_extract.text().strip(),
            catgt_cmd_string=self.ed_catgt_cmd.text().strip(),
            sync_period=float(self.ed_sync_period.value()),
            tostream_sync_params=self.ed_tostream.text().strip(),
            ks_th=self.ed_ks_th.text().strip(),
            qm_isi_thresh=float(self.ed_qm_isi.value()),
            catgt_car_mode=self.cb_catgt_car_mode.currentText().strip(),
            catgt_loccar_min_um=float(self.sp_loccar_min.value()),
            catgt_loccar_max_um=float(self.sp_loccar_max.value()),
            ks4_duplicate_spike_ms=float(self.sp_ks4_dup_ms.value()),
            ks4_min_template_size_um=float(self.sp_ks4_min_template.value()),
            c_waves_snr_um=float(self.sp_cwaves_um.value()),
            ks4_advanced_params=dict(self._ks4_adv_params),
            catgt_path=self.ed_catgt_path.text().strip(),
            tprime_path=self.ed_tprime_path.text().strip(),
            cwaves_path=self.ed_cwaves_path.text().strip(),
            ks4_repo_path=self.ed_ks4_repo.text().strip(),
            kilosort_output_tmp=self.ed_ks_tmp.text().strip(),
        )

    @staticmethod
    def _serialize_cfg(cfg: EcephysPipelineConfig) -> Dict[str, object]:
        data = asdict(cfg)
        return json.loads(json.dumps(data))

    @staticmethod
    def _normalize_completed_entry(entry: Dict[str, object]) -> Dict[str, object]:
        out = {
            "run_name": str(entry.get("run_name") or ""),
            "ks_folder": str(entry.get("ks_folder") or ""),
            "bin_file": str(entry.get("bin_file") or ""),
            "label": str(entry.get("label") or ""),
            "finished_at": str(entry.get("finished_at") or ""),
            "params_file": str(entry.get("params_file") or ""),
            "source_root": str(entry.get("source_root") or ""),
            "job_snapshot": dict(entry.get("job_snapshot") or {}),
            "cfg_snapshot": dict(entry.get("cfg_snapshot") or {}),
        }
        if not out["label"]:
            out["label"] = f"{out['run_name']} | {out['ks_folder']}"
        return out

    @staticmethod
    def _completed_tooltip_lines(entry: Dict[str, object]) -> List[str]:
        lines = [str(entry["ks_folder"])]
        if entry.get("finished_at"):
            lines.append(f"Completed: {entry['finished_at']}")
        if entry.get("bin_file"):
            lines.append(f"Input: {entry['bin_file']}")
        if entry.get("params_file"):
            lines.append(f"Params: {entry['params_file']}")
        if entry.get("source_root"):
            lines.append(f"Scanned from: {entry['source_root']}")
        return lines

    def _add_completed_item(self, entry: Dict[str, object]) -> None:
        normalized = self._normalize_completed_entry(entry)
        item = QtWidgets.QListWidgetItem(str(normalized["label"]))
        item.setData(QtCore.Qt.UserRole, normalized)
        item.setToolTip("\n".join(self._completed_tooltip_lines(normalized)))
        self.list_completed.addItem(item)

    def _upsert_completed_entry(self, entry: Dict[str, object]) -> int:
        normalized = self._normalize_completed_entry(entry)
        existing_index = next(
            (idx for idx, existing in enumerate(self.completed_runs) if existing.get("ks_folder") == normalized["ks_folder"]),
            None,
        )
        if existing_index is None:
            self.completed_runs.append(normalized)
            self._add_completed_item(normalized)
            return self.list_completed.count() - 1

        self.completed_runs[existing_index] = normalized
        existing_item = self.list_completed.item(existing_index)
        if existing_item is not None:
            existing_item.setText(str(normalized["label"]))
            existing_item.setData(QtCore.Qt.UserRole, normalized)
            existing_item.setToolTip("\n".join(self._completed_tooltip_lines(normalized)))
        return existing_index

    def _persist_completed_history(self) -> None:
        payload = [self._normalize_completed_entry(entry) for entry in self.completed_runs]
        self.settings.setValue("preproc/completed_runs_history_json", json.dumps(payload))
        self._settings_sync_timer.start()

    def _restore_completed_history(self) -> None:
        self.completed_runs.clear()
        self.list_completed.clear()
        raw_history = self.settings.value("preproc/completed_runs_history_json", "[]")
        try:
            payload = json.loads(str(raw_history) or "[]")
        except Exception:
            payload = []
        if not isinstance(payload, list):
            payload = []
        for raw_entry in payload:
            if not isinstance(raw_entry, dict):
                continue
            entry = self._normalize_completed_entry(raw_entry)
            if not entry["run_name"] or not entry["ks_folder"]:
                continue
            self.completed_runs.append(entry)
            self._add_completed_item(entry)
        if self.list_completed.count():
            self.list_completed.setCurrentRow(self.list_completed.count() - 1)
        self._refresh_job_list_view()

    def _completed_entry_from_item(self, item: QtWidgets.QListWidgetItem | None) -> Dict[str, object] | None:
        if item is None:
            return None
        raw_entry = item.data(QtCore.Qt.UserRole)
        if not isinstance(raw_entry, dict):
            return None
        return self._normalize_completed_entry(raw_entry)

    def _selected_completed_entry(self) -> Dict[str, object] | None:
        item = self.list_completed.currentItem()
        if item is None and self.list_completed.count():
            item = self.list_completed.item(self.list_completed.count() - 1)
        return self._completed_entry_from_item(item)

    def _selected_completed_entries(
        self,
        item_override: QtWidgets.QListWidgetItem | None = None,
    ) -> List[Dict[str, object]]:
        items = self.list_completed.selectedItems()
        if not items and item_override is not None:
            items = [item_override]
        if not items:
            current = self.list_completed.currentItem()
            if current is not None:
                items = [current]
        if not items and self.list_completed.count():
            last_item = self.list_completed.item(self.list_completed.count() - 1)
            if last_item is not None:
                items = [last_item]

        out: List[Dict[str, object]] = []
        for item in items:
            entry = self._completed_entry_from_item(item)
            if entry is not None:
                out.append(entry)
        return out

    def _show_completed_run_parameters(self, entry: Dict[str, object]) -> None:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Run Parameters: {entry.get('run_name', '')}")
        dlg.resize(860, 620)
        layout = QtWidgets.QVBoxLayout(dlg)

        summary = QtWidgets.QLabel(
            f"Completed: {entry.get('finished_at') or 'unknown'}\n"
            f"Kilosort folder: {entry.get('ks_folder', '')}\n"
            f"Input bin: {entry.get('bin_file', '') or 'unknown'}\n"
            f"Params file: {entry.get('params_file', '') or 'unknown'}"
        )
        summary.setWordWrap(True)
        summary.setObjectName("SectionHint")
        layout.addWidget(summary)

        viewer = QtWidgets.QPlainTextEdit()
        viewer.setReadOnly(True)
        viewer.setProperty("logView", True)
        viewer.setPlainText(
            json.dumps(
                {
                    "run_name": entry.get("run_name"),
                    "finished_at": entry.get("finished_at"),
                    "ks_folder": entry.get("ks_folder"),
                    "bin_file": entry.get("bin_file"),
                    "params_file": entry.get("params_file"),
                    "source_root": entry.get("source_root"),
                    "job_snapshot": entry.get("job_snapshot", {}),
                    "cfg_snapshot": entry.get("cfg_snapshot", {}),
                },
                indent=2,
                sort_keys=True,
            )
        )
        layout.addWidget(viewer, 1)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        btns.rejected.connect(dlg.reject)
        btns.accepted.connect(dlg.accept)
        btns.button(QtWidgets.QDialogButtonBox.Close).clicked.connect(dlg.accept)
        layout.addWidget(btns)
        dlg.exec()

    def _open_completed_folder(self, entry: Dict[str, object]) -> None:
        folder = Path(str(entry.get("ks_folder") or ""))
        if not folder.exists():
            self._append_log(f"Completed folder not found: {folder}")
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))

    def _resolve_completed_entry_bin_file(self, entry: Dict[str, object]) -> str:
        """Locate the source AP bin for a completed run on disk, or "".

        Tries the snapshot and entry bin paths; relative candidates are also
        resolved against the params-file folder and the scanned source root.
        Returns the first existing path as a string, or empty if none resolve.
        """
        params_file_raw = str(entry.get("params_file") or "").strip()
        source_root_raw = str(entry.get("source_root") or "").strip()
        params_file = Path(params_file_raw).expanduser() if params_file_raw else None
        source_root = Path(source_root_raw).expanduser() if source_root_raw else None
        snapshot = entry.get("job_snapshot") or {}
        candidates = [
            str((snapshot.get("bin_file") or "")).strip(),
            str(entry.get("bin_file") or "").strip(),
        ]
        for raw_path in candidates:
            if not raw_path:
                continue
            path = Path(raw_path).expanduser()
            search_paths = [path]
            if not path.is_absolute():
                if params_file:
                    search_paths.append(params_file.parent / path)
                if source_root:
                    search_paths.append(source_root / path)
            for candidate in search_paths:
                try:
                    resolved = candidate.resolve()
                except Exception:
                    resolved = candidate
                if resolved.exists():
                    return str(resolved)
        return ""

    def _job_from_completed_entry(self, entry: Dict[str, object]) -> Dict[str, str] | None:
        bin_file = self._resolve_completed_entry_bin_file(entry)
        if not bin_file:
            return None
        snapshot = entry.get("job_snapshot") or {}
        parsed = parse_spikeglx_bin_name(bin_file)
        return {
            "name": str(snapshot.get("name") or entry.get("run_name") or parsed.get("run_name") or "").strip(),
            "bin_file": bin_file,
            "workdir": str(Path(bin_file).parent),
            "gate_string": str(snapshot.get("gate_string") or parsed.get("gate_string") or "0").strip(),
            "trigger_string": str(snapshot.get("trigger_string") or parsed.get("trigger_string") or "0,0").strip(),
            "probe_string": str(snapshot.get("probe_string") or parsed.get("probe_string") or "0").strip(),
        }

    def _select_job_in_queue(self, bin_file: str) -> None:
        target = self._normalized_path(bin_file)
        for row in range(self.list_jobs.count()):
            item = self.list_jobs.item(row)
            if item is None:
                continue
            payload = item.data(QtCore.Qt.UserRole)
            if not isinstance(payload, dict):
                continue
            item_bin = str(payload.get("bin_file") or "")
            if self._normalized_path(item_bin) != target:
                continue
            self.list_jobs.setCurrentRow(row)
            self.list_jobs.scrollToItem(item, QtWidgets.QAbstractItemView.PositionAtCenter)
            return

    def _requeue_completed_entry(self, entry: Dict[str, object]) -> None:
        job = self._job_from_completed_entry(entry)
        if job is None:
            self._append_log(
                f"Could not locate the original AP bin for completed run {entry.get('run_name', '')}; "
                "re-run requires the source raw input."
            )
            return
        bin_file = str(job.get("bin_file") or "")
        ok, reason = validate_spikeglx_ap_bin(bin_file)
        if not ok:
            self._append_log(f"Cannot re-queue {entry.get('run_name', '')} from {bin_file}: {reason}")
            return
        norm_bin = self._normalized_path(bin_file)
        self._raw_run_catalog[norm_bin] = dict(job)
        already_queued = any(self._normalized_path(str(existing.get("bin_file") or "")) == norm_bin for existing in self.jobs)
        if not already_queued:
            self.jobs.append(job)
        queue_filter_index = self.cb_queue_filter.findData("non_processed")
        if queue_filter_index >= 0:
            self.cb_queue_filter.setCurrentIndex(queue_filter_index)
        self._refresh_job_list_view()
        self._refresh_queue_summary()
        if hasattr(self, "work_sections"):
            self.work_sections.setCurrentIndex(0)
        self._select_job_in_queue(bin_file)
        if already_queued:
            self._append_log(f"{entry.get('run_name', '')} is already in the queue.")
            return
        self._append_log(f"Re-queued {entry.get('run_name', '')} from {bin_file}. Adjust settings if needed, then run queue.")

    def _open_completed_in_postprocessing(self, entry: Dict[str, object]) -> None:
        folder = str(entry.get("ks_folder") or "")
        if folder:
            self.openPostProcessingRequested.emit(folder)

    def _open_completed_in_histology(self, entry: Dict[str, object]) -> None:
        folder = str(entry.get("ks_folder") or "")
        if folder:
            self.openHistologyRequested.emit(folder)

    def _show_completed_entry_menu(self, entry: Dict[str, object], global_pos: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu(self)
        act_open_folder = menu.addAction("Open Folder in Explorer")
        act_view_params = menu.addAction("View Run Parameters")
        act_requeue = menu.addAction("Re-run with New Parameters")
        act_post = menu.addAction("Load in Post-Processing")
        act_hist = menu.addAction("Set up Histology")
        action = menu.exec(global_pos)
        if action is act_open_folder:
            self._open_completed_folder(entry)
        elif action is act_view_params:
            self._show_completed_run_parameters(entry)
        elif action is act_requeue:
            self._requeue_completed_entry(entry)
        elif action is act_post:
            self._open_completed_in_postprocessing(entry)
        elif action is act_hist:
            self._open_completed_in_histology(entry)

    def _scan_raw_root(self) -> None:
        start = str(
            self.settings.value(
                "paths/last_raw_root",
                self.settings.value("paths/last_folder", self.ed_output.text().strip() or str(Path.cwd())),
            )
        )
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Scan Raw Root for Runs",
            start,
        )
        if not folder:
            return
        self.settings.setValue("paths/last_raw_root", folder)
        self._set_last_folder(folder)
        self._add_recent("recent_folders", folder)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            bins = discover_bin_files([folder])
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if not bins:
            self._append_log(f"No raw SpikeGLX AP bins found under {folder}")
            return
        added_catalog, queued, completed = self._ingest_bins(bins, queue_completed=False)
        self._refresh_job_list_view()
        self._refresh_queue_summary()
        if hasattr(self, "work_sections"):
            self.work_sections.setCurrentIndex(0)
        self._append_log(
            f"Scanned raw root {folder}: found {len(bins)} run(s), added {added_catalog} to catalog, queued {queued} non-processed run(s), skipped {completed} completed run(s)."
        )

    def _open_job_context_menu(self, pos: QtCore.QPoint) -> None:
        item = self.list_jobs.itemAt(pos)
        if item is None:
            return
        self.list_jobs.setCurrentItem(item)
        payload = item.data(QtCore.Qt.UserRole)
        if not isinstance(payload, dict):
            return
        entry = payload.get("completed_entry")
        if isinstance(entry, dict):
            menu = QtWidgets.QMenu(self)
            act_completed = menu.addAction("Completed Run Actions...")
            act_completed.setEnabled(False)
            menu.addSeparator()
            act_open_folder = menu.addAction("Open Folder in Explorer")
            act_view_params = menu.addAction("View Run Parameters")
            act_requeue = menu.addAction("Re-run with New Parameters")
            act_post = menu.addAction("Load in Post-Processing")
            act_remove = None
            if bool(payload.get("in_queue")):
                menu.addSeparator()
                act_remove = menu.addAction("Remove from Queue")
            action = menu.exec(self.list_jobs.viewport().mapToGlobal(pos))
            completed_entry = self._normalize_completed_entry(entry)
            if action is act_open_folder:
                self._open_completed_folder(completed_entry)
            elif action is act_view_params:
                self._show_completed_run_parameters(completed_entry)
            elif action is act_requeue:
                self._requeue_completed_entry(completed_entry)
            elif action is act_post:
                self._open_completed_in_postprocessing(completed_entry)
            elif act_remove is not None and action is act_remove:
                self._remove_selected()
            return

        if bool(payload.get("in_queue")):
            menu = QtWidgets.QMenu(self)
            act_remove = menu.addAction("Remove from Queue")
            action = menu.exec(self.list_jobs.viewport().mapToGlobal(pos))
            if action is act_remove:
                self._remove_selected()

    def _scan_completed_root(self) -> None:
        start = str(
            self.settings.value(
                "paths/last_processed_root",
                self.settings.value("paths/last_folder", self.ed_output.text().strip() or str(Path.cwd())),
            )
        )
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Scan Processed Root for Completed Runs",
            start,
        )
        if not folder:
            return
        self.settings.setValue("paths/last_processed_root", folder)
        self._set_last_folder(folder)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            discovered = discover_completed_runs(folder)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if not discovered:
            self._append_log(f"No completed Kilosort runs found under {folder}")
            return

        added = 0
        updated = 0
        last_row = -1
        existing_paths = {str(entry.get("ks_folder") or "") for entry in self.completed_runs}
        for entry in discovered:
            if str(entry.get("ks_folder") or "") in existing_paths:
                updated += 1
            else:
                added += 1
            last_row = self._upsert_completed_entry(entry)
            existing_paths.add(str(entry.get("ks_folder") or ""))

        if last_row >= 0:
            self.list_completed.setCurrentRow(last_row)
        self._persist_completed_history()
        self._refresh_job_list_view()
        self._refresh_queue_summary()
        if hasattr(self, "work_sections"):
            self.work_sections.setCurrentIndex(self.completed_section_index)
        self._append_log(
            f"Scanned {folder}: found {len(discovered)} completed run(s), added {added}, updated {updated}."
        )

    def _open_completed_context_menu(self, pos: QtCore.QPoint) -> None:
        item = self.list_completed.itemAt(pos)
        if item is None:
            return
        self.list_completed.setCurrentItem(item)
        entry = self._completed_entry_from_item(item)
        if entry is None:
            return
        self._show_completed_entry_menu(entry, self.list_completed.viewport().mapToGlobal(pos))

    def _planned_step_definitions(
        self, cfg: EcephysPipelineConfig, *, is_concat_run: bool = False
    ) -> List[tuple[str, str]]:
        """List the (step_key, label) pairs the worker will run for this config.

        Order mirrors the worker's execution order so the status panel lines up
        with reported step progress. TPrime is omitted for concatenated runs
        because they have no continuous sync stream to align against.
        """
        steps: List[tuple[str, str]] = []
        if cfg.run_catgt:
            if cfg.run_catgt_extract_only:
                label = "CatGT extract-only + AP save" if cfg.save_catgt_ap_bin else "CatGT extract-only"
                steps.append(("catgt_extract_only", label))
            else:
                steps.append(("catgt", "CatGT"))
        elif cfg.run_catgt_extract_only:
            label = "CatGT extract-only + AP save" if cfg.save_catgt_ap_bin else "CatGT extract-only"
            steps.append(("catgt_extract_only", label))
        if cfg.run_kilosort:
            steps.append(("kilosort", "Kilosort"))
        if cfg.run_kilosort_postproc:
            steps.append(("kilosort_postproc", "Kilosort Postprocessing"))
        if cfg.run_noise_templates:
            steps.append(("noise_templates", "Noise Templates"))
        if cfg.run_mean_waveforms:
            steps.append(("mean_waveforms", "Mean Waveforms"))
        if cfg.run_quality_metrics:
            steps.append(("quality_metrics", "Quality Metrics"))
        # Concatenated runs have no continuous sync / nidq stream to align against,
        # so the worker skips TPrime for them; mirror that in the step panel.
        if cfg.run_tprime and not is_concat_run:
            steps.append(("tprime", "TPrime"))
        if cfg.run_pybombcell:
            steps.append(("pybombcell", "py_bombcell"))
        return steps

    def _clear_step_status_panel(self) -> None:
        self._step_widgets.clear()
        while self.step_status_host_layout.count():
            item = self.step_status_host_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.lbl_active_run_name.setText("No active run")
        placeholder = QtWidgets.QLabel("Run a queue item to see live step status here.")
        placeholder.setObjectName("SectionHint")
        placeholder.setWordWrap(True)
        self.step_status_host_layout.addWidget(placeholder, 0)
        self.step_status_host_layout.addStretch(1)

    def _prepare_step_status_panel(
        self, run_name: str, cfg: EcephysPipelineConfig, *, is_concat_run: bool = False
    ) -> None:
        self._clear_step_status_panel()
        self.lbl_active_run_name.setText(run_name)
        for step_key, label in self._planned_step_definitions(cfg, is_concat_run=is_concat_run):
            widget = StepStatusItem(label, self.step_status_host)
            self._step_widgets[step_key] = widget
            self.step_status_host_layout.insertWidget(self.step_status_host_layout.count() - 1, widget)
        if not self._step_widgets:
            placeholder = QtWidgets.QLabel("No enabled preprocessing steps for this run.")
            placeholder.setObjectName("SectionHint")
            placeholder.setWordWrap(True)
            self.step_status_host_layout.insertWidget(self.step_status_host_layout.count() - 1, placeholder)

    def _on_worker_step_started(self, step_key: str, _label: str) -> None:
        widget = self._step_widgets.get(step_key)
        if widget is not None:
            widget.set_running()

    def _on_worker_step_progress(self, step_key: str, percent: int) -> None:
        widget = self._step_widgets.get(step_key)
        if widget is not None:
            widget.set_progress(percent)

    def _on_worker_step_finished(self, step_key: str, ok: bool) -> None:
        widget = self._step_widgets.get(step_key)
        if widget is not None:
            if ok:
                widget.set_finished()
            else:
                widget.set_failed()

    def _run_queue(self) -> None:
        if self._running:
            return
        if not self.jobs:
            self._append_log("No jobs in queue.")
            return
        self._persist_settings()
        self.settings.sync()

        self._queue = list(self.jobs)
        self._running = True
        self.btn_run.setEnabled(False)
        self.progress.setValue(0)
        self._update_concat_button_state()
        self._refresh_queue_summary()
        self._run_next()

    def _run_next(self) -> None:
        if not self._queue:
            self._running = False
            self.btn_run.setEnabled(True)
            self.progress.setValue(100)
            self._append_log("All jobs completed.")
            self._update_concat_button_state()
            self._refresh_queue_summary()
            return

        job = self._queue.pop(0)
        self._refresh_queue_summary()
        cfg = self._collect_cfg()
        overrides = job.get("cfg_overrides")
        if isinstance(overrides, dict) and overrides:
            cfg = replace(cfg, **{k: v for k, v in overrides.items() if hasattr(cfg, k)})
            self._append_log(
                f"[{job.get('name', '')}] Using per-job step overrides: "
                + ", ".join(f"{k}={overrides[k]}" for k in sorted(overrides))
            )
        job_bin = job.get("bin_file")
        job_is_concat = bool(job_bin) and is_concatenated_run_bin(job_bin)
        self._prepare_step_status_panel(
            str(job.get("name") or "Unknown run"), cfg, is_concat_run=job_is_concat
        )
        self._active_run_context = {
            "job_snapshot": dict(job),
            "cfg_snapshot": self._serialize_cfg(cfg),
        }
        worker = EcephysPipelineWorker(job, cfg)
        worker.signals.log.connect(self._append_log)
        worker.signals.progress.connect(self.progress.setValue)
        worker.signals.error.connect(self._append_log)
        worker.signals.stepStarted.connect(self._on_worker_step_started)
        worker.signals.stepProgress.connect(self._on_worker_step_progress)
        worker.signals.stepFinished.connect(self._on_worker_step_finished)
        worker.signals.finished.connect(self._on_job_finished)
        self.pool.start(worker)

    def _on_job_finished(self, result: Dict) -> None:
        status = "OK" if result.get("ok") else "FAILED"
        self._append_log(f"Job {result.get('job')} finished: {status}")
        run_name = str(result.get("job", ""))
        if not result.get("ok"):
            if self._step_widgets and not any(widget.percent_label.text() == "Failed" for widget in self._step_widgets.values()):
                first_widget = next(iter(self._step_widgets.values()))
                if first_widget.percent_label.text() == "Pending":
                    first_widget.set_failed()
        self.lbl_active_run_name.setText(f"{run_name} ({'completed' if result.get('ok') else 'failed'})")
        if result.get("ok"):
            ks_folder = str(result.get("ks_folder") or "")
            if not ks_folder:
                ks_ver = self.cb_ks_ver.currentText()
                ks_tag = {"2.0": "ks2", "2.5": "ks25", "3.0": "ks3", "4": "ks4"}.get(ks_ver, "ks4")
                ks_folder = str(
                    (Path(self.ed_output.text().strip()) / run_name / default_kilosort_output_name(ks_tag, self.ed_probe.text().strip())).resolve()
                )
            active_context = self._active_run_context or {}
            entry = self._normalize_completed_entry(
                {
                    "run_name": run_name,
                    "ks_folder": ks_folder,
                    "label": f"{run_name} | {ks_folder}",
                    "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "bin_file": str((active_context.get("job_snapshot") or {}).get("bin_file") or ""),
                    "job_snapshot": active_context.get("job_snapshot") or {},
                    "cfg_snapshot": active_context.get("cfg_snapshot") or {},
                }
            )
            target_row = self._upsert_completed_entry(entry)
            self.list_completed.setCurrentRow(target_row)
            self._persist_completed_history()
            self._refresh_job_list_view()
        self._active_run_context = None
        self._refresh_queue_summary()
        self._run_next()

    def _append_log(self, line: str) -> None:
        self.log.appendPlainText(line)

    def _copy_log(self) -> None:
        QtWidgets.QApplication.clipboard().setText(self.log.toPlainText())
        self._append_log("Log copied to clipboard.")

    def _restore_settings(self) -> None:
        self._restoring_settings = True
        try:
            output_root = self.settings.value("preproc/output_root", "")
            json_root = self.settings.value("preproc/json_root", "")
            if output_root:
                self.ed_output.setText(str(output_root))
            if json_root:
                self.ed_json.setText(str(json_root))
            self.ck_mirror_raw_hierarchy_output.setChecked(
                bool(self.settings.value("preproc/mirror_raw_hierarchy_output", True, type=bool))
            )
            queue_filter = str(self.settings.value("preproc/raw_run_filter", "non_processed"))
            queue_filter_index = max(0, self.cb_queue_filter.findData(queue_filter))
            self.cb_queue_filter.setCurrentIndex(queue_filter_index)
            self.ck_save_catgt_ap_bin.setChecked(
                bool(self.settings.value("preproc/save_catgt_ap_bin", False, type=bool))
            )
            self.cb_ks_ver.setCurrentText(str(self.settings.value("preproc/ks_ver", self.cb_ks_ver.currentText())))
            self.ed_gate.setText(str(self.settings.value("preproc/gate_string", self.ed_gate.text())))
            self.ed_trigger.setText(str(self.settings.value("preproc/trigger_string", self.ed_trigger.text())))
            self.ed_probe.setText(str(self.settings.value("preproc/probe_string", self.ed_probe.text())))
            self.ed_region.setText(str(self.settings.value("preproc/region_name", self.ed_region.text())))
            self.ed_ks_th.setText(str(self.settings.value("preproc/ks_th", self.ed_ks_th.text())))
            self.ed_ni_extract.setText(str(self.settings.value("preproc/ni_extract_string", self.ed_ni_extract.text())))
            self.ed_catgt_cmd.setText(str(self.settings.value("preproc/catgt_cmd_string", self.ed_catgt_cmd.text())))
            self.ed_tostream.setText(str(self.settings.value("preproc/tostream_sync_params", self.ed_tostream.text())))
            car_mode = self.settings.value("preproc/catgt_car_mode", "gbldmx")
            idx = self.cb_catgt_car_mode.findText(str(car_mode))
            if idx >= 0:
                self.cb_catgt_car_mode.setCurrentIndex(idx)
            self.ed_qm_isi.setValue(float(self.settings.value("preproc/qm_isi_thresh", float(self.ed_qm_isi.value()))))
            self.ed_sync_period.setValue(float(self.settings.value("preproc/sync_period", float(self.ed_sync_period.value()))))
            self.sp_loccar_min.setValue(float(self.settings.value("preproc/catgt_loccar_min_um", 40.0)))
            self.sp_loccar_max.setValue(float(self.settings.value("preproc/catgt_loccar_max_um", 160.0)))
            self.sp_ks4_dup_ms.setValue(float(self.settings.value("preproc/ks4_duplicate_spike_ms", 0.25)))
            self.sp_ks4_min_template.setValue(float(self.settings.value("preproc/ks4_min_template_size_um", 10.0)))
            self.sp_cwaves_um.setValue(float(self.settings.value("preproc/c_waves_snr_um", 160.0)))
            self.ed_catgt_path.setText(str(self.settings.value("preproc/catgt_path", self.ed_catgt_path.text())))
            self.ed_tprime_path.setText(str(self.settings.value("preproc/tprime_path", self.ed_tprime_path.text())))
            self.ed_cwaves_path.setText(str(self.settings.value("preproc/cwaves_path", self.ed_cwaves_path.text())))
            self.ed_ks4_repo.setText(str(self.settings.value("preproc/ks4_repo_path", self.ed_ks4_repo.text())))
            self.ed_ks_tmp.setText(str(self.settings.value("preproc/kilosort_output_tmp", self.ed_ks_tmp.text())))
            for key, checkbox in [
                ("preproc/run_catgt", self.ck_catgt),
                ("preproc/run_catgt_extract_only", self.ck_catgt_extract_only),
                ("preproc/run_tprime", self.ck_tprime),
                ("preproc/run_kilosort", self.ck_ks),
                ("preproc/run_kilosort_postproc", self.ck_post),
                ("preproc/run_noise_templates", self.ck_noise),
                ("preproc/run_mean_waveforms", self.ck_wvf),
                ("preproc/run_quality_metrics", self.ck_qm),
                ("preproc/run_pybombcell", self.ck_pybomb),
            ]:
                checkbox.setChecked(bool(self.settings.value(key, checkbox.isChecked(), type=bool)))
            raw_adv = str(self.settings.value("preproc/ks4_advanced_params_json", "{}"))
            try:
                self._ks4_adv_params = json.loads(raw_adv) if raw_adv else {}
            except Exception:
                self._ks4_adv_params = {}
            self._restore_completed_history()
        finally:
            self._restoring_settings = False

    def _persist_settings(self) -> None:
        if self._restoring_settings:
            return
        self.settings.setValue("preproc/output_root", self.ed_output.text().strip())
        self.settings.setValue("preproc/json_root", self.ed_json.text().strip())
        self.settings.setValue(
            "preproc/mirror_raw_hierarchy_output",
            self.ck_mirror_raw_hierarchy_output.isChecked(),
        )
        self.settings.setValue("preproc/raw_run_filter", self._queue_filter_mode())
        self.settings.setValue("preproc/save_catgt_ap_bin", self.ck_save_catgt_ap_bin.isChecked())
        self.settings.setValue("preproc/ks_ver", self.cb_ks_ver.currentText())
        self.settings.setValue("preproc/gate_string", self.ed_gate.text().strip())
        self.settings.setValue("preproc/trigger_string", self.ed_trigger.text().strip())
        self.settings.setValue("preproc/probe_string", self.ed_probe.text().strip())
        self.settings.setValue("preproc/region_name", self.ed_region.text().strip())
        self.settings.setValue("preproc/ks_th", self.ed_ks_th.text().strip())
        self.settings.setValue("preproc/qm_isi_thresh", float(self.ed_qm_isi.value()))
        self.settings.setValue("preproc/ni_extract_string", self.ed_ni_extract.text().strip())
        self.settings.setValue("preproc/catgt_cmd_string", self.ed_catgt_cmd.text().strip())
        self.settings.setValue("preproc/sync_period", float(self.ed_sync_period.value()))
        self.settings.setValue("preproc/tostream_sync_params", self.ed_tostream.text().strip())
        self.settings.setValue("preproc/catgt_car_mode", self.cb_catgt_car_mode.currentText())
        self.settings.setValue("preproc/catgt_loccar_min_um", float(self.sp_loccar_min.value()))
        self.settings.setValue("preproc/catgt_loccar_max_um", float(self.sp_loccar_max.value()))
        self.settings.setValue("preproc/ks4_duplicate_spike_ms", float(self.sp_ks4_dup_ms.value()))
        self.settings.setValue("preproc/ks4_min_template_size_um", float(self.sp_ks4_min_template.value()))
        self.settings.setValue("preproc/c_waves_snr_um", float(self.sp_cwaves_um.value()))
        self.settings.setValue("preproc/catgt_path", self.ed_catgt_path.text().strip())
        self.settings.setValue("preproc/tprime_path", self.ed_tprime_path.text().strip())
        self.settings.setValue("preproc/cwaves_path", self.ed_cwaves_path.text().strip())
        self.settings.setValue("preproc/ks4_repo_path", self.ed_ks4_repo.text().strip())
        self.settings.setValue("preproc/kilosort_output_tmp", self.ed_ks_tmp.text().strip())
        for key, checkbox in [
            ("preproc/run_catgt", self.ck_catgt),
            ("preproc/run_catgt_extract_only", self.ck_catgt_extract_only),
            ("preproc/run_tprime", self.ck_tprime),
            ("preproc/run_kilosort", self.ck_ks),
            ("preproc/run_kilosort_postproc", self.ck_post),
            ("preproc/run_noise_templates", self.ck_noise),
            ("preproc/run_mean_waveforms", self.ck_wvf),
            ("preproc/run_quality_metrics", self.ck_qm),
            ("preproc/run_pybombcell", self.ck_pybomb),
        ]:
            self.settings.setValue(key, checkbox.isChecked())
        self.settings.setValue("preproc/ks4_advanced_params_json", json.dumps(self._ks4_adv_params))
        self._settings_sync_timer.start()

    def save_settings(self) -> None:
        """Persist the current widget state as the defaults for the next launch."""
        self._persist_settings()
        self.settings.sync()
        self._append_log("Preprocessing settings saved as defaults for the next app launch.")

    def _set_last_folder(self, folder: str) -> None:
        self.settings.setValue("paths/last_folder", folder)

    def _set_last_file_dir(self, file_or_dir: str) -> None:
        p = Path(file_or_dir)
        folder = str(p if p.is_dir() else p.parent)
        self.settings.setValue("paths/last_file_dir", folder)
        self.settings.setValue("paths/last_folder", folder)

    def _get_recent(self, key: str) -> List[str]:
        value = self.settings.value(key, [])
        if isinstance(value, str):
            return [value] if value else []
        if value is None:
            return []
        return [str(v) for v in value]

    def _add_recent(self, key: str, path: str, max_items: int = 12) -> None:
        p = str(Path(path))
        rec = [r for r in self._get_recent(key) if r != p]
        rec.insert(0, p)
        self.settings.setValue(key, rec[:max_items])

    def _open_recent_file_menu(self) -> None:
        rec = [p for p in self._get_recent("recent_files") if Path(p).exists()]
        menu = QtWidgets.QMenu(self)
        if not rec:
            act = menu.addAction("No recent files")
            act.setEnabled(False)
        else:
            for p in rec:
                act = menu.addAction(p)
                act.triggered.connect(lambda checked=False, path=p: self._add_paths([path]))
        menu.exec(self.btn_recent_files.mapToGlobal(self.btn_recent_files.rect().bottomLeft()))

    def _open_recent_folder_menu(self) -> None:
        rec = [p for p in self._get_recent("recent_folders") if Path(p).exists()]
        menu = QtWidgets.QMenu(self)
        if not rec:
            act = menu.addAction("No recent folders")
            act.setEnabled(False)
        else:
            for p in rec:
                act = menu.addAction(p)
                act.triggered.connect(lambda checked=False, path=p: self._add_paths([path]))
        menu.exec(self.btn_recent_folders.mapToGlobal(self.btn_recent_folders.rect().bottomLeft()))

    def _open_selected_curation(self, item: QtWidgets.QListWidgetItem | None = None) -> None:
        entries = self._selected_completed_entries(item)
        folders = completed_run_target_folders(entries)
        if folders:
            self.openCurationRequested.emit(folders)

    def _open_selected_histology(self, item: QtWidgets.QListWidgetItem | None = None) -> None:
        entries = self._selected_completed_entries(item)
        if not entries:
            self._append_log("Select a completed run to set up histology.")
            return
        if len(entries) > 1:
            self._append_log("Histology is per probe; using the first selected run.")
        folder = str(entries[0].get("ks_folder") or "")
        if folder:
            self.openHistologyRequested.emit(folder)

    def clear_completed_history(self) -> None:
        """Drop the completed-runs history from the list, memory, and settings."""
        self.completed_runs.clear()
        self.list_completed.clear()
        self.settings.remove("preproc/completed_runs_history_json")
        self.settings.sync()
        self._refresh_queue_summary()

    def is_busy(self) -> bool:
        """Return True while a queue run or a concatenation is in progress."""
        return bool(self._running or self._concatenating)

    def _open_ks4_advanced(self) -> None:
        dlg = Ks4AdvancedDialog(self._ks4_adv_params, self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        self._ks4_adv_params = dlg.values()
        self._persist_settings()
        n = len(self._ks4_adv_params)
        self._append_log(f"KS4 advanced parameters updated ({n} values).")

    def _open_catgt_builder(self) -> None:
        dlg = CatGTStringBuilderDialog(self.ed_catgt_cmd.text().strip(), self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        self.ed_catgt_cmd.setText(dlg.value())
        self._persist_settings()
        self._append_log("CatGT command string updated from builder.")

    def _open_tprime_builder(self) -> None:
        # Labels are preserved only in the dedicated extractor field. The CatGT
        # command stores the executable flags without label suffixes.
        current_extractors = self.ed_ni_extract.text().strip()
        if not current_extractors:
            current_extractors = catgt_command_extractors(self.ed_catgt_cmd.text().strip())
        dlg = TPrimeStringBuilderDialog(
            self.ed_tostream.text().strip(),
            current_extractors,
            self,
        )
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        to_stream, extract_string = dlg.values()
        self.ed_tostream.setText(to_stream)
        self.ed_ni_extract.setText(extract_string)
        self.ed_catgt_cmd.setText(
            merge_extractors_into_catgt_command(self.ed_catgt_cmd.text().strip(), extract_string)
        )
        self._persist_settings()
        self._append_log("TPrime sync and CatGT extractor strings updated from builder.")

    def _open_bitfield_builder(self) -> None:
        dlg = BitFieldBuilderDialog(catgt_command_bf_extractors(self.ed_catgt_cmd.text().strip()), self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        self.ed_catgt_cmd.setText(
            merge_bitfields_into_catgt_command(self.ed_catgt_cmd.text().strip(), dlg.value())
        )
        self._persist_settings()
        self._append_log("CatGT bit-field extractor flags updated from builder.")

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import json
import math

from PySide6 import QtCore, QtGui, QtWidgets

from ..preprocessing import (
    default_kilosort_output_name,
    discover_bin_files,
    parse_spikeglx_bin_name,
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
from ..workers import EcephysPipelineConfig, EcephysPipelineWorker


class BinDropList(QtWidgets.QListWidget):
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


class Ks4AdvancedDialog(QtWidgets.QDialog):
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


class PreprocessingTab(QtWidgets.QWidget):
    openCurationRequested = QtCore.Signal(str)
    saveSettingsFileRequested = QtCore.Signal()
    loadSettingsFileRequested = QtCore.Signal()

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
        self.completed_runs: List[Dict[str, str]] = []
        self._queue: List[Dict[str, str]] = []
        self._running = False
        self._ks4_adv_params: Dict[str, object] = {}

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
        self.btn_recent_files = QtWidgets.QPushButton("Open recent file")
        self.btn_recent_folders = QtWidgets.QPushButton("Open recent folder")
        self.btn_remove = QtWidgets.QPushButton("Remove selected")
        self.btn_clear = QtWidgets.QPushButton("Clear")
        self.btn_run = QtWidgets.QPushButton("Run queue")
        self.btn_add_files.setProperty("role", "secondary")
        self.btn_add_folder.setProperty("role", "secondary")
        self.btn_recent_files.setProperty("role", "ghost")
        self.btn_recent_folders.setProperty("role", "ghost")
        self.btn_remove.setProperty("role", "ghost")
        self.btn_clear.setProperty("role", "ghost")
        self.btn_run.setProperty("role", "primary")
        top.addWidget(self.btn_add_files)
        top.addWidget(self.btn_add_folder)
        top.addWidget(self.btn_recent_files)
        top.addWidget(self.btn_recent_folders)
        top.addStretch(1)
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

        config = QtWidgets.QGroupBox("Integrated ecephys spike sorting preprocessing")
        config.setProperty("settingsSection", True)
        config.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        config_layout = QtWidgets.QVBoxLayout(config)
        config_layout.setSpacing(12)

        self.ck_catgt = QtWidgets.QCheckBox("CatGT")
        self.ck_catgt_extract_only = QtWidgets.QCheckBox("CatGT extract-only")
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
            1,
            2,
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
        done_hint = QtWidgets.QLabel("Jump directly from a finished run into curation or quality-metrics review.")
        done_hint.setObjectName("SectionHint")
        done_hint.setWordWrap(True)
        done_layout.addWidget(done_hint)
        self.list_completed = QtWidgets.QListWidget()
        self.list_completed.setAlternatingRowColors(True)
        self.list_completed.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list_completed.setSpacing(4)
        self.list_completed.setMinimumHeight(220)
        self.list_completed.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        done_layout.addWidget(self.list_completed, 1)
        done_row = QtWidgets.QHBoxLayout()
        self.btn_to_curation = QtWidgets.QPushButton("To Curation")
        self.btn_to_curation.setProperty("role", "secondary")
        done_row.addStretch(1)
        done_row.addWidget(self.btn_to_curation)
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
        log_layout.addLayout(log_header)
        log_layout.addWidget(self.log, 1)

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
        work_sections.add_page("Log", log_box)
        work_sections.add_page("Completed", done_box)
        work_sections.setCurrentIndex(0)
        self.work_sections = work_sections
        main.addWidget(work_sections, 1)
        main.addWidget(self.progress, 0)
        self._refresh_queue_summary()

        self.btn_add_files.clicked.connect(self._add_files)
        self.btn_add_folder.clicked.connect(self._add_folder)
        self.btn_recent_files.clicked.connect(self._open_recent_file_menu)
        self.btn_recent_folders.clicked.connect(self._open_recent_folder_menu)
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_clear.clicked.connect(self._clear)
        self.btn_run.clicked.connect(self._run_queue)
        self.btn_copy_log.clicked.connect(self._copy_log)
        self.list_jobs.filesDropped.connect(self._consume_drop)
        self.list_completed.itemDoubleClicked.connect(lambda _item: self._open_selected_curation())
        btn_output.clicked.connect(lambda: self._pick_folder(self.ed_output))
        btn_json.clicked.connect(lambda: self._pick_folder(self.ed_json))
        btn_catgt_path.clicked.connect(lambda: self._pick_folder(self.ed_catgt_path))
        btn_tprime_path.clicked.connect(lambda: self._pick_folder(self.ed_tprime_path))
        btn_cwaves_path.clicked.connect(lambda: self._pick_folder(self.ed_cwaves_path))
        btn_ks4_repo.clicked.connect(lambda: self._pick_folder(self.ed_ks4_repo))
        btn_ks_tmp.clicked.connect(lambda: self._pick_folder(self.ed_ks_tmp))
        self.ed_output.editingFinished.connect(self._persist_settings)
        self.ck_mirror_raw_hierarchy_output.toggled.connect(lambda _checked: self._persist_settings())
        self.ed_json.editingFinished.connect(self._persist_settings)
        for checkbox in [
            self.ck_catgt,
            self.ck_catgt_extract_only,
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

    def _add_paths(self, paths: List[str]) -> None:
        bins = discover_bin_files(paths)
        if not bins and paths:
            self._append_log("No valid AP files found. Expected names like *.imec0.ap.bin")
        existing = {j["bin_file"] for j in self.jobs}
        for b in bins:
            if b in existing:
                continue
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
            self.jobs.append(job)
            self.list_jobs.addItem(
                f"{parsed['run_name']}  |  g{parsed['gate_string']} t{parsed['trigger_string']} p{parsed['probe_string']}  |  {b}"
            )
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
        self._refresh_queue_summary()

    def _refresh_queue_summary(self) -> None:
        if not hasattr(self, "lbl_queue_summary"):
            return
        n_jobs = len(self.jobs)
        if self._running:
            remaining = len(self._queue)
            msg = f"{n_jobs} recording(s) loaded. Queue running with {remaining} remaining after the active job."
        elif n_jobs == 0:
            msg = "Queue is empty. Add AP .bin files or folders to begin."
        elif n_jobs == 1:
            msg = "1 recording queued and ready to run."
        else:
            msg = f"{n_jobs} recordings queued and ready to run."
        if self.completed_runs:
            msg += f" {len(self.completed_runs)} completed run(s) available in Completed."
        self.lbl_queue_summary.setText(msg)

    def _remove_selected(self) -> None:
        selected_rows = sorted({i.row() for i in self.list_jobs.selectedIndexes()}, reverse=True)
        for row in selected_rows:
            self.list_jobs.takeItem(row)
            self.jobs.pop(row)
        self._refresh_queue_summary()

    def _clear(self) -> None:
        self.list_jobs.clear()
        self.jobs.clear()
        self._refresh_queue_summary()

    def _collect_cfg(self) -> EcephysPipelineConfig:
        return EcephysPipelineConfig(
            output_root=self.ed_output.text().strip(),
            json_root=self.ed_json.text().strip(),
            mirror_raw_hierarchy_output=self.ck_mirror_raw_hierarchy_output.isChecked(),
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
        self._refresh_queue_summary()
        self._run_next()

    def _run_next(self) -> None:
        if not self._queue:
            self._running = False
            self.btn_run.setEnabled(True)
            self.progress.setValue(100)
            self._append_log("All jobs completed.")
            self._refresh_queue_summary()
            return

        job = self._queue.pop(0)
        self._refresh_queue_summary()
        worker = EcephysPipelineWorker(job, self._collect_cfg())
        worker.signals.log.connect(self._append_log)
        worker.signals.progress.connect(self.progress.setValue)
        worker.signals.error.connect(self._append_log)
        worker.signals.finished.connect(self._on_job_finished)
        self.pool.start(worker)

    def _on_job_finished(self, result: Dict) -> None:
        status = "OK" if result.get("ok") else "FAILED"
        self._append_log(f"Job {result.get('job')} finished: {status}")
        if result.get("ok"):
            run_name = str(result.get("job", ""))
            ks_folder = str(result.get("ks_folder") or "")
            if not ks_folder:
                ks_ver = self.cb_ks_ver.currentText()
                ks_tag = {"2.0": "ks2", "2.5": "ks25", "3.0": "ks3", "4": "ks4"}.get(ks_ver, "ks4")
                ks_folder = str(
                    (Path(self.ed_output.text().strip()) / run_name / default_kilosort_output_name(ks_tag, self.ed_probe.text().strip())).resolve()
                )
            label = f"{run_name} | {ks_folder}"
            self.completed_runs.append({"run_name": run_name, "ks_folder": ks_folder, "label": label})
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.UserRole, ks_folder)
            self.list_completed.addItem(item)
            self.list_completed.setCurrentItem(item)
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
                bool(self.settings.value("preproc/mirror_raw_hierarchy_output", False, type=bool))
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

    def _open_selected_curation(self) -> None:
        item = self.list_completed.currentItem()
        if item is None and self.list_completed.count():
            item = self.list_completed.item(self.list_completed.count() - 1)
        folder = item.data(QtCore.Qt.UserRole) if item is not None else None
        if folder:
            self.openCurationRequested.emit(str(folder))

    def is_busy(self) -> bool:
        return bool(self._running)

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

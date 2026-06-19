"""Histology tab: AP_histology-style probe localization, unified and renewed.

Stages (left rail):

    Setup      pick the histology folder + tool paths, see what products exist
    Preprocess load raw images, colourise, segment/extract slices, reorient, save
    Match      choose the Allen CCF plane for each slice            -> histology_ccf
    Align      warp atlas <-> histology (control points / auto)     -> atlas2histology_tform
    Trace      draw probe tracks, sample regions                    -> probe_ccf (+ CSV)
    Channels   ALF extraction + xyz_picks + per-channel region map  -> channel_locations_all_shanks
    IBL refine launch the unmodified IBL ephys-alignment GUI (optional)

All heavy lifting is delegated to :mod:`neuropyguin.histology`. The IBL-dependent
steps run through :mod:`neuropyguin.histology.ibl_launch` (subprocess), so this tab
works whether or not the IBL stack is importable in-process.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from ..side_nav import SideNavStack
from ..workers import FunctionWorker
from ..histology import atlas as hatlas
from ..histology import io_formats, matching, tracing, alignment, slice_prep, ibl_launch


PROBE_QCOLORS = [
    QtGui.QColor(*[int(c * 255) for c in rgb]) for rgb in tracing.probe_colormap(20)
]


class ImageCanvas(pg.GraphicsLayoutWidget):
    """A simple image view with pixel-coordinate click reporting and overlays."""

    clicked = QtCore.Signal(float, float)  # image x (col), y (row)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.view = self.addViewBox()
        self.view.setAspectLocked(True)
        self.view.invertY(True)
        self.img = pg.ImageItem()
        self.img.setOpts(axisOrder="row-major")
        self.view.addItem(self.img)
        self._overlays: List[pg.GraphicsObject] = []
        self.scene().sigMouseClicked.connect(self._on_click)

    def set_image(self, arr: Optional[np.ndarray], levels=None) -> None:
        if arr is None:
            self.img.clear()
            return
        self.img.setImage(np.asarray(arr), autoLevels=(levels is None), levels=levels)
        self.view.autoRange()

    def _on_click(self, ev) -> None:
        if ev.button() != QtCore.Qt.LeftButton:
            return
        p = self.view.mapSceneToView(ev.scenePos())
        self.clicked.emit(float(p.x()), float(p.y()))

    def clear_overlays(self) -> None:
        for item in self._overlays:
            self.view.removeItem(item)
        self._overlays = []

    def add_scatter(self, xs, ys, color="w", size=12) -> None:
        sp = pg.ScatterPlotItem(x=list(xs), y=list(ys), size=size,
                                brush=pg.mkBrush(color), pen=pg.mkPen("k", width=1))
        self.view.addItem(sp)
        self._overlays.append(sp)

    def add_line(self, xs, ys, color="y", width=3) -> None:
        ln = pg.PlotDataItem(x=list(xs), y=list(ys), pen=pg.mkPen(color, width=width))
        self.view.addItem(ln)
        self._overlays.append(ln)

    def add_mask_overlay(self, mask: np.ndarray, color=(255, 40, 40)) -> None:
        rgba = np.zeros((*mask.shape, 4), dtype=np.ubyte)
        rgba[mask.astype(bool)] = [*color, 160]
        item = pg.ImageItem(rgba)
        item.setOpts(axisOrder="row-major")
        self.view.addItem(item)
        self._overlays.append(item)


class HistologyTab(QtWidgets.QWidget):
    def __init__(self, thread_pool: QtCore.QThreadPool) -> None:
        super().__init__()
        self.pool = thread_pool
        self.settings = QtCore.QSettings("NeuroPyGuiN", "NeuroPyGuiN")
        self._busy_count = 0
        self._plot_theme = "Light"

        # Pipeline state
        self.folder: Optional[Path] = None
        self.atlas: Optional[hatlas.AllenCCFAtlas] = None
        self.slice_images: List[np.ndarray] = []
        self.histology_ccf: List[Dict[str, np.ndarray]] = []
        self.tforms: List[np.ndarray] = []
        self.slice_specs: List[Optional[Dict[str, np.ndarray]]] = []
        self.probe_points: Dict[Tuple[int, int], np.ndarray] = {}
        self._cur_match_slice = 0
        self._cur_align_slice = 0
        self._cur_trace_slice = 0
        self._align_hist_pts: Dict[int, List[Tuple[float, float]]] = {}
        self._align_atlas_pts: Dict[int, List[Tuple[float, float]]] = {}
        self._active_probe = 1
        self._pending_click: List[Tuple[float, float]] = []

        self._build_ui()
        self._restore_settings()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        main = QtWidgets.QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        self.nav = SideNavStack(
            "Histology", "Localize Neuropixels probes on the Allen CCF.",
        )
        main.addWidget(self.nav, 1)

        self.nav.add_page("Setup", self._build_setup_page())
        self.nav.add_page("Preprocess", self._build_preprocess_page())
        self.nav.add_page("Match atlas", self._build_match_page())
        self.nav.add_page("Align", self._build_align_page())
        self.nav.add_page("Trace probes", self._build_trace_page())
        self.nav.add_page("Channel map", self._build_channels_page())
        self.nav.add_page("IBL refine", self._build_ibl_page())
        self.nav.setCurrentIndex(0)

        # Shared log dock at the bottom.
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setFixedHeight(120)
        self.log.setObjectName("HistologyLog")
        main.addWidget(self.log, 0)

    def _section(self, title: str, hint: str = "") -> Tuple[QtWidgets.QWidget, QtWidgets.QVBoxLayout]:
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setContentsMargins(18, 16, 18, 16)
        v.setSpacing(12)
        t = QtWidgets.QLabel(title)
        t.setObjectName("FieldTitle")
        v.addWidget(t)
        if hint:
            h = QtWidgets.QLabel(hint)
            h.setObjectName("SectionHint")
            h.setWordWrap(True)
            v.addWidget(h)
        return page, v

    # ---- Setup page ----
    def _build_setup_page(self) -> QtWidgets.QWidget:
        page, v = self._section(
            "Session and tools",
            "Point at the session histology folder (where products are written) and "
            "the Allen CCF atlas. Optional paths enable the IBL refinement step.",
        )
        form = QtWidgets.QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        self.ed_folder = QtWidgets.QLineEdit()
        b_folder = QtWidgets.QPushButton("Browse...")
        b_folder.clicked.connect(self._pick_folder)
        form.addRow("Histology folder", self._with_button(self.ed_folder, b_folder))

        self.ed_raw = QtWidgets.QLineEdit()
        b_raw = QtWidgets.QPushButton("Browse...")
        b_raw.clicked.connect(lambda: self._pick_into(self.ed_raw, "Raw image folder"))
        form.addRow("Raw images", self._with_button(self.ed_raw, b_raw))

        self.ed_atlas = QtWidgets.QLineEdit(hatlas.DEFAULT_ATLAS_PATH)
        b_atlas = QtWidgets.QPushButton("Browse...")
        b_atlas.clicked.connect(lambda: self._pick_into(self.ed_atlas, "Allen CCF atlas folder"))
        form.addRow("Atlas folder", self._with_button(self.ed_atlas, b_atlas))

        self.ed_ks = QtWidgets.QLineEdit()
        b_ks = QtWidgets.QPushButton("Browse...")
        b_ks.clicked.connect(lambda: self._pick_into(self.ed_ks, "Kilosort output folder"))
        form.addRow("Kilosort folder", self._with_button(self.ed_ks, b_ks))

        self.ed_ephys = QtWidgets.QLineEdit()
        b_ephys = QtWidgets.QPushButton("Browse...")
        b_ephys.clicked.connect(lambda: self._pick_into(self.ed_ephys, "Raw ephys folder (.ap.bin)"))
        form.addRow("Ephys folder", self._with_button(self.ed_ephys, b_ephys))

        self.ed_iblapps = QtWidgets.QLineEdit(ibl_launch.DEFAULT_IBLAPPS_PATH)
        form.addRow("iblapps path", self.ed_iblapps)
        self.ed_pyexe = QtWidgets.QLineEdit()
        self.ed_pyexe.setPlaceholderText("auto-detect (interpreter with iblatlas)")
        form.addRow("IBL python", self.ed_pyexe)

        v.addLayout(form)

        row = QtWidgets.QHBoxLayout()
        b_load = QtWidgets.QPushButton("Load session")
        b_load.clicked.connect(self._load_session)
        b_save = QtWidgets.QPushButton("Save paths")
        b_save.clicked.connect(self._persist_settings)
        row.addWidget(b_load)
        row.addWidget(b_save)
        row.addStretch(1)
        v.addLayout(row)

        self.lbl_status = QtWidgets.QLabel("No session loaded.")
        self.lbl_status.setObjectName("SectionHint")
        self.lbl_status.setTextFormat(QtCore.Qt.RichText)
        v.addWidget(self.lbl_status)
        v.addStretch(1)
        return page

    # ---- Preprocess page ----
    def _build_preprocess_page(self) -> QtWidgets.QWidget:
        page, v = self._section(
            "Slice preprocessing",
            "Load raw images, optionally downsample, then save individual slice "
            "images. Use the reorient buttons to fix rotation/flip/order.",
        )
        ctl = QtWidgets.QHBoxLayout()
        self.sp_downsample = QtWidgets.QDoubleSpinBox()
        self.sp_downsample.setRange(1, 50)
        self.sp_downsample.setValue(1)
        self.sp_downsample.setPrefix("1/")
        ctl.addWidget(QtWidgets.QLabel("Downsample"))
        ctl.addWidget(self.sp_downsample)
        b_loadimg = QtWidgets.QPushButton("Load raw images")
        b_loadimg.clicked.connect(self._preproc_load)
        ctl.addWidget(b_loadimg)
        b_save_slices = QtWidgets.QPushButton("Save slices")
        b_save_slices.clicked.connect(self._preproc_save)
        ctl.addWidget(b_save_slices)
        ctl.addStretch(1)
        v.addLayout(ctl)

        nav = QtWidgets.QHBoxLayout()
        b_prev = QtWidgets.QPushButton("< Prev")
        b_prev.clicked.connect(lambda: self._preproc_step(-1))
        b_next = QtWidgets.QPushButton("Next >")
        b_next.clicked.connect(lambda: self._preproc_step(1))
        b_rotl = QtWidgets.QPushButton("Rotate -90")
        b_rotl.clicked.connect(lambda: self._preproc_rotate(-90))
        b_rotr = QtWidgets.QPushButton("Rotate +90")
        b_rotr.clicked.connect(lambda: self._preproc_rotate(90))
        b_fliph = QtWidgets.QPushButton("Flip H")
        b_fliph.clicked.connect(lambda: self._preproc_flip(True))
        b_flipv = QtWidgets.QPushButton("Flip V")
        b_flipv.clicked.connect(lambda: self._preproc_flip(False))
        for b in [b_prev, b_next, b_rotl, b_rotr, b_fliph, b_flipv]:
            nav.addWidget(b)
        nav.addStretch(1)
        self.lbl_preproc = QtWidgets.QLabel("0 / 0")
        nav.addWidget(self.lbl_preproc)
        v.addLayout(nav)

        self.canvas_preproc = ImageCanvas()
        v.addWidget(self.canvas_preproc, 1)
        return page

    # ---- Match page ----
    def _build_match_page(self) -> QtWidgets.QWidget:
        page, v = self._section(
            "Match atlas slices",
            "For each histology slice, dial in the Allen CCF plane (AP position and "
            "small tilts), then Assign. Save writes histology_ccf.mat.",
        )
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.canvas_match_hist = ImageCanvas()
        self.canvas_match_atlas = ImageCanvas()
        split.addWidget(self._titled("Histology", self.canvas_match_hist))
        split.addWidget(self._titled("Atlas plane", self.canvas_match_atlas))
        v.addWidget(split, 1)

        ctl = QtWidgets.QHBoxLayout()
        b_prev = QtWidgets.QPushButton("< Slice")
        b_prev.clicked.connect(lambda: self._match_step(-1))
        b_next = QtWidgets.QPushButton("Slice >")
        b_next.clicked.connect(lambda: self._match_step(1))
        ctl.addWidget(b_prev)
        ctl.addWidget(b_next)
        ctl.addWidget(QtWidgets.QLabel("AP"))
        self.sl_ap = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sl_ap.setRange(1, 1320)
        self.sl_ap.setValue(540)
        self.sl_ap.valueChanged.connect(self._match_update_atlas)
        ctl.addWidget(self.sl_ap, 2)
        ctl.addWidget(QtWidgets.QLabel("LR tilt"))
        self.sl_lr = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sl_lr.setRange(-15, 15)
        self.sl_lr.valueChanged.connect(self._match_update_atlas)
        ctl.addWidget(self.sl_lr, 1)
        ctl.addWidget(QtWidgets.QLabel("SI tilt"))
        self.sl_si = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sl_si.setRange(-15, 15)
        self.sl_si.valueChanged.connect(self._match_update_atlas)
        ctl.addWidget(self.sl_si, 1)
        self.cb_mode = QtWidgets.QComboBox()
        self.cb_mode.addItems(["TV", "AV", "TV-AV"])
        self.cb_mode.currentTextChanged.connect(self._match_update_atlas)
        ctl.addWidget(self.cb_mode)
        v.addLayout(ctl)

        row = QtWidgets.QHBoxLayout()
        self.lbl_match = QtWidgets.QLabel("slice 0 / 0")
        row.addWidget(self.lbl_match)
        row.addStretch(1)
        b_assign = QtWidgets.QPushButton("Assign plane to slice")
        b_assign.clicked.connect(self._match_assign)
        b_save = QtWidgets.QPushButton("Save histology_ccf")
        b_save.clicked.connect(self._match_save)
        row.addWidget(b_assign)
        row.addWidget(b_save)
        v.addLayout(row)
        return page

    # ---- Align page ----
    def _build_align_page(self) -> QtWidgets.QWidget:
        page, v = self._section(
            "Align atlas to histology",
            "Click matching landmarks on histology then atlas (>= 3 pairs) for a manual "
            "affine, or use Auto-align. Save writes atlas2histology_tform.mat.",
        )
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.canvas_align_hist = ImageCanvas()
        self.canvas_align_atlas = ImageCanvas()
        self.canvas_align_hist.clicked.connect(self._align_click_hist)
        self.canvas_align_atlas.clicked.connect(self._align_click_atlas)
        split.addWidget(self._titled("Histology (click landmarks)", self.canvas_align_hist))
        split.addWidget(self._titled("Atlas (click landmarks)", self.canvas_align_atlas))
        v.addWidget(split, 1)

        ctl = QtWidgets.QHBoxLayout()
        b_prev = QtWidgets.QPushButton("< Slice")
        b_prev.clicked.connect(lambda: self._align_step(-1))
        b_next = QtWidgets.QPushButton("Slice >")
        b_next.clicked.connect(lambda: self._align_step(1))
        b_clear = QtWidgets.QPushButton("Clear points")
        b_clear.clicked.connect(self._align_clear)
        b_auto = QtWidgets.QPushButton("Auto-align")
        b_auto.clicked.connect(self._align_auto)
        b_apply = QtWidgets.QPushButton("Apply points")
        b_apply.clicked.connect(self._align_apply_points)
        b_save = QtWidgets.QPushButton("Save tform")
        b_save.clicked.connect(self._align_save)
        for b in [b_prev, b_next, b_clear, b_auto, b_apply, b_save]:
            ctl.addWidget(b)
        ctl.addStretch(1)
        self.lbl_align = QtWidgets.QLabel("slice 0 / 0")
        ctl.addWidget(self.lbl_align)
        v.addLayout(ctl)
        return page

    # ---- Trace page ----
    def _build_trace_page(self) -> QtWidgets.QWidget:
        page, v = self._section(
            "Trace probe tracks",
            "Pick a probe number, then click two endpoints on each slice the probe "
            "crosses. Save writes probe_ccf.mat + CSV and the trajectory-area chart.",
        )
        body = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.canvas_trace = ImageCanvas()
        self.canvas_trace.clicked.connect(self._trace_click)
        body.addWidget(self._titled("Histology (click 2 points per probe)", self.canvas_trace))
        self.trace_areas = pg.GraphicsLayoutWidget()
        body.addWidget(self._titled("Trajectory areas", self.trace_areas))
        body.setSizes([700, 350])
        v.addWidget(body, 1)

        ctl = QtWidgets.QHBoxLayout()
        b_prev = QtWidgets.QPushButton("< Slice")
        b_prev.clicked.connect(lambda: self._trace_step(-1))
        b_next = QtWidgets.QPushButton("Slice >")
        b_next.clicked.connect(lambda: self._trace_step(1))
        ctl.addWidget(b_prev)
        ctl.addWidget(b_next)
        ctl.addWidget(QtWidgets.QLabel("Probe"))
        self.sp_probe = QtWidgets.QSpinBox()
        self.sp_probe.setRange(1, 20)
        self.sp_probe.valueChanged.connect(self._trace_set_probe)
        ctl.addWidget(self.sp_probe)
        b_clear = QtWidgets.QPushButton("Clear probe on slice")
        b_clear.clicked.connect(self._trace_clear)
        ctl.addWidget(b_clear)
        b_build = QtWidgets.QPushButton("Build + Save probe_ccf")
        b_build.clicked.connect(self._trace_build)
        ctl.addWidget(b_build)
        ctl.addStretch(1)
        self.lbl_trace = QtWidgets.QLabel("slice 0 / 0")
        ctl.addWidget(self.lbl_trace)
        v.addLayout(ctl)
        return page

    # ---- Channel map page ----
    def _build_channels_page(self) -> QtWidgets.QWidget:
        page, v = self._section(
            "Per-channel region map",
            "Generate xyz_picks and the channel region map directly from the probe "
            "tracks (AP_histology is enough). ALF extraction is optional and only "
            "needed if the ephys files are not present yet.",
        )
        self.ck_extract = QtWidgets.QCheckBox("Run ALF extraction first (needs Kilosort + ephys folders)")
        v.addWidget(self.ck_extract)
        self.cb_alignment = QtWidgets.QComboBox()
        self.cb_alignment.addItems(["original", "latest"])
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Alignment source"))
        row.addWidget(self.cb_alignment)
        row.addStretch(1)
        v.addLayout(row)

        btns = QtWidgets.QHBoxLayout()
        b_xyz = QtWidgets.QPushButton("Generate xyz_picks")
        b_xyz.clicked.connect(self._gen_xyz)
        b_ch = QtWidgets.QPushButton("Generate channel map")
        b_ch.clicked.connect(self._gen_channels)
        b_all = QtWidgets.QPushButton("Run all (extract -> xyz -> channels)")
        b_all.clicked.connect(self._gen_all)
        for b in [b_xyz, b_ch, b_all]:
            btns.addWidget(b)
        btns.addStretch(1)
        v.addLayout(btns)

        self.tbl_channels = QtWidgets.QTableWidget(0, 4)
        self.tbl_channels.setHorizontalHeaderLabels(["channel", "axial", "lateral", "region"])
        self.tbl_channels.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.tbl_channels, 1)
        return page

    # ---- IBL page ----
    def _build_ibl_page(self) -> QtWidgets.QWidget:
        page, v = self._section(
            "IBL ephys alignment (optional)",
            "Launch the original IBL alignment GUI to refine the probe-to-brain mapping "
            "against electrophysiology features. After saving in that GUI, regenerate "
            "the channel map with the 'latest' alignment.",
        )
        btns = QtWidgets.QHBoxLayout()
        b_launch = QtWidgets.QPushButton("Launch IBL alignment GUI")
        b_launch.clicked.connect(self._launch_ibl)
        b_refresh = QtWidgets.QPushButton("Regenerate channel map (latest)")
        b_refresh.clicked.connect(lambda: self._gen_channels(alignment_override="latest"))
        btns.addWidget(b_launch)
        btns.addWidget(b_refresh)
        btns.addStretch(1)
        v.addLayout(btns)
        note = QtWidgets.QLabel(
            "The IBL GUI opens its own window (offline mode). Select the histology "
            "folder there, refine, and save. This step is entirely optional."
        )
        note.setObjectName("SectionHint")
        note.setWordWrap(True)
        v.addWidget(note)
        v.addStretch(1)
        return page

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _with_button(edit: QtWidgets.QLineEdit, button: QtWidgets.QPushButton) -> QtWidgets.QWidget:
        host = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(host)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit, 1)
        h.addWidget(button, 0)
        return host

    @staticmethod
    def _titled(title: str, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        host = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(host)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        lbl = QtWidgets.QLabel(title)
        lbl.setObjectName("SectionHint")
        v.addWidget(lbl)
        v.addWidget(widget, 1)
        return host

    def _log(self, msg: str) -> None:
        self.log.appendPlainText(str(msg))

    def is_busy(self) -> bool:
        return self._busy_count > 0

    def set_plot_preferences(self, theme: str, show_grid: bool) -> None:
        self._plot_theme = "Dark" if str(theme).lower().startswith("dark") else "Light"
        bg = "#0b0f14" if self._plot_theme == "Dark" else "#ffffff"
        for c in [getattr(self, n, None) for n in (
            "canvas_preproc", "canvas_match_hist", "canvas_match_atlas",
            "canvas_align_hist", "canvas_align_atlas", "canvas_trace", "trace_areas",
        )]:
            if c is not None:
                c.setBackground(bg)

    def _run_bg(self, fn, on_done, *args, busy_msg: str = "") -> None:
        if busy_msg:
            self._log(busy_msg)
        self._busy_count += 1
        worker = FunctionWorker(fn, *args)

        def _finished(payload: dict) -> None:
            self._busy_count = max(0, self._busy_count - 1)
            if payload.get("ok"):
                on_done(payload.get("result"))
            else:
                self._log("Task failed.")

        worker.signals.finished.connect(_finished)
        worker.signals.error.connect(lambda m: self._log(f"Error: {m}"))
        worker.signals.log.connect(self._log)
        self.pool.start(worker)

    def _ensure_atlas(self) -> Optional[hatlas.AllenCCFAtlas]:
        if self.atlas is not None:
            return self.atlas
        path = self.ed_atlas.text().strip() or None
        if not hatlas.atlas_files_present(path):
            self._log("Atlas files not found. Download from https://osf.io/fv7ed/overview "
                      "and set the atlas folder.")
            return None
        self._log("Loading Allen CCF atlas (first use)...")
        self.atlas = hatlas.AllenCCFAtlas(path)
        self._log(f"Atlas loaded: shape {self.atlas.shape}.")
        return self.atlas

    # ----------------------------------------------------- Setup actions
    def _pick_into(self, edit: QtWidgets.QLineEdit, title: str) -> None:
        start = edit.text().strip() or str(self.settings.value("paths/last_folder", str(Path.cwd())))
        d = QtWidgets.QFileDialog.getExistingDirectory(self, title, start)
        if d:
            edit.setText(d)

    def _pick_folder(self) -> None:
        self._pick_into(self.ed_folder, "Histology folder")
        if self.ed_folder.text().strip():
            self._load_session()

    def open_histology_folder(self, folder: str) -> None:
        self.ed_folder.setText(str(folder))
        self._load_session()

    def _load_session(self) -> None:
        text = self.ed_folder.text().strip()
        if not text:
            return
        self.folder = Path(text)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.settings.setValue("histology/last_folder", text)
        # Load any slice images already saved.
        self.slice_images = [slice_prep.load_image(p) for p in slice_prep.list_tiffs(self.folder)]
        # Existing products.
        hccf = self.folder / "histology_ccf.mat"
        if hccf.exists():
            try:
                self.histology_ccf = io_formats.load_histology_ccf(hccf)
            except Exception as exc:
                self._log(f"Could not load histology_ccf.mat: {exc}")
        tfn = self.folder / "atlas2histology_tform.mat"
        if tfn.exists():
            try:
                self.tforms = io_formats.load_tforms(tfn)
            except Exception as exc:
                self._log(f"Could not load tforms: {exc}")
        self.slice_specs = [None] * max(len(self.slice_images), len(self.histology_ccf))
        self._refresh_status()
        self._preproc_show()
        self._match_show()
        self._align_show()
        self._trace_show()
        self._log(f"Loaded session: {self.folder}")

    def _refresh_status(self) -> None:
        if self.folder is None:
            self.lbl_status.setText("No session loaded.")
            return
        def mark(name: str) -> str:
            ok = (self.folder / name).exists()
            color = "#2e7d32" if ok else "#9aa3af"
            return f"<span style='color:{color}'>{'YES' if ok else 'no '}</span> {name}"
        items = [
            f"Slice images: {len(self.slice_images)}",
            mark("histology_ccf.mat"),
            mark("atlas2histology_tform.mat"),
            mark("probe_ccf.mat"),
            mark("channel_locations_all_shanks.json"),
        ]
        self.lbl_status.setText("<br>".join(items))

    # ------------------------------------------------ Preprocess actions
    def _preproc_load(self) -> None:
        raw = self.ed_raw.text().strip()
        if not raw:
            self._log("Set a raw image folder first.")
            return
        factor = float(self.sp_downsample.value())
        paths = slice_prep.list_tiffs(raw)
        if not paths:
            self._log(f"No TIFFs found in {raw}.")
            return

        def job():
            imgs = []
            for p in paths:
                im = slice_prep.load_image(p)
                if factor != 1:
                    im = slice_prep.downsample(im, factor)
                imgs.append(im)
            return imgs

        def done(result):
            self.slice_images = result
            self._cur_preproc = 0
            self._log(f"Loaded {len(result)} raw image(s).")
            self._preproc_show()

        self._run_bg(job, done, busy_msg="Loading raw images...")

    def _preproc_save(self) -> None:
        if not self.slice_images or self.folder is None:
            self._log("Nothing to save (load images and set a folder).")
            return
        out = slice_prep.save_slices(self.slice_images, self.folder)
        self._log(f"Saved {len(out)} slice image(s) to {self.folder}.")
        self._refresh_status()

    def _preproc_step(self, d: int) -> None:
        if not self.slice_images:
            return
        self._cur_preproc = int(np.clip(getattr(self, "_cur_preproc", 0) + d, 0, len(self.slice_images) - 1))
        self._preproc_show()

    def _preproc_show(self) -> None:
        idx = getattr(self, "_cur_preproc", 0)
        if not self.slice_images:
            self.canvas_preproc.set_image(None)
            self.lbl_preproc.setText("0 / 0")
            return
        idx = min(idx, len(self.slice_images) - 1)
        self.canvas_preproc.set_image(self.slice_images[idx])
        self.lbl_preproc.setText(f"{idx + 1} / {len(self.slice_images)}")

    def _preproc_rotate(self, angle: float) -> None:
        idx = getattr(self, "_cur_preproc", 0)
        if not self.slice_images:
            return
        self.slice_images[idx] = slice_prep.rotate_center(self.slice_images[idx], angle)
        self._preproc_show()

    def _preproc_flip(self, horizontal: bool) -> None:
        idx = getattr(self, "_cur_preproc", 0)
        if not self.slice_images:
            return
        self.slice_images[idx] = slice_prep.flip(self.slice_images[idx], horizontal)
        self._preproc_show()

    # ------------------------------------------------------ Match actions
    def _match_step(self, d: int) -> None:
        if not self.slice_images:
            return
        self._cur_match_slice = int(np.clip(self._cur_match_slice + d, 0, len(self.slice_images) - 1))
        self._match_show()

    def _match_show(self) -> None:
        if not self.slice_images:
            self.canvas_match_hist.set_image(None)
            self.lbl_match.setText("slice 0 / 0")
            return
        idx = min(self._cur_match_slice, len(self.slice_images) - 1)
        self.canvas_match_hist.set_image(self.slice_images[idx])
        self.lbl_match.setText(f"slice {idx + 1} / {len(self.slice_images)}")
        self._match_update_atlas()

    def _match_update_atlas(self) -> None:
        at = self._ensure_atlas()
        if at is None:
            return
        cv = hatlas.coronal_camera_vector(self.sl_lr.value(), self.sl_si.value())
        sp = hatlas.coronal_slice_point(self.sl_ap.value(), at)
        sl = at.grab_atlas_slice(sp, cv, spacing=3)
        rgb = matching.render_atlas_slice(sl, at, self.cb_mode.currentText())
        self.canvas_match_atlas.set_image(rgb)

    def _match_assign(self) -> None:
        at = self._ensure_atlas()
        if at is None or not self.slice_images:
            return
        cv = hatlas.coronal_camera_vector(self.sl_lr.value(), self.sl_si.value())
        sp = hatlas.coronal_slice_point(self.sl_ap.value(), at)
        while len(self.slice_specs) < len(self.slice_images):
            self.slice_specs.append(None)
        self.slice_specs[self._cur_match_slice] = {"slice_point": sp, "camera_vector": cv}
        n_assigned = sum(s is not None for s in self.slice_specs)
        self._log(f"Assigned plane to slice {self._cur_match_slice + 1} "
                  f"({n_assigned}/{len(self.slice_images)} assigned).")

    def _match_save(self) -> None:
        at = self._ensure_atlas()
        if at is None or self.folder is None:
            return
        specs = [s for s in self.slice_specs if s is not None]
        if len(specs) != len(self.slice_images):
            self._log("Assign a plane to every slice before saving histology_ccf.")
            return

        def job():
            hccf = matching.build_histology_ccf(at, specs, spacing=1)
            io_formats.save_histology_ccf(self.folder / "histology_ccf.mat", hccf)
            io_formats.export_histology_ccf_csv(self.folder, hccf)
            return hccf

        def done(result):
            self.histology_ccf = result
            self._log(f"Saved histology_ccf.mat ({len(result)} slices) + CSV.")
            self._refresh_status()

        self._run_bg(job, done, busy_msg="Building full-resolution histology_ccf...")

    # ------------------------------------------------------ Align actions
    def _align_step(self, d: int) -> None:
        if not self.slice_images:
            return
        self._cur_align_slice = int(np.clip(self._cur_align_slice + d, 0, len(self.slice_images) - 1))
        self._align_show()

    def _align_show(self) -> None:
        if not self.slice_images:
            return
        idx = min(self._cur_align_slice, len(self.slice_images) - 1)
        self.canvas_align_hist.set_image(self.slice_images[idx])
        if idx < len(self.histology_ccf):
            self.canvas_align_atlas.set_image(self.histology_ccf[idx]["tv_slices"])
        self.lbl_align.setText(f"slice {idx + 1} / {len(self.slice_images)}")
        self._align_redraw_points()

    def _align_redraw_points(self) -> None:
        idx = self._cur_align_slice
        self.canvas_align_hist.clear_overlays()
        self.canvas_align_atlas.clear_overlays()
        hp = self._align_hist_pts.get(idx, [])
        ap = self._align_atlas_pts.get(idx, [])
        if hp:
            self.canvas_align_hist.add_scatter([p[0] for p in hp], [p[1] for p in hp], "w")
        if ap:
            self.canvas_align_atlas.add_scatter([p[0] for p in ap], [p[1] for p in ap], "r")

    def _align_click_hist(self, x: float, y: float) -> None:
        self._align_hist_pts.setdefault(self._cur_align_slice, []).append((x, y))
        self._align_redraw_points()

    def _align_click_atlas(self, x: float, y: float) -> None:
        self._align_atlas_pts.setdefault(self._cur_align_slice, []).append((x, y))
        self._align_redraw_points()

    def _align_clear(self) -> None:
        self._align_hist_pts[self._cur_align_slice] = []
        self._align_atlas_pts[self._cur_align_slice] = []
        self._align_redraw_points()

    def _align_apply_points(self) -> None:
        idx = self._cur_align_slice
        hp = self._align_hist_pts.get(idx, [])
        ap = self._align_atlas_pts.get(idx, [])
        if len(hp) < 3 or len(hp) != len(ap):
            self._log("Need >= 3 matching point pairs (histology and atlas).")
            return
        T = alignment.fit_affine_from_points(np.array(ap), np.array(hp))
        self._set_tform(idx, T)
        self._log(f"Applied control-point affine to slice {idx + 1}.")
        self._align_overlay(idx)

    def _align_auto(self) -> None:
        idx = self._cur_align_slice
        if idx >= len(self.histology_ccf):
            self._log("Match atlas slices first (need histology_ccf).")
            return
        hist = self.slice_images[idx]
        hist_gray = hist.mean(axis=2) if hist.ndim == 3 else hist
        atlas_tv = self.histology_ccf[idx]["tv_slices"]

        def job():
            return alignment.auto_align(hist_gray, atlas_tv)

        def done(T):
            self._set_tform(idx, T)
            self._log(f"Auto-aligned slice {idx + 1}.")
            self._align_overlay(idx)

        self._run_bg(job, done, busy_msg="Auto-aligning (intensity registration)...")

    def _set_tform(self, idx: int, T: np.ndarray) -> None:
        while len(self.tforms) <= idx:
            self.tforms.append(np.eye(3))
        self.tforms[idx] = T

    def _align_overlay(self, idx: int) -> None:
        if idx >= len(self.histology_ccf) or idx >= len(self.tforms):
            return
        av = self.histology_ccf[idx]["av_slices"]
        hist = self.slice_images[idx]
        shape = hist.shape[:2]
        warped = alignment.warp_atlas(av, self.tforms[idx], shape, nearest=True)
        bound = alignment.atlas_boundaries(warped)
        self.canvas_align_hist.set_image(self.slice_images[idx])
        self._align_redraw_points()
        self.canvas_align_hist.add_mask_overlay(bound, (60, 180, 255))

    def _align_save(self) -> None:
        if self.folder is None or not self.tforms:
            self._log("Nothing to save.")
            return
        while len(self.tforms) < len(self.slice_images):
            self.tforms.append(np.eye(3))
        io_formats.save_tforms(self.folder / "atlas2histology_tform.mat", self.tforms)
        self._log("Saved atlas2histology_tform.mat.")
        self._refresh_status()

    # ------------------------------------------------------ Trace actions
    def _trace_set_probe(self, v: int) -> None:
        self._active_probe = int(v)
        self._pending_click = []

    def _trace_step(self, d: int) -> None:
        if not self.slice_images:
            return
        self._cur_trace_slice = int(np.clip(self._cur_trace_slice + d, 0, len(self.slice_images) - 1))
        self._pending_click = []
        self._trace_show()

    def _trace_show(self) -> None:
        if not self.slice_images:
            return
        idx = min(self._cur_trace_slice, len(self.slice_images) - 1)
        self.canvas_trace.set_image(self.slice_images[idx])
        self.canvas_trace.clear_overlays()
        for (s, p), pts in self.probe_points.items():
            if s == idx and pts is not None and len(pts):
                color = PROBE_QCOLORS[(p - 1) % len(PROBE_QCOLORS)]
                self.canvas_trace.add_line(pts[:, 0], pts[:, 1], color, 3)
        self.lbl_trace.setText(f"slice {idx + 1} / {len(self.slice_images)}")

    def _trace_click(self, x: float, y: float) -> None:
        self._pending_click.append((x, y))
        if len(self._pending_click) == 2:
            idx = self._cur_trace_slice
            self.probe_points[(idx, self._active_probe)] = np.array(self._pending_click, dtype=float)
            self._pending_click = []
            self._trace_show()
            self._log(f"Set probe {self._active_probe} on slice {idx + 1}.")

    def _trace_clear(self) -> None:
        self.probe_points.pop((self._cur_trace_slice, self._active_probe), None)
        self._pending_click = []
        self._trace_show()

    def _trace_build(self) -> None:
        at = self._ensure_atlas()
        if at is None or self.folder is None:
            return
        if not self.histology_ccf or not self.tforms:
            self._log("Need histology_ccf and tforms first (Match + Align).")
            return
        if not self.probe_points:
            self._log("Draw at least one probe track first.")
            return
        n_probes = max(p for _, p in self.probe_points)
        # tracing uses 0-based probe indices
        pts0 = {(s, p - 1): v for (s, p), v in self.probe_points.items()}

        def job():
            probes = tracing.build_probe_ccf(pts0, self.histology_ccf, self.tforms, at, n_probes)
            io_formats.save_probe_ccf(self.folder / "probe_ccf.mat", probes)
            io_formats.export_probe_ccf_csv(self.folder, probes)
            return probes

        def done(probes):
            self._log(f"Saved probe_ccf.mat ({len(probes)} probes) + CSV.")
            self._refresh_status()
            self._draw_trajectory_areas(probes)

        self._run_bg(job, done, busy_msg="Sampling probe trajectories through the CCF...")

    def _draw_trajectory_areas(self, probes) -> None:
        self.trace_areas.clear()
        for i, p in enumerate(probes):
            ta = p.get("trajectory_areas")
            plt = self.trace_areas.addPlot(row=0, col=i)
            plt.setTitle(f"Probe {i + 1}")
            plt.invertY(True)
            plt.hideAxis("bottom")
            if ta is None or len(ta) == 0:
                continue
            for j in range(len(ta)):
                d0 = float(ta.iloc[j].get("depth_start_um", 0))
                d1 = float(ta.iloc[j].get("depth_end_um", 0))
                hexc = str(ta.iloc[j].get("color_hex_triplet", "808080"))
                try:
                    col = QtGui.QColor(int(hexc[0:2], 16), int(hexc[2:4], 16), int(hexc[4:6], 16))
                except Exception:
                    col = QtGui.QColor(128, 128, 128)
                bar = pg.BarGraphItem(x=[0], y0=[d0], y1=[d1], width=1, brush=col)
                plt.addItem(bar)
                txt = pg.TextItem(str(ta.iloc[j].get("acronym", "")), color="k", anchor=(0, 0.5))
                txt.setPos(0.05, (d0 + d1) / 2)
                plt.addItem(txt)

    # --------------------------------------------------- Channel map / IBL
    def _ibl_kwargs(self) -> dict:
        return {
            "ibl_python": self.ed_pyexe.text().strip() or None,
            "iblapps_path": self.ed_iblapps.text().strip() or None,
            "log": self._log,
        }

    def _gen_xyz(self) -> None:
        if self.folder is None or not (self.folder / "probe_ccf.mat").exists():
            self._log("Need probe_ccf.mat (Trace stage) first.")
            return
        kw = self._ibl_kwargs()

        def job():
            rc, out = ibl_launch.run_bridge(["xyz_picks", str(self.folder)], **kw)
            return rc

        self._run_bg(job, lambda rc: self._log("xyz_picks done." if rc == 0 else "xyz_picks failed."),
                     busy_msg="Generating xyz_picks via IBL bridge...")

    def _gen_channels(self, alignment_override: Optional[str] = None) -> None:
        if self.folder is None:
            return
        align = alignment_override or self.cb_alignment.currentText()
        kw = self._ibl_kwargs()

        def job():
            rc, out = ibl_launch.run_bridge(
                ["channels", str(self.folder), "--alignment", align], **kw)
            return rc

        self._run_bg(job, lambda rc: self._on_channels_done(rc),
                     busy_msg=f"Generating channel map ({align}) via IBL bridge...")

    def _gen_all(self) -> None:
        if self.folder is None:
            return
        args = ["all", str(self.folder), "--alignment", self.cb_alignment.currentText()]
        if self.ck_extract.isChecked():
            ks = self.ed_ks.text().strip()
            ephys = self.ed_ephys.text().strip()
            if ks and ephys:
                args += ["--ks", ks, "--ephys", ephys]
        kw = self._ibl_kwargs()
        self._run_bg(lambda: ibl_launch.run_bridge(args, **kw)[0],
                     lambda rc: self._on_channels_done(rc),
                     busy_msg="Running full channel-map pipeline via IBL bridge...")

    def _on_channels_done(self, rc: int) -> None:
        if rc != 0:
            self._log("Channel map generation failed.")
            return
        self._log("Channel map generated.")
        self._refresh_status()
        self._load_channel_table()

    def _load_channel_table(self) -> None:
        import json
        fn = self.folder / "channel_locations_all_shanks.json"
        if not fn.exists():
            return
        with open(fn) as f:
            data = json.load(f)
        rows = [(k, v) for k, v in data.items() if k != "origin"]
        self.tbl_channels.setRowCount(len(rows))
        for r, (k, v) in enumerate(rows):
            self.tbl_channels.setItem(r, 0, QtWidgets.QTableWidgetItem(str(k)))
            self.tbl_channels.setItem(r, 1, QtWidgets.QTableWidgetItem(str(v.get("axial", ""))))
            self.tbl_channels.setItem(r, 2, QtWidgets.QTableWidgetItem(str(v.get("lateral", ""))))
            self.tbl_channels.setItem(r, 3, QtWidgets.QTableWidgetItem(str(v.get("brain_region", ""))))

    def _launch_ibl(self) -> None:
        if self.folder is None:
            self._log("Load a session folder first.")
            return
        try:
            ibl_launch.launch_ibl_gui(self.folder, **self._ibl_kwargs())
            self._log("IBL GUI launched (separate window).")
        except Exception as exc:
            self._log(f"Could not launch IBL GUI: {exc}")

    # ----------------------------------------------------------- settings
    def _persist_settings(self) -> None:
        self.settings.setValue("histology/last_folder", self.ed_folder.text().strip())
        self.settings.setValue("histology/raw_path", self.ed_raw.text().strip())
        self.settings.setValue("histology/atlas_path", self.ed_atlas.text().strip())
        self.settings.setValue("histology/ks_path", self.ed_ks.text().strip())
        self.settings.setValue("histology/ephys_path", self.ed_ephys.text().strip())
        self.settings.setValue("histology/iblapps_path", self.ed_iblapps.text().strip())
        self.settings.setValue("histology/python_exe", self.ed_pyexe.text().strip())
        self._log("Saved histology paths.")

    def _restore_settings(self) -> None:
        self.ed_folder.setText(str(self.settings.value("histology/last_folder", "")))
        self.ed_raw.setText(str(self.settings.value("histology/raw_path", "")))
        self.ed_atlas.setText(str(self.settings.value("histology/atlas_path", hatlas.DEFAULT_ATLAS_PATH)))
        self.ed_ks.setText(str(self.settings.value("histology/ks_path", "")))
        self.ed_ephys.setText(str(self.settings.value("histology/ephys_path", "")))
        self.ed_iblapps.setText(str(self.settings.value("histology/iblapps_path", ibl_launch.DEFAULT_IBLAPPS_PATH)))
        self.ed_pyexe.setText(str(self.settings.value("histology/python_exe", "")))

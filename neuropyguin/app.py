from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets


_KNOWN_QT_WARNING_SNIPPETS = (
    "unique connections require a pointer to member function of a QObject subclass",
)
_PREVIOUS_QT_MESSAGE_HANDLER = None
_QT_MODE_LABELS = {
    QtCore.QtMsgType.QtDebugMsg: "qt.debug",
    QtCore.QtMsgType.QtInfoMsg: "qt.info",
    QtCore.QtMsgType.QtWarningMsg: "qt.warning",
    QtCore.QtMsgType.QtCriticalMsg: "qt.critical",
    QtCore.QtMsgType.QtFatalMsg: "qt.fatal",
}


def _qt_message_filter(mode, context, message) -> None:
    text = str(message)
    if any(snippet in text for snippet in _KNOWN_QT_WARNING_SNIPPETS):
        return
    if _PREVIOUS_QT_MESSAGE_HANDLER is not None:
        _PREVIOUS_QT_MESSAGE_HANDLER(mode, context, message)
        return
    label = _QT_MODE_LABELS.get(mode, "qt")
    stream = sys.stderr if mode != QtCore.QtMsgType.QtDebugMsg else sys.stdout
    print(f"{label}: {text}", file=stream)


def _install_qt_message_filter() -> None:
    global _PREVIOUS_QT_MESSAGE_HANDLER
    if _PREVIOUS_QT_MESSAGE_HANDLER is None:
        _PREVIOUS_QT_MESSAGE_HANDLER = QtCore.qInstallMessageHandler(_qt_message_filter)


_install_qt_message_filter()
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
import pyqtgraph as pg

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from neuropyguin.side_nav import SideNavStack
    from neuropyguin.tabs.curation_tab import CurationTab
    from neuropyguin.tabs.postprocessing_tab import PostProcessingTab
    from neuropyguin.tabs.preprocessing_tab import PreprocessingTab
    from neuropyguin.tabs.histology_tab import HistologyTab
    from neuropyguin.styles import build_app_palette, build_app_qss
else:
    from .side_nav import SideNavStack
    from .tabs.curation_tab import CurationTab
    from .tabs.postprocessing_tab import PostProcessingTab
    from .tabs.preprocessing_tab import PreprocessingTab
    from .tabs.histology_tab import HistologyTab
    from .styles import build_app_palette, build_app_qss


TAB_TITLES = ["Preprocessing", "Curation", "Post Processing", "Histology"]
STARTUP_TAB_OPTIONS = ["Last Used", *TAB_TITLES]
PLOT_THEME_OPTIONS = ["Light", "Dark"]
ASSET_DIR = Path(__file__).resolve().parent / "assets"
SMALL_APP_ICON_PATH = ASSET_DIR / "small.jpg"
BIG_SPLASH_IMAGE_PATH = ASSET_DIR / "big.jpg"
WINDOWS_APP_ID = "BelloneLab.NeuroPyGuiN"


def _load_app_icon() -> QtGui.QIcon:
    pixmap = QtGui.QPixmap(str(SMALL_APP_ICON_PATH))
    if pixmap.isNull():
        return QtGui.QIcon()
    return QtGui.QIcon(pixmap)


def _load_splash_pixmap() -> QtGui.QPixmap:
    pixmap = QtGui.QPixmap(str(BIG_SPLASH_IMAGE_PATH))
    if pixmap.isNull():
        return QtGui.QPixmap()
    max_width = min(960, pixmap.width())
    if max_width > 0 and pixmap.width() > max_width:
        return pixmap.scaledToWidth(max_width, QtCore.Qt.SmoothTransformation)
    return pixmap


def _set_windows_taskbar_app_id() -> None:
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(WINDOWS_APP_ID)
    except Exception:
        pass


class PreferencesDialog(QtWidgets.QDialog):
    def __init__(
        self,
        theme: str,
        show_grid: bool,
        startup_tab: str,
        remember_geometry: bool,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setProperty("compactDialog", True)
        self.setWindowTitle("Settings")
        self.resize(720, 420)

        main = QtWidgets.QVBoxLayout(self)
        self.cb_theme = QtWidgets.QComboBox()
        self.cb_theme.addItems(["Light", "Dark"])
        idx = self.cb_theme.findText(str(theme))
        self.cb_theme.setCurrentIndex(idx if idx >= 0 else 0)
        self.cb_theme.setMaximumWidth(140)

        self.ck_grid = QtWidgets.QCheckBox("Show plot grid by default")
        self.ck_grid.setChecked(bool(show_grid))

        self.cb_startup_tab = QtWidgets.QComboBox()
        self.cb_startup_tab.addItems(STARTUP_TAB_OPTIONS)
        startup_value = str(startup_tab) if str(startup_tab) in STARTUP_TAB_OPTIONS else STARTUP_TAB_OPTIONS[0]
        self.cb_startup_tab.setCurrentText(startup_value)
        self.cb_startup_tab.setMaximumWidth(180)

        self.ck_remember_geometry = QtWidgets.QCheckBox("Remember window size and position")
        self.ck_remember_geometry.setChecked(bool(remember_geometry))

        note = QtWidgets.QLabel(
            "These settings apply globally. Folder history and recents can be cleared from File."
        )
        note.setObjectName("SectionHint")
        note.setWordWrap(True)
        main.addWidget(note)

        sections = SideNavStack(
            "Settings",
            "Switch sections from the left instead of scrolling through one long page.",
        )
        main.addWidget(sections, 1)

        def _new_page() -> tuple[QtWidgets.QWidget, QtWidgets.QVBoxLayout]:
            page = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(page)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(12)
            return page, layout

        appearance_page, appearance_layout = _new_page()
        appearance_box = QtWidgets.QGroupBox("Appearance")
        appearance_box.setProperty("settingsSection", True)
        appearance_form = QtWidgets.QFormLayout(appearance_box)
        appearance_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldsStayAtSizeHint)
        appearance_form.setFormAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        appearance_form.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        appearance_form.setHorizontalSpacing(14)
        appearance_form.setVerticalSpacing(10)
        appearance_form.addRow("Theme", self.cb_theme)
        appearance_form.addRow("Plot grid", self.ck_grid)
        appearance_layout.addWidget(appearance_box)
        appearance_layout.addStretch(1)
        sections.add_page("Appearance", appearance_page)

        startup_page, startup_layout = _new_page()
        startup_box = QtWidgets.QGroupBox("Startup and layout")
        startup_box.setProperty("settingsSection", True)
        startup_form = QtWidgets.QFormLayout(startup_box)
        startup_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldsStayAtSizeHint)
        startup_form.setFormAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        startup_form.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        startup_form.setHorizontalSpacing(14)
        startup_form.setVerticalSpacing(10)
        startup_form.addRow("Startup page", self.cb_startup_tab)
        startup_form.addRow("Window layout", self.ck_remember_geometry)
        startup_layout.addWidget(startup_box)
        startup_layout.addStretch(1)
        sections.add_page("Startup", startup_page)

        sections.setCurrentIndex(0)
        self.sections = sections

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        footer = QtWidgets.QHBoxLayout()
        btn_defaults = QtWidgets.QPushButton("Restore Defaults")
        btn_defaults.clicked.connect(self._restore_defaults)
        footer.addWidget(btn_defaults)
        footer.addStretch(1)
        footer.addWidget(btns)
        main.addLayout(footer)

    def _restore_defaults(self) -> None:
        self.cb_theme.setCurrentText("Light")
        self.ck_grid.setChecked(True)
        self.cb_startup_tab.setCurrentText("Last Used")
        self.ck_remember_geometry.setChecked(True)

    def values(self) -> dict[str, object]:
        return {
            "theme": self.cb_theme.currentText(),
            "show_grid": bool(self.ck_grid.isChecked()),
            "startup_tab": self.cb_startup_tab.currentText(),
            "remember_geometry": bool(self.ck_remember_geometry.isChecked()),
        }


class NeuroPyGuiNMainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NeuroPyGuiN")
        app_icon = _load_app_icon()
        if not app_icon.isNull():
            self.setWindowIcon(app_icon)
        self.setMinimumSize(1180, 760)
        self._set_startup_geometry()

        thread_pool = QtCore.QThreadPool.globalInstance()
        self.settings = QtCore.QSettings("NeuroPyGuiN", "NeuroPyGuiN")
        self._plot_theme = "Light"
        self._plot_grid = True
        self._apply_application_theme(str(self.settings.value("plot/theme", "Light")))

        tabs = QtWidgets.QTabWidget()
        tabs.setDocumentMode(True)
        self.pre_tab = PreprocessingTab(thread_pool)
        self.cur_tab = CurationTab(thread_pool)
        self.post_tab = PostProcessingTab(thread_pool)
        self.hist_tab = HistologyTab(thread_pool)
        for t in [self.pre_tab, self.cur_tab, self.post_tab, self.hist_tab]:
            t.setMinimumSize(0, 0)
            t.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        tabs.setMinimumSize(0, 0)
        tabs.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        for title, widget in zip(TAB_TITLES, [self.pre_tab, self.cur_tab, self.post_tab, self.hist_tab]):
            tabs.addTab(widget, title)
        self.tabs = tabs

        self._build_actions()
        self._build_menu_bar()

        self.pre_tab.openCurationRequested.connect(self._open_curation)
        self.pre_tab.openPostProcessingRequested.connect(self._open_postprocessing)
        self.pre_tab.openHistologyRequested.connect(self._open_histology)
        self.pre_tab.saveSettingsFileRequested.connect(self._export_settings_file)
        self.pre_tab.loadSettingsFileRequested.connect(self._load_settings_file)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self.setCentralWidget(tabs)
        status = self.statusBar()
        status.setSizeGripEnabled(False)
        self.bottom_busy = QtWidgets.QProgressBar()
        self.bottom_busy.setTextVisible(False)
        self.bottom_busy.setFixedHeight(5)
        self.bottom_busy.setMaximumWidth(100000)
        self.bottom_busy.setRange(0, 100)
        self.bottom_busy.setValue(0)
        self.bottom_busy.hide()
        status.addPermanentWidget(self.bottom_busy, 1)
        self.busy_timer = QtCore.QTimer(self)
        self.busy_timer.setInterval(120)
        self.busy_timer.timeout.connect(self._refresh_bottom_busy)
        self.busy_timer.start()

        self._restore_plot_preferences()
        self._restore_initial_tab()
        self._restore_window_state()
        self._apply_plot_preferences()
        self._update_action_states()

    def _build_actions(self) -> None:
        self.act_add_ap_files = QtGui.QAction("Add AP Files to Queue...", self)
        self.act_add_ap_files.setShortcut(QtGui.QKeySequence.Open)
        self.act_add_ap_files.triggered.connect(self._open_ap_files)

        self.act_add_folder = QtGui.QAction("Add Folder to Queue...", self)
        self.act_add_folder.setShortcut("Ctrl+Shift+O")
        self.act_add_folder.triggered.connect(self._open_preprocessing_folder)

        self.act_set_curation_folder = QtGui.QAction("Set Curation Folder...", self)
        self.act_set_curation_folder.setShortcut("Ctrl+Shift+C")
        self.act_set_curation_folder.triggered.connect(self._pick_curation_folder)

        self.act_open_post_folder = QtGui.QAction("Open Post Processing Folder...", self)
        self.act_open_post_folder.setShortcut("Ctrl+Shift+P")
        self.act_open_post_folder.triggered.connect(self._pick_postprocessing_folder)

        self.act_export_current = QtGui.QAction("Export Current Plot Data...", self)
        self.act_export_current.setShortcut("Ctrl+E")
        self.act_export_current.triggered.connect(self._export_current_plotted_data)

        self.act_save_bombcell = QtGui.QAction("Save Bombcell Labels...", self)
        self.act_save_bombcell.triggered.connect(self._save_bombcell_labels)

        self.act_export_units = QtGui.QAction("Export Units to H5...", self)
        self.act_export_units.triggered.connect(self._export_units_h5)

        self.act_save_settings = QtGui.QAction("Save Settings", self)
        self.act_save_settings.setShortcut(QtGui.QKeySequence.Save)
        self.act_save_settings.triggered.connect(self._save_settings)

        self.act_export_settings_file = QtGui.QAction("Save Settings to File...", self)
        self.act_export_settings_file.triggered.connect(self._export_settings_file)

        self.act_load_settings_file = QtGui.QAction("Load Settings from File...", self)
        self.act_load_settings_file.triggered.connect(self._load_settings_file)

        self.act_settings = QtGui.QAction("Settings...", self)
        self.act_settings.setShortcut("Ctrl+,")
        self.act_settings.triggered.connect(self._open_settings_dialog)

        self.act_clear_history = QtGui.QAction("Clear Folder History and Recents...", self)
        self.act_clear_history.triggered.connect(self._clear_folder_history)

        self.act_exit = QtGui.QAction("Exit", self)
        self.act_exit.setShortcut(QtGui.QKeySequence.Quit)
        self.act_exit.triggered.connect(self.close)

        self.act_tab_pre = QtGui.QAction("Preprocessing", self)
        self.act_tab_pre.setShortcut("Ctrl+1")
        self.act_tab_pre.triggered.connect(lambda: self.tabs.setCurrentWidget(self.pre_tab))

        self.act_tab_cur = QtGui.QAction("Curation", self)
        self.act_tab_cur.setShortcut("Ctrl+2")
        self.act_tab_cur.triggered.connect(lambda: self.tabs.setCurrentWidget(self.cur_tab))

        self.act_tab_post = QtGui.QAction("Post Processing", self)
        self.act_tab_post.setShortcut("Ctrl+3")
        self.act_tab_post.triggered.connect(lambda: self.tabs.setCurrentWidget(self.post_tab))

        self.act_tab_hist = QtGui.QAction("Histology", self)
        self.act_tab_hist.setShortcut("Ctrl+4")
        self.act_tab_hist.triggered.connect(lambda: self.tabs.setCurrentWidget(self.hist_tab))

        self.theme_group = QtGui.QActionGroup(self)
        self.theme_group.setExclusive(True)

        self.act_theme_light = QtGui.QAction("Light", self)
        self.act_theme_light.setCheckable(True)
        self.act_theme_light.triggered.connect(lambda checked=False: self._set_plot_preferences(theme="Light"))
        self.theme_group.addAction(self.act_theme_light)

        self.act_theme_dark = QtGui.QAction("Dark", self)
        self.act_theme_dark.setCheckable(True)
        self.act_theme_dark.triggered.connect(lambda checked=False: self._set_plot_preferences(theme="Dark"))
        self.theme_group.addAction(self.act_theme_dark)

        self.act_show_grid = QtGui.QAction("Show Grid", self)
        self.act_show_grid.setCheckable(True)
        self.act_show_grid.triggered.connect(self._set_plot_preferences)

        self.act_help_settings = QtGui.QAction("Settings Help", self)
        self.act_help_settings.setShortcut(QtGui.QKeySequence.HelpContents)
        self.act_help_settings.triggered.connect(self._show_settings_help)

        self.act_about = QtGui.QAction("About NeuroPyGuiN", self)
        self.act_about.triggered.connect(self._show_about)

    def _build_menu_bar(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.act_add_ap_files)
        file_menu.addAction(self.act_add_folder)
        file_menu.addSeparator()
        file_menu.addAction(self.act_set_curation_folder)
        file_menu.addAction(self.act_open_post_folder)
        file_menu.addSeparator()
        file_menu.addAction(self.act_export_current)
        file_menu.addAction(self.act_save_bombcell)
        file_menu.addAction(self.act_export_units)
        file_menu.addAction(self.act_save_settings)
        file_menu.addAction(self.act_export_settings_file)
        file_menu.addAction(self.act_load_settings_file)
        file_menu.addSeparator()
        file_menu.addAction(self.act_settings)
        file_menu.addAction(self.act_clear_history)
        file_menu.addSeparator()
        file_menu.addAction(self.act_exit)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.act_tab_pre)
        view_menu.addAction(self.act_tab_cur)
        view_menu.addAction(self.act_tab_post)
        view_menu.addAction(self.act_tab_hist)
        view_menu.addSeparator()
        theme_menu = view_menu.addMenu("Theme")
        theme_menu.addAction(self.act_theme_light)
        theme_menu.addAction(self.act_theme_dark)
        view_menu.addAction(self.act_show_grid)

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(self.act_help_settings)
        help_menu.addAction(self.act_about)

    def _set_startup_geometry(self) -> None:
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is None:
            self.resize(1680, 1020)
            return
        avail = screen.availableGeometry()
        w = max(1180, min(avail.width(), int(avail.width() * 0.99)))
        h = max(760, min(avail.height(), int(avail.height() * 0.97)))
        self.resize(w, h)

    def _restore_initial_tab(self) -> None:
        pref = str(self.settings.value("ui/startup_tab", "Last Used"))
        if pref == "Last Used":
            idx = int(self.settings.value("ui/last_tab_index", 0))
        else:
            idx = TAB_TITLES.index(pref) if pref in TAB_TITLES else 0
        idx = max(0, min(len(TAB_TITLES) - 1, idx))
        self.tabs.setCurrentIndex(idx)

    def _restore_window_state(self) -> None:
        remember = bool(self.settings.value("ui/remember_geometry", True, type=bool))
        if not remember:
            return
        geom = self.settings.value("ui/geometry")
        if isinstance(geom, QtCore.QByteArray) and not geom.isEmpty():
            self.restoreGeometry(geom)
        state = self.settings.value("ui/window_state")
        if isinstance(state, QtCore.QByteArray) and not state.isEmpty():
            self.restoreState(state)

    def _persist_window_state(self) -> None:
        remember = bool(self.settings.value("ui/remember_geometry", True, type=bool))
        if remember:
            self.settings.setValue("ui/geometry", self.saveGeometry())
            self.settings.setValue("ui/window_state", self.saveState())
        else:
            self.settings.remove("ui/geometry")
            self.settings.remove("ui/window_state")

    def _open_curation(self, folders) -> None:
        if isinstance(folders, str):
            targets = [folders]
        else:
            targets = [str(folder).strip() for folder in list(folders or []) if str(folder).strip()]
        if not targets:
            return
        if len(targets) == 1:
            self.cur_tab.open_ks_folder(targets[0])
        else:
            self.cur_tab.set_ks_folders(targets)
            if hasattr(self.cur_tab, "show_phy_page"):
                self.cur_tab.show_phy_page()
        self.tabs.setCurrentWidget(self.cur_tab)

    def _open_postprocessing(self, folder: str) -> None:
        self.post_tab.open_ks_folder(folder)
        self.tabs.setCurrentWidget(self.post_tab)

    def _open_histology(self, ks_folder: str) -> None:
        self.tabs.setCurrentWidget(self.hist_tab)
        self.hist_tab.setup_from_ks_folder(ks_folder)

    def _restore_plot_preferences(self) -> None:
        theme = str(self.settings.value("plot/theme", "Light"))
        grid = bool(self.settings.value("plot/grid", True, type=bool))
        self._set_plot_preferences(theme=theme, grid=grid, apply=False)

    def _apply_application_theme(self, theme: str) -> None:
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        app.setPalette(build_app_palette(theme))
        app.setStyleSheet(build_app_qss(theme))

    def _set_plot_preferences(
        self,
        checked: bool | None = None,
        *,
        theme: str | None = None,
        grid: bool | None = None,
        apply: bool = True,
    ) -> None:
        if theme is not None:
            theme_value = str(theme)
            self._plot_theme = theme_value if theme_value in PLOT_THEME_OPTIONS else PLOT_THEME_OPTIONS[0]
        if grid is not None:
            self._plot_grid = bool(grid)
        elif checked is not None:
            self._plot_grid = bool(checked)
        if apply:
            self._apply_plot_preferences()
        else:
            self._sync_menu_controls()

    def _sync_menu_controls(self) -> None:
        theme = self._plot_theme
        for action, checked in [
            (self.act_theme_light, theme == "Light"),
            (self.act_theme_dark, theme == "Dark"),
            (self.act_show_grid, self._plot_grid),
        ]:
            action.blockSignals(True)
            action.setChecked(checked)
            action.blockSignals(False)

    def _apply_plot_preferences(self) -> None:
        theme = self._plot_theme
        grid = self._plot_grid
        self.settings.setValue("plot/theme", theme)
        self.settings.setValue("plot/grid", grid)
        self._apply_application_theme(theme)
        dark = theme.lower().startswith("dark")
        pg.setConfigOption("background", "#0b0f14" if dark else "w")
        pg.setConfigOption("foreground", "#e8eef7" if dark else "k")
        for tab in [self.cur_tab, self.post_tab, self.hist_tab]:
            if hasattr(tab, "set_plot_preferences"):
                tab.set_plot_preferences(theme, grid)
        self._sync_menu_controls()

    def _choose_folder(self, title: str) -> str:
        start = str(self.settings.value("paths/last_folder", str(Path.cwd())))
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, title, start)
        if folder:
            self.settings.setValue("paths/last_folder", folder)
        return folder

    def _open_ap_files(self) -> None:
        self.tabs.setCurrentWidget(self.pre_tab)
        self.pre_tab._add_files()

    def _open_preprocessing_folder(self) -> None:
        self.tabs.setCurrentWidget(self.pre_tab)
        self.pre_tab._add_folder()

    def _pick_curation_folder(self) -> None:
        folder = self._choose_folder("Select curated Kilosort folder")
        if not folder:
            return
        self.settings.setValue("curation/phy_folder", folder)
        self.settings.setValue("curation/bomb_folder", folder)
        self.cur_tab.set_ks_folders([folder])
        self.tabs.setCurrentWidget(self.cur_tab)

    def _pick_postprocessing_folder(self) -> None:
        folder = self._choose_folder("Select Kilosort folder for post processing")
        if not folder:
            return
        self.post_tab.open_ks_folder(folder)
        self.tabs.setCurrentWidget(self.post_tab)

    def _export_current_plotted_data(self) -> None:
        current = self.tabs.currentWidget()
        exporter = getattr(current, "_export_plotted_data", None)
        if callable(exporter):
            exporter()

    def _save_bombcell_labels(self) -> None:
        self.tabs.setCurrentWidget(self.cur_tab)
        self.cur_tab._save_labels()

    def _export_units_h5(self) -> None:
        self.tabs.setCurrentWidget(self.post_tab)
        self.post_tab._export_units_file()

    def _open_settings_dialog(self) -> None:
        dlg = PreferencesDialog(
            theme=self._plot_theme,
            show_grid=self._plot_grid,
            startup_tab=str(self.settings.value("ui/startup_tab", "Last Used")),
            remember_geometry=bool(self.settings.value("ui/remember_geometry", True, type=bool)),
            parent=self,
        )
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        values = dlg.values()
        self.settings.setValue("ui/startup_tab", str(values["startup_tab"]))
        self.settings.setValue("ui/remember_geometry", bool(values["remember_geometry"]))
        self._set_plot_preferences(theme=str(values["theme"]), grid=bool(values["show_grid"]))
        self._persist_window_state()

    def _save_settings(self) -> None:
        self._persist_all_settings()
        self.statusBar().showMessage("Settings saved.", 3000)

    def _persist_all_settings(self) -> None:
        persist_preproc = getattr(self.pre_tab, "_persist_settings", None)
        if callable(persist_preproc):
            persist_preproc()

        if hasattr(self.cur_tab, "ed_phy_folder"):
            self.settings.setValue("curation/phy_folder", self.cur_tab.ed_phy_folder.text().strip())
        if hasattr(self.cur_tab, "ed_bomb_folder"):
            self.settings.setValue("curation/bomb_folder", self.cur_tab.ed_bomb_folder.text().strip())
        persist_cur = getattr(self.cur_tab, "_persist_splitter_sizes", None)
        if callable(persist_cur):
            persist_cur()

        if hasattr(self.post_tab, "ed_folder"):
            self.settings.setValue("post/last_folder", self.post_tab.ed_folder.text().strip())

        self.settings.setValue("plot/theme", self._plot_theme)
        self.settings.setValue("plot/grid", self._plot_grid)
        self._persist_window_state()
        self.settings.sync()

    @staticmethod
    def _copy_qsettings(src: QtCore.QSettings, dst: QtCore.QSettings) -> None:
        dst.clear()
        for key in src.allKeys():
            dst.setValue(key, src.value(key))
        dst.sync()

    def _reload_settings_from_store(self) -> None:
        restore_preproc = getattr(self.pre_tab, "_restore_settings", None)
        if callable(restore_preproc):
            restore_preproc()
        restore_cur = getattr(self.cur_tab, "_restore_settings", None)
        if callable(restore_cur):
            restore_cur()
        restore_post = getattr(self.post_tab, "_restore_settings", None)
        if callable(restore_post):
            restore_post()
        self._restore_plot_preferences()
        self._apply_plot_preferences()
        self._restore_window_state()
        self._update_action_states()

    def _export_settings_file(self) -> None:
        self._persist_all_settings()
        start = str(self.settings.value("paths/last_folder", str(Path.cwd())))
        fp, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Settings to File",
            str(Path(start) / "NeuroPyGuiN_settings.ini"),
            "INI files (*.ini)",
        )
        if not fp:
            return
        out_path = Path(fp)
        if out_path.suffix.lower() != ".ini":
            out_path = out_path.with_suffix(".ini")
        file_settings = QtCore.QSettings(str(out_path), QtCore.QSettings.IniFormat)
        self._copy_qsettings(self.settings, file_settings)
        self.settings.setValue("paths/last_folder", str(out_path.parent))
        self.settings.sync()
        self.pre_tab._append_log(f"Settings file saved: {out_path}")
        self.statusBar().showMessage(f"Settings exported to {out_path}", 4000)

    def _load_settings_file(self) -> None:
        start = str(self.settings.value("paths/last_folder", str(Path.cwd())))
        fp, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Settings from File",
            start,
            "INI files (*.ini)",
        )
        if not fp:
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            "Load Settings from File",
            "Replace current app settings with the selected settings file?",
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        in_path = Path(fp)
        file_settings = QtCore.QSettings(str(in_path), QtCore.QSettings.IniFormat)
        self._copy_qsettings(file_settings, self.settings)
        self.settings.setValue("paths/last_folder", str(in_path.parent))
        self.settings.sync()
        self._reload_settings_from_store()
        self.pre_tab._append_log(f"Settings file loaded: {in_path}")
        self.statusBar().showMessage(f"Settings loaded from {in_path}", 4000)

    def _clear_folder_history(self) -> None:
        answer = QtWidgets.QMessageBox.question(
            self,
            "Clear Folder History",
            "Remove saved folder history, recent-file lists, and completed-run history for this app?",
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        for key in [
            "paths/last_folder",
            "paths/last_file_dir",
            "recent_files",
            "recent_folders",
            "preproc/completed_runs_history_json",
            "curation/phy_folder",
            "curation/bomb_folder",
            "quality/last_folder",
            "post/last_folder",
        ]:
            self.settings.remove(key)
        clear_completed_history = getattr(self.pre_tab, "clear_completed_history", None)
        if callable(clear_completed_history):
            clear_completed_history()
        self.cur_tab.ed_phy_folder.clear()
        self.cur_tab.ed_bomb_folder.clear()
        self.post_tab.ed_folder.clear()
        QtWidgets.QMessageBox.information(
            self,
            "History Cleared",
            "Saved folders, recents, and completed-run history were cleared.",
        )

    def _on_tab_changed(self, index: int) -> None:
        self.settings.setValue("ui/last_tab_index", int(index))
        self._update_action_states()

    def _update_action_states(self) -> None:
        current = self.tabs.currentWidget()
        self.act_export_current.setEnabled(callable(getattr(current, "_export_plotted_data", None)))
        self.act_save_bombcell.setEnabled(current is self.cur_tab)
        self.act_export_units.setEnabled(current is self.post_tab)

    def _refresh_bottom_busy(self) -> None:
        tabs = [self.pre_tab, self.cur_tab, self.post_tab, self.hist_tab]
        busy = any(bool(getattr(t, "is_busy", lambda: False)()) for t in tabs)
        if busy:
            if self.bottom_busy.maximum() != 0:
                self.bottom_busy.setRange(0, 0)
            self.bottom_busy.show()
        else:
            if self.bottom_busy.maximum() == 0:
                self.bottom_busy.setRange(0, 100)
            self.bottom_busy.setValue(0)
            self.bottom_busy.hide()

    def _show_settings_help(self) -> None:
        txt = (
            "When to adjust default settings\n\n"
            "n_chan_bin: total channels in the binary file (including non-ephys channels).\n"
            "batch_size: samples per batch; increase on low-channel probes for better drift estimation.\n"
            "nblocks: drift correction blocks (0=off, 1=rigid, >1=non-rigid).\n"
            "Th_universal / Th_learned: spike detection thresholds.\n"
            "tmin / tmax: sorting start/end times in seconds.\n"
            "nt: waveform length / filter padding in samples.\n"
            "dmin / dminx: vertical/lateral template spacing.\n"
            "min_template_size: minimum spatial template size.\n"
            "nearest_chans / nearest_templates: neighborhood size for template assignment.\n"
            "x_centers: number of x-position template groups.\n"
            "duplicate_spike_ms: same-unit refractory artifact removal window.\n\n"
            "Use ? buttons next to fields for contextual per-setting help."
        )
        QtWidgets.QMessageBox.information(self, "NeuroPyGuiN Settings Help", txt)

    def _show_about(self) -> None:
        txt = (
            "NeuroPyGuiN\n\n"
            "Integrated preprocessing, curation, quality-metrics, and post-processing GUI for Neuropixels workflows.\n\n"
            f"Python {sys.version.split()[0]}\n"
            f"Qt {QtCore.qVersion()}"
        )
        QtWidgets.QMessageBox.about(self, "About NeuroPyGuiN", txt)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._persist_all_settings()
        super().closeEvent(event)


def main() -> int:
    _set_windows_taskbar_app_id()
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("NeuroPyGuiN")
    app.setApplicationDisplayName("NeuroPyGuiN")
    app_icon = _load_app_icon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)
    fusion = QtWidgets.QStyleFactory.create("Fusion")
    if fusion is not None:
        app.setStyle(fusion)
    app.setFont(QtGui.QFont("Segoe UI", 10))
    splash = None
    splash_pixmap = _load_splash_pixmap()
    if not splash_pixmap.isNull():
        splash = QtWidgets.QSplashScreen(
            splash_pixmap,
            QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.FramelessWindowHint,
        )
        splash.show()
        splash.showMessage(
            "Loading NeuroPyGuiN...",
            QtCore.Qt.AlignBottom | QtCore.Qt.AlignHCenter,
            QtGui.QColor("#f4f7ff"),
        )
        app.processEvents()
    win = NeuroPyGuiNMainWindow()
    win.showMaximized()
    if splash is not None:
        splash.finish(win)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

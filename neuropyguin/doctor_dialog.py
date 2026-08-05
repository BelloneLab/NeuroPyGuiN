"""Qt front-end for the environment self-check in :mod:`neuropyguin.doctor`.

:class:`DoctorDialog` runs the checks in a worker thread (importing torch and
probing CUDA takes a few seconds, which must not freeze the GUI), groups the
results by category in a tree, and shows the fix command for whichever row is
selected. The main window opens it automatically on the very first launch and
from Help > Run Diagnostics afterwards.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

from PySide6 import QtCore, QtGui, QtWidgets

from . import doctor
from .doctor import CheckResult


# Status colors that stay legible on both the light and the dark palette.
_STATUS_COLORS = {
    "Light": {doctor.OK: "#15803d", doctor.WARN: "#b45309", doctor.FAIL: "#b91c1c"},
    "Dark": {doctor.OK: "#4ade80", doctor.WARN: "#fbbf24", doctor.FAIL: "#f87171"},
}
_STATUS_WORDS = {doctor.OK: "Ready", doctor.WARN: "Check", doctor.FAIL: "Missing"}
_STATUS_DOTS = {doctor.OK: "●", doctor.WARN: "▲", doctor.FAIL: "✕"}


class _DiagnosticsSignals(QtCore.QObject):
    progress = QtCore.Signal(int, int, str)
    finished = QtCore.Signal(object)


class DiagnosticsWorker(QtCore.QRunnable):
    """Run :func:`doctor.run_diagnostics` off the GUI thread."""

    def __init__(self, settings_get: Optional[Callable[[str, object], object]]) -> None:
        super().__init__()
        self._settings_get = settings_get
        self.signals = _DiagnosticsSignals()

    @QtCore.Slot()
    def run(self) -> None:
        try:
            results = doctor.run_diagnostics(
                self._settings_get,
                progress=lambda done, total, label: self.signals.progress.emit(done, total, label),
            )
        except Exception as exc:  # noqa: BLE001 - always give the dialog something to show
            results = [
                CheckResult(
                    "doctor_error", "Application core", "Diagnostics", doctor.FAIL,
                    f"The self-check itself failed: {exc}", doctor.REQUIRED,
                )
            ]
        self.signals.finished.emit(results)


class DoctorDialog(QtWidgets.QDialog):
    """Modal report of everything the app needs, with per-row fix commands."""

    installToolsRequested = QtCore.Signal(list)

    def __init__(
        self,
        *,
        settings: QtCore.QSettings,
        pool: QtCore.QThreadPool | None = None,
        theme: str = "Light",
        first_run: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setProperty("compactDialog", True)
        self.setWindowTitle("NeuroPyGuiN Diagnostics")
        self.resize(1020, 700)
        self._settings = settings
        self._pool = pool or QtCore.QThreadPool.globalInstance()
        self._colors = _STATUS_COLORS.get("Dark" if str(theme).lower().startswith("dark") else "Light")
        self._results: List[CheckResult] = []
        self._running = False

        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(12)

        self.lbl_headline = QtWidgets.QLabel("Checking your installation...")
        self.lbl_headline.setObjectName("StepStatusRunName")
        self.lbl_headline.setWordWrap(True)
        root.addWidget(self.lbl_headline)

        intro = (
            "First launch: NeuroPyGuiN is verifying that every tool it drives is installed. "
            "Anything marked Missing blocks a workflow; Check items only disable one feature."
            if first_run
            else "Verifying every Python package, bundled toolbox, GPU runtime, and external "
                 "binary the app drives."
        )
        self.lbl_intro = QtWidgets.QLabel(intro)
        self.lbl_intro.setObjectName("SectionHint")
        self.lbl_intro.setWordWrap(True)
        root.addWidget(self.lbl_intro)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setFormat("%v / %m")
        root.addWidget(self.progress)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Component", "Status", "Detail"])
        self.tree.setRootIsDecorated(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tree.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        header = self.tree.header()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Interactive)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.tree.setColumnWidth(0, 300)
        self.tree.currentItemChanged.connect(lambda *_: self._show_selected_detail())
        root.addWidget(self.tree, 1)

        detail_box = QtWidgets.QGroupBox("Details and fix")
        detail_box.setProperty("settingsSection", True)
        detail_layout = QtWidgets.QVBoxLayout(detail_box)
        self.txt_detail = QtWidgets.QPlainTextEdit()
        self.txt_detail.setReadOnly(True)
        self.txt_detail.setMaximumHeight(120)
        self.txt_detail.setPlaceholderText("Select a row to see what it means and how to fix it.")
        detail_layout.addWidget(self.txt_detail)
        self.btn_copy_fix = QtWidgets.QPushButton("Copy fix command")
        self.btn_copy_fix.setProperty("role", "secondary")
        self.btn_copy_fix.setEnabled(False)
        self.btn_copy_fix.clicked.connect(self._copy_fix)
        fix_row = QtWidgets.QHBoxLayout()
        fix_row.setContentsMargins(0, 0, 0, 0)
        fix_row.addStretch(1)
        fix_row.addWidget(self.btn_copy_fix)
        detail_layout.addLayout(fix_row)
        root.addWidget(detail_box)

        self.ck_startup = QtWidgets.QCheckBox(
            "Re-check at every startup (a warning only appears when something required is missing)"
        )
        self.ck_startup.setChecked(bool(settings.value("doctor/check_on_startup", False, type=bool)))
        self.ck_startup.toggled.connect(
            lambda checked: self._settings.setValue("doctor/check_on_startup", bool(checked))
        )
        root.addWidget(self.ck_startup)

        self.btn_rerun = QtWidgets.QPushButton("Re-run checks")
        self.btn_rerun.setProperty("role", "secondary")
        self.btn_rerun.clicked.connect(self.start)
        self.btn_report = QtWidgets.QPushButton("Copy full report")
        self.btn_report.setProperty("role", "secondary")
        self.btn_report.clicked.connect(self._copy_report)
        self.btn_install = QtWidgets.QPushButton("Install missing libraries/tools")
        self.btn_install.setProperty("role", "primary")
        self.btn_install.setEnabled(False)
        self.btn_install.setToolTip(
            "Install missing Python packages with pip and download supported native tools."
        )
        self.btn_install.clicked.connect(self._request_install)
        self.btn_close = QtWidgets.QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)
        footer = QtWidgets.QHBoxLayout()
        footer.addWidget(self.btn_rerun)
        footer.addWidget(self.btn_report)
        footer.addWidget(self.btn_install)
        footer.addStretch(1)
        footer.addWidget(self.btn_close)
        root.addLayout(footer)

    # --- running ---------------------------------------------------------- #

    def start(self) -> None:
        """Kick off (or restart) the checks in the background."""
        if self._running:
            return
        self._running = True
        self.tree.clear()
        self.txt_detail.clear()
        self.btn_copy_fix.setEnabled(False)
        self.btn_rerun.setEnabled(False)
        self.btn_install.setEnabled(False)
        self.progress.setRange(0, 0)
        self.progress.setFormat("Checking...")
        self.progress.show()
        self.lbl_headline.setText("Checking your installation...")

        worker = DiagnosticsWorker(lambda key, default: self._settings.value(key, default))
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(self._on_finished)
        self._pool.start(worker)

    @QtCore.Slot(int, int, str)
    def _on_progress(self, done: int, total: int, label: str) -> None:
        if self.progress.maximum() != total:
            self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.progress.setFormat(f"{label}  ({done}/{total})")

    @QtCore.Slot(object)
    def _on_finished(self, results: object) -> None:
        self._running = False
        self._results = list(results) if isinstance(results, (list, tuple)) else []
        self.progress.hide()
        self.btn_rerun.setEnabled(True)
        self._populate(self._results)
        self.lbl_headline.setText(doctor.headline(self._results))
        counts = doctor.summarize(self._results)
        self.lbl_intro.setText(
            f"{counts[doctor.OK]} ready · {counts[doctor.WARN]} to check · "
            f"{counts[doctor.FAIL]} missing. Select any row for the exact fix."
        )
        install_keys = self.install_keys()
        self.btn_install.setEnabled(bool(install_keys))
        self.btn_install.setToolTip(
            "Install: " + ", ".join(install_keys)
            if install_keys
            else "No automatic installer is available for the remaining checks."
        )

    def results(self) -> List[CheckResult]:
        """Return the results of the last completed run."""
        return list(self._results)

    def install_keys(self) -> List[str]:
        """Return de-duplicated automatic installer keys for unresolved rows."""
        keys: List[str] = []
        for result in self._results:
            if result.status == doctor.OK:
                continue
            for key in result.install_keys:
                if key and key not in keys:
                    keys.append(key)
        return keys

    # --- rendering -------------------------------------------------------- #

    def _populate(self, results: Sequence[CheckResult]) -> None:
        self.tree.clear()
        bold = QtGui.QFont()
        bold.setBold(True)
        groups: dict[str, QtWidgets.QTreeWidgetItem] = {}
        first_problem: QtWidgets.QTreeWidgetItem | None = None

        for result in results:
            parent = groups.get(result.category)
            if parent is None:
                parent = QtWidgets.QTreeWidgetItem(self.tree, [result.category, "", ""])
                parent.setFont(0, bold)
                parent.setFirstColumnSpanned(False)
                parent.setFlags(QtCore.Qt.ItemIsEnabled)
                groups[result.category] = parent

            item = QtWidgets.QTreeWidgetItem(
                parent,
                [
                    result.label,
                    f"{_STATUS_DOTS.get(result.status, '?')} {_STATUS_WORDS.get(result.status, result.status)}",
                    result.detail,
                ],
            )
            color = QtGui.QColor(self._colors.get(result.status, "#888888"))
            item.setForeground(1, QtGui.QBrush(color))
            if result.status != doctor.OK:
                item.setFont(1, bold)
            item.setToolTip(2, result.detail)
            item.setData(0, QtCore.Qt.UserRole, result)
            if first_problem is None and result.status != doctor.OK:
                first_problem = item

        # Collapse healthy groups so the eye lands on what needs work.
        for category, parent in groups.items():
            has_problem = any(
                parent.child(i).data(0, QtCore.Qt.UserRole).status != doctor.OK
                for i in range(parent.childCount())
            )
            parent.setExpanded(has_problem)
            failed = sum(
                1 for i in range(parent.childCount())
                if parent.child(i).data(0, QtCore.Qt.UserRole).status == doctor.FAIL
            )
            warned = sum(
                1 for i in range(parent.childCount())
                if parent.child(i).data(0, QtCore.Qt.UserRole).status == doctor.WARN
            )
            if failed or warned:
                bits = []
                if failed:
                    bits.append(f"{failed} missing")
                if warned:
                    bits.append(f"{warned} to check")
                parent.setText(2, ", ".join(bits))
                parent.setForeground(
                    2,
                    QtGui.QBrush(QtGui.QColor(self._colors[doctor.FAIL if failed else doctor.WARN])),
                )
            else:
                parent.setText(2, "all ready")

        if first_problem is not None:
            self.tree.setCurrentItem(first_problem)
        self._show_selected_detail()

    def _selected_result(self) -> CheckResult | None:
        item = self.tree.currentItem()
        if item is None:
            return None
        value = item.data(0, QtCore.Qt.UserRole)
        return value if isinstance(value, CheckResult) else None

    def _show_selected_detail(self) -> None:
        result = self._selected_result()
        if result is None:
            self.txt_detail.clear()
            self.btn_copy_fix.setEnabled(False)
            return
        lines = [
            f"{result.label}  -  {_STATUS_WORDS.get(result.status, result.status)}"
            f"  ({result.severity})",
            "",
            result.detail,
        ]
        if result.fix:
            lines += ["", "Fix:", result.fix]
        self.txt_detail.setPlainText("\n".join(lines))
        self.btn_copy_fix.setEnabled(bool(result.fix))

    # --- actions ---------------------------------------------------------- #

    def _copy_fix(self) -> None:
        result = self._selected_result()
        if result is None or not result.fix:
            return
        QtWidgets.QApplication.clipboard().setText(result.fix)

    def _copy_report(self) -> None:
        if not self._results:
            return
        QtWidgets.QApplication.clipboard().setText(doctor.report_text(self._results))
        QtWidgets.QMessageBox.information(
            self, "Diagnostics", "The full report was copied to the clipboard."
        )

    def _request_install(self) -> None:
        self.installToolsRequested.emit(self.install_keys())
        self.accept()

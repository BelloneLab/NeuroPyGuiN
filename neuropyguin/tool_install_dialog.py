"""Review and progress windows for the external preprocessing-tool installer.

:mod:`neuropyguin.tool_installer` downloads CatGT / TPrime / C_Waves from the
SpikeGLX site and can pip-install Kilosort 4. Those steps take from seconds to
several minutes, so they run in a worker thread and stream two things back:
``(label, fraction)`` progress and raw log lines. This dialog shows both - a
percentage bar per download, an indeterminate bar while pip works, and the full
log - so the install never looks like it has hung.

The progress dialog is intentionally non-modal: closing it leaves the installation
running (the log keeps flowing into the Preprocessing tab log) and the Preprocessing
tab re-shows this window if the user clicks the button again.

:class:`ToolMaintenanceDialog` is the way back in once everything is installed: it
lists each tool with its status and location, re-checks on demand, and lets the user
tick whichever tools to install or reinstall.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from PySide6 import QtCore, QtGui, QtWidgets


# Bar resolution: fine enough that a few-megabyte download animates smoothly.
_SCALE = 1000

_RESULT_COLORS = {
    "Light": {"ok": "#15803d", "fail": "#b91c1c"},
    "Dark": {"ok": "#4ade80", "fail": "#f87171"},
}


@dataclass(frozen=True)
class ToolStatus:
    """One row of the tool review dialog."""

    key: str
    name: str
    ready: bool
    location: str
    #: False when no official package exists for this platform (native tools on macOS).
    installable: bool = True
    #: Extra context, e.g. the installed Kilosort version.
    note: str = ""


class ToolMaintenanceDialog(QtWidgets.QDialog):
    """Review the external tools, re-check them, and (re)install a chosen subset.

    The dialog owns no logic: the Preprocessing tab supplies rows through
    :meth:`set_rows` and reacts to :attr:`recheckRequested` / :attr:`installRequested`.
    """

    recheckRequested = QtCore.Signal()
    installRequested = QtCore.Signal(list)

    def __init__(
        self,
        *,
        theme: str = "Light",
        platform_label: str = "",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setProperty("compactDialog", True)
        self.setWindowTitle("Preprocessing tools")
        self.resize(820, 480)
        self._colors = _RESULT_COLORS["Dark" if str(theme).lower().startswith("dark") else "Light"]
        self._rows: List[ToolStatus] = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        self.lbl_headline = QtWidgets.QLabel("Preprocessing tools")
        self.lbl_headline.setObjectName("StepStatusRunName")
        layout.addWidget(self.lbl_headline)

        self.lbl_hint = QtWidgets.QLabel(
            f"{platform_label}Tick a tool to install it, or to download and verify it again if "
            "it is already there. Kilosort is refreshed with pip using --no-deps, so the "
            "installed PyTorch build is never touched."
        )
        self.lbl_hint.setObjectName("SectionHint")
        self.lbl_hint.setWordWrap(True)
        layout.addWidget(self.lbl_hint)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Tool", "Status", "Location"])
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        header = self.tree.header()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Interactive)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.tree.setColumnWidth(0, 220)
        self.tree.itemChanged.connect(lambda *_: self._sync_install_button())
        layout.addWidget(self.tree, 1)

        self.btn_recheck = QtWidgets.QPushButton("Re-check")
        self.btn_recheck.setProperty("role", "secondary")
        self.btn_recheck.setToolTip("Verify the configured folders and the installed Kilosort again.")
        self.btn_recheck.clicked.connect(self.recheckRequested.emit)
        self.btn_select_missing = QtWidgets.QPushButton("Select missing")
        self.btn_select_missing.setProperty("role", "secondary")
        self.btn_select_missing.clicked.connect(self._select_missing)
        self.btn_install = QtWidgets.QPushButton("Install selected")
        self.btn_install.setProperty("role", "primary")
        self.btn_install.setEnabled(False)
        self.btn_install.clicked.connect(self._emit_install)
        self.btn_close = QtWidgets.QPushButton("Close")
        self.btn_close.clicked.connect(self.reject)
        footer = QtWidgets.QHBoxLayout()
        footer.addWidget(self.btn_recheck)
        footer.addWidget(self.btn_select_missing)
        footer.addStretch(1)
        footer.addWidget(self.btn_install)
        footer.addWidget(self.btn_close)
        layout.addLayout(footer)

    # --- population ------------------------------------------------------- #

    def set_rows(self, rows: Sequence[ToolStatus]) -> None:
        """Replace the table contents, pre-ticking whatever is missing."""
        self._rows = list(rows)
        self.tree.blockSignals(True)
        self.tree.clear()
        bold = QtGui.QFont()
        bold.setBold(True)
        for row in self._rows:
            status = "Ready" if row.ready else ("Unavailable" if not row.installable else "Missing")
            detail = row.location or "not configured"
            if row.note:
                detail = f"{detail}  ({row.note})" if row.location else row.note
            item = QtWidgets.QTreeWidgetItem(self.tree, [row.name, status, detail])
            item.setData(0, QtCore.Qt.UserRole, row.key)
            item.setToolTip(2, detail)
            colour = QtGui.QColor(self._colors["ok" if row.ready else "fail"])
            item.setForeground(1, QtGui.QBrush(colour))
            if not row.ready:
                item.setFont(1, bold)
            if row.installable:
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(0, QtCore.Qt.Checked if not row.ready else QtCore.Qt.Unchecked)
            else:
                item.setFlags(QtCore.Qt.ItemIsEnabled)
                item.setToolTip(0, "No official prebuilt package exists for this platform.")
        self.tree.blockSignals(False)

        ready = sum(1 for row in self._rows if row.ready)
        total = len(self._rows)
        self.lbl_headline.setText(
            f"All {total} preprocessing tools are ready"
            if ready == total
            else f"{ready} of {total} preprocessing tools are ready"
        )
        self._sync_install_button()

    def selected_keys(self) -> List[str]:
        keys: List[str] = []
        for index in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(index)
            if item.flags() & QtCore.Qt.ItemIsUserCheckable and item.checkState(0) == QtCore.Qt.Checked:
                keys.append(str(item.data(0, QtCore.Qt.UserRole)))
        return keys

    # --- actions ---------------------------------------------------------- #

    def _select_missing(self) -> None:
        self.tree.blockSignals(True)
        for index in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(index)
            if not item.flags() & QtCore.Qt.ItemIsUserCheckable:
                continue
            key = str(item.data(0, QtCore.Qt.UserRole))
            missing = any(row.key == key and not row.ready for row in self._rows)
            item.setCheckState(0, QtCore.Qt.Checked if missing else QtCore.Qt.Unchecked)
        self.tree.blockSignals(False)
        self._sync_install_button()

    def _sync_install_button(self) -> None:
        selected = self.selected_keys()
        reinstalling = [
            key for key in selected if any(row.key == key and row.ready for row in self._rows)
        ]
        self.btn_install.setEnabled(bool(selected))
        if not selected:
            self.btn_install.setText("Install selected")
        elif reinstalling and len(reinstalling) == len(selected):
            self.btn_install.setText(f"Reinstall {len(selected)} selected")
        else:
            self.btn_install.setText(f"Install {len(selected)} selected")

    def _emit_install(self) -> None:
        selected = self.selected_keys()
        if not selected:
            return
        self.installRequested.emit(selected)
        self.accept()


class ToolInstallProgressDialog(QtWidgets.QDialog):
    """Show download/extract/verify progress for a tool-installation run."""

    def __init__(
        self,
        tool_names: Sequence[str],
        *,
        theme: str = "Light",
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setProperty("compactDialog", True)
        self.setWindowTitle("Installing preprocessing tools")
        self.setModal(False)
        self.resize(720, 460)
        self._running = True
        self._colors = _RESULT_COLORS["Dark" if str(theme).lower().startswith("dark") else "Light"]

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        names = ", ".join(tool_names) if tool_names else "selected tools"
        self.lbl_title = QtWidgets.QLabel(f"Installing {names}")
        self.lbl_title.setObjectName("StepStatusRunName")
        self.lbl_title.setWordWrap(True)
        layout.addWidget(self.lbl_title)

        self.lbl_hint = QtWidgets.QLabel(
            "Native tools are downloaded from the official SpikeGLX site into this project's "
            "tools folder. You can close this window; the installation keeps running."
        )
        self.lbl_hint.setObjectName("SectionHint")
        self.lbl_hint.setWordWrap(True)
        layout.addWidget(self.lbl_hint)

        self.lbl_step = QtWidgets.QLabel("Starting...")
        self.lbl_step.setWordWrap(True)
        layout.addWidget(self.lbl_step)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, _SCALE)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%p%")
        layout.addWidget(self.progress)

        self.txt_log = QtWidgets.QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumBlockCount(4000)
        self.txt_log.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.txt_log.setPlaceholderText("Installer output appears here.")
        layout.addWidget(self.txt_log, 1)

        self.btn_copy = QtWidgets.QPushButton("Copy log")
        self.btn_copy.setProperty("role", "secondary")
        self.btn_copy.clicked.connect(self._copy_log)
        self.btn_dismiss = QtWidgets.QPushButton("Run in background")
        self.btn_dismiss.clicked.connect(self.close)
        footer = QtWidgets.QHBoxLayout()
        footer.addWidget(self.btn_copy)
        footer.addStretch(1)
        footer.addWidget(self.btn_dismiss)
        layout.addLayout(footer)

    # --- worker-driven updates -------------------------------------------- #

    @QtCore.Slot(str, float)
    def set_progress(self, label: str, fraction: float) -> None:
        """Update the bar and step label. ``fraction`` < 0 means indeterminate."""
        if label:
            self.lbl_step.setText(label)
        if fraction < 0:
            if self.progress.maximum() != 0:
                self.progress.setRange(0, 0)  # marquee: pip gives no measurable progress
                self.progress.setFormat("Working...")
            return
        if self.progress.maximum() != _SCALE:
            self.progress.setRange(0, _SCALE)
            self.progress.setFormat("%p%")
        self.progress.setValue(int(max(0.0, min(1.0, fraction)) * _SCALE))

    @QtCore.Slot(str)
    def append_log(self, line: str) -> None:
        text = str(line).rstrip()
        if not text:
            return
        self.txt_log.appendPlainText(text)
        scrollbar = self.txt_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def finish(self, ok: bool, message: str) -> None:
        """Mark the run as finished and turn the dismiss button into Close."""
        self._running = False
        if self.progress.maximum() != _SCALE:
            self.progress.setRange(0, _SCALE)
            self.progress.setFormat("%p%")
        if ok:
            self.progress.setValue(_SCALE)
        color = self._colors["ok" if ok else "fail"]
        self.lbl_step.setText(message)
        self.lbl_step.setStyleSheet(f"color: {color}; font-weight: 700;")
        self.lbl_hint.setText(
            "Installed paths were saved to the Preprocessing tab."
            if ok
            else "Nothing else was changed. Tools that completed before the error were kept."
        )
        self.btn_dismiss.setText("Close")
        self.btn_dismiss.setProperty("role", "primary")
        self.btn_dismiss.setDefault(True)

    def is_running(self) -> bool:
        return self._running

    # --- actions ---------------------------------------------------------- #

    def _copy_log(self) -> None:
        QtWidgets.QApplication.clipboard().setText(self.txt_log.toPlainText())

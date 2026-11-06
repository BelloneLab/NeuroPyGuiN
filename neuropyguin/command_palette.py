"""Keyboard-first action finder built from the application's existing commands."""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets


class CommandPalette(QtWidgets.QDialog):
    """Search enabled commands and analysis destinations without learning menus."""

    def __init__(self, parent: QtWidgets.QWidget, entries: list[tuple[str, str, object]]) -> None:
        super().__init__(parent)
        self.setWindowTitle("Find an action")
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint, False)
        self.resize(640, 480)
        self._entries = entries
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(22, 22, 22, 18)
        title = QtWidgets.QLabel("Where would you like to go?")
        title.setObjectName("WorkspaceTitle")
        layout.addWidget(title)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search analyses, open a recording, export, settings...")
        self.search.setClearButtonEnabled(True)
        self.search.setAccessibleName("Search application actions")
        layout.addWidget(self.search)
        self.results = QtWidgets.QListWidget()
        self.results.setAccessibleName("Matching actions")
        layout.addWidget(self.results, 1)
        hint = QtWidgets.QLabel("Type to filter  ·  ↑ ↓ to choose  ·  Enter to open  ·  Esc to close")
        hint.setObjectName("SectionHint")
        layout.addWidget(hint)
        self.search.textChanged.connect(self._filter)
        self.search.returnPressed.connect(self._activate)
        self.results.itemActivated.connect(self._activate)
        self.search.installEventFilter(self)
        self._filter("")
        self.search.setFocus()

    def _filter(self, text: str) -> None:
        """Match all query words against titles and categories."""
        self.results.clear()
        words = text.casefold().split()
        for index, (label, category, _) in enumerate(self._entries):
            if all(word in f"{label} {category}".casefold() for word in words):
                item = QtWidgets.QListWidgetItem(f"{label}   ·   {category}")
                item.setData(QtCore.Qt.UserRole, index)
                self.results.addItem(item)
        if self.results.count():
            self.results.setCurrentRow(0)

    def eventFilter(self, watched, event) -> bool:
        """Keep typing focus while arrow keys select a matching action."""
        if watched is self.search and event.type() == QtCore.QEvent.KeyPress:
            if event.key() in (QtCore.Qt.Key_Up, QtCore.Qt.Key_Down):
                step = 1 if event.key() == QtCore.Qt.Key_Down else -1
                self.results.setCurrentRow(max(0, min(self.results.count() - 1, self.results.currentRow() + step)))
                return True
        return super().eventFilter(watched, event)

    def _activate(self, *args) -> None:
        """Close before dispatch so modal file dialogs have the main window owner."""
        item = self.results.currentItem()
        if item is None:
            return
        callback = self._entries[item.data(QtCore.Qt.UserRole)][2]
        self.accept()
        QtCore.QTimer.singleShot(0, callback)

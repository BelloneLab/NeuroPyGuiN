"""Side navigation widgets: a vertical-rail stacked-page container.

Provides ``SideNavStack``, a QStackedWidget paired with a column of nav
buttons, plus ``VerticalNavButton``, a push button that paints its label
rotated 90 degrees for use in a narrow vertical rail.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class VerticalNavButton(QtWidgets.QPushButton):
    """Push button that draws its label rotated 90 degrees (bottom-to-top).

    Used in the compact vertical rail where horizontal space is scarce. The
    label text is rendered by ``paintEvent`` rather than the default button
    layout, and is exposed as the tooltip and accessible name.
    """

    def __init__(self, label: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("", parent)
        self._label = label
        self.setToolTip(label)
        self.setAccessibleName(label)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

    def sizeHint(self) -> QtCore.QSize:
        fm = self.fontMetrics()
        return QtCore.QSize(max(40, fm.height() + 18), max(88, fm.horizontalAdvance(self._label) + 28))

    def minimumSizeHint(self) -> QtCore.QSize:
        return self.sizeHint()

    def paintEvent(self, event) -> None:
        opt = QtWidgets.QStyleOptionButton()
        self.initStyleOption(opt)
        opt.text = ""

        painter = QtWidgets.QStylePainter(self)
        painter.drawControl(QtWidgets.QStyle.CE_PushButtonBevel, opt)
        painter.save()
        painter.translate(self.rect().center())
        painter.rotate(-90)
        painter.setPen(opt.palette.buttonText().color())
        painter.drawText(
            QtCore.QRect(-self.height() // 2, -self.width() // 2, self.height(), self.width()),
            QtCore.Qt.AlignCenter,
            self._label,
        )
        painter.restore()


class SideNavStack(QtWidgets.QWidget):
    """A stacked-page container with a left navigation rail.

    Each page added via ``add_page`` gets a checkable nav button. Selecting a
    button (or calling ``setCurrentIndex``) switches the visible page and emits
    ``currentChanged``. The rail can render either standard horizontal buttons
    or rotated vertical-label buttons in a narrow compact rail.
    """

    currentChanged = QtCore.Signal(int)

    def __init__(
        self,
        title: str = "",
        subtitle: str = "",
        *,
        vertical_labels: bool = False,
        compact_rail: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._vertical_labels = vertical_labels
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(14)

        self.nav_frame = QtWidgets.QFrame()
        self.nav_frame.setProperty("navRail", True)
        self.nav_frame.setProperty("compactRail", compact_rail)
        self.nav_frame.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        if vertical_labels:
            rail_width = 58 if compact_rail else 72
            self.nav_frame.setMinimumWidth(rail_width)
            self.nav_frame.setMaximumWidth(rail_width)
        else:
            self.nav_frame.setMinimumWidth(210)
        nav_layout = QtWidgets.QVBoxLayout(self.nav_frame)
        if vertical_labels:
            nav_layout.setContentsMargins(8, 10, 8, 10)
            nav_layout.setSpacing(6)
        else:
            nav_layout.setContentsMargins(12, 12, 12, 12)
            nav_layout.setSpacing(8)

        if vertical_labels:
            nav_tip = "\n".join(part for part in [title, subtitle] if part)
            if nav_tip:
                self.nav_frame.setToolTip(nav_tip)
        else:
            if title:
                title_label = QtWidgets.QLabel(title)
                title_label.setObjectName("FieldTitle")
                nav_layout.addWidget(title_label)
            if subtitle:
                subtitle_label = QtWidgets.QLabel(subtitle)
                subtitle_label.setObjectName("SectionHint")
                subtitle_label.setWordWrap(True)
                nav_layout.addWidget(subtitle_label)

        self._button_layout = QtWidgets.QVBoxLayout()
        self._button_layout.setContentsMargins(0, 4 if not vertical_labels else 0, 0, 0)
        self._button_layout.setSpacing(6)
        nav_layout.addLayout(self._button_layout)
        nav_layout.addStretch(1)

        self.stack = QtWidgets.QStackedWidget()
        self.stack.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        root.addWidget(self.nav_frame, 0)
        root.addWidget(self.stack, 1)

        self._buttons: list[QtWidgets.QPushButton] = []

    def add_page(self, label: str, page: QtWidgets.QWidget) -> int:
        """Add a page and its nav button; return the page's stack index.

        The first page added is selected automatically.
        """
        index = self.stack.addWidget(page)
        if self._vertical_labels:
            button = VerticalNavButton(label)
            button.setProperty("verticalLabel", True)
        else:
            button = QtWidgets.QPushButton(label)
        button.setProperty("navButton", True)
        button.setCheckable(True)
        if self._vertical_labels:
            button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        else:
            button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        button.clicked.connect(lambda _checked=False, idx=index: self.setCurrentIndex(idx))
        self._button_layout.addWidget(button)
        self._buttons.append(button)
        if len(self._buttons) == 1:
            self.setCurrentIndex(0)
        return index

    def currentIndex(self) -> int:
        """Return the index of the currently visible page."""
        return self.stack.currentIndex()

    def currentWidget(self) -> QtWidgets.QWidget | None:
        """Return the currently visible page widget, or None if empty."""
        return self.stack.currentWidget()

    def setCurrentIndex(self, index: int) -> None:
        """Show the page at ``index`` and sync the nav buttons.

        Out-of-range indices are ignored. Buttons are toggled with signals
        blocked so updating their checked state does not re-trigger selection.
        ``currentChanged`` is emitted once the page is switched.
        """
        if index < 0 or index >= self.stack.count():
            return
        self.stack.setCurrentIndex(index)
        for idx, button in enumerate(self._buttons):
            button.blockSignals(True)
            button.setChecked(idx == index)
            button.blockSignals(False)
        self.currentChanged.emit(index)

    def setCurrentWidget(self, widget: QtWidgets.QWidget) -> None:
        """Show the page matching ``widget`` (by its stack index)."""
        self.setCurrentIndex(self.stack.indexOf(widget))

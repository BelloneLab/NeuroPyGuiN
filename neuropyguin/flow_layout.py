"""Wrapping layout that never squeezes widgets below their size hint.

``QHBoxLayout`` shrinks children once the window is narrower than the sum of
their size hints, which clips button labels ("Run queue" becomes "un queu").
``FlowLayout`` instead keeps every widget at its natural size and wraps the
overflow onto a new line, like words in a paragraph. It implements
``heightForWidth`` so parent layouts reserve exactly the rows it needs.
"""

from __future__ import annotations

from typing import List

from PySide6 import QtCore, QtWidgets


class FlowLayout(QtWidgets.QLayout):
    """Left-to-right layout that wraps items onto new rows when out of width.

    Args:
        parent: Optional widget that will own this layout.
        h_spacing: Horizontal gap between items on the same row (px).
        v_spacing: Vertical gap between rows (px).
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        h_spacing: int = 8,
        v_spacing: int = 8,
    ) -> None:
        super().__init__(parent)
        self._items: List[QtWidgets.QLayoutItem] = []
        self._h_spacing = h_spacing
        self._v_spacing = v_spacing
        self.setContentsMargins(0, 0, 0, 0)

    # ------------------------------------------------------------------ #
    # QLayout item bookkeeping
    # ------------------------------------------------------------------ #
    def addItem(self, item: QtWidgets.QLayoutItem) -> None:  # noqa: N802 (Qt API)
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QtWidgets.QLayoutItem | None:  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QtWidgets.QLayoutItem | None:  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    # ------------------------------------------------------------------ #
    # Size negotiation
    # ------------------------------------------------------------------ #
    def expandingDirections(self) -> QtCore.Qt.Orientations:  # noqa: N802
        return QtCore.Qt.Orientations(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._do_layout(QtCore.QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QtCore.QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QtCore.QSize:  # noqa: N802
        """Smallest size is the widest single item, so nothing is ever clipped."""
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())

    # ------------------------------------------------------------------ #
    # Core placement
    # ------------------------------------------------------------------ #
    def _do_layout(self, rect: QtCore.QRect, test_only: bool) -> int:
        """Place items row by row inside ``rect``; return the total height used.

        Each item keeps its ``sizeHint``. When the next item would cross the
        right edge, the cursor moves to a new row below the tallest item of
        the current row. Items in a row are vertically centred.
        """
        m = self.contentsMargins()
        area = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y = area.x(), area.y()
        row: list[tuple[QtWidgets.QLayoutItem, QtCore.QSize]] = []
        row_height = 0

        def flush_row(row_y: int) -> None:
            # Commit one row, centring each item within the row height.
            if test_only:
                return
            for it, hint in row:
                geo = it.geometry()
                it.setGeometry(
                    QtCore.QRect(
                        QtCore.QPoint(geo.x(), row_y + (row_height - hint.height()) // 2),
                        hint,
                    )
                )

        for item in self._items:
            widget = item.widget()
            if widget is not None and widget.isHidden():
                # Explicitly hidden widgets take no space (matches QBoxLayout).
                continue
            hint = item.sizeHint()
            next_x = x + hint.width()
            if row and next_x > area.right() + 1:
                flush_row(y)
                y += row_height + self._v_spacing
                x = area.x()
                next_x = x + hint.width()
                row, row_height = [], 0
            if not test_only:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), hint))
            row.append((item, hint))
            row_height = max(row_height, hint.height())
            x = next_x + self._h_spacing
        flush_row(y)
        return y + row_height - rect.y() + m.bottom()

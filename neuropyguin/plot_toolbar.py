"""Compact Matplotlib navigation on a permanent, theme-independent paper surface."""
from pathlib import Path

import matplotlib
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PySide6 import QtCore, QtGui


class PaperNavigationToolbar(NavigationToolbar2QT):
    """Keep the original dark icons legible when the surrounding app is dark.

    Matplotlib normally recolors icons using the application palette at toolbar
    construction. A white toolbar stylesheet alone therefore produces pale or
    invisible icons. This toolbar reads the original packaged icon assets.
    """

    def __init__(self, canvas, parent=None):
        super().__init__(canvas, parent)
        self.setIconSize(QtCore.QSize(18, 18))
        self.setFixedHeight(32)
        # Some application styles assign 16px default layout margins, which
        # consume the entire compact toolbar and leave 6px icon buttons.
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(2)
        self.setStyleSheet(
            "QToolBar { background: white; color: #243649; padding: 0px; spacing: 2px; border: 0px; }"
            "QToolButton { background: white; color: #243649; padding: 3px; margin: 0px; border: 0px; min-height: 0px; min-width: 0px; }"
            "QToolButton:hover, QToolButton:checked { background: #E6EEF6; }"
            "QLabel { background: white; color: #243649; padding: 0px; margin: 0px; }"
        )

    def event(self, event):
        """Reapply compact margins after Qt styles polish a newly inserted toolbar."""
        handled = super().event(event)
        if event.type() in (QtCore.QEvent.StyleChange, QtCore.QEvent.Show, QtCore.QEvent.LayoutRequest):
            layout = self.layout()
            if layout is not None and layout.contentsMargins() != QtCore.QMargins(0, 0, 0, 0):
                layout.setContentsMargins(0, 0, 0, 0)
                layout.activate()
        return handled

    def _icon(self, name):
        """Load a high-resolution dark icon without application-palette recoloring."""
        regular = Path(matplotlib.get_data_path()) / "images" / name
        large = regular.with_name(regular.name.replace(".png", "_large.png"))
        pixmap = QtGui.QPixmap(str(large if large.exists() else regular))
        pixmap.setDevicePixelRatio(self.devicePixelRatioF() or 1)
        return QtGui.QIcon(pixmap)

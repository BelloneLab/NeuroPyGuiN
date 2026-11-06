"""Keyboard navigation and signal regressions in the shared workspace shell."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6 import QtCore, QtGui, QtWidgets
from neuropyguin.command_palette import CommandPalette
from neuropyguin.side_nav import SideNavStack


def test_palette_matches_all_words_and_runs_only_selected_action():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    owner = QtWidgets.QWidget()
    calls = []
    dialog = CommandPalette(owner, [('Population geometry', 'Analysis NEMBA', lambda: calls.append('population')),
                                    ('Export figure', 'PDF SVG PNG', lambda: calls.append('export'))])
    dialog.search.setText('nemba pop')
    assert dialog.results.count() == 1
    dialog._activate()
    app.processEvents()
    assert calls == ['population']
    dialog.deleteLater()
    owner.deleteLater()


def test_side_navigation_does_not_recompute_current_page():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    nav = SideNavStack()
    nav.add_page('One', QtWidgets.QWidget())
    nav.add_page('Two', QtWidgets.QWidget())
    changes = []
    nav.currentChanged.connect(changes.append)
    nav.setCurrentIndex(0)
    nav.setCurrentIndex(1)
    nav.setCurrentIndex(1)
    nav.setCurrentIndex(-1)
    assert changes == [1]
    assert nav._buttons[1].isChecked()
    nav.deleteLater()
    app.processEvents()

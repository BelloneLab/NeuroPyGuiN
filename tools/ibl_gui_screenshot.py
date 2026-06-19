"""Construct the unmodified IBL ephys-alignment GUI (offline) and screenshot it.

Run by the neuropygui interpreter (which has PyQt5 + iblatlas + iblapps on path).
Proves the 'launch the original IBL GUI' integration works from the unified env.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "_verify_shots"
OUT.mkdir(parents=True, exist_ok=True)

from PyQt5 import QtWidgets, QtCore  # noqa: E402
from atlaselectrophysiology.ephys_atlas_gui import MainWindow  # noqa: E402


def _shoot():
    win = QtWidgets.QApplication.instance().activeWindow() or main.win  # type: ignore
    main.win.grab().save(str(OUT / "08_ibl_gui.png"))
    print("IBL GUI screenshot saved.")
    QtWidgets.QApplication.instance().quit()


class _Holder:
    win = None


main = _Holder()

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    main.win = MainWindow(offline=True)
    main.win.resize(1500, 950)
    main.win.show()
    QtCore.QTimer.singleShot(4000, _shoot)
    QtCore.QTimer.singleShot(15000, app.quit)  # hard stop
    app.exec_()

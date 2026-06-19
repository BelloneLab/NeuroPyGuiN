"""Launch the IBL ephys-alignment GUI offline, auto-loading a session folder.

The stock ``atlaselectrophysiology/ephys_atlas_gui.py`` offline entry point always
opens a folder picker and waits for a manual "Get Data" click. This launcher builds
the very same ``MainWindow(offline=True)`` but points it straight at a given
histology folder and presses "Get Data" programmatically, so the probe alignment is
displayed immediately.

It must run under an interpreter that has the IBL stack and PyQt5 (the ``neuropygui``
env), with the ``iblapps`` repo on ``PYTHONPATH`` (:mod:`ibl_launch` arranges both).
Run as::

    python -m neuropyguin.histology.ibl_gui_launcher <hist_folder>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional


def _auto_load(win, folder: Path) -> None:
    """Replicate ``MainWindow.on_folder_selected`` for a fixed folder, then load.

    Mirrors the offline wiring in ``ephys_atlas_gui.py`` exactly: set the folder
    line, read the per-shank info, populate the shank/alignment drop-downs, select
    the first shank and trigger the data load (the "Get Data" handler).
    """
    win.data_status = False
    win.folder_line.setText(str(folder))
    win.prev_alignments, shank_options = win.loaddata.get_info(folder)
    win.populate_lists(shank_options, win.shank_list, win.shank_combobox)
    win.populate_lists(win.prev_alignments, win.align_list, win.align_combobox)
    win.on_shank_selected(0)
    win.data_button_pressed()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Auto-load the IBL alignment GUI")
    parser.add_argument("folder", help="Histology session folder to load")
    args = parser.parse_args(argv)
    folder = Path(args.folder)

    from PyQt5 import QtWidgets
    from atlaselectrophysiology.ephys_atlas_gui import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow(offline=True)

    try:
        _auto_load(win, folder)
        print(f"Auto-loaded session: {folder}")
    except Exception as exc:  # keep the GUI usable; user can pick the folder manually
        print(f"Auto-load failed ({type(exc).__name__}: {exc}). "
              "Use the '...' folder button to select the session.", file=sys.stderr)

    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

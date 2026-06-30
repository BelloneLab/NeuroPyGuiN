"""Headless visual harness for the Post Processing tab.

Drives neuropyguin.tabs.postprocessing_tab.PostProcessingTab against the real
test dataset, switches through every analysis page, drives the Condition PSTH
with a behavior CSV, and writes a screenshot of each view. Used to capture a
"before/after" record while making the panel npyx-faithful.

Usage (from the neuropygui env)::

    python tools/postproc_verify.py [TAG] [KS_FOLDER] [BEHAVIOR_CSV]

Defaults:
    TAG          = baseline
    KS_FOLDER    = B:/NPX/.../imec0_ks4   (the curated VTA_NPX 29537/2 dataset)
    BEHAVIOR_CSV = Z:/#SHARE/.../behaviors_binary_threshold_v2.csv
Screenshots are written to tools/_postproc_shots/<TAG>/.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# The Qt "offscreen" platform on Windows renders text as tofu (no font DB), so
# axis labels are unreadable. Use the native windows platform (a window flashes
# briefly) so screenshots carry readable labels. Override with QT_QPA_PLATFORM=offscreen.
os.environ.setdefault("QT_QPA_PLATFORM", "windows")
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from neuropyguin.tabs.postprocessing_tab import PostProcessingTab  # noqa: E402
from neuropyguin.styles import build_app_palette, build_app_qss  # noqa: E402


TAG = sys.argv[1] if len(sys.argv) > 1 else "baseline"
KS = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
    r"B:\NPX\processedData\VTA_NPX\29237\2\spike_sorting\catgt_29537_2_trial1_g0\29537_2_trial1_g0_imec0\imec0_ks4")
BEHAVIOR = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(
    r"Z:\#SHARE\AndryA\PROJECTS\processedData\VTA_NPX\29537\2\dlc\behaviors_binary_threshold_v2.csv")
OUT = Path(__file__).resolve().parent / "_postproc_shots" / TAG

DEMO_UNITS = [8, 166, 154]          # high-FR units confirmed present in this dataset
BEHAVIOR_LABEL = "approach"          # behavior column with 88 real onsets


def _style(app: QtWidgets.QApplication) -> None:
    fusion = QtWidgets.QStyleFactory.create("Fusion")
    if fusion is not None:
        app.setStyle(fusion)
    app.setFont(QtGui.QFont("Segoe UI", 10))
    app.setPalette(build_app_palette("Light"))
    app.setStyleSheet(build_app_qss("Light"))


def _pump(app: QtWidgets.QApplication, ms: int = 250) -> None:
    end = time.time() + ms / 1000.0
    while time.time() < end:
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)


def _grab(widget: QtWidgets.QWidget, name: str, app: QtWidgets.QApplication) -> None:
    _pump(app, 400)
    OUT.mkdir(parents=True, exist_ok=True)
    ok = widget.grab().save(str(OUT / name))
    print(f"  shot: {name}  ({'ok' if ok else 'FAILED'})")


def _select_units(tab: PostProcessingTab, units: list[int]) -> list[int]:
    tab.list_units.clearSelection()
    available = {int(tab.list_units.item(i).text()): i for i in range(tab.list_units.count())}
    chosen: list[int] = []
    for u in units:
        if u in available:
            tab.list_units.item(available[u]).setSelected(True)
            chosen.append(u)
    if not chosen and tab.list_units.count():
        for i in range(min(3, tab.list_units.count())):
            tab.list_units.item(i).setSelected(True)
            chosen.append(int(tab.list_units.item(i).text()))
    return chosen


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _style(app)
    tab = PostProcessingTab(QtCore.QThreadPool.globalInstance())
    tab.resize(1700, 1040)
    tab.show()
    _pump(app, 300)

    print(f"[postproc_verify] tag={TAG}")
    print(f"  KS folder : {KS}  (exists={KS.exists()})")
    print(f"  behavior  : {BEHAVIOR}  (exists={BEHAVIOR.exists()})")

    tab.open_ks_folder(str(KS))
    _pump(app, 600)
    print(f"  units loaded: {tab.list_units.count()}")

    chosen = _select_units(tab, DEMO_UNITS)
    print(f"  selected units: {chosen}")
    _pump(app, 300)

    page_names = ["unit_basics", "raw_explorer", "correlogram", "condition_psth", "network", "advanced", "celltypes"]
    for idx, pname in enumerate(page_names):
        tab.analysis_tabs.setCurrentIndex(idx)
        _pump(app, 300)
        try:
            if pname == "condition_psth":
                # Drive the PSTH with the real behavior CSV.
                if tab.tbl_conditions.rowCount() == 0:
                    tab._add_condition_row()
                tab._apply_condition_csv_to_row(0, str(BEHAVIOR))
                # Try to select the demo behavior label if present.
                combo = tab._condition_label_combo(0)
                if combo is not None:
                    j = combo.findData(BEHAVIOR_LABEL)
                    if j >= 0:
                        combo.setCurrentIndex(j)
                _pump(app, 200)
                tab._compute_psth()
                _pump(app, 600)
            elif pname == "network":
                # New population network analysis (3-panel matplotlib figure).
                tab._compute_network()
                _pump(app, 1500)
            elif pname == "advanced":
                # Render the default curated method so the page is non-empty.
                tab._refresh_current_page()
                _pump(app, 1500)
            elif pname == "celltypes":
                # C4 must NOT auto-run; just confirm the panel/idle message renders.
                # (Optionally kick the off-thread run, but do NOT block for ~1 min.)
                _pump(app, 400)
        except Exception as exc:  # noqa: BLE001
            print(f"  !! {pname} drive error: {exc}")
        suffix = "7_celltypes" if pname == "celltypes" else f"{idx}_{pname}"
        _grab(tab, f"{suffix}.png", app)

    # Extra: Correlogram in CCG-grid mode (the canonical npyx ACG/CCG grid).
    try:
        tab.analysis_tabs.setCurrentIndex(2)
        j = tab.cb_corr_mode.findText("CCG")
        if j >= 0:
            tab.cb_corr_mode.setCurrentIndex(j)
        _pump(app, 1200)
        _grab(tab, "2b_correlogram_ccg.png", app)
    except Exception as exc:  # noqa: BLE001
        print(f"  !! ccg capture error: {exc}")

    # Extra: single-unit PSTH (per-trial raster heatmap over the 88 approach onsets).
    try:
        tab.analysis_tabs.setCurrentIndex(3)
        _select_units(tab, [DEMO_UNITS[0]])
        _pump(app, 200)
        tab._compute_psth()
        _pump(app, 600)
        _grab(tab, "3b_psth_single_unit.png", app)
    except Exception as exc:  # noqa: BLE001
        print(f"  !! single-unit psth error: {exc}")

    # Extra: settings panel COLLAPSED, so the figure goes full-width. Capture on
    # the Correlogram page; the nav tab bar must still be usable in this state.
    try:
        tab.analysis_tabs.setCurrentIndex(2)
        _select_units(tab, DEMO_UNITS)
        _pump(app, 400)
        tab._set_settings_visible(False)
        _pump(app, 500)
        _grab(tab, "6_settings_collapsed.png", app)
        # Switch analysis while collapsed to prove nav still works.
        tab.analysis_tabs.setCurrentIndex(4)
        _pump(app, 400)
        _grab(tab, "6b_collapsed_nav_switch.png", app)
        tab._set_settings_visible(True)
        _pump(app, 300)
    except Exception as exc:  # noqa: BLE001
        print(f"  !! settings-collapsed capture error: {exc}")

    print(f"[postproc_verify] done -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

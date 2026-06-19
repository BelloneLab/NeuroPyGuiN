"""End-to-end visual + numeric verification of the Histology tab.

Runs against the test session and writes a screenshot of every stage plus a
numeric report comparing the regenerated products to the reference files.

Usage (from the neuropygui env)::

    python tools/histology_verify.py [TEST_FOLDER] [OUT_DIR]

Defaults: TEST_FOLDER = B:/NPX/processedData/VTA_NPX/29237/2/histology
          OUT_DIR     = tools/_verify_shots
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from neuropyguin.tabs.histology_tab import HistologyTab  # noqa: E402
from neuropyguin.histology import io_formats, tracing  # noqa: E402
from neuropyguin.histology.atlas import AllenCCFAtlas  # noqa: E402
from neuropyguin.styles import build_app_palette, build_app_qss  # noqa: E402


def _style(app: QtWidgets.QApplication) -> None:
    fusion = QtWidgets.QStyleFactory.create("Fusion")
    if fusion is not None:
        app.setStyle(fusion)
    app.setFont(QtGui.QFont("Segoe UI", 10))
    app.setPalette(build_app_palette("Light"))
    app.setStyleSheet(build_app_qss("Light"))


TEST = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    r"B:\NPX\processedData\VTA_NPX\29237\2\histology")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent / "_verify_shots"


def _grab(tab, name, app):
    app.processEvents()
    OUT.mkdir(parents=True, exist_ok=True)
    tab.grab().save(str(OUT / name))
    print(f"  shot: {name}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="histo_verify_"))
    # Stage a working copy so the reference is never touched.
    for f in ["probe_ccf.mat", "histology_ccf.mat", "atlas2histology_tform.mat",
              "channels.localCoordinates.npy",
              "xyz_picks_shank1.json", "xyz_picks_shank2.json",
              "xyz_picks_shank3.json", "xyz_picks_shank4.json",
              "prev_alignments_shank1.json", "prev_alignments_shank2.json",
              "prev_alignments_shank3.json"]:
        if (TEST / f).exists():
            shutil.copy(TEST / f, work / f)
    # Provide a slice image (use the raw histology image as slice_1).
    imgs = list((TEST / "images").glob("*.tif")) if (TEST / "images").exists() else []
    if imgs:
        shutil.copy(imgs[0], work / "slice_1.tif")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _style(app)
    tab = HistologyTab(QtCore.QThreadPool.globalInstance())
    tab.resize(1500, 950)
    tab.show()

    # --- Setup ---
    tab.ed_atlas.setText(r"D:\AP_histology\allen_atlas_path")
    tab.open_histology_folder(str(work))
    tab.nav.setCurrentIndex(0)
    _grab(tab, "01_setup.png", app)

    # --- Preprocess (shows the loaded slice) ---
    tab.nav.setCurrentIndex(1)
    tab._preproc_show()
    _grab(tab, "02_preprocess.png", app)

    # --- Match (render atlas plane near the reference AP) ---
    tab.nav.setCurrentIndex(2)
    tab._ensure_atlas()
    # Reference slice sits near AP 820 (from histology_ccf summary).
    tab.sl_ap.setValue(820)
    tab.cb_mode.setCurrentText("AV")
    tab._match_show()
    _grab(tab, "03_match.png", app)

    # --- Align (overlay the reference atlas boundaries on histology) ---
    tab.nav.setCurrentIndex(3)
    if tab.histology_ccf and tab.tforms:
        tab._cur_align_slice = 0
        tab._align_show()
        tab._align_overlay(0)
    _grab(tab, "04_align.png", app)

    # --- Trace (build probe_ccf from reference points, draw area chart) ---
    tab.nav.setCurrentIndex(4)
    at = tab._ensure_atlas()
    ref_pts = io_formats.load_probe_ccf_points(TEST / "probe_ccf.mat")
    probes = []
    for pts in ref_pts:
        pts = np.asarray(pts, float)
        df, coords = tracing.trajectory_areas_from_points(pts, at)
        probes.append({"points": pts, "trajectory_areas": df, "trajectory_coords": coords})
    tab._draw_trajectory_areas(probes)
    _grab(tab, "05_trace_areas.png", app)

    # --- Channels (regenerate with saved alignment, compare to reference) ---
    tab.nav.setCurrentIndex(5)
    tab.cb_alignment.setCurrentText("latest")
    from neuropyguin.histology import ibl_launch
    rc, _ = ibl_launch.run_bridge(["channels", str(work), "--alignment", "latest"], log=print)
    tab._load_channel_table()
    _grab(tab, "06_channels.png", app)

    # --- IBL page ---
    tab.nav.setCurrentIndex(6)
    _grab(tab, "07_ibl.png", app)

    # ---------- numeric report ----------
    print("\n=== numeric verification ===")
    gen = json.load(open(work / "channel_locations_all_shanks.json"))
    ref = json.load(open(TEST / "channel_locations_all_shanks.json"))
    gk, rk = set(gen) - {"origin"}, set(ref) - {"origin"}
    mismatches = sum(1 for k in gk & rk if gen[k]["brain_region_id"] != ref[k]["brain_region_id"])
    max_xyz = max(
        (abs(gen[k]["x"] - ref[k]["x"]) + abs(gen[k]["y"] - ref[k]["y"]) + abs(gen[k]["z"] - ref[k]["z"])
         for k in gk & rk), default=float("nan"))
    print(f"channels: {len(gk)} (ref {len(rk)}) | key-set match: {gk == rk}")
    print(f"origin equal: {gen['origin'] == ref['origin']}")
    print(f"max xyz L1 diff (um): {max_xyz:.3e} | region_id mismatches: {mismatches}")
    ok = (gk == rk) and mismatches == 0 and max_xyz < 1e-3
    print("RESULT:", "PASS" if ok else "FAIL")
    print(f"screenshots in: {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

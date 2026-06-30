"""Regenerate README screenshots from real local NeuroPyGuiN data.

Run from the repository root with the main app environment:

    C:/Users/bellone/.conda/envs/neuropygui/python.exe tools/readme_screens.py all

The captures use the actual main window, not isolated tab widgets. This makes
the README match what a user sees: top workflow tabs, real paths, real loaded
data, and the white application theme throughout.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Set these before importing Qt. The native Windows platform gives readable
# fonts; offscreen renders tofu boxes on this machine.
os.environ["QT_QPA_PLATFORM"] = "windows"
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyqtgraph as pg  # noqa: E402
from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from neuropyguin.app import NeuroPyGuiNMainWindow  # noqa: E402
from neuropyguin.preprocessing import (  # noqa: E402
    _read_meta_keyvals,
    find_meta_for_bin,
    parse_spikeglx_bin_name,
    spikeglx_meta_channel_info,
)
from neuropyguin.styles import build_app_palette, build_app_qss  # noqa: E402


RAW_AP_BIN = Path(
    r"B:\NPX\rawData\mPFC-NAc\51543\mPFC_NAc_week1\dual"
    r"\51543_dual_recordings_mPFC_NAc_g0\51543_dual_recordings_mPFC_NAc_g0_imec0"
    r"\51543_dual_recordings_mPFC_NAc_g0_t0.imec0.ap.bin"
)
KS_FOLDER = Path(
    r"B:\NPX\processedData\VTA_NPX\29237\2\spike_sorting"
    r"\catgt_29537_2_trial1_g0\29537_2_trial1_g0_imec0\imec0_ks4"
)
BEHAVIOR_CSV = Path(
    r"Z:\#SHARE\AndryA\PROJECTS\processedData\VTA_NPX\29537\2\dlc"
    r"\behaviors_binary_threshold_v2.csv"
)
HISTO_SESSION = Path(r"B:\NPX\processedData\VTA_NPX\29237\2\histology")
ATLAS_PATH = Path(r"D:\AP_histology\allen_atlas_path")
SHOTS_DIR = REPO_ROOT / "neuropyguin" / "assets" / "screenshots"

WINDOW_SIZE = (2048, 1024)
POST_CORR_UNITS = [243, 250, 253, 255, 260, 268]


def style_app(app: QtWidgets.QApplication, theme: str = "Light") -> None:
    fusion = QtWidgets.QStyleFactory.create("Fusion")
    if fusion is not None:
        app.setStyle(fusion)
    app.setFont(QtGui.QFont("Segoe UI", 10))
    app.setPalette(build_app_palette(theme))
    app.setStyleSheet(build_app_qss(theme))
    dark = str(theme).lower().startswith("dark")
    pg.setConfigOption("background", "#0b0f14" if dark else "w")
    pg.setConfigOption("foreground", "#e8eef7" if dark else "k")


def pump(app: QtWidgets.QApplication, ms: int = 350) -> None:
    end = time.time() + ms / 1000.0
    while time.time() < end:
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)


def grab(win: QtWidgets.QWidget, name: str, app: QtWidgets.QApplication, settle_ms: int = 700) -> bool:
    pump(app, settle_ms)
    SHOTS_DIR.mkdir(parents=True, exist_ok=True)
    ok = bool(win.grab().save(str(SHOTS_DIR / name)))
    print(f"  saved {name}: {'OK' if ok else 'FAILED'}")
    return ok


def build_window(app: QtWidgets.QApplication) -> NeuroPyGuiNMainWindow:
    win = NeuroPyGuiNMainWindow()
    win.resize(*WINDOW_SIZE)
    win.show()
    win.raise_()
    win.activateWindow()
    win._set_plot_preferences(theme="Light", grid=True)
    style_app(app, "Light")
    pump(app, 600)
    return win


def select_list_items(list_widget: QtWidgets.QListWidget, values: list[int]) -> list[int]:
    list_widget.clearSelection()
    available = {}
    for i in range(list_widget.count()):
        text = list_widget.item(i).text().strip()
        try:
            available[int(float(text))] = i
        except Exception:
            continue
    selected = []
    for value in values:
        row = available.get(int(value))
        if row is None:
            continue
        item = list_widget.item(row)
        item.setSelected(True)
        selected.append(int(value))
    if selected:
        list_widget.scrollToItem(list_widget.item(available[selected[0]]))
    return selected


def _format_gb(path: Path) -> str:
    try:
        return f"{path.stat().st_size / (1024 ** 3):.2f} GB"
    except OSError:
        return "unknown size"


def _set_preprocessing_demo_steps(pre) -> None:
    pre.ck_catgt.setChecked(True)
    pre.ck_catgt_extract_only.setChecked(False)
    pre.ck_tprime.setChecked(True)
    pre.ck_ks.setChecked(True)
    pre.ck_post.setChecked(True)
    pre.ck_noise.setChecked(True)
    pre.ck_wvf.setChecked(True)
    pre.ck_qm.setChecked(True)
    pre.ck_pybomb.setChecked(True)
    pre.cb_ks_ver.setCurrentText("4")
    pre.ed_region.setText("mPFC-NAc")


def _populate_preprocessing_run_plan(pre) -> None:
    parsed = parse_spikeglx_bin_name(str(RAW_AP_BIN))
    meta_path = find_meta_for_bin(str(RAW_AP_BIN))
    meta = _read_meta_keyvals(meta_path)
    info = spikeglx_meta_channel_info(meta)
    cfg = pre._collect_cfg()
    pre._prepare_step_status_panel(parsed["run_name"], cfg)

    planned = [label for _key, label in pre._planned_step_definitions(cfg)]
    pre.log.clear()
    pre._append_log("README preprocessing preflight using a real SpikeGLX AP recording.")
    pre._append_log(f"Input AP binary: {RAW_AP_BIN}")
    pre._append_log(f"Meta sidecar: {meta_path}")
    pre._append_log(
        "Parsed run: "
        f"{parsed['run_name']} | gate {parsed['gate_string']} | "
        f"trigger {parsed['trigger_string']} | probe {parsed['probe_string']}"
    )
    pre._append_log(
        "Recording summary: "
        f"{_format_gb(RAW_AP_BIN)}, {int(info.get('ap', 0))} AP channels, "
        f"{int(info.get('lf', 0))} LF channels, {int(info.get('sy', 0))} sync channels, "
        f"{info.get('sample_rate', 0.0):.1f} Hz"
    )
    pre._append_log("Planned modules: " + ", ".join(planned))
    pre._append_log(
        "Queue is ready. Run queue will write the CatGT, Kilosort4, waveform, "
        "quality-metric, and Bombcell outputs under the configured processed-data root."
    )


def shot_preprocessing(win: NeuroPyGuiNMainWindow, app: QtWidgets.QApplication) -> None:
    print("[01] Preprocessing")
    print(f"  raw AP exists: {RAW_AP_BIN.exists()}")
    pre = win.pre_tab
    win.tabs.setCurrentWidget(pre)
    pump(app, 400)

    pre._clear()
    pre._add_paths([str(RAW_AP_BIN)])
    if pre.jobs:
        pre._select_job_in_queue(str(RAW_AP_BIN))
    _set_preprocessing_demo_steps(pre)
    pump(app, 500)
    pre.work_sections.setCurrentIndex(0)
    grab(win, "01_preprocessing_queue.png", app)

    _populate_preprocessing_run_plan(pre)
    pre.work_sections.setCurrentIndex(int(pre.log_section_index))
    pump(app, 500)
    grab(win, "01_preprocessing.png", app)


def populate_curation(cur, app: QtWidgets.QApplication) -> None:
    cur.set_plot_preferences("Light", True)
    cur.set_ks_folders([str(KS_FOLDER)])
    pump(app, 500)
    cur._main_sections.setCurrentIndex(1)
    cur._load_metrics(allow_compute=False)
    pump(app, 900)
    phy_labels = KS_FOLDER / "cluster_group.tsv"
    if phy_labels.exists():
        labels = pd.read_csv(phy_labels, sep="\t")
        if {"cluster_id", "group"}.issubset(labels.columns):
            labels["bombcell_label"] = labels["group"].astype(str).str.lower().map(
                {
                    "good": "good",
                    "mua": "mua",
                    "noise": "noise",
                    "unsorted": "noise",
                }
            ).fillna("noise")
            labels = labels.set_index("cluster_id")[["bombcell_label"]]
            cur._set_preview_labels(labels, source=f"Phy labels: {phy_labels}")
            cur._log(f"Loaded Phy labels for README screenshot: {phy_labels}")
            pump(app, 300)
    wanted = [
        "firing_rate",
        "presence_ratio",
        "isi_viol",
        "amplitude_cutoff",
        "snr",
        "isolation_distance",
        "d_prime",
        "nn_hit_rate",
    ]
    available = {cur.list_metrics.item(i).text(): i for i in range(cur.list_metrics.count())}
    cur.list_metrics.clearSelection()
    selected = []
    for metric in wanted:
        row = available.get(metric)
        if row is not None:
            cur.list_metrics.item(row).setSelected(True)
            selected.append(metric)
    print(f"  curation metrics selected: {selected}")
    cur._refresh_metric_plot()


def shot_curation(win: NeuroPyGuiNMainWindow, app: QtWidgets.QApplication) -> None:
    print("[02/04] Curation")
    print(f"  KS exists: {KS_FOLDER.exists()}")
    cur = win.cur_tab
    win.tabs.setCurrentWidget(cur)
    pump(app, 400)
    populate_curation(cur, app)
    cur._bomb_subsections.setCurrentIndex(4)
    grab(win, "02_curation.png", app)

    cur._bomb_subsections.setCurrentIndex(1)
    preferred_units = ["243", "250", "253", "255", "260", "268"]
    unit_lists = [cur.list_good, cur.list_mua, cur.list_noise, cur.list_non_soma]
    selected_widget = None
    selected_item = None
    for list_widget in unit_lists:
        for unit in preferred_units:
            matches = list_widget.findItems(unit, QtCore.Qt.MatchExactly)
            if matches:
                selected_widget = list_widget
                selected_item = matches[0]
                break
        if selected_item is not None:
            break
    if selected_item is None:
        for list_widget in unit_lists:
            if list_widget.count():
                selected_widget = list_widget
                selected_item = list_widget.item(0)
                break
    if selected_widget is not None and selected_item is not None:
        cur.tabs_units.setCurrentWidget(selected_widget)
        selected_widget.setCurrentItem(selected_item)
        selected_item.setSelected(True)
        cur._on_unit_selection_changed(selected_widget)
    else:
        cur._refresh_unit_inspector()
    pump(app, 500)
    grab(win, "04_curation_units.png", app)


def shot_postprocessing(win: NeuroPyGuiNMainWindow, app: QtWidgets.QApplication) -> None:
    print("[03] Post Processing")
    print(f"  KS exists: {KS_FOLDER.exists()}")
    post = win.post_tab
    win.tabs.setCurrentWidget(post)
    pump(app, 400)
    post.open_ks_folder(str(KS_FOLDER))
    pump(app, 1200)

    try:
        post.cb_good_source.setCurrentText("Phy")
        post.btn_good_only.setChecked(True)
        post._refresh_units_list()
    except Exception as exc:
        print(f"  good-unit filter warning: {exc}")

    selected = select_list_items(post.list_units, POST_CORR_UNITS)
    print(f"  post units selected: {selected}")
    try:
        post.analysis_tabs.setCurrentIndex(2)
        post.cb_corr_mode.setCurrentText("ACG")
        post.cb_corr_norm.setCurrentText("Hertz")
        post.cb_corr_style.setCurrentText("bar")
        post.sp_corr_bin.setValue(1.0)
        post.sp_corr_win.setValue(100.0)
        post._set_settings_visible(True)
        post._refresh_current_page()
    except Exception as exc:
        print(f"  correlogram warning: {exc}")
    pump(app, 2200)
    grab(win, "03_postprocessing.png", app, settle_ms=1000)


def stage_histology_workdir() -> Path:
    work = Path(tempfile.mkdtemp(prefix="readme_histo_"))
    for name in [
        "probe_ccf.mat",
        "histology_ccf.mat",
        "atlas2histology_tform.mat",
        "channels.localCoordinates.npy",
        "xyz_picks_shank1.json",
        "xyz_picks_shank2.json",
        "xyz_picks_shank3.json",
        "xyz_picks_shank4.json",
        "prev_alignments_shank1.json",
        "prev_alignments_shank2.json",
        "prev_alignments_shank3.json",
        "channel_locations_all_shanks.json",
        "clusters.channels.npy",
        "clusters.depths.npy",
        "clusters.amps.npy",
        "spikes.clusters.npy",
        "spikes.times.npy",
        "spikes.depths.npy",
    ]:
        src = HISTO_SESSION / name
        if src.exists():
            shutil.copy(src, work / name)
    images = sorted((HISTO_SESSION / "images").glob("*.tif")) if (HISTO_SESSION / "images").exists() else []
    if images:
        shutil.copy(images[0], work / "slice_1.tif")
    return work


def shot_histology(win: NeuroPyGuiNMainWindow, app: QtWidgets.QApplication) -> None:
    print("[05-09] Histology")
    print(f"  histology exists: {HISTO_SESSION.exists()}  atlas exists: {ATLAS_PATH.exists()}")
    from neuropyguin.histology import io_formats, tracing

    work = stage_histology_workdir()
    hist = win.hist_tab
    win.tabs.setCurrentWidget(hist)
    pump(app, 400)
    try:
        hist.ed_atlas.setText(str(ATLAS_PATH))
        hist.open_histology_folder(str(work))
        pump(app, 700)

        hist.nav.setCurrentIndex(0)
        grab(win, "05_histology_setup.png", app)

        hist.nav.setCurrentIndex(2)
        hist._ensure_atlas()
        hist.sl_ap.setValue(820)
        hist.cb_mode.setCurrentText("AV")
        hist._match_show()
        grab(win, "06_histology_match.png", app, settle_ms=1000)

        hist.nav.setCurrentIndex(4)
        atlas = hist._ensure_atlas()
        points = io_formats.load_probe_ccf_points(HISTO_SESSION / "probe_ccf.mat")
        probes = []
        for pts in points:
            pts = np.asarray(pts, dtype=float)
            areas, coords = tracing.trajectory_areas_from_points(pts, atlas)
            probes.append({"points": pts, "trajectory_areas": areas, "trajectory_coords": coords})
        hist._draw_trajectory_areas(probes)
        print(f"  traced probes: {len(probes)}")
        grab(win, "07_histology_trace.png", app, settle_ms=1000)

        hist.nav.setCurrentIndex(5)
        hist.cb_alignment.setCurrentText("latest")
        hist._load_channel_table()
        hist._plot_unit_distribution()
        for _ in range(50):
            pump(app, 200)
            if int(getattr(hist, "_busy_count", 0)) <= 0:
                break
        grab(win, "08_histology_channels.png", app, settle_ms=900)

        hist.nav.setCurrentIndex(6)
        grab(win, "09_ibl_alignment.png", app)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def save_settings_snapshot() -> dict[str, object]:
    settings = QtCore.QSettings("NeuroPyGuiN", "NeuroPyGuiN")
    return {key: settings.value(key) for key in settings.allKeys()}


def restore_settings_snapshot(snapshot: dict[str, object]) -> None:
    settings = QtCore.QSettings("NeuroPyGuiN", "NeuroPyGuiN")
    settings.clear()
    for key, value in snapshot.items():
        settings.setValue(key, value)
    settings.sync()


def main() -> int:
    which = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    snapshot = save_settings_snapshot()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    style_app(app, "Light")
    win = build_window(app)
    try:
        if which in ("all", "pre"):
            shot_preprocessing(win, app)
        if which in ("all", "cur"):
            shot_curation(win, app)
        if which in ("all", "post"):
            shot_postprocessing(win, app)
        if which in ("all", "histo"):
            shot_histology(win, app)
        if which == "ibl":
            shot_histology(win, app)
    finally:
        win.hide()
        restore_settings_snapshot(snapshot)
    print(f"\nDone. Screenshots in: {SHOTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

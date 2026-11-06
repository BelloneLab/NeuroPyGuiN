"""Regression coverage for scientific figure integrity and workbench exports."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.colors import to_rgba

from neuropyguin.publication import export_figure_bundle, publication_canvas
from neuropyguin.unit_figures import (
    _extract_spike_waveforms, _session_duration_s, _draw_firing_rate, _Theme,
    unit_basics_figure, network_figure,
)


def test_publication_export_keeps_data_and_restores_dark_theme(tmp_path):
    """White-paper export must not mutate the user's plotted data or theme."""
    figure = Figure(figsize=(3, 2))
    figure.set_facecolor("#0b0f14")
    axis = figure.subplots()
    axis.set_facecolor("#0b0f14")
    axis.set_title("Example", color="#e8eef7")
    line, = axis.plot([0, 1], [3, 2], color="#0072B2")
    before = figure.get_facecolor()
    paths = export_figure_bundle(figure, tmp_path, "example", tables=[("values.csv", pd.DataFrame({"x": [0, 1]}))])
    assert {p.suffix for p in paths} == {".pdf", ".png", ".svg", ".csv", ".json"}
    assert all(path.stat().st_size > 0 for path in paths)
    from PIL import Image
    assert abs(Image.open(tmp_path / "example.png").width - 183 / 25.4 * 600) < 2
    assert figure.get_facecolor() == before
    assert np.array_equal(line.get_ydata(), [3, 2])
    assert axis.title.get_color() == "#e8eef7"
    second = export_figure_bundle(figure, tmp_path, "example")
    assert second[0].name == "example_2.pdf"
    assert "<text" in (tmp_path / "example.svg").read_text()


def test_publication_restores_theme_when_saving_raises():
    figure = Figure()
    figure.set_facecolor("#0b0f14")
    try:
        with publication_canvas(figure):
            assert figure.get_facecolor() == to_rgba("white")
            raise RuntimeError("disk failure")
    except RuntimeError:
        pass
    assert figure.get_facecolor() == to_rgba("#0b0f14")


def test_waveform_extraction_respects_channel_map(tmp_path):
    """Template column 0 maps to file channel 2, not file channel 0."""
    path = tmp_path / "raw.bin"
    raw = np.zeros((100, 3), dtype=np.int16)
    spikes = np.array([20, 40, 60, 80])
    raw[spikes, 0] = 11
    raw[spikes, 2] = 73
    raw.tofile(path)
    dataset = SimpleNamespace(ap_bin_path=path, sample_rate=1000, n_channels=3,
                              bit_uV=2, spike_times=spikes, spike_clusters=np.ones(4),
                              channel_map=np.array([2, 0, 1]))
    _, mean, sem, used = _extract_spike_waveforms(dataset, 1, np.array([0]), half_ms=4)
    assert used == 4
    assert mean[4, 0] == 146
    assert sem[4, 0] == 0


def test_duration_uses_recording_tail_and_rate_partial_bin(tmp_path):
    path = tmp_path / "raw.bin"
    np.zeros((1500, 1), dtype=np.int16).tofile(path)
    dataset = SimpleNamespace(ap_bin_path=path, sample_rate=1000, n_channels=1,
                              spike_times=np.array([100, 1200]),
                              unit_spike_times_s=lambda unit: np.array([0.1, 1.2]))
    assert _session_duration_s(dataset) == 1.5
    axis = Figure().subplots()
    _draw_firing_rate(axis, dataset, [1], _Theme(False), bin_s=1)
    assert np.allclose(axis.lines[0].get_ydata(), [1, 2])


def test_unit_window_controls_have_visible_effect(tmp_path, monkeypatch):
    """Previously the raster window and smoothing controls were silently ignored."""
    from neuropyguin import unit_figures
    monkeypatch.setattr(unit_figures, "_draw_acg", lambda *args, **kwargs: None)
    dataset = SimpleNamespace(ap_bin_path=None, spike_times=np.array([100, 500, 1200, 1800]),
                              sample_rate=1000, channel_positions=None, channel_map=None,
                              mean_template_waveform=lambda unit: None,
                              unit_spike_times_s=lambda unit: np.array([0.1, 0.5, 1.2, 1.8]),
                              spike_clusters=np.ones(4), ks_folder=tmp_path)
    figure = unit_basics_figure(dataset, [1], window_start_s=1, window_s=0.5, show_ifr=True)
    raster = [axis for axis in figure.axes if axis.get_ylabel() == "unit"][0]
    assert raster.get_xlim() == (1, 1.5)
    assert len(raster.collections[0].get_segments()) == 1
    assert any(axis.get_ylabel() == "unit 1 rate (Hz)" for axis in figure.axes)


def test_network_does_not_claim_synaptic_significance():
    figure = network_figure({"units": [1, 2], "corr_matrix": np.eye(2),
                             "connections": np.zeros((2, 2)), "n_significant": 2})
    text = " ".join(axis.get_title() for axis in figure.axes) + figure._suptitle.get_text()
    assert "significant" not in text.lower()
    assert "threshold crossings" in text


def test_unit_filter_batches_selection_and_preserves_analysis_navigation(tmp_path, monkeypatch):
    """A large filter operation emits one refresh, rather than one per unit."""
    from PySide6 import QtCore, QtWidgets
    from neuropyguin.tabs.postprocessing_tab import PostProcessingTab
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    settings = QtCore.QSettings(str(tmp_path / "settings.ini"), QtCore.QSettings.IniFormat)
    monkeypatch.setattr(QtCore, "QSettings", lambda *args: settings)
    widget = PostProcessingTab(QtCore.QThreadPool.globalInstance())
    widget._all_units = list(range(2000))
    calls = []
    monkeypatch.setattr(widget, "_schedule_refresh", lambda: calls.append("refresh"))
    widget._refresh_units_list()
    assert calls == ["refresh"]
    assert widget._selected_units() == [0]
    assert widget.analysis_tabs.count() == widget.view_tabs.count() == 8
    widget.analysis_tabs.setCurrentIndex(7)
    assert widget.view_tabs.currentIndex() == 7
    assert widget.settings_panel.isHidden()
    widget.close()


def test_dataset_reload_clears_cached_analyses(tmp_path, monkeypatch):
    """Results and export tables must never leak between recordings."""
    from PySide6 import QtCore, QtWidgets
    from neuropyguin.tabs import postprocessing_tab
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    settings = QtCore.QSettings(str(tmp_path / "settings.ini"), QtCore.QSettings.IniFormat)
    monkeypatch.setattr(QtCore, "QSettings", lambda *args: settings)
    widget = postprocessing_tab.PostProcessingTab(QtCore.QThreadPool.globalInstance())
    dataset = SimpleNamespace(ks_folder=tmp_path, units=np.array([2, 4]), sample_rate=1000,
                              spike_times=np.array([1, 2000]), channel_map=None,
                              mean_template_waveform=lambda unit: None)
    monkeypatch.setattr(postprocessing_tab.NeuropixelsDataset, "load", lambda folder: dataset)
    monkeypatch.setattr(widget.population_widget, "set_dataset", lambda *args: None)
    monkeypatch.setattr(widget, "_refresh_current_page", lambda: None)
    widget.results = {"psth": "old recording"}
    widget._export_payloads = {"network": ["old recording"]}
    widget.ed_folder.setText(str(tmp_path))
    widget._load_dataset()
    assert widget.results == {}
    assert widget._export_payloads == {}
    assert widget._selected_units() == [2]
    widget._refresh_timer.stop()
    widget.close()


def test_waveform_summary_cache_invalidates_for_calibration(tmp_path, monkeypatch):
    from neuropyguin import unit_figures
    path = tmp_path / "raw.bin"
    path.write_bytes(b"\0\0" * 100)
    dataset = SimpleNamespace(ap_bin_path=path, sample_rate=1000, bit_uV=1,
                              n_channels=1, channel_map=np.array([0]))
    reads = []
    monkeypatch.setattr(unit_figures, "_read_spike_waveforms",
                        lambda *args, **kwargs: reads.append(1) or (None, None, None, 0))
    unit_figures._extract_spike_waveforms(dataset, 1, np.array([0]))
    unit_figures._extract_spike_waveforms(dataset, 1, np.array([0]))
    assert len(reads) == 1
    dataset.bit_uV = 2
    unit_figures._extract_spike_waveforms(dataset, 1, np.array([0]))
    assert len(reads) == 2


def test_background_analysis_freezes_inputs_until_completion(tmp_path, monkeypatch):
    from PySide6 import QtCore, QtWidgets
    from neuropyguin.tabs.postprocessing_tab import PostProcessingTab
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    settings = QtCore.QSettings(str(tmp_path / "settings.ini"), QtCore.QSettings.IniFormat)
    monkeypatch.setattr(QtCore, "QSettings", lambda *args: settings)
    widget = PostProcessingTab(QtCore.QThreadPool.globalInstance())
    # Hold the task without executing it to inspect the exact in-flight UI state.
    monkeypatch.setattr(widget.pool, "start", lambda worker: None)
    widget._start_analysis("psth", lambda: {})
    assert not widget.list_units.isEnabled()
    assert not widget.sp_psth_pre.isEnabled()
    assert not widget.btn_publication.isEnabled()
    assert widget.analysis_tabs.isEnabled()
    widget._on_analysis_finished({"ok": False})
    assert widget.list_units.isEnabled()
    assert widget.sp_psth_pre.isEnabled()
    assert widget.btn_publication.isEnabled()
    widget.close()


def test_isi_annotation_stays_outside_data_region():
    from neuropyguin.unit_figures import _draw_isi
    dataset = SimpleNamespace(unit_spike_times_s=lambda unit: np.arange(0, 1, .003))
    axis = Figure().subplots()
    _draw_isi(axis, dataset, 1, _Theme(False), max_ms=200)
    assert "RPV" in axis.get_title()
    assert not axis.texts

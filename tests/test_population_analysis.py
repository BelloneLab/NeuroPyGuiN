"""Regression tests for exposure, alignment, leakage protection and exports."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from neuropyguin.population_analysis import (
    PopulationConfig, align_continuous_behavior, analyze_population, bin_population,
    load_result, nemba_embedding, result_tables, save_result, temporal_decode,
)


def source(trains):
    """Small read-only dataset stand-in with explicit spike times in seconds."""
    return SimpleNamespace(units=np.arange(len(trains)), ks_folder=Path("/nonexistent/synthetic_test"),
                           unit_spike_times_s=lambda u: np.asarray(trains[u], float))


def test_complete_bins_are_half_open_and_rate_exposure_exact():
    dataset = source([[0, 0.1, 1, 1.05], [0.2, 0.9]])
    ids, times, counts = bin_population(dataset, [0, 1], PopulationConfig(stop_s=1.05))
    assert counts.shape == (10, 2)
    assert counts[:, 0].sum() == 2  # Spike at the excluded end cannot inflate rate.
    np.testing.assert_allclose(times, np.arange(0.05, 1, 0.1))
    assert counts[:, 1].sum() == 2


def test_binning_rejects_bad_inputs_and_oversized_allocations():
    dataset = source([[0.1], [0.2]])
    for config in [PopulationConfig(bin_s=0), PopulationConfig(stop_s=-1),
                   PopulationConfig(stop_s=1e9), PopulationConfig(start_s=np.nan)]:
        with pytest.raises(ValueError):
            bin_population(dataset, [0, 1], config)
    with pytest.raises(ValueError, match="absent"):
        bin_population(dataset, [0, 2], PopulationConfig())


def test_behavior_no_extrapolation_nan_bridging_or_long_gaps(tmp_path):
    path = tmp_path / "behavior.csv"
    pd.DataFrame({"time_s": [0, 1, 2, 3, 10, 11], "speed": [0, 2, np.nan, 6, 20, 22]}).to_csv(path, index=False)
    aligned = align_continuous_behavior(path, "speed", [-1, 0, 0.5, 1, 1.5, 2.5, 3, 5, 10, 11, 12], 1.1)
    np.testing.assert_allclose(aligned, [np.nan, 0, 1, 2, np.nan, np.nan, 6, np.nan, 20, 22, np.nan], equal_nan=True)
    pd.DataFrame({"time": [0, 0, 1], "speed": [0, 2, 3]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        align_continuous_behavior(path, "speed", [0.5])


def test_blocked_decoder_predicts_known_signal_and_beats_shift_null():
    rng = np.random.default_rng(45)
    X = rng.normal(size=(250, 5))
    y = X[:, 0] * 2 - X[:, 2] + rng.normal(scale=0.05, size=250)
    result = temporal_decode(X, y, gap_bins=4, permutations=19)
    assert result["r2"] > 0.98
    assert result["baseline_r2"] < 0.05
    assert result["p_shift"] <= 0.05
    assert len(result["folds"]) == 5
    assert result["folds"][2]["n_train"] == 192  # 250 - 50 test - 8 purge.
    assert result["n_evaluated"] == 250


def test_decoder_training_mean_is_fitted_without_test_bins():
    X = np.arange(100)[:, None]
    y = np.r_[np.zeros(80), np.full(20, 100.0)]
    result = temporal_decode(X, y, gap_bins=2, permutations=0)
    np.testing.assert_allclose(result["baseline"][80:], 0)
    assert result["p_shift"] is None


def test_constant_neurons_removed_and_safe_result_round_trip(tmp_path):
    dataset = source([np.arange(0.1, 10, .3), np.arange(.2, 10, .7), []])
    result = analyze_population(dataset, [0, 1, 2], PopulationConfig(stop_s=10))
    assert result["metadata"]["excluded_constant_units"] == [2]
    assert result["embedding"].shape == (100, 2)
    assert result["metadata"]["embedding_dimensions_used"] == 2
    save_result(result, tmp_path / "result.npz")
    loaded = load_result(tmp_path / "result.npz")
    np.testing.assert_array_equal(loaded["counts"], result["counts"])
    assert len(result_tables(loaded)) == 3


def test_nemba_adapter_uses_verified_api_and_preserves_clock(monkeypatch):
    calls = {}
    class Embedder:
        def __init__(self, backbone, **kwargs):
            calls["backbone"] = backbone
            calls["kwargs"] = kwargs
        def fit_matrix(self, X, *, times, counts):
            calls["times"] = times.copy()
            calls["counts"] = counts.copy()
            self.Z = X.T
    module = SimpleNamespace(Embedder=Embedder, __file__=__file__, __version__="test")
    import neuropyguin.population_analysis as analysis
    monkeypatch.setattr(analysis.importlib, "import_module", lambda name: module)
    counts = np.arange(30).reshape(10, 3) % 7
    times = np.arange(10) * 0.1 + 5
    Z, metadata = nemba_embedding(counts, times, PopulationConfig(method="nemba:tcn"))
    assert Z.shape == (10, 3)
    np.testing.assert_array_equal(calls["counts"], counts.T)
    np.testing.assert_array_equal(calls["times"], times)
    assert calls["kwargs"]["smooth_sigma"] == 0
    assert calls["kwargs"]["device"] == "cpu"


def test_population_figure_and_export_bundle(tmp_path):
    from neuropyguin.tabs.population_widget import population_figure
    from neuropyguin.publication import export_figure_bundle
    dataset = source([np.arange(0.1, 10, .3), np.arange(.2, 10, .7)])
    result = analyze_population(dataset, [0, 1], PopulationConfig(stop_s=10))
    fig = population_figure(result)
    paths = export_figure_bundle(fig, tmp_path, "population", tables=result_tables(result), metadata=result["metadata"])
    assert {p.suffix for p in paths} >= {".pdf", ".svg", ".png", ".csv", ".json"}
    assert all(p.stat().st_size > 0 for p in paths)


def test_widget_runs_isolated_worker_and_reenables_controls(tmp_path):
    """Exercise the actual subprocess protocol and result render without a desktop."""
    from PySide6 import QtCore, QtWidgets
    from neuropyguin.tabs.population_widget import PopulationWidget
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    rng = np.random.default_rng(12)
    seconds = np.sort(rng.uniform(0, 20, 1200))
    clusters = rng.integers(0, 6, len(seconds))
    samples = (seconds * 1000).astype(np.int64)
    np.save(tmp_path / "spike_times.npy", samples)
    np.save(tmp_path / "spike_clusters.npy", clusters)
    dataset = SimpleNamespace(units=np.arange(6), ks_folder=tmp_path, spike_times=samples, sample_rate=1000)
    widget = PopulationWidget()
    widget.set_dataset(dataset)
    widget.stop.setValue(20)
    widget._run()
    assert not widget.method.isEnabled()
    timer = QtCore.QElapsedTimer()
    timer.start()
    while widget._process is not None and timer.elapsed() < 15000:
        app.processEvents()
        QtCore.QThread.msleep(10)
    try:
        assert widget.result is not None
        assert not widget.busy
        assert widget.method.isEnabled()
        assert widget.export_button.isEnabled()
        assert widget.toolbar.canvas is widget.canvas
        assert widget.toolbar._id_press in widget.canvas.callbacks.callbacks["button_press_event"]
        assert widget.toolbar._id_release in widget.canvas.callbacks.callbacks["button_release_event"]
        # Picking an embedding point links the exact recording time to all
        # time axes while preserving live navigation callbacks.
        scatter = next(artist for ax in widget.figure.axes for artist in ax.collections
                       if hasattr(artist, "_population_indices"))
        widget._pick_embedding(SimpleNamespace(artist=scatter, ind=[0]))
        assert len(widget._cursor_lines) == 2
        assert "Selected time" in widget.status.text()
        assert widget.result["counts"].shape == (200, 6)
        assert widget.result["metadata"]["source_files"]
        assert widget.result["metadata"]["sample_rate_hz"] == 1000
    finally:
        widget.shutdown()
        widget.close()


def test_batched_null_matches_separately_refitted_shift_scores():
    """The fast multi-target solve must match independent ridge evaluations."""
    rng = np.random.default_rng(9)
    X = rng.normal(size=(100, 6))
    y = rng.normal(size=100)
    result = temporal_decode(X, y, gap_bins=3, permutations=5)
    expected = [temporal_decode(X, np.roll(y, int(shift)), gap_bins=3, permutations=0)["r2"]
                for shift in result["null_shifts_bins"]]
    np.testing.assert_allclose(result["null_scores"], expected, atol=1e-12)


def test_styled_compact_workspace_preserves_plot_height():
    """A realistic embedded viewport must retain paper space in the dark theme."""
    from PySide6 import QtWidgets
    from neuropyguin.styles import build_app_qss, build_app_palette
    from neuropyguin.tabs.population_widget import PopulationWidget
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    parent = QtWidgets.QWidget()
    parent.setPalette(build_app_palette("Dark"))
    parent.setStyleSheet(build_app_qss("Dark"))
    parent.resize(1050, 420)
    layout = QtWidgets.QVBoxLayout(parent)
    layout.setContentsMargins(0, 0, 0, 0)
    widget = PopulationWidget()
    layout.addWidget(widget)
    parent.show()
    app.processEvents()
    try:
        assert widget.canvas.height() >= 520
        assert widget.paper_scroll.height() >= 240
        assert widget.toolbar.height() == 32
        assert widget.toolbar.widgetForAction(widget.toolbar._actions["home"]).height() >= 18
        # Replacing the figure in an already visible parent keeps toolbar
        # buttons visible and large enough for their 18px icons.
        from matplotlib.figure import Figure
        widget._install_figure(Figure(figsize=(8, 6)))
        app.processEvents()
        assert widget.toolbar.isVisible()
        assert widget.toolbar.widgetForAction(widget.toolbar._actions["home"]).height() >= 18
        assert parent.height() <= 420
        widget._show_settings()
        app.processEvents()
        assert widget.configuration.isVisible()
        assert widget.canvas.height() >= 520
        assert widget.paper_scroll.height() >= 240
        widget.configuration.hide()
    finally:
        parent.close()

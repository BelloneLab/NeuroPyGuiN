"""Dataset isolation, responsive review, and reproducible queue regressions."""
from pathlib import Path
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtWidgets

from neuropyguin.ks_output_resolver import KILOSORT_REQUIRED_FILES
from neuropyguin.tabs import curation_tab as curation
from neuropyguin.tabs import preprocessing_tab as preprocessing


@pytest.fixture(scope="module")
def application():
    """Reuse the process application without touching desktop settings."""
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def review(application, tmp_path, monkeypatch):
    """Create an isolated QC tab with a temporary preferences store."""
    original_settings = QtCore.QSettings
    monkeypatch.setattr(QtCore, "QSettings", lambda *args: original_settings(str(tmp_path / "settings.ini"), original_settings.IniFormat))
    monkeypatch.setattr(curation, "summarize_saved_pybombcell_results", lambda *args, **kwargs: {})
    widget = curation.CurationTab(QtCore.QThreadPool())
    yield widget
    widget.close()
    widget.deleteLater()
    application.processEvents()


def _recording(root: Path, name: str) -> Path:
    """Write minimal synthetic sorter output, never real recording data."""
    folder = root / name
    folder.mkdir()
    np.save(folder / "spike_times.npy", np.array([1, 2, 3]))
    np.save(folder / "spike_clusters.npy", np.array([0, 0, 1]))
    for name in KILOSORT_REQUIRED_FILES:
        if not (folder / name).exists():
            np.save(folder / name, np.zeros(1))
    (folder / "params.py").write_text("sample_rate = 20000\n")
    return folder


def test_empty_dataset_never_uses_working_directory(review, monkeypatch):
    """No-recording actions cannot silently inspect or process the project root."""
    monkeypatch.setattr(curation, "_preferred_metrics_file", lambda _: pytest.fail("Empty dataset must not resolve metrics"))
    review._load_metrics()
    review._run_pybombcell()
    assert not review.btn_load_metrics.isEnabled()
    assert not review.btn_save_labels.isEnabled()
    assert review._busy_count == 0


def test_dataset_switch_clears_metrics_labels_and_file_watchers(review, tmp_path):
    """Programmatic navigation must never save the preceding recording's labels."""
    first = _recording(tmp_path, "first")
    second = _recording(tmp_path, "second")
    review.set_ks_folders([str(first)])
    review.metrics_df = pd.DataFrame({"firing_rate": [3.]}, index=[0])
    review._metrics_folder = first.resolve()
    review._set_preview_labels(pd.DataFrame({"bombcell_label": ["good"]}, index=[0]), source="test")
    review.watcher.addPath(str(first / "params.py"))
    assert review.btn_save_labels.isEnabled()
    review.set_ks_folders([str(second)], preserve_existing=True)
    assert review.metrics_df.empty
    assert review.preview_labels.empty
    assert review.watcher.files() == []
    assert not review.btn_save_labels.isEnabled()


def test_refresh_preserves_batch_selection(review, tmp_path):
    """Refreshing batch summaries keeps every explicitly selected recording."""
    folders = [_recording(tmp_path, name) for name in ("first", "second", "third")]
    review.set_ks_folders([str(path) for path in folders])
    review.list_ks_folders.selectAll()
    review._refresh_folder_item_texts()
    assert set(review._selected_ks_folders()) == {str(path) for path in folders}


def test_select_all_metrics_redraws_only_once(review, monkeypatch):
    """Batch metric selection has one redraw irrespective of the metric count."""
    review.list_metrics.addItems([f"metric_{i}" for i in range(20)])
    redraws = []
    monkeypatch.setattr(review, "_refresh_metric_plot", lambda: redraws.append(True))
    review._select_all_metrics()
    assert len(review.list_metrics.selectedItems()) == 20
    assert len(redraws) == 1


def test_histogram_accepts_missing_and_infinite_metrics(review):
    """Non-finite QC values are missing observations, not invalid histogram ranges."""
    review.metrics_df = pd.DataFrame({"rate": [1., 2., np.inf, np.nan]})
    review.list_metrics.addItem("rate")
    review.list_metrics.item(0).setSelected(True)
    review._refresh_metric_plot()
    assert review.metrics_grid.ci.items


def test_queue_clear_and_remove_are_rejected_while_running():
    """An active worker's queue cannot be silently cleared through a menu action."""
    class State:
        _running = True
        _concatenating = False
        jobs = [{"name": "keep"}]
        messages = []
        def _append_log(self, text):
            self.messages.append(text)
    state = State()
    preprocessing.PreprocessingTab._clear(state)
    preprocessing.PreprocessingTab._remove_selected(state)
    assert state.jobs == [{"name": "keep"}]
    assert len(state.messages) == 2


def test_queue_keeps_original_settings_and_retries_only_failed_runs(application, tmp_path, monkeypatch):
    """Changing controls mid-queue cannot change later jobs or rerun successes."""
    original_settings = QtCore.QSettings
    monkeypatch.setattr(QtCore, "QSettings", lambda *args: original_settings(str(tmp_path / "queue.ini"), original_settings.IniFormat))
    monkeypatch.setattr(preprocessing.PreprocessingTab, "_offer_missing_tools", lambda self: None)
    monkeypatch.setattr(preprocessing.PreprocessingTab, "_refresh_tool_status", lambda self: [])
    class Pool:
        """Capture actual workers without invoking external scientific programs."""
        def __init__(self):
            self.workers = []
        def start(self, worker):
            self.workers.append(worker)
    pool = Pool()
    widget = preprocessing.PreprocessingTab(pool)
    widget.ed_output.setText(str(tmp_path / "output"))
    first = {"name": "first", "bin_file": str(tmp_path / "first_g0_t0.imec0.ap.bin")}
    second = {"name": "second", "bin_file": str(tmp_path / "second_g0_t0.imec0.ap.bin")}
    for job in (first, second):
        job.update(gate_string="0", trigger_string="0", probe_string="0")
    widget.jobs = [first, second]
    widget._run_queue()
    original_region = pool.workers[0].cfg.region_name
    widget.ed_region.setText("changed during run")
    widget._on_job_finished({"ok": True, "job": "first", "ks_folder": str(tmp_path / "sorted")})
    assert len(pool.workers) == 2
    assert pool.workers[1].cfg.region_name == original_region
    assert widget.jobs == [second]
    widget._on_job_finished({"ok": False, "job": "second"})
    assert widget.jobs == [second]
    assert not widget._running
    widget.close()
    widget.deleteLater()
    application.processEvents()

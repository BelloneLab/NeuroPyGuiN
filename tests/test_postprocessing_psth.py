from __future__ import annotations

from pathlib import Path

import numpy as np

from neuropyguin.postproc_engine import NeuropixelsDataset, cluster_synced_units
from neuropyguin.postproc_events import inspect_event_csv, load_event_times


def _dataset(tmp_path: Path, spike_times: np.ndarray, spike_clusters: np.ndarray) -> NeuropixelsDataset:
    return NeuropixelsDataset(
        ks_folder=tmp_path,
        sample_rate=1000.0,
        n_channels=1,
        bit_uV=1.0,
        ap_bin_path=None,
        spike_times=np.asarray(spike_times, dtype=float),
        spike_clusters=np.asarray(spike_clusters, dtype=int),
        units=np.unique(np.asarray(spike_clusters, dtype=int)),
        channel_map=None,
        channel_positions=None,
        templates=None,
        spike_templates=None,
    )


def test_inspect_event_csv_detects_event_type_and_onset_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "events.csv"
    csv_path.write_text(
        "event_type,cue_onset_s\n"
        "cue_door_mouse_interact,374.68\n"
        "cue_door_mouse_no_interact,430.4\n"
        "cue_door_mouse_interact,498.28\n",
        encoding="utf-8",
    )

    info = inspect_event_csv(csv_path)

    assert info["time_column"] == "cue_onset_s"
    assert info["label_column"] == "event_type"
    assert info["labels"] == ["cue_door_mouse_interact", "cue_door_mouse_no_interact"]


def test_load_event_times_filters_selected_event_label(tmp_path: Path) -> None:
    csv_path = tmp_path / "events.csv"
    csv_path.write_text(
        "event_type,cue_onset_s\n"
        "a,1.0\n"
        "b,2.0\n"
        "a,3.5\n",
        encoding="utf-8",
    )

    values = load_event_times(csv_path, selected_label="a").to_numpy(dtype=float)

    assert np.allclose(values, np.array([1.0, 3.5], dtype=float))


def test_psth_counts_zero_spike_trials_in_denominator(tmp_path: Path) -> None:
    ds = _dataset(tmp_path, spike_times=np.array([105.0]), spike_clusters=np.array([1]))

    t_ms, rate = ds.psth([1], np.array([0.1, 1.1]), pre_s=0.0, post_s=0.02, bin_ms=10.0)

    assert np.allclose(t_ms, np.array([5.0, 15.0], dtype=float))
    assert np.allclose(rate, np.array([50.0, 0.0], dtype=float))


def test_psth_trials_returns_one_row_per_trial(tmp_path: Path) -> None:
    ds = _dataset(tmp_path, spike_times=np.array([105.0, 215.0]), spike_clusters=np.array([1, 1]))

    t_ms, trial_mat = ds.psth_trials(1, np.array([0.1, 0.2]), pre_s=0.0, post_s=0.02, bin_ms=10.0)

    assert np.allclose(t_ms, np.array([5.0, 15.0], dtype=float))
    assert trial_mat.shape == (2, 2)
    assert np.allclose(trial_mat[0], np.array([100.0, 0.0], dtype=float))
    assert np.allclose(trial_mat[1], np.array([0.0, 100.0], dtype=float))


def test_psth_by_unit_returns_per_unit_heatmap_rows(tmp_path: Path) -> None:
    ds = _dataset(
        tmp_path,
        spike_times=np.array([105.0, 115.0]),
        spike_clusters=np.array([1, 2]),
    )

    t_ms, unit_ids, mat = ds.psth_by_unit([1, 2], np.array([0.1]), pre_s=0.0, post_s=0.02, bin_ms=10.0)

    assert np.allclose(t_ms, np.array([5.0, 15.0], dtype=float))
    assert unit_ids == [1, 2]
    assert mat.shape == (2, 2)
    assert np.allclose(mat[0], np.array([100.0, 0.0], dtype=float))
    assert np.allclose(mat[1], np.array([0.0, 100.0], dtype=float))


def test_psth_by_unit_averages_trials_within_each_unit(tmp_path: Path) -> None:
    ds = _dataset(
        tmp_path,
        spike_times=np.array([105.0, 215.0, 108.0, 208.0]),
        spike_clusters=np.array([1, 1, 2, 2]),
    )

    t_ms, unit_ids, mat = ds.psth_by_unit([1, 2], np.array([0.1, 0.2]), pre_s=0.0, post_s=0.02, bin_ms=10.0)

    assert np.allclose(t_ms, np.array([5.0, 15.0], dtype=float))
    assert unit_ids == [1, 2]
    assert np.allclose(mat[0], np.array([50.0, 50.0], dtype=float))
    assert np.allclose(mat[1], np.array([100.0, 0.0], dtype=float))


def test_cluster_synced_units_groups_and_orders_strong_pairs() -> None:
    mat = np.array(
        [
            [0.0, 9.0, 1.0, 0.0],
            [8.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 7.0],
            [0.0, 0.0, 6.0, 0.0],
        ],
        dtype=float,
    )

    grouped = cluster_synced_units([10, 11, 12, 13], mat, threshold=5.0)

    assert grouped["sorted_units"].tolist() == [10, 11, 12, 13]
    assert grouped["group_labels"].tolist() == [1, 1, 2, 2]
    assert grouped["threshold"] == 5.0

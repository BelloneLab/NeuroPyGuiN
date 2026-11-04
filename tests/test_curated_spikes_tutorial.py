"""Small synthetic recordings test the tutorial without touching lab data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import loadmat

from tutorials.curated_spikes_tutorial import (
    analyze_session,
    calculate_firing_rates,
    find_sorter_folder,
    load_integer_vector,
    read_labels,
    read_recording_info,
)


@pytest.fixture
def example_sorter(tmp_path):
    """Build a tiny 320-second, 10-Hz fake recording with two saved clusters.

    These are test values, not real neural observations. The new current ID7
    replaces original ID0, so a relabeled/merged cluster cannot be selected by
    accidentally using spike_templates instead of spike_clusters.
    """
    sorter = tmp_path / "source" / "imec0_ks4"
    sorter.mkdir(parents=True)
    binary = sorter.parent / "run_g0_tcat.imec0.ap.bin"
    np.zeros((3200, 2), dtype=np.int16).tofile(binary)
    binary.with_suffix(".meta").write_text("imSampRate=10\nnSavedChans=2\n", encoding="utf-8")
    (sorter / "params.py").write_text(
        f"dat_path = {str(binary)!r}\nsample_rate = 10\nn_channels_dat = 2\ndtype = 'int16'\n",
        encoding="utf-8",
    )
    np.save(sorter / "spike_times.npy", np.array([0, 10, 2999, 3000, 3100], dtype=np.int64))
    np.save(sorter / "spike_clusters.npy", np.array([7, 2, 7, 7, 2], dtype=np.int32))
    np.save(sorter / "spike_templates.npy", np.array([0, 2, 0, 0, 2], dtype=np.int32))
    # ID99 has no spikes: a stale good label must not inflate the neuron count.
    (sorter / "cluster_group.tsv").write_text(
        "cluster_id\tgroup\n2\tmua\n7\tgood\n99\tgood\n", encoding="utf-8")
    (sorter / "cluster_KSLabel.tsv").write_text(
        "cluster_id\tKSLabel\n0\tgood\n2\tgood\n", encoding="utf-8")
    (sorter / "bombcell_labels.csv").write_text(
        "cluster_id,bombcell_label\n0,good\n2,mua\n", encoding="utf-8")
    return sorter


def test_pipeline_uses_current_ids_exports_matrix_and_preserves_inputs(example_sorter, tmp_path):
    """Round-trip both Python and MATLAB exports, including time endpoints."""
    before = {p: p.read_bytes() for p in example_sorter.iterdir() if p.is_file()}
    result = analyze_session(example_sorter, tmp_path / "outputs", make_figures=False)
    assert result["good_cluster_ids"].tolist() == [7]
    assert result["firing_rates_hz"].shape == (1, 300)
    assert result["spike_counts"].sum() == 2  # Spike at exactly300s is excluded.
    assert result["spike_counts"][0, 0] == 1
    assert result["spike_counts"][0, 299] == 1
    assert result["mean_firing_rates_hz"][0] == pytest.approx(2 / 300)
    assert result["good_spike_trains_s"][0].tolist() == [0, 299.9, 300]
    assert result["audit"]["n_spikes_with_changed_cluster_assignment"] == 3
    assert result["audit"]["n_current_ids_missing_from_kilosort"] == 1
    assert result["audit"]["inactive_labeled_cluster_ids"] == [99]
    output = result["output_dir"]
    with np.load(output / "firing_rates.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["firing_rates_hz"], result["firing_rates_hz"])
        assert data["time_origin"].item() == "first saved AP sample"
    matlab = loadmat(output / "firing_rates.mat")
    assert matlab["firing_rates_hz"].shape == (1, 300)
    assert matlab["good_cluster_ids"].shape == (1, 1)
    csv = pd.read_csv(output / "firing_rates.csv", index_col="cluster_id")
    assert csv.shape == (1, 300)
    assert csv.index.tolist() == [7]
    changed = pd.read_csv(output / "changed_vs_kilosort.csv")
    assert changed["cluster_id"].tolist() == [2]  # New ID7 has no same-ID baseline label.
    with np.load(output / "curated_good_spikes.npz", allow_pickle=False) as data:
        assert data["cluster_ids"].tolist() == [7, 7, 7]
    assert all(p.read_bytes() == content for p, content in before.items())


def test_noncontiguous_ids_and_internal_boundary():
    """A spike at1s belongs to the second bin; rows follow supplied IDs."""
    counts, rates, mean, edges = calculate_firing_rates(
        np.array([0, 9, 10, 19, 20]), np.array([42, 42, 7, 7, 42]),
        np.array([7, 42]), 10, 2, 1,
    )
    np.testing.assert_array_equal(counts, [[0, 2], [2, 0]])
    np.testing.assert_array_equal(rates, counts)
    np.testing.assert_array_equal(mean, [1, 1])
    np.testing.assert_array_equal(edges, [0, 1, 2])


def test_partial_last_bin_uses_actual_exposure():
    counts, rates, mean, edges = calculate_firing_rates(
        np.array([0, 10, 20, 24, 25]), np.ones(5, dtype=int), np.array([1]), 10, 2.5, 1)
    np.testing.assert_array_equal(counts, [[1, 1, 2]])
    np.testing.assert_array_equal(rates, [[1, 1, 4]])
    assert mean[0] == pytest.approx(4 / 2.5)
    assert not np.isclose(mean[0], rates.mean())  # Unweighted bin mean is wrong here.


def test_decimal_bin_boundary_has_no_floating_point_off_by_one():
    """0.3s is the start of bin3, not the end of bin2 at a 10Hz sample clock."""
    counts, _, _, _ = calculate_firing_rates(
        np.array([3, 10]), np.array([1, 1]), np.array([1]), 10, 1, 0.1)
    assert counts.sum() == 1
    assert counts[0, 3] == 1
    # The same rounding issue must not include a spike exactly at the endpoint.
    counts, _, _, _ = calculate_firing_rates(
        np.array([3]), np.array([1]), np.array([1]), 10, 0.3, 0.1)
    assert counts.sum() == 0


@pytest.mark.parametrize("duration,width,rate", [(0, 1, 10), (300, 0, 10), (300, 1, 0),
                                                  (float("nan"), 1, 10), (300, float("inf"), 10)])
def test_invalid_rate_parameters(duration, width, rate):
    with pytest.raises(ValueError, match="finite and positive"):
        calculate_firing_rates(np.array([0]), np.array([1]), np.array([1]), rate, duration, width)


def test_zero_good_neurons_preserves_time_axis():
    counts, rates, mean, _ = calculate_firing_rates(
        np.array([0]), np.array([1]), np.array([], dtype=int), 10, 300, 1)
    assert counts.shape == rates.shape == (0, 300)
    assert mean.shape == (0,)


def test_missing_manual_groups_never_falls_back_to_automatic(example_sorter, tmp_path):
    (example_sorter / "cluster_group.tsv").unlink()
    with pytest.raises(FileNotFoundError, match="cluster_group.tsv"):
        analyze_session(example_sorter, tmp_path / "outputs", make_figures=False)


def test_missing_one_manual_label_is_unsorted_not_automatic_good(example_sorter, tmp_path):
    (example_sorter / "cluster_group.tsv").write_text("cluster_id\tgroup\n7\tgood\n", encoding="utf-8")
    result = analyze_session(example_sorter, tmp_path / "outputs", make_figures=False)
    assert result["good_cluster_ids"].tolist() == [7]
    assert result["audit"]["unlabeled_active_cluster_ids"] == [2]


def test_duplicate_label_ids_rejected(tmp_path):
    path = tmp_path / "groups.tsv"
    path.write_text("cluster_id\tgroup\n7\tgood\n7\tnoise\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        read_labels(path, "group")


def test_vector_loader_preserves_one_spike_and_rejects_two_columns(tmp_path):
    path = tmp_path / "vector.npy"
    np.save(path, np.array([[3]], dtype=np.int32))
    assert load_integer_vector(path).shape == (1,)
    np.save(path, np.array([[1, 2], [3, 4]], dtype=np.int32))
    with pytest.raises(ValueError, match="1-D"):
        load_integer_vector(path)


def test_short_recording_fails_without_padding(example_sorter, tmp_path):
    with pytest.raises(ValueError, match="shorter"):
        analyze_session(example_sorter, tmp_path / "outputs", duration_s=400, make_figures=False)


def test_params_are_parsed_not_executed(example_sorter):
    marker = example_sorter / "must_not_be_created"
    with (example_sorter / "params.py").open("a", encoding="utf-8") as stream:
        stream.write(f"\nopen({str(marker)!r}, 'w').write('unsafe')\n")
    assert read_recording_info(example_sorter)["sample_rate_hz"] == 10
    assert not marker.exists()


def test_binary_partial_frames_rejected(example_sorter):
    info = read_recording_info(example_sorter)
    with Path(info["binary_path"]).open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(ValueError, match="whole number"):
        read_recording_info(example_sorter)


def test_ambiguous_parent_rejected(example_sorter):
    other = example_sorter.parent / "imec0_ks4_2"
    other.mkdir()
    np.save(other / "spike_times.npy", np.array([0]))
    np.save(other / "spike_clusters.npy", np.array([0]))
    with pytest.raises(ValueError, match="found 2"):
        find_sorter_folder(example_sorter.parent)


def test_output_inside_source_rejected(example_sorter):
    with pytest.raises(ValueError, match="outside the source"):
        analyze_session(example_sorter, example_sorter / "derived", make_figures=False)


def test_reruns_get_distinct_directories(example_sorter, tmp_path):
    a = analyze_session(example_sorter, tmp_path / "outputs", make_figures=False)
    b = analyze_session(example_sorter, tmp_path / "outputs", make_figures=False)
    assert a["output_dir"] != b["output_dir"]


def test_no_good_clusters_still_exports_empty_matrix_and_figures(example_sorter, tmp_path):
    (example_sorter / "cluster_group.tsv").write_text(
        "cluster_id\tgroup\n2\tmua\n7\tnoise\n", encoding="utf-8")
    result = analyze_session(example_sorter, tmp_path / "outputs", make_figures=True)
    assert result["firing_rates_hz"].shape == (0, 300)
    for extension in ("png", "pdf", "svg"):
        assert (result["output_dir"] / f"curation_and_firing_rates.{extension}").stat().st_size > 0

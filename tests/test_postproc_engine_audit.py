"""Numerical and data-integrity regressions for the postprocessing audit."""
from pathlib import Path
import numpy as np
import pytest
from neuropyguin.postproc_engine import NeuropixelsDataset


def dataset(folder, times, labels):
    """Build a tiny explicit synthetic recording without any external data."""
    return NeuropixelsDataset(Path(folder), 1000., 1, 1., None, np.asarray(times),
        np.asarray(labels), np.unique(labels), None, None, None, None)


def test_windowed_ccg_matches_exhaustive_pairs(tmp_path):
    rng = np.random.default_rng(81)
    times = np.sort(rng.uniform(0, 20_000, 800))
    labels = rng.integers(0, 2, times.size)
    ds = dataset(tmp_path, times, labels)
    for zero in (False, True):
        centers, actual = ds.correlogram(0, 1, 2, 50, zero, 'Counts')
        a, b = times[labels == 0] / 1000, times[labels == 1] / 1000
        pairs = (b[None, :] - a[:, None]).ravel()
        pairs = pairs[(pairs >= -.05) & (pairs <= .05)]
        if zero:
            pairs = pairs[np.abs(pairs) > 1e-12]
        expected = np.histogram(pairs, np.arange(-.05, .05 + .002, .002))[0]
        np.testing.assert_array_equal(actual, expected)


def test_unsorted_spikes_are_sorted_without_mutating_source(tmp_path):
    times = np.array([50, 10, 30])
    ds = dataset(tmp_path, times, [1, 1, 1])
    np.testing.assert_allclose(ds.unit_spike_times_s(1), [.01, .03, .05])
    np.testing.assert_array_equal(times, [50, 10, 30])
    assert not ds.unit_spike_times_s(1).flags.writeable


def test_singleton_load_and_relative_raw_path(tmp_path):
    np.save(tmp_path / 'spike_times.npy', np.array([[5]]))
    np.save(tmp_path / 'spike_clusters.npy', np.array([[2]]))
    np.save(tmp_path / 'channel_map.npy', np.array([[0]]))
    (tmp_path / 'params.py').write_text("sample_rate = 1000\ndat_path = 'signal.ap.bin'\nn_channels_dat = 1\n")
    (tmp_path / 'signal.ap.bin').write_bytes(b'\0' * 100)
    ds = NeuropixelsDataset.load(str(tmp_path))
    assert ds.spike_times.shape == (1,)
    assert ds.channel_map.shape == (1,)
    assert ds.ap_bin_path == tmp_path / 'signal.ap.bin'
    np.testing.assert_allclose(ds.unit_spike_times_s(2), [.005])


def test_recuration_invalidates_disk_cache(tmp_path):
    np.save(tmp_path / 'spike_times.npy', [10, 20, 100])
    np.save(tmp_path / 'spike_clusters.npy', [1, 1, 2])
    first = NeuropixelsDataset.load(str(tmp_path))
    first._cache_set('test', {}, {'value': np.array([4])})
    assert NeuropixelsDataset.load(str(tmp_path))._cache_get('test', {}) is not None
    np.save(tmp_path / 'spike_clusters.npy', [1, 2, 2])
    assert NeuropixelsDataset.load(str(tmp_path))._cache_get('test', {}) is None


def test_corrupt_and_unwritable_cache_are_optional(tmp_path):
    ds = dataset(tmp_path, [10, 20], [1, 1])
    ds.cache_dir.mkdir()
    ds._cache_path('test', {}).write_bytes(b'broken')
    assert ds._cache_get('test', {}) is None
    blocked = tmp_path / 'blocked'
    blocked.write_text('not a directory')
    ds.ks_folder = blocked
    ds._cache_set('test', {}, {'value': np.array([4])})
    assert ds._cache_get('test', {}) is None


def test_array_shape_and_rate_validation(tmp_path):
    with pytest.raises(ValueError, match='same length'):
        dataset(tmp_path, [1, 2], [1])
    ds = dataset(tmp_path, [1], [1])
    ds.sample_rate = 0
    with pytest.raises(ValueError, match='Sample rate'):
        ds.__post_init__()


def test_psth_windowed_equals_explicit_histograms(tmp_path):
    rng = np.random.default_rng(9)
    times = np.sort(rng.uniform(0, 20_000, 1000))
    ds = dataset(tmp_path, times, np.ones(times.size))
    events = np.array([1, 5, 12, 19])
    _, actual = ds.psth_trials(1, events, .2, .4, 10)
    edges = np.linspace(-.2, .4, 61)  # Exactly 60 bins; no phantom floating-point tail.
    expected = []
    for event in events:
        rel = times / 1000 - event
        expected.append(np.histogram(rel[(rel >= -.2) & (rel <= .4)], edges)[0] / .01)
    np.testing.assert_allclose(actual, expected)


def test_params_literals_comments_and_single_path_list(tmp_path):
    from neuropyguin.postproc_engine import _parse_params_py
    source = tmp_path / 'params.py'
    source.write_text("sample_rate = 1000.0 # Hz\ndat_path = [r'raw.ap.bin']\nn_channels_dat = 4\n")
    parsed = _parse_params_py(source)
    assert parsed == {'sample_rate': '1000.0', 'dat_path': 'raw.ap.bin', 'n_channels_dat': '4'}


def test_psth_partial_bin_uses_actual_exposure(tmp_path):
    ds = dataset(tmp_path, [101, 121], [1, 1])
    centers, rates = ds.psth_trials(1, np.array([.1]), 0, .025, 10)
    np.testing.assert_allclose(centers, [5, 15, 22.5])
    np.testing.assert_allclose(rates, [[100, 0, 200]])
    np.testing.assert_allclose(ds.psth([1], np.array([.1]), 0, .025, 10)[1], rates[0])


@pytest.mark.parametrize('pre,post,width', [(-1, 1, 10), (0, 0, 10), (0, 1, 0), (0, 1, float('nan'))])
def test_psth_rejects_invalid_windows(tmp_path, pre, post, width):
    ds = dataset(tmp_path, [101], [1])
    with pytest.raises(ValueError, match='PSTH'):
        ds.psth_trials(1, np.array([.1]), pre, post, width)

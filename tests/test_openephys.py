"""Open Ephys -> SpikeGLX staging: discovery, meta synthesis, geometry, events, paths."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from openephys_fake import (  # noqa: E402
    NP1_PN,
    NP2_4S_PN,
    FakeProbe,
    make_openephys_session,
    np2_4shank_positions,
)

from neuropyguin.ecephys_runtime import ensure_ecephys_on_sys_path  # noqa: E402
from neuropyguin.openephys import (  # noqa: E402
    OE_EVENTS_CSV_NAME,
    discover_openephys_recordings,
    gain_fields_for_bit_volts,
    is_openephys_staged_bin,
    openephys_provenance,
    stage_openephys_recording,
)
from neuropyguin.preprocessing import (  # noqa: E402
    discover_bin_files,
    parse_kilosort_params_dat_path,
    parse_spikeglx_bin_name,
    validate_spikeglx_ap_bin,
)

BIT_VOLTS = 0.1949999928474426


def _noise(n=6000, ch=384, seed=0):
    return np.random.default_rng(seed).integers(-60, 60, (n, ch), dtype=np.int16)


def _read_meta(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        k, v = line.split("=", 1)
        out[k.lstrip("~")] = v
    return out


@pytest.fixture()
def two_probe_session(tmp_path: Path):
    x2, y2, s2 = np2_4shank_positions()
    probes = [
        FakeProbe("ProbeA", NP1_PN, _noise(seed=1), ttl_samples=[1500, 4500]),
        FakeProbe("ProbeB", NP2_4S_PN, _noise(seed=2), x=x2, y=y2, shank=s2, with_lfp=False),
    ]
    sess = make_openephys_session(
        tmp_path / "rawData" / "mouse7", probes, ni_ttl_seconds=[0.05, 0.12], ts_offset_s=12.5, clock_scale=1.0002
    )
    return sess, probes


def test_discovery_finds_ap_streams_and_pairs_lfp(two_probe_session):
    sess, _ = two_probe_session
    recs, skipped = discover_openephys_recordings([sess.root])
    assert [(r.probe_label, r.probe_index) for r in recs] == [("ProbeA", 0), ("ProbeB", 1)]
    assert recs[0].lfp is not None and recs[0].lfp.sample_rate == 2500.0
    assert recs[1].lfp is None
    assert recs[0].run_name == "2024-05-01_12-00-00_e1r1"
    assert skipped == []


def test_discovery_accepts_dat_oebin_and_settings_paths(two_probe_session):
    sess, _ = two_probe_session
    for path in (sess.ap_dat["ProbeA"], sess.recording_dir / "structure.oebin", sess.record_node / "settings.xml"):
        recs, _ = discover_openephys_recordings([path])
        assert {r.probe_label for r in recs} == {"ProbeA", "ProbeB"}


def test_staging_links_data_without_copy_and_names_like_spikeglx(two_probe_session, tmp_path):
    sess, probes = two_probe_session
    recs, _ = discover_openephys_recordings([sess.root])
    staged = stage_openephys_recording(recs[0], tmp_path / "processed")
    assert staged.link_mode in {"hardlink", "symlink"}
    assert staged.ap_bin.name == "2024-05-01_12-00-00_e1r1_g0_t0.imec0.ap.bin"
    # Mirrors the rawData hierarchy under the stage root, SpikeGLX run-folder style.
    assert staged.ap_bin.parent.parent.parent == tmp_path / "processed" / "mouse7" / "2024-05-01_12-00-00"
    data = np.fromfile(staged.ap_bin, dtype=np.int16).reshape(-1, 384)
    np.testing.assert_array_equal(data, probes[0].data)
    assert validate_spikeglx_ap_bin(str(staged.ap_bin)) == (True, "")
    assert parse_spikeglx_bin_name(str(staged.ap_bin))["probe_string"] == "0"
    assert is_openephys_staged_bin(staged.ap_bin)
    assert staged.lf_bin is not None and staged.lf_bin.with_suffix(".meta").exists()
    prov = openephys_provenance(staged.ap_bin)
    assert prov["source_ap_dat"] == str(sess.ap_dat["ProbeA"])
    # Idempotent: re-staging reuses the existing link.
    again = stage_openephys_recording(recs[0], tmp_path / "processed")
    assert again.link_mode == "existing"


@pytest.mark.parametrize("probe_type", ["0", "2013", "21"])
def test_gain_fields_round_trip_bit_volts_in_ecephys_and_bombcell_formulas(probe_type):
    f = gain_fields_for_bit_volts(BIT_VOLTS, probe_type)
    vmax, vmin, imax, gain = (float(f["imAiRangeMax"]), float(f["imAiRangeMin"]), int(f["imMaxInt"]), float(f["imChan0apGain"]))
    ecephys = 1e6 * (vmax - vmin) / gain / (2 * imax)
    bombcell_gain = 80.0 if probe_type in {"21", "2013"} else gain
    bombcell = vmax * 1e6 / imax / bombcell_gain
    assert ecephys == pytest.approx(BIT_VOLTS, rel=1e-6)
    assert bombcell == pytest.approx(BIT_VOLTS, rel=2e-5)


def test_ecephys_reads_staged_meta_and_reproduces_geometry(two_probe_session, tmp_path):
    ensure_ecephys_on_sys_path()
    from ecephys_spike_sorting.common.SGLXMetaToCoords import MetaToCoords
    from ecephys_spike_sorting.scripts.helpers import SpikeGLX_utils

    sess, probes = two_probe_session
    recs, _ = discover_openephys_recordings([sess.root])
    for rec, probe in zip(recs, probes):
        staged = stage_openephys_recording(rec, tmp_path / "processed")
        ptype, fs, n_ch, _ref, uv, vpitch, hpitch, ncol, n_ap, n_sy, use_geom = SpikeGLX_utils.EphysParams(str(staged.ap_meta))
        assert (fs, n_ch, n_ap, n_sy, use_geom) == (30000.0, 384, 384, 0, True)
        assert uv == pytest.approx(BIT_VOLTS, rel=1e-6)
        x, y, shank, connected = MetaToCoords(staged.ap_meta, -1, destFullPath=str(tmp_path / f"{rec.probe_label}.mat"))
        pitch = 250.0 if probe.part_number == NP2_4S_PN else 0.0
        np.testing.assert_allclose(x + shank * pitch, probe.x)  # back to Open Ephys' across-shank x
        np.testing.assert_allclose(y, probe.y)
        np.testing.assert_array_equal(shank, probe.shank)
        assert connected.all()
        meta = _read_meta(staged.ap_meta)
        assert meta["imroTbl"].startswith(f"({meta['imDatPrb_type']},384)")


def test_ttl_events_are_mapped_onto_the_ap_clock(two_probe_session, tmp_path):
    sess, _ = two_probe_session
    recs, _ = discover_openephys_recordings([sess.root])
    staged = stage_openephys_recording(recs[0], tmp_path / "processed")
    assert staged.events_csv is not None and staged.events_csv.name == OE_EVENTS_CSV_NAME
    import pandas as pd

    ev = pd.read_csv(staged.events_csv)
    ap_rise = ev[(ev.edge == "rising") & ev.stream.str.contains("ProbeA")].time_s.to_numpy()
    ni_rise = ev[(ev.edge == "rising") & ev.stream.str.contains("NI-DAQ")].time_s.to_numpy()
    # 12.5 s timestamp offset and 200 ppm clock drift are both removed.
    np.testing.assert_allclose(ap_rise, [1500 / 30000, 4500 / 30000], atol=1e-6)
    np.testing.assert_allclose(ni_rise, [0.05, 0.12], atol=1e-6)
    assert set(ev.sync) == {"timestamps"}


def test_legacy_gui_05_layout_is_staged(tmp_path):
    data = _noise(seed=3)
    sess = make_openephys_session(tmp_path, [FakeProbe("ProbeA", NP1_PN, data, ttl_samples=[600])], legacy=True)
    recs, _ = discover_openephys_recordings([sess.root])
    assert len(recs) == 1 and recs[0].probe_index == 0 and recs[0].lfp is not None
    staged = stage_openephys_recording(recs[0], tmp_path / "out")
    meta = _read_meta(staged.ap_meta)
    assert meta["firstSample"] == "123456"
    import pandas as pd

    ev = pd.read_csv(staged.events_csv)
    assert ev.time_s.iloc[0] == pytest.approx(600 / 30000)
    assert set(ev.sync) == {"same_stream"}


def test_missing_settings_xml_falls_back_to_bank0_with_warning(tmp_path):
    sess = make_openephys_session(tmp_path, [FakeProbe("ProbeA", NP1_PN, _noise(seed=4))], write_settings=False)
    recs, _ = discover_openephys_recordings([sess.root])
    staged = stage_openephys_recording(recs[0], tmp_path / "out")
    assert any("settings.xml" in w for w in staged.warnings)
    meta = _read_meta(staged.ap_meta)
    first = meta["snsGeomMap"].split(")(")[1:5]
    assert first == ["0:27:0:1", "0:59:0:1", "0:11:20:1", "0:43:20:1"]


def test_non_neuropixels_streams_are_reported_not_staged(tmp_path):
    import json

    sess = make_openephys_session(tmp_path, [FakeProbe("ProbeA", NP1_PN, _noise(seed=5))])
    oebin = sess.recording_dir / "structure.oebin"
    data = json.loads(oebin.read_text())
    data["continuous"].append(
        {"folder_name": "Acquisition_Board-100.Rhythm Data/", "sample_rate": 30000.0,
         "source_processor_name": "Acquisition Board", "source_processor_id": 101, "num_channels": 16,
         "stream_name": "Rhythm Data", "channels": [{"channel_name": "CH1", "bit_volts": 0.195}]}
    )
    oebin.write_text(json.dumps(data))
    recs, skipped = discover_openephys_recordings([sess.root])
    assert len(recs) == 1
    assert any("not a Neuropixels stream" in reason for _w, reason in skipped)


def test_symlinked_bins_keep_their_staged_path(tmp_path):
    target = tmp_path / "raw" / "continuous.dat"
    target.parent.mkdir()
    target.write_bytes(b"\0" * 768)
    link = tmp_path / "stage" / "run_g0" / "run_g0_imec0" / "run_g0_t0.imec0.ap.bin"
    link.parent.mkdir(parents=True)
    os.symlink(target, link)
    assert discover_bin_files([str(tmp_path / "stage")]) == [str(link)]
    params = link.parent / "imec0_ks4" / "params.py"
    params.parent.mkdir()
    params.write_text(f"dat_path = '{link.as_posix()}'\n")
    assert parse_kilosort_params_dat_path(params) == str(link)


def test_ks4_helper_takes_channel_count_from_meta_not_schema_default(tmp_path):
    """The ks4_helper schema defaults n_chan_bin to 385; the meta must win."""
    ensure_ecephys_on_sys_path()
    pytest.importorskip("kilosort")
    from ecephys_spike_sorting.modules.ks4_helper.__main__ import _get_ks_params

    meta = tmp_path / "x.ap.meta"
    meta.write_text("nSavedChans=384\nimSampRate=30000.5\n")
    settings = _get_ks_params(meta, {"n_chan_bin": 385, "fs": 30000.0, "tmax": -1, "nblocks": 3}, b_seed=False)
    assert settings["n_chan_bin"] == 384
    assert settings["fs"] == 30000.5
    assert settings["nblocks"] == 3


def test_openephys_catgt_command_drops_extractors_and_gfix_only():
    from neuropyguin.openephys import openephys_catgt_command

    cmd = "-prb_fld -out_prb_fld -apfilter=butter,12,300,10000 -gfix=0.4,0.10,0.02 -xd=2,0,384,6,500 -xa=0,0,0,1,3,500"
    out, notes = openephys_catgt_command(cmd)
    assert out == "-prb_fld -out_prb_fld -apfilter=butter,12,300,10000"
    assert any("extractors" in n for n in notes) and any("-gfix" in n for n in notes)
    assert openephys_catgt_command("-prb_fld -apfilter=butter,12,300,10000") == ("-prb_fld -apfilter=butter,12,300,10000", [])


def test_c_waves_scale_estimator_recovers_uv_per_count():
    """mean_waveforms rescales C_Waves output when its built-in gain != meta bit_volts."""
    ensure_ecephys_on_sys_path()
    from ecephys_spike_sorting.modules.mean_waveforms.__main__ import _measure_uv_per_count

    rng = np.random.default_rng(0)
    n_t, n_ch, pre = 82, 8, 20
    shape = -np.exp(-0.5 * ((np.arange(n_t) - pre) / 3.0) ** 2) * 400  # counts
    data = rng.normal(0, 20, (200_000, n_ch)).astype(np.int16)
    times = np.sort(rng.choice(np.arange(100, 199_800), 600, replace=False))
    for t in times:
        data[t - pre:t - pre + n_t, 3] += shape.astype(np.int16)
    clusters = np.zeros(len(times), dtype=np.int64)
    true_scale = 3.0273  # what C_Waves would apply
    mw = np.zeros((1, n_ch, n_t))
    mw[0, 3] = shape * true_scale
    est = _measure_uv_per_count(mw, data, times, clusters, pre)
    assert est == pytest.approx(true_scale, rel=0.02)

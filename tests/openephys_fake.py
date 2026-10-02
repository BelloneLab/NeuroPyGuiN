"""Build small, format-faithful Open Ephys GUI recordings for tests and benchmarks.

``make_openephys_session`` writes the folder tree the Open Ephys GUI produces:

    <session>/Record Node 101/settings.xml
    <session>/Record Node 101/experiment1/recording1/structure.oebin
        continuous/<proc>.<stream>/continuous.dat (+ sample_numbers.npy, timestamps.npy)
        events/<proc>.<stream>/TTL/{states,sample_numbers,timestamps,full_words}.npy

for GUI >= 0.6, or the 0.5.x variant (numeric sub-index folders, int sample
numbers in ``timestamps.npy`` and ``channel_states.npy`` / ``channels.npy``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

NP1_PN = "PRB_1_4_0480_1"
NP2_4S_PN = "NP2014"


@dataclass
class FakeProbe:
    """One probe to write. ``data`` is (n_samples, n_channels) int16 AP data."""

    name: str = "ProbeA"
    part_number: str = NP1_PN
    data: Optional[np.ndarray] = None
    x: Optional[np.ndarray] = None  # Open Ephys convention: x includes shank offset
    y: Optional[np.ndarray] = None
    shank: Optional[np.ndarray] = None
    with_lfp: bool = True
    ttl_samples: Sequence[int] = ()  # rising edges on line 1 of this probe's AP stream
    ttl_width: int = 300


@dataclass
class FakeSession:
    root: Path
    record_node: Path
    recording_dir: Path
    ap_dat: Dict[str, Path] = field(default_factory=dict)


def np1_bank0_positions(n: int = 384):
    ch = np.arange(n)
    row = ch // 2
    x = np.where(row % 2 == 0, 27.0, 11.0) + (ch % 2) * 32.0
    return x, row * 20.0, np.zeros(n, dtype=int)


def np2_4shank_positions(n: int = 384):
    """96 sites per shank on 4 shanks; x carries the 250 um shank pitch (OE style)."""
    ch = np.arange(n)
    shank = ch // 96
    local = ch % 96
    x = 27.0 + (local % 2) * 32.0 + shank * 250.0
    y = (local // 2) * 15.0
    return x, y, shank


def _settings_xml(probes: Sequence[FakeProbe], version: str) -> str:
    blocks = []
    for i, p in enumerate(probes):
        n = p.data.shape[1]
        chans = " ".join(f'CH{c}="0:{int(p.shank[c])}"' for c in range(n))
        xs = " ".join(f'CH{c}="{p.x[c]:g}"' for c in range(n))
        ys = " ".join(f'CH{c}="{p.y[c]:g}"' for c in range(n))
        blocks.append(
            f'<NP_PROBE slot="2" port="{i + 1}" dock="1" probe_serial_number="1800000{i}" '
            f'probe_part_number="{p.part_number}" probe_name="Neuropixels" custom_probe_name="{p.name}" '
            f'ap_gain="500" lfp_gain="250" reference_channel="Ext" isEnabled="1">'
            f"<CHANNELS {chans}/><ELECTRODE_XPOS {xs}/><ELECTRODE_YPOS {ys}/></NP_PROBE>"
        )
    return (
        f'<?xml version="1.0" encoding="UTF-8"?><SETTINGS><INFO><VERSION>{version}</VERSION></INFO>'
        '<SIGNALCHAIN><PROCESSOR name="Neuropix-PXI" nodeId="100"><EDITOR displayName="Neuropix-PXI">'
        + "".join(blocks)
        + "</EDITOR></PROCESSOR></SIGNALCHAIN></SETTINGS>"
    )


def make_openephys_session(
    root: Path,
    probes: Sequence[FakeProbe],
    *,
    legacy: bool = False,
    first_sample: int = 123_456,
    ts_offset_s: float = 10.0,
    clock_scale: float = 1.0,
    ni_ttl_seconds: Sequence[float] = (),
    write_settings: bool = True,
    session_name: str = "2024-05-01_12-00-00",
) -> FakeSession:
    """Write a fake Open Ephys session and return its key paths.

    ``ts_offset_s``/``clock_scale`` make the synchronized ``timestamps.npy`` differ
    from ``sample / fs`` so tests can check the AP-clock mapping. ``ni_ttl_seconds``
    adds an NI-DAQ TTL stream whose events are expressed in synchronized seconds.
    """
    version = "0.5.5" if legacy else "0.6.7"
    session = Path(root) / session_name
    node = session / "Record Node 101"
    rec = node / "experiment1" / "recording1"
    (rec / "continuous").mkdir(parents=True, exist_ok=True)
    out = FakeSession(root=session, record_node=node, recording_dir=rec)
    fs_ap, fs_lf = 30000.0, 2500.0
    continuous: List[dict] = []
    events: List[dict] = []

    for i, p in enumerate(probes):
        if p.x is None:
            p.x, p.y, p.shank = np1_bank0_positions(p.data.shape[1])
        n_samp, n_ch = p.data.shape
        bands = [("AP", fs_ap, p.data)]
        if p.with_lfp:
            lf = p.data[:: int(fs_ap // fs_lf)].copy()
            bands.append(("LFP", fs_lf, lf))
        for b_idx, (band, fs, arr) in enumerate(bands):
            if legacy:
                folder = f"Neuropix-PXI-100.{2 * i + b_idx}"
                stream_name = ""
            else:
                folder = f"Neuropix-PXI-100.{p.name}-{band}"
                stream_name = f"{p.name}-{band}"
            d = rec / "continuous" / folder
            d.mkdir(parents=True, exist_ok=True)
            np.ascontiguousarray(arr, dtype=np.int16).tofile(d / "continuous.dat")
            first = first_sample if band == "AP" else first_sample // int(fs_ap // fs_lf)
            sn = np.arange(first, first + arr.shape[0], dtype=np.int64)
            if legacy:
                np.save(d / "timestamps.npy", sn)
            else:
                np.save(d / "sample_numbers.npy", sn)
                np.save(d / "timestamps.npy", ts_offset_s + (sn - first) / fs * clock_scale)
            entry = {
                "folder_name": folder + "/",
                "sample_rate": fs,
                "source_processor_name": "Neuropix-PXI",
                "source_processor_id": 100,
                "recorded_processor": "Neuropix-PXI",
                "recorded_processor_id": 100,
                "num_channels": n_ch,
                "channels": [
                    {"channel_name": f"{band if band == 'AP' else 'LFP'}{c + 1}", "bit_volts": 0.1949999928474426, "units": "uV"}
                    for c in range(n_ch)
                ],
            }
            if legacy:
                entry["source_processor_sub_idx"] = 2 * i + b_idx
            else:
                entry["stream_name"] = stream_name
            continuous.append(entry)
            if band == "AP":
                out.ap_dat[p.name] = d / "continuous.dat"
                if p.ttl_samples:
                    ev_folder = f"{folder}/TTL" if not legacy else f"{folder}/TTL_1"
                    ed = rec / "events" / ev_folder
                    ed.mkdir(parents=True, exist_ok=True)
                    rel = np.asarray(p.ttl_samples, dtype=np.int64)
                    samples = np.sort(np.concatenate([rel, rel + p.ttl_width])) + first
                    rising = np.isin(samples - first, rel)
                    if legacy:
                        np.save(ed / "timestamps.npy", samples)
                        np.save(ed / "channel_states.npy", np.where(rising, 1, -1).astype(np.int64))
                        np.save(ed / "channels.npy", np.ones(len(samples), dtype=np.int64))
                    else:
                        np.save(ed / "sample_numbers.npy", samples)
                        np.save(ed / "timestamps.npy", ts_offset_s + (samples - first) / fs_ap * clock_scale)
                        np.save(ed / "states.npy", np.where(rising, 1, -1).astype(np.int16))
                        np.save(ed / "full_words.npy", rising.astype(np.uint64))
                    events.append({"folder_name": ev_folder + "/", "sample_rate": fs_ap, "stream_name": stream_name})

    if ni_ttl_seconds and not legacy:
        folder = "NI-DAQmx-102.PXIe-6341/TTL"
        ed = rec / "events" / folder
        ed.mkdir(parents=True, exist_ok=True)
        t = np.asarray(ni_ttl_seconds, dtype=float)
        stamps = np.sort(np.concatenate([t, t + 0.01]))
        rising = np.isin(stamps, t)
        np.save(ed / "timestamps.npy", ts_offset_s + stamps * clock_scale)
        np.save(ed / "sample_numbers.npy", (stamps * 30000.0).astype(np.int64))
        np.save(ed / "states.npy", np.where(rising, 3, -3).astype(np.int16))
        events.append({"folder_name": folder + "/", "sample_rate": 30000.0, "stream_name": "PXIe-6341"})

    oebin = {"GUI version": version, "continuous": continuous, "events": events, "spikes": []}
    (rec / "structure.oebin").write_text(json.dumps(oebin, indent=2), encoding="utf-8")
    if write_settings:
        node.mkdir(parents=True, exist_ok=True)
        (node / "settings.xml").write_text(_settings_xml(probes, version), encoding="utf-8")
    return out

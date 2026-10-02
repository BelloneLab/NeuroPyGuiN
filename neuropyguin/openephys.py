"""Open Ephys GUI recordings -> SpikeGLX-compatible staged recordings.

The whole preprocessing pipeline (queue, output layouts, Kilosort helper,
C_Waves, quality metrics, BombCell, curation) is built around SpikeGLX naming
(``<run>_g0_t0.imec0.ap.bin``) and the ``.ap.meta`` sidecar it reads sampling
rate, channel count, gain and probe geometry from. Rather than forking every
consumer, this module *stages* each Open Ephys Neuropixels stream as a
SpikeGLX-shaped pair:

* ``<run>_g0_t0.imec<P>.ap.bin``  - a symlink (or hard link) to the untouched
  Open Ephys ``continuous.dat`` (zero copy; a copy is the last resort),
* ``<run>_g0_t0.imec<P>.ap.meta`` - a synthesized meta whose gain fields are
  chosen so every SpikeGLX formula recovers Open Ephys' ``bit_volts`` exactly,
  and whose ``~snsGeomMap`` holds the electrode positions from ``settings.xml``,
* ``openephys_events.csv``        - TTL edges converted onto the AP sample
  clock (the job TPrime does for SpikeGLX), ready for Post Processing,
* ``openephys_source.json``       - provenance of everything above.

Supported layouts
-----------------
* Open Ephys GUI >= 0.6 / 1.x binary format: ``continuous/<proc>.<stream>/``
  with ``continuous.dat``, ``sample_numbers.npy``, ``timestamps.npy`` (seconds)
  and ``events/<proc>.<stream>/TTL/`` with ``sample_numbers.npy``,
  ``timestamps.npy`` and ``states.npy`` (signed, 1-based line numbers).
* Open Ephys GUI 0.5.x binary format: ``continuous/<proc>.<subidx>/`` with
  ``continuous.dat`` and ``timestamps.npy`` (int sample numbers) and
  ``events/<proc>.<subidx>/TTL_1/`` with ``timestamps.npy``,
  ``channel_states.npy`` and ``channels.npy``.

Only Neuropixels streams (Neuropix-PXI / OneBox plugins) are staged; other
continuous streams (Intan acquisition board, NI-DAQ analog) are reported as
skipped with a reason, since the pipeline needs Neuropixels geometry.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: Meta key that marks a staged Open Ephys recording (value: source .dat path).
OE_SOURCE_META_KEY = "oeSourceDat"
#: Provenance file written next to every staged AP bin.
OE_PROVENANCE_NAME = "openephys_source.json"
#: Long-format TTL table written in the staged run folder.
OE_EVENTS_CSV_NAME = "openephys_events.csv"

# Neuropixels geometry tables, matching SpikeGLX conventions (um).
# [nShank, shankWidth, shankPitch, even_xOff, odd_xOff, horizPitch, vertPitch]
_NP1_GEOM = (1, 70.0, 0.0, 27.0, 11.0, 32.0, 20.0)
_NP2_SS_GEOM = (1, 70.0, 0.0, 27.0, 27.0, 32.0, 15.0)
_NP2_4S_GEOM = (4, 70.0, 250.0, 27.0, 27.0, 32.0, 15.0)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class OpenEphysStream:
    """One continuous stream described by a ``structure.oebin`` entry."""

    oebin_path: Path
    folder_name: str
    stream_name: str
    processor_name: str
    processor_id: int
    sub_idx: Optional[int]
    sample_rate: float
    num_channels: int
    bit_volts: float
    channel_names: List[str]
    gui_version: str

    @property
    def recording_dir(self) -> Path:
        """``.../experimentN/recordingM`` folder holding the oebin."""
        return self.oebin_path.parent

    @property
    def dat_path(self) -> Path:
        return self.recording_dir / "continuous" / self.folder_name / "continuous.dat"

    @property
    def stream_dir(self) -> Path:
        return self.dat_path.parent

    @property
    def is_legacy(self) -> bool:
        """True for GUI 0.5.x layouts (numeric sub-index folder names)."""
        return self.sub_idx is not None and not self.stream_name

    @property
    def label(self) -> str:
        """Human-readable stream label, e.g. ``Neuropix-PXI-100.ProbeA-AP``."""
        return self.folder_name


@dataclass
class OpenEphysAPRecording:
    """A Neuropixels AP stream plus its context, ready to be staged."""

    ap: OpenEphysStream
    lfp: Optional[OpenEphysStream]
    session_dir: Path
    record_node: str
    experiment_index: int
    recording_index: int
    probe_index: int
    probe_label: str
    run_name: str

    def describe(self) -> str:
        return f"{self.run_name} [{self.ap.label}] {self.ap.dat_path}"


@dataclass
class ProbeGeometry:
    """Electrode layout and identity used to synthesize the SpikeGLX meta."""

    part_number: str
    serial_number: str
    probe_type: str
    n_shank: int
    shank_width: float
    shank_pitch: float
    shank: np.ndarray
    x: np.ndarray
    y: np.ndarray
    source: str
    warnings: List[str] = field(default_factory=list)


@dataclass
class StagedOpenEphysRecording:
    """Result of :func:`stage_openephys_recording`."""

    run_name: str
    probe_index: int
    ap_bin: Path
    ap_meta: Path
    lf_bin: Optional[Path]
    events_csv: Optional[Path]
    provenance: Path
    link_mode: str
    warnings: List[str]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def read_oebin(oebin_path: str | Path) -> List[OpenEphysStream]:
    """Parse a ``structure.oebin`` and return its continuous streams."""
    path = Path(oebin_path)
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    version = str(data.get("GUI version", "") or "")
    streams: List[OpenEphysStream] = []
    for entry in data.get("continuous", []) or []:
        folder = str(entry.get("folder_name", "")).strip().strip("/\\")
        if not folder:
            continue
        channels = entry.get("channels", []) or []
        bit_volts = 0.0
        if channels:
            try:
                bit_volts = float(channels[0].get("bit_volts", 0.0) or 0.0)
            except Exception:
                bit_volts = 0.0
        sub_idx = entry.get("source_processor_sub_idx")
        streams.append(
            OpenEphysStream(
                oebin_path=path,
                folder_name=folder,
                stream_name=str(entry.get("stream_name", "") or ""),
                processor_name=str(
                    entry.get("source_processor_name") or entry.get("recorded_processor") or ""
                ),
                processor_id=int(entry.get("source_processor_id") or entry.get("recorded_processor_id") or 0),
                sub_idx=int(sub_idx) if sub_idx is not None else None,
                sample_rate=float(entry.get("sample_rate", 0.0) or 0.0),
                num_channels=int(entry.get("num_channels", len(channels)) or len(channels)),
                bit_volts=bit_volts,
                channel_names=[str(c.get("channel_name", "")) for c in channels],
                gui_version=version,
            )
        )
    return streams


def _is_neuropixels_stream(stream: OpenEphysStream) -> bool:
    text = f"{stream.processor_name} {stream.folder_name}".lower()
    return any(token in text for token in ("neuropix", "onebox", "probe"))


def _is_lfp_stream(stream: OpenEphysStream) -> bool:
    """LFP band: explicit ``-LFP`` suffix, ``LFP`` channel names, or ~2.5 kHz rate."""
    name = (stream.stream_name or stream.folder_name).upper()
    if name.endswith("-LFP") or name.endswith(".LFP"):
        return True
    if stream.channel_names and stream.channel_names[0].upper().startswith("LFP"):
        return True
    return 0 < stream.sample_rate < 10000


def _probe_label_for(stream: OpenEphysStream, fallback_index: int) -> Tuple[str, int]:
    """Return ``(label, index)`` such as ``("ProbeA", 0)`` for a Neuropixels stream."""
    name = stream.stream_name or stream.folder_name.split(".", 1)[-1]
    m = re.search(r"Probe\s*([A-Z])", name, flags=re.IGNORECASE)
    if m:
        letter = m.group(1).upper()
        return f"Probe{letter}", ord(letter) - ord("A")
    if stream.sub_idx is not None:
        # GUI 0.5: sub-index 0/1 = probe A AP/LFP, 2/3 = probe B, ...
        idx = stream.sub_idx // 2
        return f"Probe{chr(ord('A') + idx)}", idx
    return f"Probe{chr(ord('A') + fallback_index)}", fallback_index


def _sanitize(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(text)).strip("_")
    return cleaned or "openephys"


def _session_context(oebin_path: Path) -> Tuple[Path, str, int, int]:
    """Return ``(session_dir, record_node, experiment_idx, recording_idx)``.

    Expected layout: ``<session>/<Record Node N>/experimentE/recordingR/structure.oebin``.
    Falls back gracefully when a level is missing (e.g. a copied recording folder).
    """
    rec_dir = oebin_path.parent
    exp_dir = rec_dir.parent
    rec_m = re.search(r"(\d+)$", rec_dir.name)
    exp_m = re.search(r"(\d+)$", exp_dir.name)
    rec_idx = int(rec_m.group(1)) if rec_m else 1
    exp_idx = int(exp_m.group(1)) if exp_m and exp_dir.name.lower().startswith("experiment") else 1
    node_dir = exp_dir.parent if exp_dir.name.lower().startswith("experiment") else exp_dir
    if node_dir.name.lower().startswith("record node"):
        return node_dir.parent, node_dir.name, exp_idx, rec_idx
    return node_dir, "", exp_idx, rec_idx


def _find_oebin_files(paths: Sequence[str | Path]) -> List[Path]:
    found: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            if p.name == "structure.oebin":
                found.append(p)
                continue
            # A continuous.dat / settings.xml / any file inside a recording tree:
            # walk up to the nearest folder that owns a structure.oebin.
            for parent in p.parents:
                cand = parent / "structure.oebin"
                if cand.exists():
                    found.append(cand)
                    break
                if parent.name.lower().startswith("record node"):
                    found.extend(parent.rglob("structure.oebin"))
                    break
        elif p.is_dir():
            found.extend(p.rglob("structure.oebin"))
    return sorted({f.resolve() for f in found})


def discover_openephys_recordings(
    paths: Sequence[str | Path],
) -> Tuple[List[OpenEphysAPRecording], List[Tuple[str, str]]]:
    """Find Neuropixels AP streams in Open Ephys recordings under ``paths``.

    ``paths`` may mix folders (searched recursively), ``structure.oebin`` files,
    ``continuous.dat`` files and ``settings.xml`` files. Returns
    ``(recordings, skipped)`` where ``skipped`` lists ``(stream, reason)`` pairs
    for continuous streams that cannot be sorted (non-Neuropixels, LFP-only,
    missing data file).
    """
    oebins = _find_oebin_files(paths)
    recordings: List[OpenEphysAPRecording] = []
    skipped: List[Tuple[str, str]] = []
    # Record nodes per session decide whether the node name must enter run names.
    nodes_per_session: Dict[Path, set] = {}
    parsed: List[Tuple[Path, List[OpenEphysStream]]] = []
    for oebin in oebins:
        try:
            streams = read_oebin(oebin)
        except Exception as exc:  # malformed JSON, permissions, ...
            skipped.append((str(oebin), f"unreadable structure.oebin ({exc})"))
            continue
        session_dir, node, _e, _r = _session_context(oebin)
        nodes_per_session.setdefault(session_dir, set()).add(node)
        parsed.append((oebin, streams))

    for oebin, streams in parsed:
        session_dir, node, exp_idx, rec_idx = _session_context(oebin)
        np_streams = [s for s in streams if _is_neuropixels_stream(s)]
        for s in streams:
            if s not in np_streams:
                skipped.append(
                    (f"{oebin.parent}/{s.folder_name}", "not a Neuropixels stream (no probe geometry)")
                )
        ap_streams = [s for s in np_streams if not _is_lfp_stream(s)]
        lfp_streams = [s for s in np_streams if _is_lfp_stream(s)]
        for order, ap in enumerate(ap_streams):
            if not ap.dat_path.exists():
                skipped.append((str(ap.dat_path), "continuous.dat missing"))
                continue
            label, probe_idx = _probe_label_for(ap, order)
            lfp = next(
                (l for l in lfp_streams if _probe_label_for(l, order)[0] == label and l.processor_id == ap.processor_id),
                None,
            )
            parts = [_sanitize(session_dir.name)]
            if len(nodes_per_session.get(session_dir, ())) > 1 and node:
                parts.append(_sanitize(node.replace("Record Node", "RN")))
            parts.append(f"e{exp_idx}r{rec_idx}")
            recordings.append(
                OpenEphysAPRecording(
                    ap=ap,
                    lfp=lfp,
                    session_dir=session_dir,
                    record_node=node,
                    experiment_index=exp_idx,
                    recording_index=rec_idx,
                    probe_index=probe_idx,
                    probe_label=label,
                    run_name="_".join(parts),
                )
            )
        for l in lfp_streams:
            if not any(r.lfp is l for r in recordings):
                skipped.append((f"{oebin.parent}/{l.folder_name}", "LFP-band stream (sorted from its AP stream)"))
    return recordings, skipped


# ---------------------------------------------------------------------------
# Probe geometry
# ---------------------------------------------------------------------------


def probe_type_for_part_number(part_number: str) -> str:
    """Map a Neuropixels part number to SpikeGLX's ``imDatPrb_type`` code."""
    pn = str(part_number or "").strip().upper()
    if not pn or pn.startswith("PRB_1_") or pn in {"NP1000", "NP1010", "NP1011", "NP1012", "NP1013",
                                                    "NP1015", "NP1016", "NP1017"}:
        return "0"
    if pn.startswith("PRB2_1_") or pn == "NP2000":
        return "21"
    if pn.startswith("PRB2_4_") or pn == "NP2010":
        return "24"
    if pn in {"NP2003", "NP2004"}:
        return "2003"
    if pn in {"NP2013", "NP2014"}:
        return "2013"
    m = re.fullmatch(r"NP(\d{4})", pn)
    return m.group(1) if m else "0"


def _is_np2_family(probe_type: str) -> bool:
    return probe_type in {"21", "24", "2003", "2013", "2020"}


def _geometry_table(probe_type: str) -> Tuple[int, float, float, float, float, float, float]:
    if probe_type in {"24", "2013"}:
        return _NP2_4S_GEOM
    if _is_np2_family(probe_type):
        return _NP2_SS_GEOM
    return _NP1_GEOM


def default_bank0_geometry(part_number: str, n_channels: int) -> ProbeGeometry:
    """Bank-0 electrode positions for a probe when settings.xml is unavailable."""
    ptype = probe_type_for_part_number(part_number)
    n_shank, width, pitch, even_x, odd_x, hpitch, vpitch = _geometry_table(ptype)
    ch = np.arange(n_channels)
    row = ch // 2
    col = ch % 2
    x = np.where(row % 2 == 0, even_x, odd_x) + col * hpitch
    y = row * vpitch
    return ProbeGeometry(
        part_number=part_number or ("PRB_1_4_0480_1" if ptype == "0" else part_number),
        serial_number="",
        probe_type=ptype,
        n_shank=n_shank,
        shank_width=width,
        shank_pitch=pitch,
        shank=np.zeros(n_channels, dtype=int),
        x=x.astype(float),
        y=y.astype(float),
        source="default bank-0 layout (settings.xml probe map not found)",
        warnings=[
            "Electrode positions were not found in settings.xml; assumed the default bank-0 "
            "layout. If you recorded a custom channel map, Kilosort drift correction and unit "
            "depths will be wrong."
        ],
    )


def _settings_xml_for(oebin_path: Path, experiment_index: int) -> Optional[Path]:
    """``settings.xml`` lives in the Record Node folder (``settings_N.xml`` for experiment N>1)."""
    rec_dir = oebin_path.parent
    candidates = []
    for base in (rec_dir.parent.parent, rec_dir.parent, rec_dir):
        if experiment_index > 1:
            candidates.append(base / f"settings_{experiment_index}.xml")
        candidates.append(base / "settings.xml")
    return next((c for c in candidates if c.exists()), None)


def _channel_sort_key(name: str) -> int:
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else 0


def _probe_elements(root: ET.Element) -> List[ET.Element]:
    return [el for el in root.iter() if el.tag.upper() == "NP_PROBE"]


def parse_probe_geometry(
    settings_xml: Optional[Path],
    recording: OpenEphysAPRecording,
) -> ProbeGeometry:
    """Read electrode positions for ``recording``'s probe from ``settings.xml``.

    Matches the ``NP_PROBE`` element by its custom probe name (``ProbeA``) when
    present, otherwise by probe order. Positions come from ``ELECTRODE_XPOS`` /
    ``ELECTRODE_YPOS`` and shanks from the ``CHANNELS`` values (``bank:shank``).
    Open Ephys stores x across shanks (``shank * pitch + x_local``); the SpikeGLX
    geom map wants the within-shank x, so the shank offset is removed.
    """
    n_ch = recording.ap.num_channels
    if settings_xml is None:
        return default_bank0_geometry("", n_ch)
    try:
        root = ET.parse(settings_xml).getroot()
    except Exception:
        return default_bank0_geometry("", n_ch)
    probes = _probe_elements(root)
    if not probes:
        return default_bank0_geometry("", n_ch)

    chosen: Optional[ET.Element] = None
    for el in probes:
        names = {str(el.attrib.get(k, "")).strip().lower() for k in ("custom_probe_name", "probe_name", "name")}
        if recording.probe_label.lower() in names:
            chosen = el
            break
    if chosen is None:
        idx = recording.probe_index if 0 <= recording.probe_index < len(probes) else 0
        chosen = probes[idx]

    part_number = str(chosen.attrib.get("probe_part_number", "") or "")
    serial = str(chosen.attrib.get("probe_serial_number", "") or "")
    xpos_el = next((c for c in chosen if c.tag.upper() == "ELECTRODE_XPOS"), None)
    ypos_el = next((c for c in chosen if c.tag.upper() == "ELECTRODE_YPOS"), None)
    chan_el = next((c for c in chosen if c.tag.upper() == "CHANNELS"), None)
    if xpos_el is None or ypos_el is None or not xpos_el.attrib:
        geom = default_bank0_geometry(part_number, n_ch)
        geom.serial_number = serial
        return geom

    keys = sorted(xpos_el.attrib.keys(), key=_channel_sort_key)[:n_ch]
    x = np.array([float(xpos_el.attrib[k]) for k in keys])
    y = np.array([float(ypos_el.attrib.get(k, 0.0)) for k in keys])
    shank = np.zeros(len(keys), dtype=int)
    if chan_el is not None:
        for i, k in enumerate(keys):
            val = str(chan_el.attrib.get(k, ""))
            if ":" in val:
                try:
                    shank[i] = int(val.split(":")[1])
                except ValueError:
                    shank[i] = 0

    ptype = probe_type_for_part_number(part_number)
    n_shank_tab, width, pitch, *_ = _geometry_table(ptype)
    n_shank = max(n_shank_tab, int(shank.max()) + 1 if shank.size else 1)
    if n_shank > 1 and pitch <= 0:
        pitch = 250.0
    # Strip the across-shank offset Open Ephys bakes into x.
    offset = shank * pitch
    x_local = np.where(x >= offset, x - offset, x)

    warnings: List[str] = []
    if len(keys) < n_ch:
        warnings.append(
            f"settings.xml lists {len(keys)} electrode positions for {n_ch} channels; "
            "missing channels were placed at the default bank-0 layout."
        )
        fill = default_bank0_geometry(part_number, n_ch)
        x_local = np.concatenate([x_local, fill.x[len(keys):]])
        y = np.concatenate([y, fill.y[len(keys):]])
        shank = np.concatenate([shank, fill.shank[len(keys):]])
    return ProbeGeometry(
        part_number=part_number or "PRB_1_4_0480_1",
        serial_number=serial,
        probe_type=ptype,
        n_shank=n_shank,
        shank_width=width,
        shank_pitch=pitch if n_shank > 1 else 0.0,
        shank=shank.astype(int),
        x=x_local.astype(float),
        y=y.astype(float),
        source=f"settings.xml ({settings_xml.name}, probe {chosen.attrib.get('custom_probe_name', recording.probe_label)})",
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# SpikeGLX meta synthesis
# ---------------------------------------------------------------------------


def gain_fields_for_bit_volts(bit_volts: float, probe_type: str) -> Dict[str, str]:
    """Gain-related meta fields that reproduce ``bit_volts`` (uV per bit).

    SpikeGLX consumers compute uV/bit as ``Vmax / Imax / gain`` (BombCell) or
    ``1e6 * (Vmax - Vmin) / gain / (2 * Imax)`` (ecephys). For NP1-family probe
    codes we keep Imax = 512 and solve for the gain; for NP2-family codes BombCell
    hard-codes gain = 80, so we keep 80 and solve for Imax instead. Both paths
    agree with ``bit_volts`` to better than 1e-4 relative error.
    """
    bv = float(bit_volts) if bit_volts and bit_volts > 0 else 0.195
    vmax = 0.6
    if _is_np2_family(probe_type):
        imax = int(round(vmax * 1e6 / (80.0 * bv)))
        gain = 2 * vmax * 1e6 / (2 * imax * bv)
    else:
        imax = 512
        gain = 2 * vmax * 1e6 / (2 * imax * bv)
    return {
        "imAiRangeMax": f"{vmax:g}",
        "imAiRangeMin": f"{-vmax:g}",
        "imMaxInt": str(imax),
        "imChan0apGain": f"{gain:.6f}",
        "imChan0lfGain": f"{gain:.6f}",
    }


def _rows_and_cols(geom: ProbeGeometry) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Per-channel (row, col) on the probe grid plus grid size ``(n_cols, n_rows)``.

    Row is ``y / vertical pitch``; column is the rank of x among the sites that
    share the same shank and row, which reproduces SpikeGLX's ``electrode % 2``
    convention for staggered NP1 rows (smaller x = column 0).
    """
    vpitch = _geometry_table(geom.probe_type)[6]
    rows = np.round(geom.y / vpitch).astype(int) if vpitch else np.zeros(geom.y.shape, dtype=int)
    row_xs: Dict[Tuple[int, int], List[float]] = {}
    for s, r, xv in zip(geom.shank, rows, geom.x):
        row_xs.setdefault((int(s), int(r)), []).append(float(xv))
    for key in row_xs:
        row_xs[key] = sorted(set(row_xs[key]))
    cols = np.array(
        [row_xs[(int(s), int(r))].index(float(xv)) for s, r, xv in zip(geom.shank, rows, geom.x)], dtype=int
    )
    n_cols = max((len(v) for v in row_xs.values()), default=1)
    n_rows = int(rows.max()) + 1 if rows.size else 1
    return rows, cols, n_cols, n_rows


def _shank_map_string(geom: ProbeGeometry) -> str:
    """``~snsShankMap`` entries ``(shank:col:row:use)`` derived from the positions."""
    rows, cols, n_cols, n_rows = _rows_and_cols(geom)
    entries = "".join(f"({int(s)}:{int(c)}:{int(r)}:1)" for s, c, r in zip(geom.shank, cols, rows))
    return f"({geom.n_shank},{n_cols},{n_rows})" + entries


def _imro_table_string(geom: ProbeGeometry, gain: str) -> str:
    """``~imroTbl`` in the per-family SpikeGLX format.

    CatGT refuses (segfaults) without an imro table. Electrode indices are
    rebuilt from the positions (``row * 2 + col``). For NP1-family tables the AP
    and LF gains carry the synthesized gain, so CatGT's mV-based ``-gfix``
    thresholds stay consistent with Open Ephys' ``bit_volts`` scaling.
    """
    rows, cols, _n_cols, _n_rows = _rows_and_cols(geom)
    elec = rows * 2 + cols
    bank = elec // 384
    n = len(elec)
    pt = geom.probe_type
    if pt in {"24", "2013"}:  # NP2.0 4-shank: (chan shank bank refid elec)
        body = "".join(f"({c} {int(geom.shank[c])} {int(bank[c])} 0 {int(elec[c] % 384)})" for c in range(n))
    elif _is_np2_family(pt):  # NP2.0 1-shank: (chan bank-mask refid elec)
        body = "".join(f"({c} {1 << int(bank[c])} 0 {int(elec[c])})" for c in range(n))
    else:  # NP1 family: (chan bank refid apgain lfgain apfilt)
        body = "".join(f"({c} {int(bank[c])} 0 {gain} {gain} 1)" for c in range(n))
    return f"({pt},{n})" + body


def build_spikeglx_meta(
    stream: OpenEphysStream,
    geometry: ProbeGeometry,
    *,
    band: str,
    n_samples: int,
    first_sample: int,
    file_name: str,
) -> Dict[str, str]:
    """Synthesize the SpikeGLX meta fields the pipeline reads.

    ``band`` is ``"ap"`` or ``"lf"``. Open Ephys does not record the imec SYNC
    channel as data, so ``snsApLfSy`` declares zero sync channels; sync edges
    arrive separately as TTL events and are exported to CSV instead.
    """
    n_ch = stream.num_channels
    is_ap = band == "ap"
    counts = f"{n_ch},0,0" if is_ap else f"0,{n_ch},0"
    prefix = "AP" if is_ap else "LF"
    meta: Dict[str, str] = {
        "acqApLfSy": counts,
        "appVersion": f"OpenEphys-{stream.gui_version or 'unknown'}",
        "fileName": file_name,
        "fileSizeBytes": str(int(n_samples) * n_ch * 2),
        "fileTimeSecs": f"{n_samples / stream.sample_rate:.6f}" if stream.sample_rate else "0",
        "firstSample": str(int(first_sample)),
        "imDatPrb_pn": geometry.part_number,
        "imDatPrb_sn": geometry.serial_number,
        "imDatPrb_type": geometry.probe_type,
        "imSampRate": f"{stream.sample_rate:.6f}",
        "nSavedChans": str(n_ch),
        "snsApLfSy": counts,
        "snsSaveChanSubset": "all",
        "typeThis": "imec",
        OE_SOURCE_META_KEY: str(stream.dat_path),
        "oeStream": stream.label,
        "oeBitVolts": f"{stream.bit_volts:.12g}",
        "oeGeometrySource": geometry.source,
    }
    meta.update(gain_fields_for_bit_volts(stream.bit_volts, geometry.probe_type))
    meta["~imroTbl"] = _imro_table_string(geometry, meta["imChan0apGain"])
    meta["~snsChanMap"] = f"({counts})" + "".join(f"({prefix}{i};{i}:{i})" for i in range(n_ch))
    meta["~snsShankMap"] = _shank_map_string(geometry)
    meta["~snsGeomMap"] = (
        f"({geometry.part_number},{geometry.n_shank},{geometry.shank_pitch:g},{geometry.shank_width:g})"
        + "".join(
            f"({int(s)}:{float(xv):g}:{float(yv):g}:1)" for s, xv, yv in zip(geometry.shank, geometry.x, geometry.y)
        )
    )
    return meta


def write_meta(path: Path, meta: Dict[str, str]) -> None:
    """Write ``key=value`` lines, plain keys first and ``~`` keys last (SpikeGLX order)."""
    plain = sorted(k for k in meta if not k.startswith("~"))
    tilde = sorted(k for k in meta if k.startswith("~"))
    path.write_text("".join(f"{k}={meta[k]}\n" for k in plain + tilde), encoding="utf-8")


# ---------------------------------------------------------------------------
# Timing and events
# ---------------------------------------------------------------------------


def _load_npy(path: Path) -> Optional[np.ndarray]:
    try:
        return np.load(path, mmap_mode="r") if path.exists() else None
    except Exception:
        return None


@dataclass
class _StreamClock:
    """First/last sample numbers and synchronized timestamps of the AP stream."""

    first_sample: int
    n_samples: int
    sample_rate: float
    ts_first: Optional[float]
    ts_last: Optional[float]

    def seconds_from_timestamps(self, ts: np.ndarray) -> np.ndarray:
        """Map synchronized seconds onto the AP sample clock (linear, like TPrime)."""
        if self.ts_first is None or self.ts_last is None or self.n_samples < 2 or self.ts_last <= self.ts_first:
            return np.asarray(ts, dtype=float) - (self.ts_first or 0.0)
        span_ap = (self.n_samples - 1) / self.sample_rate
        scale = span_ap / (self.ts_last - self.ts_first)
        return (np.asarray(ts, dtype=float) - self.ts_first) * scale


def _ap_clock(stream: OpenEphysStream, n_samples: int) -> _StreamClock:
    sn = _load_npy(stream.stream_dir / "sample_numbers.npy")
    ts = _load_npy(stream.stream_dir / "timestamps.npy")
    first = 0
    ts_first = ts_last = None
    if sn is not None and len(sn):
        first = int(sn[0])
        if ts is not None and len(ts) and np.issubdtype(ts.dtype, np.floating):
            ts_first, ts_last = float(ts[0]), float(ts[-1])
    elif ts is not None and len(ts):
        # GUI 0.5: timestamps.npy holds integer sample numbers.
        first = int(ts[0])
    return _StreamClock(first, n_samples, stream.sample_rate, ts_first, ts_last)


def _event_entries(oebin_path: Path) -> List[Dict[str, object]]:
    try:
        data = json.loads(oebin_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return []
    return [e for e in data.get("events", []) or [] if "TTL" in str(e.get("folder_name", "")).upper()]


def export_ttl_events(
    recording: OpenEphysAPRecording,
    clock: _StreamClock,
    out_dir: Path,
) -> Tuple[Optional[Path], List[str]]:
    """Write every TTL edge of the recording to ``openephys_events.csv``.

    Times are seconds on the AP stream's sample clock, so they share the time
    base of Kilosort spike times. Synchronization strategy, best first:

    1. GUI >= 0.6 ``timestamps.npy`` (synchronized seconds) mapped linearly onto
       the AP samples (``sync="timestamps"``);
    2. events recorded on the AP stream itself: sample number difference
       (``sync="same_stream"``);
    3. otherwise raw sample numbers of the other stream, flagged
       ``sync="unsynchronized"`` so the user knows drift is uncorrected.
    """
    warnings: List[str] = []
    rows: List[Tuple[float, str, str, int, str, int, str]] = []
    ap = recording.ap
    for entry in _event_entries(ap.oebin_path):
        folder = str(entry.get("folder_name", "")).strip("/\\")
        ev_dir = ap.recording_dir / "events" / folder
        if not ev_dir.is_dir():
            continue
        ev_rate = float(entry.get("sample_rate", ap.sample_rate) or ap.sample_rate)
        stream_label = folder.split("/")[0].split("\\")[0]
        same_stream = stream_label == ap.folder_name
        states = _load_npy(ev_dir / "states.npy")
        if states is not None:  # GUI >= 0.6
            samples = _load_npy(ev_dir / "sample_numbers.npy")
            stamps = _load_npy(ev_dir / "timestamps.npy")
            lines = np.abs(np.asarray(states)).astype(int)
            rising = np.asarray(states) > 0
        else:  # GUI 0.5
            ch_states = _load_npy(ev_dir / "channel_states.npy")
            channels = _load_npy(ev_dir / "channels.npy")
            samples = _load_npy(ev_dir / "timestamps.npy")
            stamps = None
            if ch_states is None or samples is None:
                continue
            lines = np.asarray(channels if channels is not None else np.abs(ch_states)).astype(int)
            rising = np.asarray(ch_states) > 0
        if samples is None or len(samples) == 0:
            continue
        samples = np.asarray(samples).astype(np.int64)
        if stamps is not None and len(stamps) == len(samples) and clock.ts_first is not None:
            times = clock.seconds_from_timestamps(np.asarray(stamps))
            sync = "timestamps"
        elif same_stream:
            times = (samples - clock.first_sample) / clock.sample_rate
            sync = "same_stream"
        else:
            times = samples / ev_rate - clock.first_sample / clock.sample_rate
            sync = "unsynchronized"
            warnings.append(
                f"TTL events from {stream_label} could not be synchronized to the AP clock "
                "(no synchronized timestamps); exported with sample-number timing."
            )
        for t, line, up, sn in zip(times, lines, rising, samples):
            edge = "rising" if up else "falling"
            rows.append((float(t), f"{stream_label} TTL{int(line)} {edge}", stream_label, int(line), edge, int(sn), sync))
    if not rows:
        return None, warnings
    rows.sort(key=lambda r: r[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / OE_EVENTS_CSV_NAME
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("time_s,event_type,stream,line,edge,sample_number,sync\n")
        for t, etype, stream_label, line, edge, sn, sync in rows:
            f.write(f"{t:.9f},{etype},{stream_label},{line},{edge},{sn},{sync}\n")
    return csv_path, warnings


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------


def _relative_session_parts(session_dir: Path) -> Tuple[str, ...]:
    """Session path below a ``rawData`` folder, or just the session folder name."""
    parts = session_dir.resolve().parts
    lowered = [p.lower() for p in parts]
    if "rawdata" in lowered:
        rel = parts[lowered.index("rawdata") + 1 :]
        if rel:
            return tuple(rel)
    return (_sanitize(session_dir.name),)


def staged_ap_bin_path(recording: OpenEphysAPRecording, stage_root: str | Path) -> Path:
    """Where :func:`stage_openephys_recording` puts the SpikeGLX-named AP bin.

    ``<stage_root>/<session>/<run>_g0/<run>_g0_imec<P>/<run>_g0_t0.imec<P>.ap.bin``
    mirrors a SpikeGLX run folder, so the existing ``mirror`` output layout lands
    results in ``<stage_root>/<session>/spike_sorting``.
    """
    run = recording.run_name
    p = recording.probe_index
    base = Path(stage_root).expanduser().joinpath(*_relative_session_parts(recording.session_dir))
    return base / f"{run}_g0" / f"{run}_g0_imec{p}" / f"{run}_g0_t0.imec{p}.ap.bin"


def _link_or_copy(
    src: Path,
    dst: Path,
    progress: Optional[Callable[[int, int], None]] = None,
) -> str:
    """Create ``dst`` pointing at ``src``; return ``"hardlink"``, ``"symlink"`` or ``"copy"``.

    A hard link is preferred: it costs no space and is a genuine directory entry,
    so code that calls ``Path.resolve()`` keeps the staged SpikeGLX path. Symlinks
    (across filesystems) are the fallback; a chunked copy is the last resort
    (e.g. Windows without symlink rights and the output on another drive).
    """
    if dst.is_symlink() or dst.exists():
        try:
            if dst.resolve() == src.resolve() or (dst.exists() and os.path.samefile(dst, src)):
                return "existing"
        except OSError:
            pass
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        pass
    try:
        os.symlink(src, dst)
        return "symlink"
    except (OSError, NotImplementedError):
        pass
    total = src.stat().st_size
    done = 0
    chunk = 64 * 1024 * 1024
    tmp = dst.with_suffix(dst.suffix + ".part")
    with src.open("rb") as fin, tmp.open("wb") as fout:
        while True:
            buf = fin.read(chunk)
            if not buf:
                break
            fout.write(buf)
            done += len(buf)
            if progress is not None:
                progress(done, total)
    tmp.replace(dst)
    return "copy"


def stage_openephys_recording(
    recording: OpenEphysAPRecording,
    stage_root: str | Path,
    *,
    progress: Optional[Callable[[int, int], None]] = None,
) -> StagedOpenEphysRecording:
    """Stage one Open Ephys Neuropixels recording as a SpikeGLX-shaped run.

    Idempotent: re-staging the same recording reuses the existing link and
    rewrites only the small sidecar files.
    """
    ap = recording.ap
    warnings: List[str] = []
    size = ap.dat_path.stat().st_size
    frame = 2 * ap.num_channels
    n_samples = size // frame
    if size % frame:
        warnings.append(
            f"{ap.dat_path.name} size is not a multiple of {ap.num_channels} int16 channels; "
            f"the trailing {size % frame} bytes are ignored."
        )
    settings = _settings_xml_for(ap.oebin_path, recording.experiment_index)
    geometry = parse_probe_geometry(settings, recording)
    warnings.extend(geometry.warnings)

    ap_bin = staged_ap_bin_path(recording, stage_root)
    ap_bin.parent.mkdir(parents=True, exist_ok=True)
    link_mode = _link_or_copy(ap.dat_path, ap_bin, progress)
    if link_mode == "copy":
        warnings.append("Could not link continuous.dat (different drive / no permission); a full copy was made.")

    clock = _ap_clock(ap, n_samples)
    ap_meta = ap_bin.with_suffix(".meta")
    write_meta(
        ap_meta,
        build_spikeglx_meta(
            ap, geometry, band="ap", n_samples=n_samples, first_sample=clock.first_sample, file_name=str(ap_bin)
        ),
    )

    lf_bin: Optional[Path] = None
    if recording.lfp is not None and recording.lfp.dat_path.exists():
        lfp = recording.lfp
        lf_bin = ap_bin.with_name(ap_bin.name.replace(".ap.bin", ".lf.bin"))
        _link_or_copy(lfp.dat_path, lf_bin, progress)
        lf_samples = lfp.dat_path.stat().st_size // (2 * lfp.num_channels)
        lf_clock = _ap_clock(lfp, lf_samples)
        write_meta(
            lf_bin.with_suffix(".meta"),
            build_spikeglx_meta(
                lfp, geometry, band="lf", n_samples=lf_samples, first_sample=lf_clock.first_sample, file_name=str(lf_bin)
            ),
        )

    events_csv, ev_warnings = export_ttl_events(recording, clock, ap_bin.parent)
    warnings.extend(ev_warnings)

    provenance = ap_bin.parent / OE_PROVENANCE_NAME
    provenance.write_text(
        json.dumps(
            {
                "format": "openephys-binary",
                "gui_version": ap.gui_version,
                "run_name": recording.run_name,
                "probe_label": recording.probe_label,
                "probe_index": recording.probe_index,
                "record_node": recording.record_node,
                "experiment": recording.experiment_index,
                "recording": recording.recording_index,
                "source_oebin": str(ap.oebin_path),
                "source_ap_dat": str(ap.dat_path),
                "source_lfp_dat": str(recording.lfp.dat_path) if recording.lfp else "",
                "settings_xml": str(settings) if settings else "",
                "geometry_source": geometry.source,
                "probe_part_number": geometry.part_number,
                "sample_rate": ap.sample_rate,
                "num_channels": ap.num_channels,
                "n_samples": int(n_samples),
                "first_sample": clock.first_sample,
                "bit_volts": ap.bit_volts,
                "link_mode": link_mode,
                "staged_ap_bin": str(ap_bin),
                "events_csv": str(events_csv) if events_csv else "",
                "warnings": warnings,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return StagedOpenEphysRecording(
        run_name=recording.run_name,
        probe_index=recording.probe_index,
        ap_bin=ap_bin,
        ap_meta=ap_meta,
        lf_bin=lf_bin,
        events_csv=events_csv,
        provenance=provenance,
        link_mode=link_mode,
        warnings=warnings,
    )


def openephys_catgt_command(catgt_cmd: str) -> Tuple[str, List[str]]:
    """Adapt a CatGT flag string for an Open Ephys run; return ``(command, notes)``.

    * Event extractors (``-xa/-xd/-xia/-xid``) are removed: Open Ephys records no
      imec SYNC channel and no nidq stream (TTLs are exported to CSV instead).
    * ``-gfix`` is removed: CatGT converts its mV thresholds with built-in
      per-probe gains and ignores the meta for NP2-type probes, so on Open Ephys'
      0.195 uV/bit counts it flags every spike as an artifact (benchmark on real
      NP2014 data: 198 edits/s vs 0.07 for the same data recorded by SpikeGLX).
      Filtering and CAR are unaffected (r = 0.99 against native SpikeGLX).
    """
    notes: List[str] = []
    kept: List[str] = []
    dropped_x: List[str] = []
    for tok in str(catgt_cmd).split():
        low = tok.lower()
        if re.match(r"^-(xa|xd|xia|xid)=", low):
            dropped_x.append(tok)
        elif low.startswith("-gfix"):
            notes.append(
                "CatGT -gfix disabled: its mV thresholds assume SpikeGLX gains and would blank real "
                "spikes in Open Ephys data. Filtering and CAR still run."
            )
        else:
            kept.append(tok)
    if dropped_x:
        notes.insert(0, "Open Ephys has no SYNC channel or nidq stream; dropped CatGT extractors " + " ".join(dropped_x))
    return " ".join(kept), notes


def is_openephys_staged_bin(bin_file: str | Path) -> bool:
    """True when ``bin_file``'s meta was synthesized from an Open Ephys stream."""
    meta = Path(str(bin_file).replace(".ap.bin", ".ap.meta"))
    if not meta.exists():
        meta = Path(bin_file).with_suffix(".meta")
    if not meta.exists():
        return False
    try:
        with meta.open(encoding="utf-8", errors="ignore") as f:
            return any(line.startswith(OE_SOURCE_META_KEY + "=") for line in f)
    except OSError:
        return False


def openephys_provenance(bin_file: str | Path) -> Dict[str, object]:
    """Load ``openephys_source.json`` next to a staged bin (empty dict if absent)."""
    path = Path(bin_file).parent / OE_PROVENANCE_NAME
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

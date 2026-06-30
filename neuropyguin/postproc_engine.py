"""Post-processing analysis engine for Kilosort/SpikeGLX recordings.

This module provides the :class:`NeuropixelsDataset` loader plus pure-computation
helpers (correlograms, PSTHs, synchrony, raw-trace extraction) and an HDF5 unit
exporter. It performs no GUI work, so functions here are safe to call from worker
threads. Per-dataset results are cached on disk under ``NeuroPyGuiN_cache``.
"""

from __future__ import annotations

import hashlib
import json
import numbers
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from scipy import signal


def cluster_synced_units(
    units: Iterable[int],
    matrix: np.ndarray,
    threshold: Optional[float] = None,
    min_group_size: int = 2,
) -> Dict[str, np.ndarray | float]:
    """Group and order units by high pairwise synchrony scores.

    The input matrix is treated as a pairwise affinity matrix. Strong off-diagonal
    edges define connected components; components are sorted by their internal
    synchrony and units within each component are sorted by total synchrony.
    Singleton or weakly connected units receive group id 0.
    """
    unit_arr = np.asarray([int(u) for u in units], dtype=np.int64)
    mat = np.asarray(matrix, dtype=float)
    n_units = int(unit_arr.size)
    if n_units == 0:
        return {
            "order": np.asarray([], dtype=np.int64),
            "sorted_units": np.asarray([], dtype=np.int64),
            "group_labels": np.asarray([], dtype=np.int64),
            "threshold": float("nan"),
        }
    if mat.shape != (n_units, n_units):
        order = np.arange(n_units, dtype=np.int64)
        return {
            "order": order,
            "sorted_units": unit_arr.copy(),
            "group_labels": np.zeros(n_units, dtype=np.int64),
            "threshold": float("nan"),
        }

    score = np.nan_to_num(0.5 * (mat + mat.T), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(score, 0.0)
    tri = score[np.triu_indices(n_units, k=1)]
    positive = tri[np.isfinite(tri) & (tri > 0.0)]
    if positive.size == 0:
        order = np.arange(n_units, dtype=np.int64)
        return {
            "order": order,
            "sorted_units": unit_arr.copy(),
            "group_labels": np.zeros(n_units, dtype=np.int64),
            "threshold": float("nan"),
        }

    if threshold is None:
        median = float(np.nanmedian(positive))
        mad = float(np.nanmedian(np.abs(positive - median)))
        robust_high = median + 1.4826 * mad
        quantile_high = float(np.nanpercentile(positive, 75.0))
        threshold_value = max(quantile_high, robust_high)
    else:
        threshold_value = float(threshold)
    if not np.isfinite(threshold_value):
        threshold_value = float(np.nanmax(positive))

    adjacency = score >= threshold_value
    np.fill_diagonal(adjacency, False)
    visited = np.zeros(n_units, dtype=bool)
    components: list[list[int]] = []
    for start in range(n_units):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component: list[int] = []
        while stack:
            idx = stack.pop()
            component.append(idx)
            for neighbor in np.flatnonzero(adjacency[idx]):
                ni = int(neighbor)
                if not visited[ni]:
                    visited[ni] = True
                    stack.append(ni)
        components.append(component)

    def component_strength(component: list[int]) -> float:
        if len(component) < 2:
            return float(np.sum(score[component[0]]))
        sub = score[np.ix_(component, component)]
        vals = sub[np.triu_indices(len(component), k=1)]
        return float(np.nanmean(vals)) if vals.size else 0.0

    components.sort(
        key=lambda comp: (
            -int(len(comp) >= max(2, int(min_group_size))),
            -component_strength(comp),
            min(comp),
        )
    )

    group_by_original = np.zeros(n_units, dtype=np.int64)
    ordered: list[int] = []
    next_group_id = 1
    for component in components:
        component = sorted(
            component,
            key=lambda idx: (-float(np.sum(score[idx, component])), idx),
        )
        if len(component) >= max(2, int(min_group_size)):
            for idx in component:
                group_by_original[idx] = next_group_id
            next_group_id += 1
        ordered.extend(component)

    order = np.asarray(ordered, dtype=np.int64)
    return {
        "order": order,
        "sorted_units": unit_arr[order],
        "group_labels": group_by_original[order],
        "threshold": float(threshold_value),
    }


def _parse_meta(meta_path: Path) -> Dict[str, str]:
    """Parse a SpikeGLX ``.meta`` file into a key=value string mapping."""
    out: Dict[str, str] = {}
    if not meta_path.exists():
        return out
    for ln in meta_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _parse_params_py(params_path: Path) -> Dict[str, str]:
    """Extract the handful of fields we need from a Phy/Kilosort ``params.py``."""
    out: Dict[str, str] = {}
    if not params_path.exists():
        return out
    txt = params_path.read_text(encoding="utf-8", errors="ignore")
    for key in ["dat_path", "sample_rate", "n_channels_dat"]:
        m = re.search(rf"^{key}\s*=\s*(.+)$", txt, flags=re.MULTILINE)
        if m:
            out[key] = m.group(1).strip().strip("'").strip('"')
    return out


def _hash_payload(payload: Dict) -> str:
    """Return a short, deterministic hash of a JSON-serializable cache key."""
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


@dataclass
class NeuropixelsDataset:
    """In-memory view of a Kilosort output folder plus its SpikeGLX metadata.

    Spike arrays are memory-mapped where possible; analysis methods return plain
    NumPy arrays and cache expensive results under :pyattr:`cache_dir`.
    """

    ks_folder: Path
    sample_rate: float
    n_channels: int
    bit_uV: float
    ap_bin_path: Optional[Path]
    spike_times: np.ndarray
    spike_clusters: np.ndarray
    units: np.ndarray
    channel_map: Optional[np.ndarray]
    channel_positions: Optional[np.ndarray]
    templates: Optional[np.ndarray]
    spike_templates: Optional[np.ndarray]

    @property
    def cache_dir(self) -> Path:
        """Return (creating if needed) the per-dataset on-disk cache directory."""
        p = self.ks_folder / "NeuroPyGuiN_cache"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _cache_get(self, name: str, payload: Dict) -> Optional[Dict[str, np.ndarray]]:
        key = _hash_payload(payload)
        fp = self.cache_dir / f"{name}_{key}.npz"
        if not fp.exists():
            return None
        data = np.load(fp, allow_pickle=False)
        return {k: data[k] for k in data.files}

    def _cache_set(self, name: str, payload: Dict, arrays: Dict[str, np.ndarray]) -> None:
        key = _hash_payload(payload)
        fp = self.cache_dir / f"{name}_{key}.npz"
        np.savez_compressed(fp, **arrays)

    @classmethod
    def load(cls, ks_folder: str) -> "NeuropixelsDataset":
        """Build a dataset from a Kilosort folder, reading params.py and the AP meta.

        Sample rate, channel count, and bit-to-microvolt scaling are taken from the
        SpikeGLX ``.ap.meta`` when available, falling back to ``params.py`` defaults.
        """
        root = Path(ks_folder)
        if not root.exists():
            raise RuntimeError(f"Folder does not exist: {root}")

        spk_t = np.load(root / "spike_times.npy", mmap_mode="r").squeeze()
        spk_c = np.load(root / "spike_clusters.npy", mmap_mode="r").squeeze()
        units = np.unique(spk_c).astype(int)

        params = _parse_params_py(root / "params.py")
        dat_path = params.get("dat_path", "")
        ap_bin = Path(dat_path) if dat_path else None
        if ap_bin is not None and not ap_bin.exists():
            ap_bin = None
        if ap_bin is None:
            # Fallback: walk up and try to find matching ap.bin.
            candidates = list(root.parent.rglob("*.ap.bin"))
            ap_bin = candidates[0] if candidates else None

        sample_rate = float(params.get("sample_rate", "30000")) if "sample_rate" in params else 30000.0
        n_channels = int(float(params.get("n_channels_dat", "385"))) if "n_channels_dat" in params else 385
        bit_uV = 2.34375
        if ap_bin is not None:
            meta = _parse_meta(Path(str(ap_bin).replace(".ap.bin", ".ap.meta")))
            if meta:
                sample_rate = float(meta.get("imSampRate", sample_rate))
                n_channels = int(meta.get("nSavedChans", n_channels))
                try:
                    rng = float(meta.get("imAiRangeMax", "0.6")) - float(meta.get("imAiRangeMin", "-0.6"))
                    gain = float(meta.get("imChan0apGain", "80"))
                    max_int = float(meta.get("imMaxInt", "512"))
                    bit_uV = (1e6) * (rng / gain) / (2 * max_int)
                except Exception:
                    pass

        channel_map = None
        if (root / "channel_map.npy").exists():
            channel_map = np.load(root / "channel_map.npy").squeeze()

        channel_positions = None
        if (root / "channel_positions.npy").exists():
            channel_positions = np.load(root / "channel_positions.npy")

        templates = None
        spike_templates = None
        if (root / "templates.npy").exists():
            templates = np.load(root / "templates.npy", mmap_mode="r")
        if (root / "spike_templates.npy").exists():
            spike_templates = np.load(root / "spike_templates.npy", mmap_mode="r").squeeze()

        return cls(
            ks_folder=root,
            sample_rate=sample_rate,
            n_channels=n_channels,
            bit_uV=bit_uV,
            ap_bin_path=ap_bin,
            spike_times=np.asarray(spk_t),
            spike_clusters=np.asarray(spk_c),
            units=units,
            channel_map=channel_map,
            channel_positions=channel_positions,
            templates=np.asarray(templates) if templates is not None else None,
            spike_templates=np.asarray(spike_templates) if spike_templates is not None else None,
        )

    def unit_spike_times_s(self, unit: int) -> np.ndarray:
        """Return the spike times of one unit, in seconds."""
        m = self.spike_clusters == int(unit)
        return self.spike_times[m].astype(float) / float(self.sample_rate)

    def isi_hist(self, unit: int, max_ms: float, bins: int = 80) -> Tuple[np.ndarray, np.ndarray]:
        """Return (bin edges, counts) for a unit's inter-spike-interval histogram."""
        t = self.unit_spike_times_s(unit)
        if t.size < 2:
            return np.array([]), np.array([])
        isi_ms = np.diff(t) * 1000.0
        hist, edges = np.histogram(isi_ms, bins=bins, range=(0, max_ms))
        return edges, hist.astype(float)

    def correlogram(
        self,
        unit_a: int,
        unit_b: int,
        bin_ms: float,
        win_ms: float,
        remove_zero: bool = False,
        normalize: str = "Hertz",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (bin centers in ms, values) for the cross/auto-correlogram of two units.

        ``normalize`` selects the y-axis unit following the NeuroPyxels convention:
        ``"Counts"`` returns raw spike-pair counts, ``"Hertz"`` (the default)
        returns a firing rate ``counts / (N_ref_spikes * bin_s)`` that is invariant
        to bin width and reference-spike count and matches npyx's ``acg``/``ccg``.
        Result is cached on disk. ``remove_zero`` drops the exact-zero lag (used for
        autocorrelograms to suppress the trivial self-coincidence peak).
        """
        payload = {
            "ua": int(unit_a), "ub": int(unit_b), "bin_ms": float(bin_ms),
            "win_ms": float(win_ms), "rz": bool(remove_zero), "norm": str(normalize),
        }
        cached = self._cache_get("corr", payload)
        if cached is not None:
            return cached["centers"], cached["counts"]

        t1 = self.unit_spike_times_s(unit_a)
        t2 = self.unit_spike_times_s(unit_b)
        if t1.size == 0 or t2.size == 0:
            return np.array([]), np.array([])
        bin_s = max(bin_ms, 0.1) / 1000.0
        win_s = max(win_ms, 1.0) / 1000.0
        edges = np.arange(-win_s, win_s + bin_s, bin_s)
        all_dt: List[np.ndarray] = []
        for t in t1:
            dt = t2 - t
            m = (dt >= -win_s) & (dt <= win_s)
            if np.any(m):
                all_dt.append(dt[m])
        centers = 0.5 * (edges[:-1] + edges[1:]) * 1000.0
        if not all_dt:
            return centers, np.zeros_like(centers)
        vals = np.concatenate(all_dt)
        if remove_zero:
            vals = vals[np.abs(vals) > 1e-12]
        counts, _ = np.histogram(vals, bins=edges)
        out = counts.astype(float)
        if str(normalize).lower().startswith("hert"):
            n_ref = float(t1.size)
            if n_ref > 0:
                out = out / (n_ref * bin_s)
        self._cache_set("corr", payload, {"centers": centers, "counts": out})
        return centers, out

    def mean_template_waveform(self, unit: int) -> Optional[np.ndarray]:
        """Return the unit's dominant Kilosort template (in microvolts), or None.

        The template most frequently assigned to the unit's spikes is selected and
        scaled by ``bit_uV``. Returns None when templates are unavailable.
        """
        if self.templates is None or self.spike_templates is None:
            return None
        m = self.spike_clusters == int(unit)
        if not np.any(m):
            return None
        tmp_ids = self.spike_templates[m]
        if tmp_ids.size == 0:
            return None
        ids, counts = np.unique(tmp_ids, return_counts=True)
        tid = int(ids[np.argmax(counts)])
        w = np.asarray(self.templates[tid], dtype=float)
        return w * float(self.bit_uV)

    def population_synchrony(self, bin_ms: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """Return (bin centers in ms, total spike counts) binning all units together.

        Result is cached on disk.
        """
        payload = {"bin_ms": float(bin_ms)}
        cached = self._cache_get("sync", payload)
        if cached is not None:
            return cached["t_ms"], cached["counts"]

        t = self.spike_times.astype(float) / float(self.sample_rate)
        if t.size == 0:
            return np.array([]), np.array([])
        dt = max(bin_ms, 0.5) / 1000.0
        edges = np.arange(float(t.min()), float(t.max()) + dt, dt)
        counts, _ = np.histogram(t, bins=edges)
        centers_ms = 1000.0 * (0.5 * (edges[:-1] + edges[1:]))
        out = counts.astype(float)
        self._cache_set("sync", payload, {"t_ms": centers_ms, "counts": out})
        return centers_ms, out

    def psth_trials(
        self,
        unit: int,
        event_times_s: np.ndarray,
        pre_s: float,
        post_s: float,
        bin_ms: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Per-trial PSTH for one unit: return (bin centers in ms, trials x bins firing rate)."""
        ev = np.asarray(event_times_s, dtype=float)
        ev = ev[np.isfinite(ev)]
        if ev.size == 0:
            return np.array([]), np.zeros((0, 0), dtype=float)

        bin_s = max(bin_ms, 0.5) / 1000.0
        edges = np.arange(-pre_s, post_s + bin_s, bin_s)
        centers_ms = 1000.0 * (0.5 * (edges[:-1] + edges[1:]))
        trial_mat = np.zeros((ev.size, edges.size - 1), dtype=float)
        st = self.unit_spike_times_s(int(unit))
        if st.size == 0:
            return centers_ms, trial_mat

        for row, e in enumerate(ev):
            rel = st - e
            m = (rel >= -pre_s) & (rel <= post_s)
            if np.any(m):
                h, _ = np.histogram(rel[m], bins=edges)
                trial_mat[row] = h.astype(float) / bin_s
        return centers_ms, trial_mat

    def psth(self, units: Iterable[int], event_times_s: np.ndarray, pre_s: float, post_s: float, bin_ms: float) -> Tuple[np.ndarray, np.ndarray]:
        """Pooled PSTH across units and events: return (bin centers in ms, mean firing rate)."""
        units = list(units)
        ev = np.asarray(event_times_s, dtype=float)
        ev = ev[np.isfinite(ev)]
        if len(units) == 0 or ev.size == 0:
            return np.array([]), np.array([])

        bin_s = max(bin_ms, 0.5) / 1000.0
        edges = np.arange(-pre_s, post_s + bin_s, bin_s)
        total = np.zeros(edges.size - 1, dtype=float)
        n_trials = int(len(units) * ev.size)
        for u in units:
            st = self.unit_spike_times_s(int(u))
            for e in ev:
                rel = st - e
                m = (rel >= -pre_s) & (rel <= post_s)
                if np.any(m):
                    h, _ = np.histogram(rel[m], bins=edges)
                    total += h
        if n_trials == 0:
            return np.array([]), np.array([])
        rate = total / (n_trials * bin_s)
        centers_ms = 1000.0 * (0.5 * (edges[:-1] + edges[1:]))
        return centers_ms, rate

    def psth_by_unit(
        self,
        units: Iterable[int],
        event_times_s: np.ndarray,
        pre_s: float,
        post_s: float,
        bin_ms: float,
    ) -> Tuple[np.ndarray, List[int], np.ndarray]:
        """Per-unit PSTH: return (bin centers in ms, unit ids, units x bins mean firing rate)."""
        unit_ids = [int(u) for u in units]
        ev = np.asarray(event_times_s, dtype=float)
        ev = ev[np.isfinite(ev)]
        if len(unit_ids) == 0 or ev.size == 0:
            return np.array([]), [], np.zeros((0, 0), dtype=float)

        centers_ms = np.array([], dtype=float)
        mat: Optional[np.ndarray] = None

        for row, unit in enumerate(unit_ids):
            unit_centers_ms, trial_mat = self.psth_trials(unit, ev, pre_s=pre_s, post_s=post_s, bin_ms=bin_ms)
            if centers_ms.size == 0:
                centers_ms = np.asarray(unit_centers_ms, dtype=float)
                mat = np.zeros((len(unit_ids), centers_ms.size), dtype=float)
            if mat is None:
                return np.array([]), [], np.zeros((0, 0), dtype=float)
            if trial_mat.size == 0:
                continue
            mat[row] = np.nanmean(trial_mat, axis=0)
        if mat is None:
            return np.array([]), [], np.zeros((0, 0), dtype=float)
        return centers_ms, unit_ids, mat

    def load_raw_chunk_uv(
        self,
        t0_s: float,
        dur_s: float,
        max_channels: int = 24,
        center_channel: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Read a raw AP-band chunk in microvolts from the memory-mapped binary.

        Returns (time vector in s, samples x channels data, file channel ids, channel
        order indices). Selects up to ``max_channels`` channels centered on
        ``center_channel`` when given.
        """
        if self.ap_bin_path is None or not self.ap_bin_path.exists():
            raise RuntimeError("AP binary file not found from params/meta.")
        t0_s = max(0.0, float(t0_s))
        dur_s = max(0.05, float(dur_s))
        n0 = int(round(t0_s * self.sample_rate))
        n1 = int(round((t0_s + dur_s) * self.sample_rate))
        if n1 <= n0:
            n1 = n0 + int(self.sample_rate * 0.1)

        data = np.memmap(self.ap_bin_path, dtype=np.int16, mode="r")
        n_samples = data.size // int(self.n_channels)
        n0 = min(max(0, n0), max(0, n_samples - 1))
        n1 = min(max(n0 + 1, n1), n_samples)
        block = data[n0 * self.n_channels : n1 * self.n_channels].reshape(-1, self.n_channels)
        if self.channel_map is not None and self.channel_map.size > 0:
            ch_idx = np.asarray(self.channel_map).astype(int)
            ch_idx = ch_idx[ch_idx >= 0]
            if ch_idx.size == 0:
                ch_idx = np.arange(self.n_channels)
        else:
            ch_idx = np.arange(self.n_channels)
        count = min(max(1, int(max_channels)), int(ch_idx.size))
        if center_channel is None:
            start = 0
        else:
            center = int(np.clip(int(center_channel), 0, max(int(ch_idx.size) - 1, 0)))
            half = count // 2
            start = max(0, center - half)
            start = min(start, max(int(ch_idx.size) - count, 0))
        stop = start + count
        order_idx = np.arange(start, stop, dtype=int)
        file_channels = ch_idx[order_idx]
        block = block[:, file_channels].astype(float) * float(self.bit_uV)
        t = np.arange(block.shape[0], dtype=float) / float(self.sample_rate) + (n0 / float(self.sample_rate))
        return t, block, file_channels.astype(int), order_idx

    def raw_explorer_chunk(
        self,
        t0_s: float,
        dur_s: float,
        max_channels: int = 24,
        hp_hz: float = 0.0,
        lp_hz: float = 0.0,
        downsample: int = 1,
        center_channel: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Raw chunk for the explorer view: load, optionally band-filter, and downsample.

        Wraps :pymeth:`load_raw_chunk_uv`, applies an optional Butterworth filter
        (high-pass, low-pass, or band-pass depending on ``hp_hz``/``lp_hz``), then
        decimates by ``downsample``. Result is cached on disk.
        """
        payload = {
            "t0": float(t0_s),
            "dur": float(dur_s),
            "ch": int(max_channels),
            "hp": float(hp_hz),
            "lp": float(lp_hz),
            "ds": int(downsample),
            "center": None if center_channel is None else int(center_channel),
        }
        cached = self._cache_get("rawx", payload)
        if cached is not None:
            return cached["t"], cached["x"], cached["channel_ids"], cached["channel_order"]

        t, x, channel_ids, channel_order = self.load_raw_chunk_uv(
            t0_s=t0_s,
            dur_s=dur_s,
            max_channels=max_channels,
            center_channel=center_channel,
        )
        fs = float(self.sample_rate)
        hp = float(max(0.0, hp_hz))
        lp = float(max(0.0, lp_hz))
        if hp > 0.0 or lp > 0.0:
            nyq = fs * 0.5
            if hp > 0.0 and lp > hp and lp < nyq:
                b, a = signal.butter(3, [hp / nyq, lp / nyq], btype="bandpass")
                x = signal.filtfilt(b, a, x, axis=0)
            elif hp > 0.0 and hp < nyq:
                b, a = signal.butter(3, hp / nyq, btype="highpass")
                x = signal.filtfilt(b, a, x, axis=0)
            elif lp > 0.0 and lp < nyq:
                b, a = signal.butter(3, lp / nyq, btype="lowpass")
                x = signal.filtfilt(b, a, x, axis=0)

        ds = max(1, int(downsample))
        if ds > 1:
            t = t[::ds]
            x = x[::ds, :]
        self._cache_set(
            "rawx",
            payload,
            {
                "t": t,
                "x": x,
                "channel_ids": channel_ids.astype(int),
                "channel_order": channel_order.astype(int),
            },
        )
        return t, x, channel_ids.astype(int), channel_order.astype(int)

    def psth_conditions(
        self,
        units: Iterable[int],
        condition_events: Dict[str, np.ndarray],
        pre_s: float,
        post_s: float,
        bin_ms: float,
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Pooled PSTH per named condition: return (bin centers in ms, labels, conditions x bins)."""
        labels = list(condition_events.keys())
        all_rows: List[np.ndarray] = []
        t_ref: Optional[np.ndarray] = None
        for lab in labels:
            t_ms, rate = self.psth(units, condition_events[lab], pre_s=pre_s, post_s=post_s, bin_ms=bin_ms)
            if t_ref is None:
                t_ref = t_ms
            if t_ms.size == 0:
                if t_ref is None:
                    t_ref = np.array([])
                row = np.zeros_like(t_ref, dtype=float)
            else:
                row = rate
            all_rows.append(row)
        if t_ref is None:
            return np.array([]), [], np.zeros((0, 0))
        mat = np.vstack(all_rows) if all_rows else np.zeros((0, t_ref.size))
        return t_ref, labels, mat

    def ccg_matrix(self, units: Iterable[int], bin_ms: float, win_ms: float) -> np.ndarray:
        """Return an N x N matrix of peak correlogram counts between every unit pair."""
        u = [int(v) for v in units]
        n = len(u)
        if n == 0:
            return np.zeros((0, 0))
        mat = np.zeros((n, n), dtype=float)
        for i, ua in enumerate(u):
            for j, ub in enumerate(u):
                centers, counts = self.correlogram(ua, ub, bin_ms=bin_ms, win_ms=win_ms, remove_zero=(ua == ub))
                if counts.size:
                    mat[i, j] = float(np.nanmax(counts))
                else:
                    mat[i, j] = 0.0
        return mat

    def synchrony_over_time(self, bin_ms: float = 10.0, window_s: float = 2.0, step_s: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Return (window centers in s, synchrony index) using a sliding coefficient of variation.

        Within each sliding window the index is std/mean of the population spike-count
        histogram (0 where the mean is non-positive).
        """
        t_ms, counts = self.population_synchrony(bin_ms=bin_ms)
        if t_ms.size == 0:
            return np.array([]), np.array([])
        t_s = t_ms / 1000.0
        win = max(window_s, 0.2)
        step = max(step_s, 0.05)
        centers = np.arange(t_s.min() + 0.5 * win, t_s.max() - 0.5 * win + step, step)
        sync_idx = np.zeros_like(centers)
        for k, c in enumerate(centers):
            m = (t_s >= c - 0.5 * win) & (t_s <= c + 0.5 * win)
            if not np.any(m):
                sync_idx[k] = 0.0
            else:
                arr = counts[m]
                mu = float(np.mean(arr))
                sd = float(np.std(arr))
                sync_idx[k] = 0.0 if mu <= 1e-12 else sd / mu
        return centers, sync_idx

    def _unit_depth_um(self, unit: int) -> float:
        """Return the depth (um, probe y) of a unit's peak template channel, or NaN."""
        if self.channel_positions is None:
            return float("nan")
        wvf = self.mean_template_waveform(unit)
        if wvf is None or wvf.ndim != 2 or wvf.shape[1] == 0:
            return float("nan")
        best = int(np.nanargmax(np.nanmax(np.abs(wvf), axis=0)))
        pos = np.asarray(self.channel_positions, dtype=float)
        if pos.ndim == 2 and pos.shape[1] >= 2 and best < pos.shape[0]:
            return float(pos[best, 1])
        return float("nan")

    def network_analysis(
        self,
        units: Iterable[int],
        *,
        bin_ms: float = 25.0,
        compute_connections: bool = True,
        conn_bin_ms: float = 0.5,
        conn_win_ms: float = 50.0,
        conn_z: float = 5.0,
        max_conn_units: int = 16,
    ) -> Dict[str, object]:
        """Population network analysis for the selected units.

        Returns a dict with a hierarchically-sorted pairwise spike-count (noise)
        correlation matrix, the Okun population-coupling per unit (z-scored), unit
        depths (um), and an optional signed significant-connection matrix derived
        from short-latency CCG deviations. All matrices/vectors are returned in the
        same sorted display order; ``labels`` holds the unit ids in that order.
        """
        u = [int(x) for x in units]
        n = len(u)
        empty = {
            "units": u, "labels": [str(x) for x in u],
            "corr_matrix": np.zeros((n, n), dtype=float), "corr_bin_ms": float(bin_ms),
            "population_coupling": np.zeros(n, dtype=float), "depths_um": None,
            "connections": None, "n_significant": 0,
        }
        if n == 0:
            return empty

        fs = float(self.sample_rate)
        t_end = float(self.spike_times.max()) / fs if self.spike_times.size else 0.0
        if t_end <= 0:
            return empty
        bin_s = max(float(bin_ms), 1.0) / 1000.0
        edges = np.arange(0.0, t_end + bin_s, bin_s)
        nb = max(edges.size - 1, 1)
        counts = np.zeros((n, nb), dtype=float)
        for i, uu in enumerate(u):
            st = self.unit_spike_times_s(uu)
            if st.size:
                h, _ = np.histogram(st, bins=edges)
                counts[i] = h.astype(float)

        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(counts) if n > 1 else np.array([[1.0]])
        corr = np.atleast_2d(np.nan_to_num(np.asarray(corr, dtype=float), nan=0.0))

        order = np.arange(n, dtype=int)
        if n >= 3:
            try:
                from scipy.cluster.hierarchy import leaves_list, linkage
                from scipy.spatial.distance import squareform
                dist = 1.0 - corr
                dist = 0.5 * (dist + dist.T)
                np.fill_diagonal(dist, 0.0)
                dist[dist < 0] = 0.0
                z = linkage(squareform(dist, checks=False), method="average")
                order = np.asarray(leaves_list(z), dtype=int)
            except Exception:
                order = np.arange(n, dtype=int)

        # Okun population coupling: each unit's binned rate vs the rest-of-population rate.
        pop = counts.sum(axis=0)
        coupling = np.zeros(n, dtype=float)
        for i in range(n):
            rest = pop - counts[i]
            if np.std(counts[i]) > 0 and np.std(rest) > 0:
                coupling[i] = float(np.corrcoef(counts[i], rest)[0, 1])
        if n > 1 and np.std(coupling) > 0:
            coupling = (coupling - np.mean(coupling)) / np.std(coupling)

        depths = np.array([self._unit_depth_um(uu) for uu in u], dtype=float)

        connections = None
        n_sig = 0
        if compute_connections and 2 <= n <= int(max_conn_units):
            conn = np.zeros((n, n), dtype=float)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    centers, c = self.correlogram(
                        u[i], u[j], bin_ms=conn_bin_ms, win_ms=conn_win_ms,
                        remove_zero=False, normalize="Counts",
                    )
                    if centers.size == 0 or c.size == 0:
                        continue
                    flank = np.abs(centers) > 10.0
                    near = (np.abs(centers) >= 0.5) & (np.abs(centers) <= 4.0)
                    if not np.any(flank) or not np.any(near):
                        continue
                    mu = float(np.mean(c[flank]))
                    sd = float(np.std(c[flank]))
                    if sd <= 1e-9:
                        continue
                    seg = c[near]
                    z_peak = (float(np.max(seg)) - mu) / sd
                    z_trough = (float(np.min(seg)) - mu) / sd
                    conn[i, j] = z_peak if abs(z_peak) >= abs(z_trough) else z_trough
            connections = conn[np.ix_(order, order)]
            off = ~np.eye(n, dtype=bool)
            n_sig = int(np.sum(np.abs(conn[off]) >= float(conn_z)))

        depths_sorted = depths[order]
        return {
            "units": u,
            "labels": [str(u[k]) for k in order],
            "corr_matrix": corr[np.ix_(order, order)],
            "corr_bin_ms": float(bin_ms),
            "population_coupling": coupling[order],
            "depths_um": depths_sorted if np.any(np.isfinite(depths_sorted)) else None,
            "connections": connections,
            "n_significant": n_sig,
        }


_H5_TEXT_DTYPE = h5py.string_dtype(encoding="utf-8")


def _is_missing_value(value: object) -> bool:
    """Return True for None or pandas/NumPy NaN-like scalars (array-likes count as present)."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _unit_row_dict(df: Optional[pd.DataFrame], unit: int) -> Dict[str, object]:
    """Look up a unit's row (by int or str index) and return its non-missing fields as a dict."""
    if df is None or df.empty:
        return {}
    row = None
    if unit in df.index:
        row = df.loc[unit]
    else:
        su = str(unit)
        if su in df.index:
            row = df.loc[su]
    if row is None:
        return {}
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    out: Dict[str, object] = {}
    for key, value in row.items():
        if _is_missing_value(value):
            continue
        if isinstance(value, np.ndarray) and value.ndim == 0:
            value = value.item()
        elif isinstance(value, np.generic):
            value = value.item()
        out[str(key)] = value
    return out


def _h5_safe_name(name: str, used: set[str]) -> str:
    """Sanitize a name into a unique HDF5-safe dataset key, recording it in ``used``."""
    base = re.sub(r"[^\w.-]+", "_", str(name)).strip("._")
    if not base:
        base = "field"
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _value_to_text(value: object) -> str:
    """Render an arbitrary value as a stable text string (JSON for containers/arrays)."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _value_to_text(value.item())
        return json.dumps(np.asarray(value).tolist(), ensure_ascii=False)
    if isinstance(value, np.generic):
        return _value_to_text(value.item())
    if isinstance(value, (list, tuple, dict)):
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return str(value)
    return str(value)


def _write_text_array(group: h5py.Group, name: str, values: List[str]) -> None:
    """Write a list of strings as a variable-length UTF-8 dataset."""
    group.create_dataset(name, data=np.asarray(values, dtype=object), dtype=_H5_TEXT_DTYPE)


def _write_scalar_dataset(group: h5py.Group, name: str, value: object, original_name: Optional[str] = None) -> None:
    """Write one value to ``group`` using an HDF5-native dtype, falling back to text.

    Scalars use the closest native type (bool/int/float/str); arrays and numeric
    sequences are stored gzip-compressed; anything else is serialized via
    :func:`_value_to_text`. The pre-sanitization key is recorded in ``original_name``.
    """
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            value = value.item()
        else:
            ds = group.create_dataset(name, data=value, compression="gzip")
            if original_name is not None:
                ds.attrs["original_name"] = str(original_name)
            return
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, (str, bytes, Path)):
        ds = group.create_dataset(name, data=str(value), dtype=_H5_TEXT_DTYPE)
    elif isinstance(value, (bool, np.bool_)):
        ds = group.create_dataset(name, data=np.bool_(value))
    elif isinstance(value, numbers.Integral):
        ds = group.create_dataset(name, data=np.int64(value))
    elif isinstance(value, numbers.Real):
        ds = group.create_dataset(name, data=np.float64(value))
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value)
        if arr.dtype.kind in {"U", "S", "O"}:
            ds = group.create_dataset(name, data=np.asarray([_value_to_text(v) for v in value], dtype=object), dtype=_H5_TEXT_DTYPE)
        else:
            ds = group.create_dataset(name, data=arr, compression="gzip")
    else:
        ds = group.create_dataset(name, data=_value_to_text(value), dtype=_H5_TEXT_DTYPE)

    if original_name is not None:
        ds.attrs["original_name"] = str(original_name)


def _write_mapping_group(group: h5py.Group, mapping: Dict[str, object]) -> int:
    """Write a mapping as an HDF5 group and return the number of non-missing entries.

    Each value is stored under an ``entries`` subgroup (native dtype where possible),
    alongside parallel ``fields`` and ``values_as_text`` arrays for easy inspection.
    """
    fields: List[str] = []
    values_as_text: List[str] = []
    entries_group = group.create_group("entries")
    used_names: set[str] = set()

    for key, value in mapping.items():
        if _is_missing_value(value):
            continue
        fields.append(str(key))
        values_as_text.append(_value_to_text(value))
        ds_name = _h5_safe_name(str(key), used_names)
        _write_scalar_dataset(entries_group, ds_name, value, original_name=str(key))

    group.attrs["entry_count"] = int(len(fields))
    _write_text_array(group, "fields", fields)
    _write_text_array(group, "values_as_text", values_as_text)
    return len(fields)


def _best_channel_info(dataset: NeuropixelsDataset, unit: int) -> Tuple[Optional[int], Optional[int]]:
    """Return (template channel index, mapped channel id) of the unit's peak-amplitude channel.

    Returns (None, None) when no usable mean template waveform is available.
    """
    waveform = dataset.mean_template_waveform(unit)
    if waveform is None or waveform.ndim != 2 or waveform.shape[1] == 0:
        return None, None
    best_idx = int(np.nanargmax(np.max(np.abs(waveform), axis=0)))
    if dataset.channel_map is not None and np.asarray(dataset.channel_map).size > best_idx:
        best_ch = int(np.asarray(dataset.channel_map).squeeze()[best_idx])
    else:
        best_ch = best_idx
    return best_idx, best_ch


def export_units_h5(
    dataset: NeuropixelsDataset,
    output_path: str | Path,
    units: Iterable[int],
    labels_df: Optional[pd.DataFrame] = None,
    metrics_df: Optional[pd.DataFrame] = None,
    label_sources: Optional[Dict[str, pd.DataFrame]] = None,
    good_units: Optional[Iterable[int]] = None,
    good_source: str = "Auto",
    export_mode: str = "all",
    progress_callback: Optional[Callable[[int], None]] = None,
) -> Dict[str, int]:
    """Export the selected units to a structured HDF5 file.

    Each unit group holds its spike times (samples and seconds), derived fields,
    combined and per-source labels, metrics, and the mean template waveform.
    ``progress_callback`` (if given) is invoked with an integer percentage from 5
    to 100. Returns a small summary dict of exported and good unit counts.
    """
    units_list = [int(u) for u in units]
    good_unit_set = {int(u) for u in good_units or []}
    label_sources = label_sources or {}
    out_path = Path(output_path)

    if progress_callback is not None:
        progress_callback(5)

    spike_clusters = np.asarray(dataset.spike_clusters, dtype=np.int64)
    spike_times = np.asarray(dataset.spike_times, dtype=np.int64)
    order = np.argsort(spike_clusters, kind="mergesort")
    sorted_clusters = spike_clusters[order]
    sorted_times = spike_times[order]
    unit_starts = np.searchsorted(sorted_clusters, np.asarray(units_list, dtype=np.int64), side="left")
    unit_ends = np.searchsorted(sorted_clusters, np.asarray(units_list, dtype=np.int64), side="right")

    with h5py.File(out_path, "w") as h5:
        h5.attrs["format"] = "NeuroPyGuiN_unit_export"
        h5.attrs["format_version"] = 1
        h5.attrs["source_ks_folder"] = str(dataset.ks_folder)
        h5.attrs["sample_rate_hz"] = float(dataset.sample_rate)
        h5.attrs["n_channels"] = int(dataset.n_channels)
        h5.attrs["bit_uV"] = float(dataset.bit_uV)
        h5.attrs["export_mode"] = str(export_mode)
        h5.attrs["good_source"] = str(good_source)
        h5.attrs["exported_unit_count"] = int(len(units_list))
        h5.attrs["dataset_unit_count"] = int(len(dataset.units))
        h5.attrs["good_unit_count"] = int(len(good_unit_set))

        summary = h5.create_group("summary")
        summary.create_dataset("all_unit_ids", data=np.asarray(dataset.units, dtype=np.int64))
        summary.create_dataset("exported_unit_ids", data=np.asarray(units_list, dtype=np.int64))
        summary.create_dataset("good_unit_ids", data=np.asarray(sorted(good_unit_set), dtype=np.int64))
        summary.create_dataset("exported_is_good", data=np.asarray([u in good_unit_set for u in units_list], dtype=np.bool_))
        _write_text_array(summary, "label_sources", [str(name) for name in sorted(label_sources.keys())])
        _write_text_array(summary, "combined_label_fields", [str(col) for col in getattr(labels_df, "columns", [])])
        _write_text_array(summary, "metric_fields", [str(col) for col in getattr(metrics_df, "columns", [])])

        units_group = h5.create_group("units")
        source_name_cache: Dict[str, str] = {}
        used_source_names: set[str] = set()
        for source_name in sorted(label_sources.keys()):
            source_name_cache[str(source_name)] = _h5_safe_name(str(source_name), used_source_names)

        total_units = max(len(units_list), 1)
        for idx, unit in enumerate(units_list):
            unit_group = units_group.create_group(f"unit_{unit}")
            unit_group.attrs["unit_id"] = int(unit)
            unit_group.attrs["export_index"] = int(idx)
            unit_group.attrs["is_good"] = bool(unit in good_unit_set)

            spike_samples = np.asarray(sorted_times[unit_starts[idx] : unit_ends[idx]], dtype=np.int64)
            spike_seconds = spike_samples.astype(np.float64) / float(dataset.sample_rate)

            spike_group = unit_group.create_group("spike_times")
            spike_group.create_dataset("samples", data=spike_samples, compression="gzip")
            spike_group.create_dataset("seconds", data=spike_seconds, compression="gzip")

            derived: Dict[str, object] = {
                "spike_count": int(spike_samples.size),
                "is_good": bool(unit in good_unit_set),
            }
            best_idx, best_ch = _best_channel_info(dataset, unit)
            if best_idx is not None:
                derived["best_channel_index"] = int(best_idx)
            if best_ch is not None:
                derived["best_channel_id"] = int(best_ch)
            _write_mapping_group(unit_group.create_group("derived"), derived)

            labels_root = unit_group.create_group("labels")
            _write_mapping_group(labels_root.create_group("combined"), _unit_row_dict(labels_df, unit))
            labels_by_source = labels_root.create_group("by_source")
            for source_name, source_df in label_sources.items():
                source_group = labels_by_source.create_group(source_name_cache[str(source_name)])
                source_group.attrs["source_name"] = str(source_name)
                _write_mapping_group(source_group, _unit_row_dict(source_df, unit))

            _write_mapping_group(unit_group.create_group("metrics"), _unit_row_dict(metrics_df, unit))

            waveform = dataset.mean_template_waveform(unit)
            if waveform is not None:
                unit_group.create_dataset(
                    "mean_template_waveform_uv",
                    data=np.asarray(waveform, dtype=np.float32),
                    compression="gzip",
                )

            if progress_callback is not None:
                progress_callback(5 + int(95 * (idx + 1) / total_units))

    return {
        "exported_unit_count": int(len(units_list)),
        "good_unit_count": int(len(good_unit_set)),
    }

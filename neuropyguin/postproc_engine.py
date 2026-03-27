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


def _parse_meta(meta_path: Path) -> Dict[str, str]:
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
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


@dataclass
class NeuropixelsDataset:
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
        m = self.spike_clusters == int(unit)
        return self.spike_times[m].astype(float) / float(self.sample_rate)

    def isi_hist(self, unit: int, max_ms: float, bins: int = 80) -> Tuple[np.ndarray, np.ndarray]:
        t = self.unit_spike_times_s(unit)
        if t.size < 2:
            return np.array([]), np.array([])
        isi_ms = np.diff(t) * 1000.0
        hist, edges = np.histogram(isi_ms, bins=bins, range=(0, max_ms))
        return edges, hist.astype(float)

    def correlogram(self, unit_a: int, unit_b: int, bin_ms: float, win_ms: float, remove_zero: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        payload = {"ua": int(unit_a), "ub": int(unit_b), "bin_ms": float(bin_ms), "win_ms": float(win_ms), "rz": bool(remove_zero)}
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
        if not all_dt:
            centers = 0.5 * (edges[:-1] + edges[1:]) * 1000.0
            counts = np.zeros_like(centers)
            return centers, counts
        vals = np.concatenate(all_dt)
        if remove_zero:
            vals = vals[np.abs(vals) > 1e-12]
        counts, _ = np.histogram(vals, bins=edges)
        centers = 0.5 * (edges[:-1] + edges[1:]) * 1000.0
        out_counts = counts.astype(float)
        self._cache_set("corr", payload, {"centers": centers, "counts": out_counts})
        return centers, out_counts

    def mean_template_waveform(self, unit: int) -> Optional[np.ndarray]:
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

    def psth(self, units: Iterable[int], event_times_s: np.ndarray, pre_s: float, post_s: float, bin_ms: float) -> Tuple[np.ndarray, np.ndarray]:
        units = list(units)
        ev = np.asarray(event_times_s, dtype=float)
        ev = ev[np.isfinite(ev)]
        if len(units) == 0 or ev.size == 0:
            return np.array([]), np.array([])

        bin_s = max(bin_ms, 0.5) / 1000.0
        edges = np.arange(-pre_s, post_s + bin_s, bin_s)
        total = np.zeros(edges.size - 1, dtype=float)
        n_trials = 0
        for u in units:
            st = self.unit_spike_times_s(int(u))
            if st.size == 0:
                continue
            for e in ev:
                rel = st - e
                m = (rel >= -pre_s) & (rel <= post_s)
                if np.any(m):
                    h, _ = np.histogram(rel[m], bins=edges)
                    total += h
                    n_trials += 1
        if n_trials == 0:
            return np.array([]), np.array([])
        rate = total / (n_trials * bin_s)
        centers_ms = 1000.0 * (0.5 * (edges[:-1] + edges[1:]))
        return centers_ms, rate

    def load_raw_chunk_uv(
        self,
        t0_s: float,
        dur_s: float,
        max_channels: int = 24,
        center_channel: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


_H5_TEXT_DTYPE = h5py.string_dtype(encoding="utf-8")


def _is_missing_value(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _unit_row_dict(df: Optional[pd.DataFrame], unit: int) -> Dict[str, object]:
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
    group.create_dataset(name, data=np.asarray(values, dtype=object), dtype=_H5_TEXT_DTYPE)


def _write_scalar_dataset(group: h5py.Group, name: str, value: object, original_name: Optional[str] = None) -> None:
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

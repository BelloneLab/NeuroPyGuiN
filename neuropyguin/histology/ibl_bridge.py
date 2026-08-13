"""Bridge from AP_histology products to the IBL ephys-alignment world.

Three jobs, all depending on the IBL stack (``iblatlas`` / ``ibllib`` /
``atlaselectrophysiology``):

1. ``compute_xyz_picks``   probe_ccf.mat -> xyz_picks_shankN.json
                           (native port of D:\\NPX\\process_histology.py)
2. ``extract_alf``         KS4/catGT -> ALF ephys files in the histology folder
                           (wraps atlaselectrophysiology.extract_files.extract_data)
3. ``compute_channel_locations``
                           xyz_picks + channels.localCoordinates.npy ->
                           channel_locations_shankN.json and the merged
                           channel_locations_all_shanks.json
                           (native port of load_data_local.LoadDataLocal, plus the
                           all-shanks merge that did not previously exist)

This module is written to run **under an interpreter that has the IBL stack**
(the ``neuropygui`` env, or any env with ibllib/iblatlas). It imports the heavy
IBL packages lazily, so it can also be imported by the NeuroPyGuiN process
(ks4_ece) purely for orchestration. It is runnable as a CLI so the GUI can
dispatch it to the right interpreter via subprocess::

    python -m neuropyguin.histology.ibl_bridge xyz_picks   <hist_folder>
    python -m neuropyguin.histology.ibl_bridge extract_alf <ks_dir> <ephys_dir> <out_dir>
    python -m neuropyguin.histology.ibl_bridge channels    <hist_folder> [--alignment original|latest]
    python -m neuropyguin.histology.ibl_bridge all         <hist_folder> --ks <ks_dir> --ephys <ephys_dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# 1. xyz_picks
# ---------------------------------------------------------------------------

_ALLEN_BREGMA_MLAPDV_UM = np.array([5739.0, 5400.0, 332.0], dtype=np.float64)


def compute_xyz_picks(
    probe_ccf_path: str | Path,
    out_folder: str | Path,
    res: int = 10,
    brain_atlas=None,
    mode: str = "fast",
) -> List[Path]:
    """Convert ``probe_ccf.mat`` to per-shank ``xyz_picks_shankN.json``.

    ``mode='fast'`` is the default used by NeuroPyGuiN. It avoids importing the
    heavy IBL atlas stack and writes the two points needed downstream directly
    from ``probe_ccf.mat``. The old exact IBL surface-intersection path remains
    available with ``mode='ibl'``.
    """
    mode = str(mode or "fast").lower()
    if mode not in {"auto", "fast", "ibl"}:
        raise ValueError(f"Unknown xyz_picks mode: {mode!r}")
    if brain_atlas is not None:
        mode = "ibl"

    if mode in {"auto", "fast"}:
        try:
            return _compute_xyz_picks_fast(probe_ccf_path, out_folder, res=res)
        except Exception as exc:
            if mode == "fast":
                raise
            print(f"Fast xyz_picks unavailable ({exc}); falling back to IBL extractor.")
    return _compute_xyz_picks_ibl(probe_ccf_path, out_folder, res=res, brain_atlas=brain_atlas)


def _compute_xyz_picks_ibl(
    probe_ccf_path: str | Path,
    out_folder: str | Path,
    res: int = 10,
    brain_atlas=None,
) -> List[Path]:
    """Faithful port of ``D:\\NPX\\process_histology.py`` using IBL classes."""
    import scipy.io as sio
    from iblatlas.atlas import AllenAtlas, Insertion

    probe_ccf_path = Path(probe_ccf_path)
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    ba = brain_atlas or AllenAtlas(res)
    mat = sio.loadmat(str(probe_ccf_path))
    pc = mat["probe_ccf"]
    valid_rows, skipped = _valid_probe_ccf_rows(pc)
    if not valid_rows:
        raise ValueError("probe_ccf.mat contains no probes with at least two finite CCF points")
    _clear_xyz_pick_outputs(out_folder)

    written: List[Path] = []
    for p, _row, points in valid_rows:
        ccf_apdvml = points * res
        bregma_apdvml = ba.ccf2xyz(ccf_apdvml, ccf_order="apdvml") * 1e6
        ins = Insertion.from_track(bregma_apdvml / 1e6, brain_atlas=ba)
        xyz_picks = {"xyz_picks": (ins.xyz * 1e6).tolist()}
        fn = _write_xyz_pick(out_folder, p, xyz_picks["xyz_picks"])
        written.append(fn)
    _write_single_xyz_pick_alias(out_folder, written)
    if skipped:
        print(f"Skipped empty/invalid probe slot(s): {', '.join(map(str, skipped))}.")
    return written


def _compute_xyz_picks_fast(
    probe_ccf_path: str | Path,
    out_folder: str | Path,
    res: int = 10,
) -> List[Path]:
    """Fast xyz_picks writer that avoids IBL imports and atlas-volume loading.

    AP_histology saves CCF coordinates as 1-based [AP, DV, ML] voxel positions.
    IBL's ``AllenAtlas.ccf2xyz(..., ccf_order='apdvml')`` is an affine transform
    around bregma, so we can reproduce it directly. The tip is the same PCA line
    projection used by ``Insertion.from_track``. The entry uses the stored
    ``trajectory_coords`` first in-brain coordinate, nudged one voxel anterior and
    dorsal to match IBL's surface-entry convention closely. If older files lack
    ``trajectory_coords``, the shallowest point on the fitted line is used.
    """
    import scipy.io as sio

    probe_ccf_path = Path(probe_ccf_path)
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    pc = sio.loadmat(str(probe_ccf_path))["probe_ccf"]

    valid_rows, skipped = _valid_probe_ccf_rows(pc)
    if not valid_rows:
        raise ValueError("probe_ccf.mat contains no probes with at least two finite CCF points")
    _clear_xyz_pick_outputs(out_folder)

    written: List[Path] = []
    for p, row, points in valid_rows:
        xyz_um = _ccf_apdvml_to_xyz_um(points, res=res)
        entry = _fast_entry_xyz_um(row, xyz_um, res=res)
        tip = _project_deepest_tip_um(xyz_um)
        fn = _write_xyz_pick(out_folder, p, np.vstack([entry, tip]).tolist())
        written.append(fn)
    _write_single_xyz_pick_alias(out_folder, written)
    if skipped:
        print(f"Skipped empty/invalid probe slot(s): {', '.join(map(str, skipped))}.")
    print(f"Fast xyz_picks wrote {len(written)} file(s).")
    return written


def _probe_ccf_points(row) -> np.ndarray:
    points = row["points"] if getattr(row, "dtype", None).names else row[0]
    return _normalise_ccf_points(points)


def _normalise_ccf_points(points) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        if arr.size < 3:
            return np.zeros((0, 3), dtype=np.float64)
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        return np.zeros((0, 3), dtype=np.float64)
    if arr.shape[1] < 3 and arr.shape[0] >= 3:
        arr = arr.T
    if arr.shape[1] < 3:
        return np.zeros((0, 3), dtype=np.float64)
    arr = arr[:, :3]
    return arr[np.all(np.isfinite(arr), axis=1)]


def _valid_probe_ccf_rows(pc) -> tuple[List[tuple[int, object, np.ndarray]], List[int]]:
    valid: List[tuple[int, object, np.ndarray]] = []
    skipped: List[int] = []
    for p in range(pc.shape[0]):
        row = pc[p, 0]
        points = _probe_ccf_points(row)
        if points.shape[0] < 2:
            skipped.append(p + 1)
            continue
        valid.append((p, row, points))
    return valid, skipped


def _probe_ccf_trajectory_coords(row) -> np.ndarray:
    names = getattr(row, "dtype", None).names
    if names and "trajectory_coords" in names:
        return np.asarray(row["trajectory_coords"], dtype=np.float64)
    try:
        return np.asarray(row[2], dtype=np.float64)
    except Exception:
        return np.zeros((0, 3), dtype=np.float64)


def _ccf_apdvml_to_xyz_um(ccf_apdvml: np.ndarray, res: int = 10) -> np.ndarray:
    ccf = np.asarray(ccf_apdvml, dtype=np.float64)
    out = np.empty_like(ccf, dtype=np.float64)
    out[..., 0] = ccf[..., 2] * float(res) - _ALLEN_BREGMA_MLAPDV_UM[0]
    out[..., 1] = _ALLEN_BREGMA_MLAPDV_UM[1] - ccf[..., 0] * float(res)
    out[..., 2] = _ALLEN_BREGMA_MLAPDV_UM[2] - ccf[..., 1] * float(res)
    return out


def _project_deepest_tip_um(xyz_um: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz_um, dtype=np.float64)
    center = xyz.mean(axis=0)
    vector = np.linalg.svd(xyz - center, full_matrices=False)[2][0]
    deepest = xyz[int(np.argmin(xyz[:, 2]))]
    return center + np.dot(deepest - center, vector) / np.dot(vector, vector) * vector


def _fast_entry_xyz_um(row, xyz_um: np.ndarray, res: int = 10) -> np.ndarray:
    coords = _probe_ccf_trajectory_coords(row)
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3) if coords.size else coords
    if coords.size:
        entry_ccf = coords[0].copy()
        entry_ccf[:2] -= 1.0
        return _ccf_apdvml_to_xyz_um(entry_ccf, res=res)

    center = xyz_um.mean(axis=0)
    vector = np.linalg.svd(xyz_um - center, full_matrices=False)[2][0]
    shallowest = xyz_um[int(np.argmax(xyz_um[:, 2]))]
    return center + np.dot(shallowest - center, vector) / np.dot(vector, vector) * vector


def _write_xyz_pick(out_folder: Path, zero_based_idx: int, xyz_picks) -> Path:
    fn = out_folder / f"xyz_picks_shank{zero_based_idx + 1}.json"
    with open(fn, "w") as f:
        json.dump({"xyz_picks": xyz_picks}, f, indent=2)
    return fn


def _clear_xyz_pick_outputs(out_folder: Path) -> None:
    for fp in [out_folder / "xyz_picks.json", *out_folder.glob("xyz_picks_shank*.json")]:
        try:
            fp.unlink()
        except FileNotFoundError:
            pass


def _write_single_xyz_pick_alias(out_folder: Path, written: List[Path]) -> None:
    """Write the IBL-style single-probe name when only one shank is present."""
    if len(written) != 1:
        return
    with open(written[0]) as f:
        payload = json.load(f)
    with open(out_folder / "xyz_picks.json", "w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# 2. ALF extraction
# ---------------------------------------------------------------------------

_FAST_ALF_FILES = (
    "channels.localCoordinates.npy",
    "channels.rawInd.npy",
    "clusters.channels.npy",
    "clusters.depths.npy",
    "spikes.clusters.npy",
    "spikes.depths.npy",
    "spikes.times.npy",
    "spikes.amps.npy",
    "clusters.waveforms.npy",
    "clusters.peakToTrough.npy",
)


def extract_alf(ks_dir: str | Path, ephys_path: str | Path, out_dir: str | Path,
                compute_rms: bool = False, mode: str = "auto",
                force: bool = False) -> Path:
    """Extract the ALF files used by the histology workflow into ``out_dir``.

    ``mode='auto'`` (the GUI default) first uses a fast Kilosort-native exporter.
    It writes the spikes/clusters/channel arrays consumed by this app and the IBL
    alignment GUI without loading heavy Kilosort feature matrices. If the required
    Kilosort files are absent, it falls back to the stock IBL extractor.

    The per-channel RMS/QC map (``extract_rmsmap``) streams the **entire** raw AP
    binary window-by-window and is by far the slowest part (many minutes, and
    I/O-bound when the binary is on a network drive). It is only consumed by the
    IBL alignment GUI's RMS display, not by xyz_picks, the channel map or the unit
    distribution, so it is skipped unless ``compute_rms`` is set.
    """
    mode = str(mode or "auto").lower()
    if mode not in {"auto", "fast", "ibl"}:
        raise ValueError(f"Unknown ALF extraction mode: {mode!r}")

    ks_dir = Path(ks_dir)
    ephys_path = Path(ephys_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if mode in {"auto", "fast"}:
        try:
            _extract_alf_fast(ks_dir, ephys_path, out_dir, force=force)
        except Exception as exc:
            if mode == "fast":
                raise
            print(f"Fast ALF extraction unavailable ({exc}); falling back to IBL extractor.")
        else:
            if compute_rms:
                _extract_rms_maps(ephys_path, out_dir)
            return out_dir

    _extract_alf_ibl(ks_dir, ephys_path, out_dir, compute_rms=compute_rms, force=force)
    return out_dir


def _extract_alf_ibl(ks_dir: Path, ephys_path: Path, out_dir: Path,
                     compute_rms: bool = False, force: bool = False) -> None:
    """Run the stock IBL converter. This is slower but maximally complete."""
    from atlaselectrophysiology.extract_files import ks2_to_alf, extract_rmsmap, _sample2v
    import spikeglx

    efiles = spikeglx.glob_ephys_files(ephys_path)
    for efile in efiles:
        if efile.get("ap") and efile.ap.exists():
            ks2_to_alf(ks_dir, ephys_path, out_dir, bin_file=efile.ap,
                       ampfactor=_sample2v(efile.ap), label=None, force=force)
            if compute_rms:
                extract_rmsmap(efile.ap, out_folder=out_dir, spectra=False)
        if compute_rms and efile.get("lf") and efile.lf.exists():
            extract_rmsmap(efile.lf, out_folder=out_dir)


def _extract_rms_maps(ephys_path: Path, out_dir: Path) -> None:
    """Compute optional RMS maps using IBL's raw-binary streamer."""
    from atlaselectrophysiology.extract_files import extract_rmsmap
    import spikeglx

    efiles = spikeglx.glob_ephys_files(ephys_path)
    for efile in efiles:
        if efile.get("ap") and efile.ap.exists():
            extract_rmsmap(efile.ap, out_folder=out_dir, spectra=False)
        if efile.get("lf") and efile.lf.exists():
            extract_rmsmap(efile.lf, out_folder=out_dir)


def _extract_alf_fast(ks_dir: Path, ephys_path: Path, out_dir: Path,
                      force: bool = False) -> None:
    """Write the minimal ALF set directly from Kilosort outputs.

    This avoids ``pc_features.npy`` and the full phy/ephys metric pass that make
    the stock converter slow on network drives. Large spike vectors are streamed
    in chunks sized from available RAM.
    """
    if not force and _fast_alf_complete(out_dir):
        print("Fast ALF outputs already present; skipping Kilosort conversion.")
        return

    times_f = ks_dir / "spike_times.npy"
    clusters_f = ks_dir / "spike_clusters.npy"
    channels_f = ks_dir / "channel_positions.npy"
    missing = [str(p) for p in (times_f, clusters_f, channels_f) if not p.exists()]
    if missing:
        raise FileNotFoundError("missing fast Kilosort inputs: " + ", ".join(missing))

    sample_rate = _read_sample_rate(ks_dir, ephys_path)
    channel_positions = np.asarray(np.load(channels_f), dtype=np.float32)
    if channel_positions.ndim != 2 or channel_positions.shape[1] < 2:
        raise ValueError(f"Expected channel_positions.npy to be (n_channels, 2), got {channel_positions.shape}")
    np.save(out_dir / "channels.localCoordinates.npy", channel_positions[:, :2])
    if (ks_dir / "channel_map.npy").exists():
        _copy_optional_array(ks_dir / "channel_map.npy", out_dir / "channels.rawInd.npy")
    else:
        np.save(out_dir / "channels.rawInd.npy", np.arange(channel_positions.shape[0], dtype=np.int32))

    spike_times = np.load(times_f, mmap_mode="r")
    spike_clusters = np.load(clusters_f, mmap_mode="r")
    n_spikes = int(np.asarray(spike_times).shape[0])
    if int(np.asarray(spike_clusters).shape[0]) != n_spikes:
        raise ValueError("spike_times.npy and spike_clusters.npy have different lengths")

    max_cluster = int(np.max(spike_clusters)) if n_spikes else -1
    max_cluster = max(max_cluster, _max_metric_cluster_id(ks_dir))
    n_clusters = max_cluster + 1
    if n_clusters <= 0:
        raise ValueError("No clusters found in spike_clusters.npy")

    chunk = _auto_spike_chunk_size(n_spikes)
    print(
        f"Fast ALF extraction: {n_spikes:,} spikes, {n_clusters:,} cluster slots, "
        f"{sample_rate:g} Hz, chunk {chunk:,} spikes."
    )
    counts, amp_sums = _write_spike_core_arrays(
        ks_dir, out_dir, spike_times, spike_clusters, sample_rate, n_clusters, chunk
    )
    channels, depths, amps = _cluster_arrays(
        ks_dir, channel_positions[:, :2], n_clusters, counts, amp_sums
    )
    np.save(out_dir / "clusters.channels.npy", channels.astype(np.int32, copy=False))
    np.save(out_dir / "clusters.depths.npy", depths.astype(np.float32, copy=False))
    np.save(out_dir / "clusters.amps.npy", amps.astype(np.float32, copy=False))
    _write_cluster_waveforms(ks_dir, out_dir, n_clusters, channels)
    _write_spike_depths_from_clusters(out_dir, spike_clusters, depths, chunk)
    print("Fast ALF extraction complete.")


def _fast_alf_complete(out_dir: Path) -> bool:
    return all((out_dir / name).exists() for name in _FAST_ALF_FILES)


def _read_sample_rate(ks_dir: Path, ephys_path: Path) -> float:
    """Read the AP sample rate from Kilosort params, SpikeGLX meta, or fallback."""
    for name in ("params.py", "old_params.py"):
        fp = ks_dir / name
        if not fp.exists():
            continue
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        m = re.search(r"\bsample_rate\s*=\s*([0-9]+(?:\.[0-9]*)?)", text)
        if m:
            return float(m.group(1))

    if ephys_path.exists():
        try:
            metas = list(ephys_path.glob("*.ap.meta"))
            if not metas:
                metas = list(ephys_path.glob("*/*.ap.meta"))
            if not metas:
                metas = [next(ephys_path.rglob("*.ap.meta"))]
        except (OSError, StopIteration):
            metas = []
        for meta in metas:
            try:
                for line in meta.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if line.startswith("imSampRate="):
                        return float(line.split("=", 1)[1])
            except (OSError, ValueError):
                pass

    print("Could not read sample rate; using 30000 Hz.")
    return 30000.0


def _auto_spike_chunk_size(n_spikes: int) -> int:
    """Pick a chunk size from available RAM and CPU count."""
    available = 2 * 1024 ** 3
    try:
        import psutil
        available = int(psutil.virtual_memory().available)
    except Exception:
        pass
    cpu = os.cpu_count() or 1
    target = min(max(available // max(cpu, 2), 64 * 1024 ** 2), 512 * 1024 ** 2)
    # Core pass touches time, cluster, optional amp/template arrays and output buffers.
    bytes_per_spike = 40
    chunk = int(target // bytes_per_spike)
    return max(100_000, min(max(n_spikes, 1), min(chunk, 5_000_000)))


def _copy_optional_array(src: Path, dst: Path) -> None:
    if src.exists():
        np.save(dst, np.asarray(np.load(src, mmap_mode="r")))


def _max_metric_cluster_id(ks_dir: Path) -> int:
    metrics = ks_dir / "metrics.csv"
    if not metrics.exists():
        return -1
    max_id = -1
    try:
        with open(metrics, newline="", encoding="utf-8", errors="ignore") as f:
            for row in csv.DictReader(f):
                cid = _safe_int(row.get("cluster_id"), default=-1)
                max_id = max(max_id, cid)
    except OSError:
        return -1
    return max_id


def _write_spike_core_arrays(ks_dir: Path, out_dir: Path, spike_times, spike_clusters,
                             sample_rate: float, n_clusters: int, chunk: int):
    """Stream spike vectors to ALF and return cluster count/amplitude sums."""
    from numpy.lib.format import open_memmap

    n_spikes = int(np.asarray(spike_times).shape[0])
    times_out = open_memmap(out_dir / "spikes.times.npy", mode="w+",
                            dtype=np.float64, shape=(n_spikes,))
    clusters_out = open_memmap(out_dir / "spikes.clusters.npy", mode="w+",
                               dtype=np.int32, shape=(n_spikes,))

    amps = _matching_mmap(ks_dir / "amplitudes.npy", n_spikes)
    amps_out = None
    if amps is not None:
        amps_out = open_memmap(out_dir / "spikes.amps.npy", mode="w+",
                               dtype=np.float32, shape=(n_spikes,))
    else:
        amps_out = open_memmap(out_dir / "spikes.amps.npy", mode="w+",
                               dtype=np.float32, shape=(n_spikes,))

    templates = _matching_mmap(ks_dir / "spike_templates.npy", n_spikes)
    templates_out = None
    if templates is not None:
        templates_out = open_memmap(out_dir / "spikes.templates.npy", mode="w+",
                                    dtype=np.int32, shape=(n_spikes,))

    counts = np.zeros(n_clusters, dtype=np.int64)
    amp_sums = np.zeros(n_clusters, dtype=np.float64)
    for start in range(0, n_spikes, chunk):
        stop = min(start + chunk, n_spikes)
        sl = slice(start, stop)
        clu = np.asarray(spike_clusters[sl]).reshape(-1).astype(np.int64, copy=False)
        times_out[sl] = np.asarray(spike_times[sl]).reshape(-1).astype(np.float64, copy=False) / sample_rate
        clusters_out[sl] = clu.astype(np.int32, copy=False)
        counts += np.bincount(clu, minlength=n_clusters)[:n_clusters]
        if amps is not None and amps_out is not None:
            amp = np.asarray(amps[sl]).reshape(-1).astype(np.float32, copy=False)
            amps_out[sl] = amp
            amp_sums += np.bincount(clu, weights=amp, minlength=n_clusters)[:n_clusters]
        elif amps_out is not None:
            amps_out[sl] = np.zeros(stop - start, dtype=np.float32)
        if templates is not None and templates_out is not None:
            templates_out[sl] = np.asarray(templates[sl]).reshape(-1).astype(np.int32, copy=False)

    del times_out, clusters_out, amps_out, templates_out
    return counts, amp_sums


def _matching_mmap(path: Path, n_rows: int):
    if not path.exists():
        return None
    arr = np.load(path, mmap_mode="r")
    if int(np.asarray(arr).shape[0]) != int(n_rows):
        print(f"Skipping {path.name}: length {np.asarray(arr).shape[0]} does not match {n_rows} spikes.")
        return None
    return arr


def _cluster_arrays(ks_dir: Path, channel_positions: np.ndarray, n_clusters: int,
                    counts: np.ndarray, amp_sums: np.ndarray):
    """Build clusters.channels/depths/amps arrays indexed by cluster id."""
    n_channels = int(channel_positions.shape[0])
    fallback_channel = int(np.argsort(channel_positions[:, 1])[n_channels // 2])
    channels = np.full(n_clusters, fallback_channel, dtype=np.int32)
    depths = np.full(n_clusters, float(channel_positions[fallback_channel, 1]), dtype=np.float32)
    amps = np.zeros(n_clusters, dtype=np.float32)
    np.divide(amp_sums, counts, out=amps, where=counts > 0)

    filled = np.zeros(n_clusters, dtype=bool)
    metrics = ks_dir / "metrics.csv"
    if metrics.exists():
        try:
            with open(metrics, newline="", encoding="utf-8", errors="ignore") as f:
                for row in csv.DictReader(f):
                    cid = _safe_int(row.get("cluster_id"), default=-1)
                    if cid < 0 or cid >= n_clusters:
                        continue
                    peak = _safe_int(row.get("peak_channel"), default=-1)
                    if 0 <= peak < n_channels:
                        channels[cid] = peak
                        depths[cid] = float(channel_positions[peak, 1])
                        filled[cid] = True
                    amp = _safe_float(row.get("amplitude"), default=np.nan)
                    if np.isfinite(amp):
                        amps[cid] = float(amp)
        except OSError:
            pass

    missing = (~filled) & (counts > 0)
    if np.any(missing):
        template_channels = _dominant_template_channels(ks_dir, n_clusters)
        if template_channels is not None:
            valid = missing & (template_channels >= 0) & (template_channels < n_channels)
            channels[valid] = template_channels[valid].astype(np.int32, copy=False)
            depths[valid] = channel_positions[channels[valid], 1].astype(np.float32, copy=False)
            filled[valid] = True

    # Empty cluster ids still need valid rows because spikes.clusters indexes these arrays.
    depths[~np.isfinite(depths)] = float(np.nanmedian(channel_positions[:, 1]))
    return channels, depths, amps


def _dominant_template_channels(ks_dir: Path, n_clusters: int):
    """Fallback cluster channel estimate from Kilosort templates."""
    st_f = ks_dir / "spike_templates.npy"
    sc_f = ks_dir / "spike_clusters.npy"
    templates_f = ks_dir / "templates.npy"
    if not (st_f.exists() and sc_f.exists() and templates_f.exists()):
        return None
    spike_templates = np.load(st_f, mmap_mode="r")
    spike_clusters = np.load(sc_f, mmap_mode="r")
    if int(np.asarray(spike_templates).shape[0]) != int(np.asarray(spike_clusters).shape[0]):
        return None
    templates = np.load(templates_f, mmap_mode="r")
    n_templates = int(np.asarray(templates).shape[0])
    if n_templates <= 0:
        return None

    ptp = np.ptp(np.asarray(templates), axis=1)
    template_peak = np.argmax(ptp, axis=1).astype(np.int32)
    del ptp

    counts = np.zeros((n_clusters, n_templates), dtype=np.uint32)
    chunk = _auto_spike_chunk_size(int(np.asarray(spike_clusters).shape[0]))
    for start in range(0, int(np.asarray(spike_clusters).shape[0]), chunk):
        stop = min(start + chunk, int(np.asarray(spike_clusters).shape[0]))
        clu = np.asarray(spike_clusters[start:stop]).reshape(-1).astype(np.int64, copy=False)
        tpl = np.asarray(spike_templates[start:stop]).reshape(-1).astype(np.int64, copy=False)
        ok = (clu >= 0) & (clu < n_clusters) & (tpl >= 0) & (tpl < n_templates)
        np.add.at(counts, (clu[ok], tpl[ok]), 1)
    dominant = np.argmax(counts, axis=1)
    has_any = counts.max(axis=1) > 0
    channels = np.full(n_clusters, -1, dtype=np.int32)
    channels[has_any] = template_peak[dominant[has_any]]
    return channels


def _write_cluster_waveforms(
    ks_dir: Path,
    out_dir: Path,
    n_clusters: int,
    channels: np.ndarray,
) -> None:
    """Write the cluster waveform arrays required by the IBL alignment GUI.

    The GUI only indexes ``clusters.waveforms[:, :, 0]`` and
    ``clusters.peakToTrough``. Kilosort templates already carry the needed shape,
    so this writes one representative peak-channel waveform per cluster without
    running the slow full IBL extraction path.
    """
    templates_f = ks_dir / "templates.npy"
    dominant = _dominant_templates_by_cluster(ks_dir, n_clusters)
    if templates_f.exists() and dominant is not None:
        templates = np.load(templates_f, mmap_mode="r")
        shape = np.asarray(templates).shape
        if len(shape) >= 3 and shape[0] > 0 and shape[1] > 0 and shape[2] > 0:
            n_time = int(shape[1])
            waveforms = np.zeros((n_clusters, n_time, 1), dtype=np.float32)
            for cid in range(n_clusters):
                tpl = int(dominant[cid]) if cid < dominant.size else -1
                if tpl < 0 or tpl >= shape[0]:
                    continue
                ch = int(channels[cid]) if cid < len(channels) else 0
                ch = max(0, min(ch, shape[2] - 1))
                waveforms[cid, :, 0] = np.asarray(templates[tpl, :, ch], dtype=np.float32)
            np.save(out_dir / "clusters.waveforms.npy", waveforms)
            np.save(out_dir / "clusters.peakToTrough.npy", _peak_to_trough_ms(waveforms[:, :, 0]))
            return

    # Last-resort placeholders keep the GUI load path alive when templates are absent.
    waveforms = np.zeros((n_clusters, 82, 1), dtype=np.float32)
    np.save(out_dir / "clusters.waveforms.npy", waveforms)
    np.save(out_dir / "clusters.peakToTrough.npy", np.zeros(n_clusters, dtype=np.float32))


def _dominant_templates_by_cluster(ks_dir: Path, n_clusters: int) -> Optional[np.ndarray]:
    """Return the most frequent Kilosort template id for each cluster id."""
    st_f = ks_dir / "spike_templates.npy"
    sc_f = ks_dir / "spike_clusters.npy"
    templates_f = ks_dir / "templates.npy"
    if not (st_f.exists() and sc_f.exists() and templates_f.exists()):
        return None
    spike_templates = np.load(st_f, mmap_mode="r")
    spike_clusters = np.load(sc_f, mmap_mode="r")
    if int(np.asarray(spike_templates).shape[0]) != int(np.asarray(spike_clusters).shape[0]):
        return None
    n_templates = int(np.asarray(np.load(templates_f, mmap_mode="r")).shape[0])
    if n_templates <= 0:
        return None

    counts = np.zeros((n_clusters, n_templates), dtype=np.uint32)
    n_spikes = int(np.asarray(spike_clusters).shape[0])
    chunk = _auto_spike_chunk_size(n_spikes)
    for start in range(0, n_spikes, chunk):
        stop = min(start + chunk, n_spikes)
        clu = np.asarray(spike_clusters[start:stop]).reshape(-1).astype(np.int64, copy=False)
        tpl = np.asarray(spike_templates[start:stop]).reshape(-1).astype(np.int64, copy=False)
        ok = (clu >= 0) & (clu < n_clusters) & (tpl >= 0) & (tpl < n_templates)
        np.add.at(counts, (clu[ok], tpl[ok]), 1)
    dominant = np.argmax(counts, axis=1).astype(np.int32)
    dominant[counts.max(axis=1) == 0] = -1
    return dominant


def _peak_to_trough_ms(waveforms: np.ndarray, sample_rate: float = 30000.0) -> np.ndarray:
    """Estimate peak-to-trough duration in milliseconds for each cluster waveform."""
    arr = np.asarray(waveforms, dtype=np.float32)
    out = np.zeros(arr.shape[0], dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] == 0:
        return out
    for i, wf in enumerate(arr):
        if not np.any(np.isfinite(wf)) or np.allclose(wf, 0):
            continue
        trough = int(np.nanargmin(wf))
        if trough + 1 < wf.size:
            peak = trough + int(np.nanargmax(wf[trough:]))
        else:
            peak = trough
        out[i] = float(peak - trough) * 1000.0 / float(sample_rate)
    return out


def _write_spike_depths_from_clusters(out_dir: Path, spike_clusters, depths: np.ndarray,
                                      chunk: int) -> None:
    from numpy.lib.format import open_memmap

    n_spikes = int(np.asarray(spike_clusters).shape[0])
    out = open_memmap(out_dir / "spikes.depths.npy", mode="w+",
                      dtype=np.float32, shape=(n_spikes,))
    for start in range(0, n_spikes, chunk):
        stop = min(start + chunk, n_spikes)
        clu = np.asarray(spike_clusters[start:stop]).reshape(-1).astype(np.int64, copy=False)
        ok = (clu >= 0) & (clu < len(depths))
        vals = np.full(stop - start, np.nan, dtype=np.float32)
        vals[ok] = depths[clu[ok]]
        out[start:stop] = vals
    del out


def _safe_int(value, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# 3. channel locations + all-shanks merge  (port of load_data_local.py)
# ---------------------------------------------------------------------------

def _shank_split(chn_coords_all: np.ndarray):
    """Return (n_shanks, list_of (orig_idx, chn_coords)) like load_data_local."""
    chn_x = np.unique(chn_coords_all[:, 0])
    n_shanks = int(np.sum(np.diff(chn_x) > 100) + 1)
    out = []
    if n_shanks == 1:
        out.append((None, chn_coords_all))
        return n_shanks, out
    for i in range(n_shanks):
        lo, hi = chn_x[i * 2], chn_x[i * 2 + 1]
        mask = (chn_coords_all[:, 0] >= lo) & (chn_coords_all[:, 0] <= hi)
        out.append((np.where(mask)[0], chn_coords_all[mask, :]))
    return n_shanks, out


def _latest_alignment_key(keys):
    """Most recent alignment by timestamp.

    Keys are ISO timestamps; our auto proposals carry an ``auto_`` prefix. A plain
    string sort would rank ``auto_2026...`` after a later manual ``2026...`` upload
    (``'a' > '2'``), so parse the timestamp and pick the genuinely latest, so a
    refined GUI alignment always wins over the earlier auto proposal.
    """
    def stamp(k):
        s = k[5:] if k.startswith("auto_") else k
        try:
            return datetime.fromisoformat(s)
        except ValueError:
            return datetime.min
    return max(keys, key=lambda k: (stamp(k), k))


def _read_alignment(folder: Path, shank_idx: int, n_shanks: int, which: str):
    """Return (feature_prev, track_prev) or (None, None) for the 'original' track."""
    if which == "original":
        return None, None
    name = "prev_alignments.json" if n_shanks == 1 else f"prev_alignments_shank{shank_idx + 1}.json"
    fp = folder / name
    if not fp.exists():
        return None, None
    with open(fp) as f:
        aligns = json.load(f)
    if not aligns:
        return None, None
    key = _latest_alignment_key(aligns.keys())
    feature = np.array(aligns[key][0])
    track = np.array(aligns[key][1])
    return feature, track


def _xyz_pick_files_for_recorded_shank(hist_folder: Path, shank_idx: int, n_shanks: int) -> List[Path]:
    if n_shanks == 1:
        canonical = hist_folder / "xyz_picks.json"
        if canonical.exists():
            return [canonical]
        sparse = _single_valid_probe_xyz_pick(hist_folder)
        if sparse is not None:
            return [sparse]
        return sorted(hist_folder.glob("xyz_picks_shank*.json"))
    return sorted(hist_folder.glob(f"xyz_picks_shank{shank_idx + 1}.json"))


def _single_valid_probe_xyz_pick(hist_folder: Path) -> Optional[Path]:
    probe_ccf = hist_folder / "probe_ccf.mat"
    if not probe_ccf.exists():
        return None
    try:
        import scipy.io as sio
        pc = sio.loadmat(str(probe_ccf))["probe_ccf"]
        valid_rows, _skipped = _valid_probe_ccf_rows(pc)
    except Exception:
        return None
    if len(valid_rows) != 1:
        return None
    fn = hist_folder / f"xyz_picks_shank{valid_rows[0][0] + 1}.json"
    return fn if fn.exists() else None


def _channel_dict_for_shank(brain_regions, chn_coords, orig_idx) -> Dict[str, dict]:
    """Port of load_data_local.create_channel_dict (keys channel_0..)."""
    out: Dict[str, dict] = {}
    n = brain_regions["id"].size
    for i in range(n):
        channel = {
            "x": float(brain_regions["xyz"][i, 0] * 1e6),
            "y": float(brain_regions["xyz"][i, 1] * 1e6),
            "z": float(brain_regions["xyz"][i, 2] * 1e6),
            "axial": float(chn_coords[i, 1]),
            "lateral": float(chn_coords[i, 0]),
            "brain_region_id": int(brain_regions["id"][i]),
            "brain_region": str(brain_regions["acronym"][i]),
        }
        if orig_idx is not None:
            channel["original_channel_idx"] = int(orig_idx[i])
        out[f"channel_{i}"] = channel
    return out


def _resolve_local_coordinates(hist_folder: Path, ks_dir: Optional[str | Path] = None) -> np.ndarray:
    """Locate (or synthesise) ``channels.localCoordinates.npy`` for the session.

    ALF extraction writes this file, but it is just the per-channel probe geometry
    (lateral/axial micrometres) that Kilosort already stores as
    ``channel_positions.npy``. When the ALF file is absent we reuse the Kilosort
    geometry and cache it as the ALF name, so the "AP_histology is enough" path
    works without re-extracting the raw ephys (and the IBL GUI finds it too).
    """
    local = hist_folder / "channels.localCoordinates.npy"
    if local.exists():
        return np.load(local)

    seen: set[str] = set()
    search: List[Path] = []
    for base in ([Path(ks_dir)] if ks_dir else []) + [
        hist_folder, hist_folder.parent, hist_folder.parent.parent
    ]:
        key = str(base)
        if key not in seen:
            seen.add(key)
            search.append(base)

    for base in search:
        cand = base / "channel_positions.npy"
        if cand.exists():
            coords = np.asarray(np.load(cand), dtype=np.float64)
            np.save(local, coords)  # cache for re-runs and the IBL alignment GUI
            print(f"Derived channels.localCoordinates.npy from {cand}")
            return coords

    raise FileNotFoundError(
        "channels.localCoordinates.npy not found and no Kilosort channel_positions.npy "
        f"could be located near {hist_folder}. Either run ALF extraction (Channel map tab "
        "-> 'Run ALF extraction first', with the Kilosort and ephys folders set) or set the "
        "Kilosort folder on the Setup tab so the probe geometry can be reused."
    )


def compute_channel_locations(
    hist_folder: str | Path,
    out_folder: Optional[str | Path] = None,
    alignment: str = "original",
    brain_atlas=None,
    write_per_shank: bool = True,
    ks_dir: Optional[str | Path] = None,
) -> Dict[str, Path]:
    """Compute per-channel CCF locations for every shank and the merged file.

    ``alignment='original'`` uses the un-refined track from the histology picks
    (the "AP_histology is enough" path). ``alignment='latest'`` reuses the most
    recent saved IBL alignment (``prev_alignments_shankN.json``) if present,
    reproducing the refined channel maps.

    Writes ``channel_locations_shankN.json`` (per shank) and the merged
    ``channel_locations_all_shanks.json`` (keyed by original channel index).
    """
    from iblatlas.atlas import AllenAtlas, ALLEN_CCF_LANDMARKS_MLAPDV_UM
    from ibllib.pipes.ephys_alignment import EphysAlignment

    hist_folder = Path(hist_folder)
    out_folder = Path(out_folder) if out_folder else hist_folder
    out_folder.mkdir(parents=True, exist_ok=True)
    ba = brain_atlas or AllenAtlas(25)

    chn_coords_all = _resolve_local_coordinates(hist_folder, ks_dir)
    n_shanks, shanks = _shank_split(chn_coords_all)

    all_shanks: Dict[str, dict] = {}
    written: Dict[str, Path] = {}

    for shank_idx, (orig_idx, chn_coords) in enumerate(shanks):
        # xyz picks for this shank
        picks = _xyz_pick_files_for_recorded_shank(hist_folder, shank_idx, n_shanks)
        if not picks:
            continue
        with open(picks[0]) as f:
            xyz_picks = np.array(json.load(f)["xyz_picks"]) / 1e6
        chn_depths = chn_coords[:, 1]

        feature_prev, track_prev = _read_alignment(hist_folder, shank_idx, n_shanks, alignment)
        ephysalign = EphysAlignment(
            xyz_picks, chn_depths, brain_atlas=ba,
            feature_prev=feature_prev, track_prev=track_prev,
        )
        feature = ephysalign.feature_init if feature_prev is None else feature_prev
        track = ephysalign.track_init if track_prev is None else track_prev
        xyz_channels = ephysalign.get_channel_locations(feature, track)

        brain_regions = ba.regions.get(ba.get_labels(xyz_channels))
        brain_regions["xyz"] = xyz_channels
        brain_regions["lateral"] = chn_coords[:, 0]
        brain_regions["axial"] = chn_coords[:, 1]

        chan_dict = _channel_dict_for_shank(brain_regions, chn_coords, orig_idx)

        if write_per_shank:
            per = dict(chan_dict)
            per["origin"] = {"bregma": ALLEN_CCF_LANDMARKS_MLAPDV_UM["bregma"].tolist()}
            name = "channel_locations.json" if n_shanks == 1 else \
                f"channel_locations_shank{shank_idx + 1}.json"
            fp = out_folder / name
            with open(fp, "w") as f:
                json.dump(per, f, indent=2, separators=(",", ": "))
            written[name] = fp

        # Merge into the all-shanks dict, keyed by original channel index.
        for i, ch in enumerate(chan_dict.values()):
            key = str(int(orig_idx[i])) if orig_idx is not None else str(i)
            all_shanks[key] = ch

    merged = {"origin": {"bregma": ALLEN_CCF_LANDMARKS_MLAPDV_UM["bregma"].tolist()}}
    for key in sorted(all_shanks, key=lambda k: int(k)):
        merged[key] = all_shanks[key]
    all_fp = out_folder / "channel_locations_all_shanks.json"
    with open(all_fp, "w") as f:
        json.dump(merged, f, indent=4)
    written["channel_locations_all_shanks.json"] = all_fp
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv: Optional[List[str]] = None) -> int:
    """Parse the CLI subcommand and dispatch to the matching bridge function."""
    parser = argparse.ArgumentParser(description="AP_histology -> IBL bridge")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_xyz = sub.add_parser("xyz_picks")
    p_xyz.add_argument("hist_folder")
    p_xyz.add_argument("--probe-ccf", default=None)
    p_xyz.add_argument("--res", type=int, default=10)
    p_xyz.add_argument("--mode", choices=["auto", "fast", "ibl"], default="fast",
                       help="xyz_picks backend: fast local conversion by default")

    p_alf = sub.add_parser("extract_alf")
    p_alf.add_argument("ks_dir")
    p_alf.add_argument("ephys_dir")
    p_alf.add_argument("out_dir")
    p_alf.add_argument("--mode", choices=["auto", "fast", "ibl"], default="auto",
                       help="ALF extraction backend: fast Kilosort-native exporter by default")
    p_alf.add_argument("--force", action="store_true",
                       help="overwrite existing ALF arrays instead of reusing them")
    p_alf.add_argument("--rms", action="store_true",
                       help="also compute the slow RMS/QC map (streams the whole raw AP binary)")

    p_ch = sub.add_parser("channels")
    p_ch.add_argument("hist_folder")
    p_ch.add_argument("--alignment", choices=["original", "latest"], default="original")
    p_ch.add_argument("--ks", default=None)

    p_all = sub.add_parser("all")
    p_all.add_argument("hist_folder")
    p_all.add_argument("--ks", default=None)
    p_all.add_argument("--ephys", default=None)
    p_all.add_argument("--alignment", choices=["original", "latest"], default="original")
    p_all.add_argument("--xyz-mode", choices=["auto", "fast", "ibl"], default="fast",
                       help="xyz_picks backend used before channel map generation")
    p_all.add_argument("--alf-mode", choices=["auto", "fast", "ibl"], default="auto",
                       help="ALF extraction backend used when --ephys is provided")
    p_all.add_argument("--force-alf", action="store_true",
                       help="overwrite existing ALF arrays during the optional ALF step")
    p_all.add_argument("--rms", action="store_true",
                       help="also compute the slow RMS/QC map (streams the whole raw AP binary)")

    p_prop = sub.add_parser("propose_align")
    p_prop.add_argument("hist_folder")
    p_prop.add_argument("--atlas", default=None)

    args = parser.parse_args(argv)

    if args.cmd == "xyz_picks":
        hf = Path(args.hist_folder)
        ccf = args.probe_ccf or (hf / "probe_ccf.mat")
        out = compute_xyz_picks(ccf, hf, res=args.res, mode=args.mode)
        print(json.dumps({"xyz_picks": [str(p) for p in out]}))
    elif args.cmd == "extract_alf":
        out = extract_alf(args.ks_dir, args.ephys_dir, args.out_dir,
                          compute_rms=args.rms, mode=args.mode, force=args.force)
        print(json.dumps({"alf_out": str(out)}))
    elif args.cmd == "channels":
        out = compute_channel_locations(args.hist_folder, alignment=args.alignment, ks_dir=args.ks)
        print(json.dumps({k: str(v) for k, v in out.items()}))
    elif args.cmd == "all":
        hf = Path(args.hist_folder)
        if args.ks and args.ephys:
            extract_alf(args.ks, args.ephys, hf, compute_rms=args.rms,
                        mode=args.alf_mode, force=args.force_alf)
        compute_xyz_picks(hf / "probe_ccf.mat", hf, mode=args.xyz_mode)
        out = compute_channel_locations(hf, alignment=args.alignment, ks_dir=args.ks)
        print(json.dumps({k: str(v) for k, v in out.items()}))
    elif args.cmd == "propose_align":
        from .auto_align import propose_alignment
        out = propose_alignment(args.hist_folder, atlas_path=args.atlas)
        for s in out["shanks"]:
            print(f"shank {s['shank']}: offset {s['offset_um']:+.0f} um  "
                  f"(conf {s['confidence']:.2f}, peak {s['peak_corr']:.2f})  "
                  f"{'GOOD' if s['good'] else 'low confidence'}")
        if not out["pairing_ok"]:
            print(f"WARNING: {out['n_recorded_shanks']} recorded shanks vs "
                  f"{out['n_xyz_picks']} xyz_picks tracks - check pairing")
        print(json.dumps({"report": out.get("report", ""),
                          "pairing_ok": out["pairing_ok"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

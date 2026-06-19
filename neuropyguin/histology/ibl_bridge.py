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
import glob
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# 1. xyz_picks  (port of process_histology.py)
# ---------------------------------------------------------------------------

def compute_xyz_picks(
    probe_ccf_path: str | Path,
    out_folder: str | Path,
    res: int = 10,
    brain_atlas=None,
) -> List[Path]:
    """Convert ``probe_ccf.mat`` to per-shank ``xyz_picks_shankN.json``.

    Faithful port of ``D:\\NPX\\process_histology.py``: each shank's CCF points
    (10um voxels, [AP, DV, ML]) are scaled to microns, mapped to bregma xyz via
    ``AllenAtlas.ccf2xyz`` and reduced to the tip/top of a best-fit insertion.
    """
    import scipy.io as sio
    from iblatlas.atlas import AllenAtlas, Insertion

    probe_ccf_path = Path(probe_ccf_path)
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    ba = brain_atlas or AllenAtlas(res)
    mat = sio.loadmat(str(probe_ccf_path))
    pc = mat["probe_ccf"]
    n_shanks = pc.shape[0]

    written: List[Path] = []
    for p in range(n_shanks):
        points = np.array(pc[p][0][0], dtype=np.float64)  # (N,3) [AP,DV,ML] voxels
        ccf_apdvml = points * res
        bregma_apdvml = ba.ccf2xyz(ccf_apdvml, ccf_order="apdvml") * 1e6
        ins = Insertion.from_track(bregma_apdvml / 1e6, brain_atlas=ba)
        xyz_picks = {"xyz_picks": (ins.xyz * 1e6).tolist()}
        fn = out_folder / f"xyz_picks_shank{p + 1}.json"
        with open(fn, "w") as f:
            json.dump(xyz_picks, f, indent=2)
        written.append(fn)
    return written


# ---------------------------------------------------------------------------
# 2. ALF extraction  (wrap atlaselectrophysiology.extract_files.extract_data)
# ---------------------------------------------------------------------------

def extract_alf(ks_dir: str | Path, ephys_path: str | Path, out_dir: str | Path,
                compute_rms: bool = False) -> Path:
    """Run the IBL ALF extraction (spikes/clusters/channels) into ``out_dir``.

    The per-channel RMS/QC map (``extract_rmsmap``) streams the **entire** raw AP
    binary window-by-window and is by far the slowest part (many minutes, and
    I/O-bound when the binary is on a network drive). It is only consumed by the
    IBL alignment GUI's RMS display, not by xyz_picks, the channel map or the unit
    distribution, so it is skipped unless ``compute_rms`` is set.
    """
    from atlaselectrophysiology.extract_files import ks2_to_alf, extract_rmsmap, _sample2v
    import spikeglx

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    efiles = spikeglx.glob_ephys_files(Path(ephys_path))
    for efile in efiles:
        if efile.get("ap") and efile.ap.exists():
            ks2_to_alf(Path(ks_dir), Path(ephys_path), out_dir, bin_file=efile.ap,
                       ampfactor=_sample2v(efile.ap), label=None, force=True)
            if compute_rms:
                extract_rmsmap(efile.ap, out_folder=out_dir, spectra=False)
        if compute_rms and efile.get("lf") and efile.lf.exists():
            extract_rmsmap(efile.lf, out_folder=out_dir)
    return out_dir


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
        if n_shanks == 1:
            picks = sorted(hist_folder.glob("*xyz_picks.json"))
        else:
            picks = sorted(hist_folder.glob(f"*xyz_picks_shank{shank_idx + 1}.json"))
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
    parser = argparse.ArgumentParser(description="AP_histology -> IBL bridge")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_xyz = sub.add_parser("xyz_picks")
    p_xyz.add_argument("hist_folder")
    p_xyz.add_argument("--probe-ccf", default=None)
    p_xyz.add_argument("--res", type=int, default=10)

    p_alf = sub.add_parser("extract_alf")
    p_alf.add_argument("ks_dir")
    p_alf.add_argument("ephys_dir")
    p_alf.add_argument("out_dir")
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
    p_all.add_argument("--rms", action="store_true",
                       help="also compute the slow RMS/QC map (streams the whole raw AP binary)")

    p_prop = sub.add_parser("propose_align")
    p_prop.add_argument("hist_folder")
    p_prop.add_argument("--atlas", default=None)

    args = parser.parse_args(argv)

    if args.cmd == "xyz_picks":
        hf = Path(args.hist_folder)
        ccf = args.probe_ccf or (hf / "probe_ccf.mat")
        out = compute_xyz_picks(ccf, hf, res=args.res)
        print(json.dumps({"xyz_picks": [str(p) for p in out]}))
    elif args.cmd == "extract_alf":
        out = extract_alf(args.ks_dir, args.ephys_dir, args.out_dir, compute_rms=args.rms)
        print(json.dumps({"alf_out": str(out)}))
    elif args.cmd == "channels":
        out = compute_channel_locations(args.hist_folder, alignment=args.alignment, ks_dir=args.ks)
        print(json.dumps({k: str(v) for k, v in out.items()}))
    elif args.cmd == "all":
        hf = Path(args.hist_folder)
        if args.ks and args.ephys:
            extract_alf(args.ks, args.ephys, hf, compute_rms=args.rms)
        compute_xyz_picks(hf / "probe_ccf.mat", hf)
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

"""Objective, reviewable probe-to-atlas alignment proposal.

Human alignment in the IBL GUI is biased: people anchor region boundaries so their
region of interest lands where they expect units. This proposes an alignment from
the data instead.

The electrophysiology already encodes anatomy: gray matter (nuclei, cortical and
hippocampal cell layers) fires, while fiber tracts and ventricles are near-silent.
For each recorded shank we therefore find the rigid vertical **offset** that best
matches the measured firing-rate depth profile to a *tissue template* built from
the atlas regions along the histology track (gray = 1, fiber = 0.15, ventricle/out
= 0). Offset-only is deliberate: scaling/warping is where bias and error re-enter.

Outputs (all in the histology folder):
  * ``prev_alignments_shankN.json``  the proposal as an entry the IBL GUI loads
  * ``alignment_report.md``          human-readable recommendations + how to apply
  * ``alignment_proposal_shankN.png``per-shank diagnostic figure

A confidence score is reported. Deep homogeneous gray matter (e.g. midbrain) has
weak ephys landmarks, so the proposal honestly flags low-confidence shanks where
the histology track should be trusted as-is rather than "corrected".
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import List

import numpy as np


# --------------------------------------------------------------------------- #
# tissue template + small numeric helpers
# --------------------------------------------------------------------------- #
def _tissue_classifier(atlas_path):
    """Return (vectorized id->weight, id->acronym). Gray=1, fiber=0.15, vent/out=0."""
    from . import atlas as hatlas

    st = hatlas.load_structure_tree(hatlas.resolve_atlas_path(atlas_path) / hatlas._STRUCTURE_FN)
    id2path = {int(r["id"]): str(r.get("structure_id_path", "")) for _, r in st.iterrows()}
    id2acr = {int(r["id"]): str(r["acronym"]) for _, r in st.iterrows()}

    def weight(rid):
        rid = int(rid)
        p = id2path.get(rid, "")
        if "/1009/" in p:      # fiber tracts
            return 0.15
        if "/73/" in p:        # ventricular systems
            return 0.0
        if id2acr.get(rid, "") in ("root", "void") or rid in (0, 997):
            return 0.0
        return 1.0

    return np.vectorize(weight, otypes=[float]), id2acr


def _z(a: np.ndarray) -> np.ndarray:
    """Z-score an array (mean 0, unit std); pass through unchanged if std is 0."""
    a = np.asarray(a, float)
    a = a - a.mean()
    s = a.std()
    return a / s if s > 0 else a


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two arrays (0.0 if either is constant)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.mean(_z(a) * _z(b)))


def _firing_per_depth(spike_depths: np.ndarray, query_depths: np.ndarray, bw_um=30.0) -> np.ndarray:
    """Smoothed spike density (a.u.) evaluated at each channel depth."""
    if spike_depths.size == 0 or query_depths.size == 0:
        return np.zeros(len(query_depths))
    from scipy.ndimage import gaussian_filter1d

    lo, hi = query_depths.min() - 100.0, query_depths.max() + 100.0
    edges = np.arange(lo, hi + 10.0, 10.0)
    h = np.histogram(spike_depths, bins=edges)[0].astype(float)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    h = gaussian_filter1d(h, max(bw_um / 10.0, 1.0))
    return np.interp(query_depths, ctr, h)


def _shank_groups(chn_all: np.ndarray):
    """Split channels into shank groups by lateral gaps >100um. Returns list of
    (orig_idx, chn_coords, lateral_median)."""
    x = np.unique(chn_all[:, 0])
    n_shanks = int(np.sum(np.diff(x) > 100) + 1)
    out = []
    if n_shanks == 1:
        out.append((np.arange(len(chn_all)), chn_all, float(np.median(chn_all[:, 0]))))
        return out
    for i in range(n_shanks):
        lo, hi = x[i * 2], x[i * 2 + 1]
        mask = (chn_all[:, 0] >= lo) & (chn_all[:, 0] <= hi)
        out.append((np.where(mask)[0], chn_all[mask, :], float(np.median(chn_all[mask, 0]))))
    return out


# --------------------------------------------------------------------------- #
# main entry
# --------------------------------------------------------------------------- #
def propose_alignment(hist_folder, atlas_path=None, brain_atlas=None,
                      offset_grid_um=None, write=True, rigid=True) -> dict:
    """Compute a per-shank alignment proposal; optionally write GUI/report files."""
    from iblatlas.atlas import AllenAtlas
    from ibllib.pipes.ephys_alignment import EphysAlignment

    hist_folder = Path(hist_folder)
    ba = brain_atlas or AllenAtlas(25)
    weight_fn, id2acr = _tissue_classifier(atlas_path)
    grid = np.arange(-600, 601, 20) if offset_grid_um is None else np.asarray(offset_grid_um)

    chn_all = np.load(hist_folder / "channels.localCoordinates.npy")
    clu_ch = np.asarray(np.load(hist_folder / "clusters.channels.npy")).astype(int)
    s_dep = np.asarray(np.load(hist_folder / "spikes.depths.npy")).astype(float)
    s_clu = np.asarray(np.load(hist_folder / "spikes.clusters.npy")).astype(int)
    clu_lat = chn_all[clu_ch, 0]

    groups = _shank_groups(chn_all)
    n_shanks = len(groups)
    n_xyz = len(sorted(hist_folder.glob("xyz_picks*.json")))

    shanks: List[dict] = []
    for si, (orig_idx, chn_coords, lat_med) in enumerate(groups):
        picks = (sorted(hist_folder.glob("*xyz_picks.json")) if n_shanks == 1
                 else sorted(hist_folder.glob(f"*xyz_picks_shank{si + 1}.json")))
        if not picks:
            continue
        xyz = np.array(json.loads(picks[0].read_text())["xyz_picks"]) / 1e6
        chn_depths = chn_coords[:, 1].astype(float)
        ea = EphysAlignment(xyz, chn_depths, brain_atlas=ba)
        feat0 = np.asarray(ea.feature_init, float)

        sel = np.where(np.abs(clu_lat - lat_med) <= 40)[0]
        sd = s_dep[np.isin(s_clu, sel)]
        fr = _firing_per_depth(sd, chn_depths)

        corrs = []
        for off in grid:
            track = feat0 + off * 1e-6
            xyz_ch = ea.get_channel_locations(feat0, track)
            rids = ba.regions.get(ba.get_labels(xyz_ch))["id"]
            corrs.append(_corr(fr, weight_fn(rids)))
        corrs = np.array(corrs)
        best_i = int(np.argmax(corrs))
        best_off = float(grid[best_i])
        peak = float(corrs[best_i])
        far = corrs[np.abs(grid - best_off) > 120]
        conf = float(peak - far.max()) if far.size else peak
        good = bool(peak >= 0.30 and conf >= 0.12)

        # regions spanned (original) for the report
        rids0 = ba.regions.get(ba.get_labels(ea.get_channel_locations(feat0, feat0)))["id"]
        acrs = []
        for r in rids0:
            a = id2acr.get(int(r), "")
            if a and (not acrs or acrs[-1] != a):
                acrs.append(a)

        shanks.append({
            "shank": si + 1, "n_clusters": int(len(sel)), "n_spikes": int(sd.size),
            "depth_um": [float(chn_depths.min()), float(chn_depths.max())],
            "offset_um": best_off, "peak_corr": peak, "confidence": conf, "good": good,
            "regions": acrs, "grid": grid.tolist(), "corrs": corrs.tolist(),
            "feature": feat0.tolist(), "track": (feat0 + best_off * 1e-6).tolist(),
            "firing": fr.tolist(), "chn_depths": chn_depths.tolist(),
            "xyz_pick_name": picks[0].name,
        })

    # Rigid-probe constraint: a multi-shank probe is one rigid body inserted at a
    # single angle with coplanar tips, so the depth offset is shared across shanks.
    # Propagate the confidence-weighted offset of the confident shank(s) to the
    # low-confidence ones (whose own ephys can't pin the depth) instead of leaving
    # them unshifted.
    confident = [s for s in shanks if s["good"]]
    shared = None
    if rigid and confident and len(shanks) > 1:
        w = np.array([s["confidence"] for s in confident], float)
        shared = float(np.average([s["offset_um"] for s in confident], weights=w))
    for s in shanks:
        if s["good"]:
            s["applied_offset_um"], s["source"] = s["offset_um"], "own"
        elif shared is not None:
            s["applied_offset_um"], s["source"] = shared, "shared (rigid probe)"
        else:
            s["applied_offset_um"], s["source"] = 0.0, "none (kept original)"

    summary = {
        "n_recorded_shanks": n_shanks, "n_xyz_picks": n_xyz,
        "pairing_ok": (n_shanks == n_xyz), "shanks": shanks,
        "shared_offset_um": shared,
    }
    if write:
        _write_proposals(hist_folder, shanks)
        _write_figures(hist_folder, shanks, weight_fn, ba)
        summary["report"] = str(_write_report(hist_folder, summary))
    return summary


def _write_proposals(hist_folder: Path, shanks: List[dict]) -> None:
    """Write an ``auto_`` prev_alignments entry the IBL GUI lists, for every shank.

    Each shank uses its applied offset: its own value when confident, the shared
    rigid-probe offset when not (and 0 if nothing is confident). The entry is always
    written so the option is visible in the GUI drop-down for every shank.
    """
    key = "auto_" + datetime.datetime.now().replace(microsecond=0).isoformat()
    multi = len(shanks) > 1
    for s in shanks:
        name = (f"prev_alignments_shank{s['shank']}.json" if multi else "prev_alignments.json")
        fp = hist_folder / name
        data = {}
        if fp.exists():
            try:
                data = json.loads(fp.read_text())
            except (OSError, ValueError):
                data = {}
        off = float(s.get("applied_offset_um", s["offset_um"] if s["good"] else 0.0))
        f2 = np.asarray(s["feature"], float)
        # 3 collinear points (same offset) so the GUI has an interior reference line
        # to draw; a bare 2-point alignment can crash its rendering on reload.
        feature = np.array([f2[0], 0.5 * (f2[0] + f2[-1]), f2[-1]])
        data[key] = [feature.tolist(), (feature + off * 1e-6).tolist()]
        fp.write_text(json.dumps(data, indent=2))


def _write_figures(hist_folder: Path, shanks: List[dict], weight_fn, ba) -> None:
    """Save a per-shank diagnostic PNG (firing profile + offset-vs-correlation).

    Silently returns if matplotlib is unavailable so the proposal still completes.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    from ibllib.pipes.ephys_alignment import EphysAlignment  # noqa: F401 (atlas already loaded)
    for s in shanks:
        fr = np.asarray(s["firing"]); cd = np.asarray(s["chn_depths"])
        grid = np.asarray(s["grid"]); corrs = np.asarray(s["corrs"])
        fig, ax = plt.subplots(1, 2, figsize=(8.5, 9))
        ax[0].plot(_z(fr), cd, "0.2", lw=1.5)
        ax[0].set_title(f"shank {s['shank']} firing (z)")
        ax[0].set_ylabel("depth from tip (um)")
        ax[1].plot(corrs, grid, "C2-o", ms=3)
        ax[1].axhline(s["offset_um"], color="C3", ls="--",
                      label=f"proposed {s['offset_um']:+.0f}um")
        ax[1].axhline(0, color="0.6", ls=":")
        ax[1].set_xlabel("firing-template corr"); ax[1].set_ylabel("offset (um)")
        tag = "GOOD" if s["good"] else "LOW confidence"
        ax[1].set_title(f"{tag}  peak={s['peak_corr']:.2f} conf={s['confidence']:.2f}")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(hist_folder / f"alignment_proposal_shank{s['shank']}.png", dpi=110)
        plt.close(fig)


def _write_report(hist_folder: Path, summary: dict) -> Path:
    """Write the human-readable ``alignment_report.md`` and return its path."""
    lines = ["# Probe alignment proposal", "",
             f"_Generated {datetime.datetime.now().replace(microsecond=0).isoformat()}_", "",
             "Objective offset that best matches firing to atlas gray/white structure ",
             "(no human bias). Offset-only; review before accepting.", ""]
    if not summary["pairing_ok"]:
        lines += [
            f"> **Warning:** {summary['n_recorded_shanks']} recorded shank(s) but "
            f"{summary['n_xyz_picks']} xyz_picks track(s). The recorded shanks may be "
            "paired with the wrong traced tracks. Verify the per-shank regions look "
            "anatomically right before trusting any alignment.", ""]
    if summary.get("shared_offset_um") is not None:
        lines += [
            f"Rigid-probe constraint applied: shared offset "
            f"**{summary['shared_offset_um']:+.0f} um** from the confident shank(s) is "
            "used for the low-confidence shanks (the probe is one rigid body, coplanar "
            "tips, single insertion angle).", ""]
    lines += ["| shank | applied offset (um) | source | confidence | units | depth (um) | regions |",
              "|---|---|---|---|---|---|---|"]
    for s in summary["shanks"]:
        regs = ", ".join(s["regions"][:10])
        lines.append(
            f"| {s['shank']} | {s.get('applied_offset_um', 0):+.0f} | {s.get('source', '')} | "
            f"{s['confidence']:.2f} (peak {s['peak_corr']:.2f}, raw {s['offset_um']:+.0f}) | "
            f"{s['n_clusters']} | {s['depth_um'][0]:.0f}-{s['depth_um'][1]:.0f} | {regs} |")
    lines += ["",
              "## How to apply",
              "1. Open the IBL alignment GUI (Histology -> IBL refine -> Launch).",
              "2. In the alignment drop-down (top right) pick the entry starting with "
              "`auto_` for this shank, then press **Get Data**.",
              "3. Review: the region bars should line up with the firing/RMS features.",
              "4. If good, **Upload** to save it; if it looks worse than `original`, "
              "select `original` and align manually.",
              "",
              "Low-confidence shanks (deep homogeneous gray matter such as midbrain) have "
              "few ephys landmarks, so the histology track is already your best estimate and "
              "the proposed offset is small/uncertain by design."]
    fp = hist_folder / "alignment_report.md"
    fp.write_text("\n".join(lines), encoding="utf-8")
    return fp

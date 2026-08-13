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
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    a = a - a.mean()
    s = a.std()
    return a / s if s > 0 else a


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two arrays (0.0 if either is constant)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.mean(_z(a) * _z(b)))


def _unit_interval(a: np.ndarray) -> np.ndarray:
    """Robustly map values to 0..1 using percentile clipping."""
    a = np.nan_to_num(np.asarray(a, float), nan=0.0, posinf=0.0, neginf=0.0)
    if a.size == 0:
        return a
    lo, hi = np.percentile(a, [2.0, 98.0])
    if hi <= lo:
        return np.zeros_like(a, dtype=float)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


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


def _firing_feature_sets(spike_depths: np.ndarray, query_depths: np.ndarray,
                         bandwidths_um=(35.0, 70.0, 120.0)) -> list[dict]:
    """Return multi-scale firing and firing-transition profiles."""
    from scipy.ndimage import gaussian_filter1d

    out = []
    for bw in bandwidths_um:
        fr = _firing_per_depth(spike_depths, query_depths, bw_um=bw)
        log_fr = np.log1p(fr)
        active = _unit_interval(log_fr)
        if len(log_fr) > 1:
            edge = np.abs(np.gradient(_z(log_fr)))
            edge = gaussian_filter1d(edge, max(1.0, bw / 80.0))
        else:
            edge = np.zeros_like(log_fr, dtype=float)
        out.append({
            "bw_um": float(bw),
            "firing": log_fr,
            "active": active,
            "edge": edge,
        })
    return out


def _atlas_profiles(ea, feature: np.ndarray, grid_um: np.ndarray, ba, weight_fn) -> list[dict]:
    """Precompute atlas tissue and boundary profiles for each offset."""
    from scipy.ndimage import gaussian_filter1d

    out = []
    for off in grid_um:
        track = np.asarray(feature, float) + float(off) * 1e-6
        xyz_ch = ea.get_channel_locations(feature, track)
        rids = np.asarray(ba.regions.get(ba.get_labels(xyz_ch))["id"], dtype=int)
        tissue = np.asarray(weight_fn(rids), dtype=float)
        boundary = np.zeros_like(tissue, dtype=float)
        if tissue.size > 1:
            changes = (rids[1:] != rids[:-1]).astype(float)
            boundary[:-1] += changes
            boundary[1:] += changes
            boundary += np.abs(np.gradient(tissue))
            boundary = gaussian_filter1d(boundary, 1.0)
        out.append({
            "offset_um": float(off),
            "rids": rids,
            "tissue": tissue,
            "boundary": boundary,
            "brain_fraction": float(np.mean(tissue > 0.05)) if tissue.size else 0.0,
        })
    return out


def _score_offset_curve(ea, feature: np.ndarray, chn_depths: np.ndarray,
                        spike_depths: np.ndarray, grid_um: np.ndarray, ba,
                        weight_fn) -> dict:
    """Score offsets with multi-scale firing, tissue and atlas-boundary cues."""
    atlas = _atlas_profiles(ea, feature, grid_um, ba, weight_fn)
    firing_sets = _firing_feature_sets(spike_depths, chn_depths)
    per_scale = []

    for fs in firing_sets:
        curve = []
        for ap in atlas:
            tissue = ap["tissue"]
            boundary = ap["boundary"]
            brain_fraction = ap["brain_fraction"]
            tissue_score = _corr(fs["firing"], tissue)
            boundary_score = _corr(fs["edge"], boundary)
            outside = 1.0 - np.clip(tissue, 0.0, 1.0)
            outside_penalty = float(np.mean(fs["active"] * outside)) if outside.size else 0.0
            brain_penalty = max(0.0, 0.55 - brain_fraction) * 0.35
            curve.append(0.62 * tissue_score + 0.28 * boundary_score
                         - 0.18 * outside_penalty - brain_penalty)
        per_scale.append(np.asarray(curve, dtype=float))

    if per_scale:
        score = np.median(np.vstack(per_scale), axis=0)
        scale_offsets = [float(grid_um[int(np.nanargmax(c))]) for c in per_scale]
    else:
        score = np.zeros_like(grid_um, dtype=float)
        scale_offsets = []

    return {
        "score": np.nan_to_num(score, nan=-1.0, posinf=-1.0, neginf=-1.0),
        "per_scale": per_scale,
        "scale_offsets_um": scale_offsets,
        "atlas": atlas,
        "firing_sets": firing_sets,
    }


def _contiguous_width_um(grid_um: np.ndarray, mask: np.ndarray, center_i: int) -> float:
    """Width of the contiguous True run around center_i."""
    if mask.size == 0 or not mask[center_i]:
        return 0.0
    left = right = int(center_i)
    while left > 0 and mask[left - 1]:
        left -= 1
    while right + 1 < mask.size and mask[right + 1]:
        right += 1
    step = float(np.median(np.diff(grid_um))) if grid_um.size > 1 else 0.0
    return float((right - left + 1) * max(step, 1.0))


def _peak_metrics(grid_um: np.ndarray, curve: np.ndarray,
                  scale_offsets_um=None) -> dict:
    """Describe whether a score peak is sharp, stable and away from grid edges."""
    grid_um = np.asarray(grid_um, dtype=float)
    curve = np.nan_to_num(np.asarray(curve, dtype=float),
                          nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    if grid_um.size == 0 or curve.size == 0 or not np.isfinite(curve).any():
        return {
            "offset_um": 0.0, "peak": 0.0, "margin": 0.0, "confidence": 0.0,
            "plateau_width_um": 0.0, "stability_um": float("inf"),
            "stability_score": 0.0, "edge_limited": True, "good": False,
        }

    best_i = int(np.nanargmax(curve))
    best_off = float(grid_um[best_i])
    peak = float(curve[best_i])
    far = curve[np.abs(grid_um - best_off) > 120.0]
    far_best = float(np.nanmax(far)) if far.size else float(np.nanmin(curve))
    margin = float(max(0.0, peak - far_best))

    scales = np.asarray(scale_offsets_um or [best_off], dtype=float)
    stability_um = float(np.std(scales)) if scales.size else 0.0
    stability_score = float(np.exp(-stability_um / 140.0))
    drop = max(0.05, margin * 0.5)
    plateau_width_um = _contiguous_width_um(grid_um, curve >= peak - drop, best_i)
    plateau_factor = min(1.0, 180.0 / max(plateau_width_um, 1.0))
    edge_limited = bool(best_i == 0 or best_i == curve.size - 1)
    confidence = float(margin * (0.35 + 0.65 * stability_score) * plateau_factor)
    if edge_limited:
        confidence *= 0.35
    good = bool(
        peak >= 0.20
        and margin >= 0.10
        and confidence >= 0.07
        and stability_score >= 0.45
        and not edge_limited
    )
    return {
        "offset_um": best_off,
        "peak": peak,
        "margin": margin,
        "confidence": confidence,
        "plateau_width_um": float(plateau_width_um),
        "stability_um": stability_um,
        "stability_score": stability_score,
        "edge_limited": edge_limited,
        "good": good,
    }


def _consensus_offset(shanks: list[dict], grid_um: np.ndarray) -> tuple:
    """Pick the best shared rigid offset from supported score curves."""
    eligible = [s for s in shanks if s.get("good") and s.get("confidence", 0.0) > 0.0]
    if not eligible:
        return None, {}, []

    grid_um = np.asarray(grid_um, dtype=float)
    combined = np.zeros_like(grid_um, dtype=float)
    source_shanks = []
    for s in eligible:
        curve = np.asarray(s.get("score_curve", s.get("corrs", [])), dtype=float)
        if curve.size != grid_um.size or not np.isfinite(curve).any():
            continue
        baseline = float(np.nanpercentile(curve, 25.0))
        peak = float(np.nanmax(curve))
        if peak <= baseline:
            continue
        shape = np.clip((curve - baseline) / (peak - baseline), 0.0, 1.5)
        combined += float(s["confidence"]) * shape
        source_shanks.append(int(s["shank"]))

    if not source_shanks:
        return None, {}, []
    metrics = _peak_metrics(grid_um, combined)
    if not metrics["good"]:
        return None, metrics, source_shanks
    return float(metrics["offset_um"]), metrics, source_shanks


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
        score_data = _score_offset_curve(ea, feat0, chn_depths, sd, grid, ba, weight_fn)
        score_curve = score_data["score"]
        metrics = _peak_metrics(grid, score_curve, score_data["scale_offsets_um"])
        best_off = float(metrics["offset_um"])
        peak = float(metrics["peak"])
        conf = float(metrics["confidence"])
        good = bool(metrics["good"])
        fr = score_data["firing_sets"][1]["firing"] if score_data["firing_sets"] else np.zeros_like(chn_depths)

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
            "score_margin": float(metrics["margin"]),
            "score_stability_um": float(metrics["stability_um"]),
            "score_stability": float(metrics["stability_score"]),
            "plateau_width_um": float(metrics["plateau_width_um"]),
            "edge_limited": bool(metrics["edge_limited"]),
            "scale_offsets_um": [float(v) for v in score_data["scale_offsets_um"]],
            "regions": acrs, "grid": grid.tolist(), "corrs": score_curve.tolist(),
            "score_curve": score_curve.tolist(),
            "feature": feat0.tolist(), "track": (feat0 + best_off * 1e-6).tolist(),
            "firing": fr.tolist(), "chn_depths": chn_depths.tolist(),
            "xyz_pick_name": picks[0].name,
        })

    # Rigid-probe constraint: a multi-shank probe is one rigid body inserted at a
    # single angle with coplanar tips, so the depth offset is shared across shanks.
    # Propagate the confidence-weighted offset of the confident shank(s) to the
    # low-confidence ones (whose own ephys can't pin the depth) instead of leaving
    # them unshifted.
    shared = None
    shared_metrics = {}
    shared_sources: list[int] = []
    if rigid and len(shanks) > 1:
        shared, shared_metrics, shared_sources = _consensus_offset(shanks, grid)
    for s in shanks:
        if shared is not None:
            s["applied_offset_um"] = shared
            if s["shank"] in shared_sources:
                s["source"] = "rigid consensus (own support)"
            else:
                s["source"] = "rigid consensus"
        elif s["good"]:
            s["applied_offset_um"], s["source"] = s["offset_um"], "own"
        else:
            s["applied_offset_um"], s["source"] = 0.0, "none (kept original)"

    summary = {
        "n_recorded_shanks": n_shanks, "n_xyz_picks": n_xyz,
        "pairing_ok": (n_shanks == n_xyz), "shanks": shanks,
        "shared_offset_um": shared,
        "shared_metrics": shared_metrics,
        "shared_sources": shared_sources,
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
        if s.get("applied_offset_um") is not None:
            ax[1].axhline(s["applied_offset_um"], color="C0", ls="-.",
                          label=f"applied {s['applied_offset_um']:+.0f}um")
        ax[1].axhline(0, color="0.6", ls=":")
        ax[1].set_xlabel("robust firing-atlas score"); ax[1].set_ylabel("offset (um)")
        tag = "GOOD" if s["good"] else "LOW confidence"
        ax[1].set_title(
            f"{tag}\npeak={s['peak_corr']:.2f}  m={s.get('score_margin', 0):.2f}  "
            f"c={s['confidence']:.2f}")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(hist_folder / f"alignment_proposal_shank{s['shank']}.png", dpi=110)
        plt.close(fig)


def _write_report(hist_folder: Path, summary: dict) -> Path:
    """Write the human-readable ``alignment_report.md`` and return its path."""
    lines = ["# Probe alignment proposal", "",
             f"_Generated {datetime.datetime.now().replace(microsecond=0).isoformat()}_", "",
             "Objective offset from multi-scale firing profiles, atlas tissue class, ",
             "and region-boundary landmarks. Offset-only; review before accepting.", ""]
    if not summary["pairing_ok"]:
        lines += [
            f"> **Warning:** {summary['n_recorded_shanks']} recorded shank(s) but "
            f"{summary['n_xyz_picks']} xyz_picks track(s). The recorded shanks may be "
            "paired with the wrong traced tracks. Verify the per-shank regions look "
            "anatomically right before trusting any alignment.", ""]
    if summary.get("shared_offset_um") is not None:
        sm = summary.get("shared_metrics", {})
        src = ", ".join(str(s) for s in summary.get("shared_sources", []))
        lines += [
            f"Rigid-probe constraint applied: shared offset "
            f"**{summary['shared_offset_um']:+.0f} um** from shank(s) {src or 'n/a'} is "
            "used across the probe. Consensus is chosen from the supported score curve, "
            "not by averaging conflicting offsets.",
            f"Consensus confidence: {sm.get('confidence', 0.0):.2f}; "
            f"margin: {sm.get('margin', 0.0):.2f}.", ""]
    lines += ["| shank | applied offset (um) | raw offset (um) | source | confidence | peak score | margin | scale peaks (um) | units | depth (um) | regions |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary["shanks"]:
        regs = ", ".join(s["regions"][:10])
        scales = ", ".join(f"{v:+.0f}" for v in s.get("scale_offsets_um", []))
        flags = []
        if s.get("edge_limited"):
            flags.append("edge")
        if s.get("plateau_width_um", 0) > 220:
            flags.append(f"plateau {s['plateau_width_um']:.0f} um")
        flag_text = f" [{', '.join(flags)}]" if flags else ""
        lines.append(
            f"| {s['shank']} | {s.get('applied_offset_um', 0):+.0f} | {s.get('offset_um', 0):+.0f} | "
            f"{s.get('source', '')} | "
            f"{s['confidence']:.2f}{flag_text} | {s['peak_corr']:.2f} | "
            f"{s.get('score_margin', 0):.2f} | {scales} | "
            f"{s['n_clusters']} | {s['depth_um'][0]:.0f}-{s['depth_um'][1]:.0f} | {regs} |")
    lines += ["",
              "## Audit notes",
              "- A shank is marked confident only when the best offset has a clear margin, "
              "is stable across smoothing scales, and is not pinned at the search-grid edge.",
              "- Broad plateaus are treated cautiously because they usually mean the shank "
              "is in homogeneous tissue and lacks ephys landmarks.",
              "- For multi-shank probes, low-confidence shanks inherit the rigid consensus "
              "only when at least one shank has a sharp supported score peak.",
              "",
              "## How to apply",
              "1. Open the IBL alignment GUI (Histology -> IBL refine -> Launch).",
              "2. In the alignment drop-down (top right) pick the entry starting with "
              "`auto_` for this shank, then press **Get Data**.",
              "3. Review: the region bars should line up with the firing/RMS features.",
              "4. If good, **Upload** to save it; if it looks worse than `original`, "
              "select `original` and align manually.",
              "",
              "Low-confidence shanks have few reliable ephys landmarks on their own. When "
              "a rigid consensus is available, review the inherited shared offset against "
              "the histology track; otherwise the original alignment is kept."]
    fp = hist_folder / "alignment_report.md"
    fp.write_text("\n".join(lines), encoding="utf-8")
    return fp

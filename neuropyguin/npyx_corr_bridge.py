"""Bridge between the GUI and the npyx.corr correlation toolbox.

This module exposes a small, stable surface (method listing, per-method
metadata, and a single dispatcher) that wraps the many correlation routines
in ``npyx.corr``. ``run_method`` resolves the datapath, computes the requested
analysis, and returns a plain dict ("payload") describing how to render the
result. The npyx package itself is imported lazily so the GUI can start even
when npyx (or its heavy dependencies) is not yet importable.
"""

from __future__ import annotations

from itertools import islice
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import psutil
from scipy import signal as scipy_signal
import inspect


# Curated, user-facing analysis menu. This is deliberately a SHORT list of the
# correlation analyses a systems neuroscientist actually wants, not a 1:1 dump of
# npyx.corr. Internal helpers (numba kernels, joblib workers, cache-name builders,
# feasibility checks) and text-only repr methods are intentionally excluded; the
# run_method dispatcher still understands the wider set if called directly.
# key, clear label
METHOD_LABELS: List[Tuple[str, str]] = [
    ("acg_3D", "3D ACG vs Firing Rate"),
    ("ccg_3D", "3D CCG vs Firing Rate"),
    ("scaled_acg", "Scaled ACG (cell-type)"),
    ("StarkAbeles2009_ccg_sig", "Monosynaptic CCG significance (Stark-Abeles)"),
    ("spike_time_tiling_coefficient", "Spike Time Tiling Coefficient (STTC)"),
    ("get_cisi", "Cross-ISI Distribution"),
]

# Minimum distinct units each method needs (the GUI uses this to guide the user).
METHOD_MIN_UNITS: Dict[str, int] = {
    "ccg": 2, "get_cm": 2, "pearson_corr_trn": 2, "correlation_index": 2,
    "get_cisi": 2, "synchrony_zscore": 2, "fraction_pop_sync": 2,
    "spike_time_tiling_coefficient": 2, "ccg_sig_stack": 2,
    "StarkAbeles2009_ccg_sig": 2, "ccg_3D": 2,
}

METHOD_META: Dict[str, Dict[str, object]] = {
    "default": {
        "description": "Compute advanced correlation analysis on selected units.",
        "params": {"bin_ms": 0.5, "win_ms": 100.0},
    },
    "ccg": {
        "description": "Cross-correlogram across selected pairs; highlights pairwise lag-locked co-firing structure.",
        "params": {"bin_ms": 0.5, "win_ms": 100.0, "normalize": "Counts"},
    },
    "ccg_hz": {
        "description": "Cross-correlogram in Hertz for selected pairs (rate-normalized).",
        "params": {"bin_ms": 0.5, "win_ms": 100.0, "rate_corrected": False},
    },
    "ccg_sig_stack": {
        "description": "Significance scan of pairwise CCGs; reports significant pairs and modulation strength.",
        "params": {"bin_ms": 0.5, "win_ms": 100.0, "p_th": 0.02, "n_consec_bins": 3, "sgn": 0},
    },
    "gen_sfc": {
        "description": "Build functional correlation graph matrix from significant correlogram interactions.",
        "params": {"bin_ms": 0.5, "win_ms": 100.0, "corr_type": "connections", "metric": "amp_z"},
    },
    "StarkAbeles2009_ccg_sig": {
        "description": "Computes CCG predictor and bin-wise p-values from convolution model.",
        "params": {"bin_ms": 0.5, "win_ms": 100.0, "W_sd_ms": 10.0, "sgn": -1},
    },
    "StarkAbeles2009_ccg_significance": {
        "description": "Tests CCG modulation significance using Stark-Abeles method.",
        "params": {"bin_ms": 0.5, "win_ms": 100.0, "p_th": 0.02, "n_consec": 3, "sgn": 0, "W_sd_ms": 10.0},
    },
    "pearson_corr_trn": {
        "description": "Pearson correlation between binned spike trains from selected units.",
        "params": {"bin_ms": 5.0},
    },
    "correlation_index": {
        "description": "Wong-Meister-Shatz correlation index over selected spike trains.",
        "params": {"dt_ms": 2.0},
    },
    "synchrony": {
        "description": "Synchrony score over selected unit pairs using central CCG window.",
        "params": {"bin_ms": 0.5, "win_ms": 100.0, "sync_win_ms": 1.0},
    },
    "synchrony_zscore": {
        "description": "Z-scored synchrony score over selected unit pairs.",
        "params": {"bin_ms": 0.5, "win_ms": 100.0, "sync_win_ms": 1.0},
    },
    "synchrony_regehr": {
        "description": "Regehr-style synchrony ratio over selected unit pairs.",
        "params": {"bin_ms": 0.5, "win_ms": 100.0, "sync_win_ms": 1.0},
    },
}


PAIRWISE_ONLY_METHODS = {
    "ccg_hz",
    "ccg",
    "ccg_3D",
    "crosscorrelate_cyrille",
    "crosscorr_cyrille",
    "KopelowitzCohen2014_ccg_significance",
    "StarkAbeles2009_ccg_sig",
    "StarkAbeles2009_ccg_significance",
    "get_cisi",
    "synchrony_regehr",
    "synchrony",
    "synchrony_zscore",
    "synchrony_deltaproba",
    "cisi_numba_para",
    "cisi_numba",
    "cisi_chunk",
    "par_process",
    "get_cisi_parprocess",
    "covariance",
    "convert_ccg_to_covariance",
    "cofiring_tags",
    "spike_time_tiling_coefficient",
}


def _ensure_npyx():
    """Import and return ``npyx.corr``, adding npyx to ``sys.path`` if needed.

    Prefers an npyx package vendored next to NeuroPyGuiN and falls back to a
    sibling NeuroPyxels checkout. Imported lazily so app startup does not depend
    on npyx being importable.
    """
    import sys

    # Prefer vendored npyx package bundled inside NeuroPyGuiN for standalone use.
    app_root = Path(__file__).resolve().parents[1]  # .../NeuroPyGuiN
    vendored_parent = app_root
    vendored_pkg = vendored_parent / "npyx"
    if vendored_pkg.exists():
        s = str(vendored_parent)
        if s not in sys.path:
            sys.path.insert(0, s)
    else:
        # Backward-compatible fallback to sibling NeuroPyxels repo.
        repo_root = Path(__file__).resolve().parents[2]  # .../neuropixels_processing
        npyx_repo = repo_root / "NeuroPyxels"
        if npyx_repo.exists():
            s = str(npyx_repo)
            if s not in sys.path:
                sys.path.insert(0, s)
    import npyx.corr as corr  # type: ignore
    return corr


def _has_metadata_folder(dp: Path) -> bool:
    """Return True if ``dp`` holds SpikeGLX (.ap/.lf.meta) or OpenEphys (.oebin) metadata."""
    if not dp.exists() or not dp.is_dir():
        return False
    if any(dp.glob("*.ap.meta")) or any(dp.glob("*.lf.meta")):
        return True
    if (dp / "structure.oebin").exists() or any(dp.glob("*.oebin")):
        return True
    return False


def _has_spike_sorting_outputs(dp: Path) -> bool:
    """Return True if ``dp`` contains Kilosort/phy spike_times.npy and spike_clusters.npy."""
    return dp.exists() and dp.is_dir() and (dp / "spike_times.npy").exists() and (dp / "spike_clusters.npy").exists()


def _score_candidate(candidate: Path, source: Path) -> tuple[int, int, int]:
    """
    Rank candidate folders:
    1) fewer parent hops from source
    2) shallower path (fewer components)
    3) shorter string (stable tie-break)
    """
    hops = 0
    cur = source
    while cur != candidate and candidate not in cur.parents:
        if cur.parent == cur:
            hops = 10_000
            break
        cur = cur.parent
        hops += 1
    return (hops, len(candidate.parts), len(str(candidate)))


def resolve_analysis_datapath(dp: str) -> str:
    """
    Resolve an analysis datapath that contains SpikeGLX/OpenEphys metadata.
    Fallback order: current folder -> parents -> nearby child folders.
    """
    # Some callers may pass a path ending with "."; trim problematic suffixes.
    cleaned = str(dp).strip().rstrip(" .\\/") or str(dp).strip()
    base = Path(cleaned).resolve()

    # If the requested folder is already a spike-sorting output folder, keep it.
    # Advanced correlation methods need the exact unit set from spike_clusters.npy.
    if _has_spike_sorting_outputs(base):
        return str(base)

    candidates: List[Path] = []

    # 1) Exact folder + parents.
    if _has_metadata_folder(base):
        candidates.append(base)
    for parent in base.parents:
        if _has_metadata_folder(parent):
            candidates.append(parent)

    # 2) Immediate children of base/parents (common: "..._imec0/imec0_ks4").
    # pathlib.Parents slicing is not supported in some Python versions.
    frontier = [base] + list(islice(base.parents, 6))
    for node in frontier:
        if not node.exists() or not node.is_dir():
            continue
        for child in node.iterdir():
            if child.is_dir() and _has_metadata_folder(child):
                candidates.append(child)

    # 3) Limited recursive scan near base parent for unusual layouts.
    anchor = base.parent if base.parent.exists() else base
    patterns = ("*.ap.meta", "*.lf.meta", "*.oebin")
    for pat in patterns:
        for hit in anchor.rglob(pat):
            parent = hit.parent
            if _has_metadata_folder(parent):
                candidates.append(parent)

    if candidates:
        # unique + closest to requested folder
        uniq = list({str(c): c for c in candidates}.values())
        best = sorted(uniq, key=lambda c: _score_candidate(c, base))[0]
        return str(best)

    # Return cleaned base path if nothing found; downstream keeps original error text.
    return str(base)


def method_options() -> List[Tuple[str, str]]:
    """Return the ``(method_key, human_label)`` pairs for populating the method picker."""
    return METHOD_LABELS


def method_metadata(method_key: str) -> Dict[str, object]:
    """Return UI metadata (``description`` and default ``params``) for a method.

    Starts from the shared ``default`` entry, overlays any explicit ``METHOD_META``
    entry, and for methods without an explicit entry infers default params and a
    one-line description from the npyx function signature and docstring.
    """
    meta = METHOD_META.get("default", {}).copy()
    specific = METHOD_META.get(method_key, {})
    meta.update(specific)
    # Auto-populate method-specific defaults from function signature for methods
    # not explicitly described above.
    if method_key not in METHOD_META:
        try:
            corr = _ensure_npyx()
            fn = getattr(corr, method_key, None)
            if fn is not None:
                sig = inspect.signature(fn)
                params: Dict[str, object] = {}
                skip = {
                    "dp", "u", "u1", "u2", "U", "U_src", "U_trg", "units",
                    "times", "clusters", "spk1", "spk2", "trains", "train", "L",
                }
                for name, p in sig.parameters.items():
                    if name in skip:
                        continue
                    if p.default is inspect._empty:
                        continue
                    dv = p.default
                    if isinstance(dv, (int, float, str, bool)):
                        params[name] = dv
                if params:
                    meta["params"] = params
                doc = (fn.__doc__ or "").strip()
                if doc:
                    first = doc.splitlines()[0].strip()
                    if first:
                        meta["description"] = first
        except Exception:
            pass
    return meta


def _meta_payload(payload: Dict[str, object], requested_dp: str, resolved_dp: str) -> Dict[str, object]:
    """Annotate a render payload in place with the requested and resolved datapaths."""
    payload["requested_dp"] = requested_dp
    payload["resolved_dp"] = resolved_dp
    return payload


def _samples_from_ms(ms: float, fs: int = 30000) -> int:
    """Convert a duration in milliseconds to a sample count (at least 1) at rate ``fs``."""
    return int(max(1, round(float(ms) * float(fs) / 1000.0)))


def _lag_axis(cbin_ms: float, cwin_ms: float, n_bins: int) -> np.ndarray:
    """Build a symmetric lag axis (in ms) of length ``n_bins`` for a correlogram.

    Uses the natural bin edges when they are long enough, otherwise synthesizes
    an evenly spaced, zero-centred axis from the bin step.
    """
    x = np.arange(-float(cwin_ms) / 2.0, float(cwin_ms) / 2.0 + float(cbin_ms), float(cbin_ms), dtype=float)
    if x.size >= n_bins:
        return x[:n_bins]
    if x.size > 1:
        step = float(np.median(np.diff(x)))
    else:
        step = float(cbin_ms)
    return np.arange(n_bins, dtype=float) * step - (n_bins // 2) * step


def _train(corr, dp: str, u: int) -> np.ndarray:
    """Return unit ``u``'s spike train (sample indices) as an int64 array."""
    return np.asarray(corr.trn(dp, int(u)), dtype=np.int64)


def _train_list(corr, dp: str, units: List[int]) -> List[np.ndarray]:
    """Return the spike trains for ``units`` in order."""
    return [_train(corr, dp, int(u)) for u in units]


def _cross_counts(corr, dp: str, u0: int, u1: int, cbin: float, cwin: float) -> np.ndarray:
    """Return the count-normalized cross-correlogram of (u0, u1) as a 1D array."""
    m = np.asarray(corr.ccg(dp, [int(u0), int(u1)], cbin, cwin, normalize="Counts"), dtype=float)
    return m[0, 1] if m.ndim == 3 and m.shape[0] > 1 else np.ravel(m)


def run_method(method_key: str, dp: str, units: List[int], bin_ms: float, win_ms: float, params: Dict[str, object] | None = None, fs: float = 30000.0) -> Dict[str, object]:
    """Dispatch a single npyx.corr analysis and return a render payload.

    Resolves the datapath, de-duplicates ``units`` (preserving order), validates
    the selection, then routes to the per-method branch matching ``method_key``.
    Each branch returns a dict (via ``_meta_payload``) describing the result and
    how to render it; an unrecognized ``method_key`` returns a fallback text payload.
    ``fs`` is the recording sample rate (Hz), threaded in from the dataset so the
    lag/tiling axes are correct for non-30 kHz probes.
    """
    corr = _ensure_npyx()
    fs = int(round(float(fs))) if fs else 30000
    requested_dp = dp
    dp = resolve_analysis_datapath(dp)
    params = params or {}
    units = [int(u) for u in units]
    units = list(dict.fromkeys(units))
    if not units:
        raise RuntimeError("Select at least one unit.")
    if method_key in PAIRWISE_ONLY_METHODS and len(units) < 2:
        return _meta_payload(
            {
                "kind": "text",
                "title": str(method_key),
                "text": "Select at least two distinct units for this pairwise method.",
            },
            requested_dp,
            dp,
        )

    def _p(name: str, default):
        return params.get(name, default)

    u0 = int(units[0])
    u1 = int(units[1]) if len(units) > 1 else int(units[0])
    cbin = float(max(0.1, _p("bin_ms", bin_ms)))
    cwin = float(max(5.0, _p("win_ms", win_ms)))
    bins = np.arange(-cwin / 2.0, cwin / 2.0 + cbin, cbin, dtype=float)

    # Compatibility fix for older scipy API expected by npyx implementation.
    if not hasattr(scipy_signal, "triang") and hasattr(scipy_signal, "windows") and hasattr(scipy_signal.windows, "triang"):
        scipy_signal.triang = scipy_signal.windows.triang  # type: ignore[attr-defined]
    if not hasattr(corr.sgnl, "triang") and hasattr(scipy_signal, "triang"):
        corr.sgnl.triang = scipy_signal.triang

    if method_key == "acg":
        y = np.asarray(corr.acg(dp, u0, cbin, cwin, fs=fs, normalize="Hertz"), dtype=float)
        return _meta_payload({"kind": "line", "title": f"ACG u{u0}", "x": bins[: y.size], "y": y,
                              "x_label": "Lag (ms)", "y_label": "Firing rate (Hz)"}, requested_dp, dp)

    if method_key == "ccg_hz":
        pairs = [(int(a), int(b)) for i, a in enumerate(units) for b in units[i + 1 :]]
        if not pairs:
            pairs = [(u0, u1)]
        items = []
        for a, b in pairs:
            y = np.asarray(corr.ccg_hz(dp, a, b, cbin, cwin, rate_corrected=bool(_p("rate_corrected", False))), dtype=float)
            items.append({"u1": a, "u2": b, "x": bins[: y.size], "y": y})
        return _meta_payload({"kind": "corr_pairs", "title": "CCG Hz pairs", "items": items}, requested_dp, dp)

    if method_key == "ccg":
        pairs = [(int(a), int(b)) for i, a in enumerate(units) for b in units[i + 1 :]]
        if not pairs:
            pairs = [(u0, u1)]
        items = []
        for a, b in pairs:
            y = _cross_counts(corr, dp, a, b, cbin, cwin)
            items.append({"u1": a, "u2": b, "x": bins[: y.size], "y": y})
        return _meta_payload({"kind": "corr_pairs", "title": "CCG count pairs", "items": items}, requested_dp, dp)

    if method_key == "ccg_3D":
        mat, t_bins, f_bins = corr.ccg_3D(dp, [u0, u1], cbin, cwin, normalize="Hertz")
        return _meta_payload({
            "kind": "image",
            "title": f"CCG 3D u{u0}->{u1}",
            "mat": np.asarray(mat, dtype=float),
            "x": np.asarray(t_bins, dtype=float),
            "y": np.asarray(f_bins, dtype=float),
            "x_label": "Lag (ms)", "y_label": "Firing rate (Hz)", "cbar_label": "CCG (Hz)",
        }, requested_dp, dp)

    if method_key == "acg_3D":
        mat, t_bins, f_bins = corr.acg_3D(dp, u0, cbin, cwin, normalize="Hertz")
        return _meta_payload({
            "kind": "image",
            "title": f"ACG 3D u{u0}",
            "mat": np.asarray(mat, dtype=float),
            "x": np.asarray(t_bins, dtype=float),
            "y": np.asarray(f_bins, dtype=float),
            "x_label": "Lag (ms)", "y_label": "Firing rate (Hz)", "cbar_label": "ACG (Hz)",
        }, requested_dp, dp)

    if method_key == "convert_acg_log":
        acg = np.asarray(corr.acg(dp, u0, cbin, cwin, normalize="Hertz"), dtype=float)
        y, x = corr.convert_acg_log(acg, cbin=cbin, cwin=cwin, n_log_bins=100)
        return _meta_payload({"kind": "line", "title": f"Log-ACG u{u0}", "x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)}, requested_dp, dp)

    if method_key == "ccg_stack":
        stack, _ustack = corr.ccg_stack(dp, U_src=units, U_trg=units, cbin=cbin, cwin=cwin, all_to_all=True, sav=False, again=True)
        # Reduce 3D stack to a 2D peak matrix for plotting
        mat = np.nanmax(np.asarray(stack, dtype=float), axis=2)
        return _meta_payload({"kind": "image", "title": "CCG Stack Peak Matrix", "mat": mat}, requested_dp, dp)

    if method_key == "get_cm":
        mat = np.asarray(corr.get_cm(dp, units, cbin=cbin, cwin=cwin, corrEvaluator="CCG"), dtype=float)
        return _meta_payload({"kind": "image", "title": "Correlation Matrix (CCG synchrony)", "mat": mat,
                              "x_label": "Unit #", "y_label": "Unit #", "cbar_label": "CCG synchrony"}, requested_dp, dp)

    if method_key in {"synchrony_regehr", "synchrony", "synchrony_zscore", "synchrony_deltaproba"}:
        sync_win = float(_p("sync_win_ms", 1.0))
        pairs = [(int(a), int(b)) for i, a in enumerate(units) for b in units[i + 1 :]]
        if not pairs:
            pairs = [(u0, u1)]
        labels: List[str] = []
        vals: List[float] = []
        traces: List[Dict[str, object]] = []
        for a, b in pairs:
            ccg = np.asarray(corr.ccg_hz(dp, a, b, cbin, cwin), dtype=float)
            if method_key == "synchrony_regehr":
                v = float(corr.synchrony_regehr(ccg, cbin=cbin, sync_win=sync_win))
            elif method_key == "synchrony":
                v = float(corr.synchrony(ccg, cbin=cbin, sync_win=sync_win))
            elif method_key == "synchrony_zscore":
                v = float(corr.synchrony_zscore(ccg, cbin=cbin, sync_win=sync_win))
            else:
                v = float(corr.synchrony_deltaproba(ccg, cbin=cbin, sync_win=sync_win))
            labels.append(f"{a}-{b}")
            vals.append(v)
            traces.append({"name": f"{a}-{b}", "x": bins[: ccg.size], "y": ccg})
        return _meta_payload(
            {
                "kind": "pair_bars",
                "title": method_key,
                "labels": labels,
                "values": vals,
                "mean": float(np.nanmean(vals)) if vals else np.nan,
                "sem": float(np.nanstd(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
                "traces": traces,
            },
            requested_dp,
            dp,
        )

    if method_key == "KopelowitzCohen2014_ccg_significance":
        ccg = np.asarray(corr.ccg_hz(dp, u0, u1, cbin, cwin), dtype=float)
        out = corr.KopelowitzCohen2014_ccg_significance(ccg, cbin=cbin, cwin=cwin, ret_values=True)
        return _meta_payload({"kind": "text", "title": "Kopelowitz-Cohen significance", "text": str(out)}, requested_dp, dp)

    if method_key == "StarkAbeles2009_ccg_sig":
        ccg_counts = np.asarray(_cross_counts(corr, dp, u0, u1, cbin, cwin), dtype=float)
        ccg_counts = np.rint(np.clip(ccg_counts, 0, None)).astype(np.int64)
        w_sd_ms = float(_p("W_sd_ms", 10.0))
        pred, pvals = corr.StarkAbeles2009_ccg_sig(ccg_counts, W=max(3, int(round(w_sd_ms / cbin))), WINTYPE="gauss", CALCP=True, sgn=int(_p("sgn", -1)))
        pred = np.asarray(pred, dtype=float).reshape(-1)
        pvals = np.asarray(pvals, dtype=float).reshape(-1)
        x = _lag_axis(cbin, cwin, pred.size)
        return _meta_payload(
            {"kind": "multi_line", "title": "Stark-Abeles predictor + p-values", "x": x, "series": [{"name": "predictor", "y": pred}, {"name": "p_values", "y": pvals}]},
            requested_dp,
            dp,
        )

    if method_key == "StarkAbeles2009_ccg_significance":
        ccg = np.asarray(_cross_counts(corr, dp, u0, u1, cbin, cwin), dtype=float)
        out = corr.StarkAbeles2009_ccg_significance(
            np.rint(np.clip(ccg, 0, None)).astype(np.int64),
            cbin=cbin,
            p_th=float(_p("p_th", 0.02)),
            n_consec=int(_p("n_consec", 3)),
            sgn=int(_p("sgn", 0)),
            W_sd=float(_p("W_sd_ms", 10.0)),
            ret_values=True,
        )
        return _meta_payload({"kind": "text", "title": "Stark-Abeles significance", "text": str(out)}, requested_dp, dp)

    if method_key == "get_cisi":
        t1 = _train(corr, dp, u0)
        t2 = _train(corr, dp, u1)
        cisi = np.asarray(corr.get_cisi(t1, t2, direction=0), dtype=float)
        cisi = cisi[np.isfinite(cisi)]
        if cisi.size == 0:
            return _meta_payload({"kind": "text", "title": "Cross-ISI", "text": "No finite cross-ISI values."}, requested_dp, dp)
        cisi_ms = cisi / float(fs) * 1000.0  # samples -> ms
        hist, edges = np.histogram(cisi_ms, bins=min(120, max(20, int(np.sqrt(cisi_ms.size)))))
        return _meta_payload({"kind": "hist", "title": f"Cross-ISI u{u0}<->u{u1}", "x": edges[:-1], "y": hist.astype(float),
                              "w": np.diff(edges), "x_label": "Cross-ISI (ms)", "y_label": "Count"}, requested_dp, dp)

    if method_key == "pearson_corr":
        rows = [np.asarray(corr.trnb(dp, u, 5), dtype=float) for u in units]
        M = np.vstack(rows)
        mat = np.asarray(corr.pearson_corr(M), dtype=float)
        return _meta_payload({"kind": "image", "title": "Pearson Correlation Matrix", "mat": mat}, requested_dp, dp)

    if method_key == "PSDxy":
        f, pxy = corr.PSDxy(dp, units, cbin, ret=True, sav=False, verbose=False)
        pxy = np.asarray(pxy, dtype=float)
        diag = np.array([pxy[i, i, :] for i in range(min(pxy.shape[0], pxy.shape[1]))], dtype=float)
        return _meta_payload({"kind": "image", "title": "PSD (auto spectra by unit)", "mat": diag, "x": np.asarray(f, dtype=float)}, requested_dp, dp)

    if method_key == "canUse_Nbins":
        ok = bool(corr.canUse_Nbins(a=0.05, w=cwin, b=cbin, n_bins=3))
        return _meta_payload({"kind": "scalar", "title": "Can use 3-bin significance triplets", "value": float(ok)}, requested_dp, dp)

    if method_key == "make_phy_like_spikeClustersTimes":
        # Only cluster ids are needed here; the spike times are intentionally discarded.
        _times, clusters = corr.make_phy_like_spikeClustersTimes(dp, units)
        clusters = np.asarray(clusters, dtype=int)
        uniq, counts = np.unique(clusters, return_counts=True)
        return _meta_payload({"kind": "hist", "title": "Phy-like spike cluster counts", "x": uniq.astype(float), "y": counts.astype(float), "w": np.ones_like(uniq, dtype=float)}, requested_dp, dp)

    if method_key == "make_matrix_2xNevents":
        trains = {int(u): _train(corr, dp, int(u)) for u in units[: min(8, len(units))]}
        m = np.asarray(corr.make_matrix_2xNevents(trains), dtype=float)
        if m.ndim != 2 or m.shape[1] == 0:
            return _meta_payload({"kind": "text", "title": "2xN events matrix", "text": "No events available."}, requested_dp, dp)
        max_cols = min(4000, m.shape[1])
        return _meta_payload({"kind": "image", "title": "2xN events matrix (truncated)", "mat": m[:, :max_cols]}, requested_dp, dp)

    if method_key == "crosscorrelate_cyrille":
        m = np.asarray(corr.crosscorrelate_cyrille(dp, cbin, cwin, units[: max(2, len(units))]), dtype=float)
        pairs = [(int(a), int(b), i, j) for i, a in enumerate(units) for j, b in enumerate(units) if j > i]
        if not pairs:
            pairs = [(u0, u1, 0, 1 if len(units) > 1 else 0)]
        items: List[Dict[str, object]] = []
        for a, b, i, j in pairs:
            if m.ndim == 3 and m.shape[0] > max(i, j):
                y = np.asarray(m[i, j], dtype=float).ravel()
            else:
                y = np.ravel(m)
            x = _lag_axis(cbin, cwin, y.size)
            items.append({"u1": a, "u2": b, "x": x, "y": y})
        return _meta_payload({"kind": "corr_pairs", "title": "Crosscorrelate Cyrille", "items": items}, requested_dp, dp)

    if method_key == "crosscorr_cyrille":
        times, clusters = corr.make_phy_like_spikeClustersTimes(dp, units)
        m = np.asarray(corr.crosscorr_cyrille(np.asarray(times), np.asarray(clusters), cwin, cbin), dtype=float)
        pairs = [(int(a), int(b), i, j) for i, a in enumerate(units) for j, b in enumerate(units) if j > i]
        if not pairs:
            pairs = [(u0, u1, 0, 1 if len(units) > 1 else 0)]
        items: List[Dict[str, object]] = []
        for a, b, i, j in pairs:
            if m.ndim == 3 and m.shape[0] > max(i, j):
                y = np.asarray(m[i, j], dtype=float).ravel()
            else:
                y = np.ravel(m)
            x = _lag_axis(cbin, cwin, y.size)
            items.append({"u1": a, "u2": b, "x": x, "y": y})
        return _meta_payload({"kind": "corr_pairs", "title": "Crosscorr Cyrille", "items": items}, requested_dp, dp)

    if method_key == "get_log_bins_samples":
        y = np.asarray(corr.get_log_bins_samples(log_window_end=max(cwin / 2.0, cbin), n_log_bins=100, fs=30000), dtype=float)
        return _meta_payload({"kind": "line", "title": "Log-bin edges in samples", "x": np.arange(y.size, dtype=float), "y": y}, requested_dp, dp)

    if method_key in {"ccg_2d_numba", "ccg_2d"}:
        t1 = _train(corr, dp, u0)
        t2 = _train(corr, dp, u1)
        bs_samp = _samples_from_ms(cbin)
        ws_samp = _samples_from_ms(cwin)
        fn = corr.ccg_2d_numba if method_key == "ccg_2d_numba" else corr.ccg_2d
        mat = np.asarray(fn(t1, t2, bs_samp, ws_samp), dtype=float)
        return _meta_payload({"kind": "image", "title": f"{method_key} matrix", "mat": mat}, requested_dp, dp)

    if method_key == "scaled_acg":
        acgs, isi_mode, _, _, _ = corr.scaled_acg(dp, units)
        acgs = np.asarray(acgs, dtype=float)
        return _meta_payload({"kind": "image", "title": "Scaled ACG (units x lag)", "mat": acgs,
                              "x_label": "Scaled lag bin", "y_label": "Unit #", "cbar_label": "ACG (Hz)",
                              "text": f"Median ISI mode: {np.nanmedian(np.asarray(isi_mode, dtype=float)):.3f} ms"}, requested_dp, dp)

    if method_key in {"crosscorr_vs_firing_rate", "ccg_vs_fr"}:
        t1 = _train(corr, dp, u0)
        t2 = _train(corr, dp, u1)
        out = corr.crosscorr_vs_firing_rate(t1, t2, cwin, cbin) if method_key == "crosscorr_vs_firing_rate" else corr.ccg_vs_fr(t1, t2, cwin, cbin)
        bins_f, mat = out
        return _meta_payload({"kind": "image", "title": f"{method_key} (FR x lag)", "mat": np.asarray(mat, dtype=float), "y": np.asarray(bins_f, dtype=float), "x": _lag_axis(cbin, cwin, np.asarray(mat).shape[1])}, requested_dp, dp)

    if method_key == "get_ccgstack_fullname":
        fn, fnu = corr.get_ccgstack_fullname(name="ui", cbin=cbin, cwin=cwin, normalize="Counts", periods="all")
        return _meta_payload({"kind": "text", "title": "CCG stack cache names", "text": f"{fn}\n{fnu}"}, requested_dp, dp)

    if method_key == "compute_ccgs_bulk":
        sel = units[: min(len(units), 6)]
        if len(sel) < 2:
            return _meta_payload({"kind": "text", "title": "compute_ccgs_bulk", "text": "Need at least two selected units."}, requested_dp, dp)
        inputs = []
        pairs: List[Tuple[int, int]] = []
        for i in range(len(sel)):
            for j in range(i + 1, len(sel)):
                u_a, u_b = int(sel[i]), int(sel[j])
                pairs.append((u_a, u_b))
                inputs.append((dp, [u_a, u_b], cbin, cwin, 30000, "Counts", 1, 1, 0, "all", 0, None))
        outs = corr.compute_ccgs_bulk(inputs, parallel=True)
        n = len(sel)
        mat = np.full((n, n), np.nan, dtype=float)
        idx = {int(u): k for k, u in enumerate(sel)}
        for (u_a, u_b), o in zip(pairs, outs):
            ccg_pair = np.asarray(o, dtype=float)
            y = ccg_pair[0, 1] if ccg_pair.ndim == 3 else np.ravel(ccg_pair)
            v = float(np.nanmax(y)) if y.size else np.nan
            i, j = idx[u_a], idx[u_b]
            mat[i, j] = v
            mat[j, i] = v
        return _meta_payload({"kind": "image", "title": "Bulk CCG peak matrix", "mat": mat}, requested_dp, dp)

    if method_key == "get_ustack_i":
        if len(units) < 2:
            return _meta_payload({"kind": "text", "title": "get_ustack_i", "text": "Need at least two selected units."}, requested_dp, dp)
        _, ustack = corr.ccg_stack(dp, U_src=units, U_trg=units, cbin=cbin, cwin=cwin, all_to_all=True, sav=False, again=True)
        q = np.array([[u0, u1]], dtype=np.int64)
        out = corr.get_ustack_i(q, np.asarray(ustack))
        return _meta_payload({"kind": "text", "title": "Pair index in stack", "text": str(np.asarray(out).tolist())}, requested_dp, dp)

    if method_key == "get_cross_features":
        ccg_counts = np.asarray(_cross_counts(corr, dp, u0, u1, cbin, cwin), dtype=float)
        crosses = corr.get_ccg_sig(ccg_counts, cbin=cbin, cwin=cwin, p_th=0.02, n_consec_bins=3, sgn=0, ret_features=False, only_max=True)
        if not crosses:
            return _meta_payload({"kind": "text", "title": "Cross features", "text": "No significant modulation found."}, requested_dp, dp)
        feat = corr.get_cross_features(crosses[0], cbin=cbin, cwin=cwin)
        return _meta_payload({"kind": "text", "title": "Cross features", "text": str(feat)}, requested_dp, dp)

    if method_key == "get_ccg_sig":
        ccg_counts = np.asarray(_cross_counts(corr, dp, u0, u1, cbin, cwin), dtype=float)
        w_sd = float(max(float(_p("W_sd_ms", 10.0)), cbin))
        feat = corr.get_ccg_sig(
            ccg_counts,
            cbin=cbin,
            cwin=cwin,
            p_th=float(_p("p_th", 0.02)),
            n_consec_bins=int(_p("n_consec_bins", 3)),
            sgn=int(_p("sgn", 0)),
            W_sd=w_sd,
            test=str(_p("test", "Poisson_Stark")),
            ret_features=True,
            only_max=False,
        )
        return _meta_payload({"kind": "text", "title": "CCG significance features", "text": str(feat)}, requested_dp, dp)

    if method_key == "ccg_sig_stack":
        if len(units) < 2:
            return _meta_payload({"kind": "text", "title": "Significant CCG stack", "text": "Need at least two selected units."}, requested_dp, dp)
        p_th = float(_p("p_th", 0.02))
        n_consec = int(_p("n_consec_bins", 3))
        sgn = int(_p("sgn", 0))
        test = str(_p("test", "Poisson_Stark"))
        w_sd = float(max(float(_p("W_sd_ms", 10.0)), cbin))

        # In-process pairwise significance scan avoids joblib worker import issues.
        items = []
        for i, a in enumerate(units):
            for b in units[i + 1 :]:
                ccg_counts = np.asarray(_cross_counts(corr, dp, int(a), int(b), cbin, cwin), dtype=float)
                sig = corr.get_ccg_sig(
                    ccg_counts,
                    cbin=cbin,
                    cwin=cwin,
                    p_th=p_th,
                    n_consec_bins=n_consec,
                    sgn=sgn,
                    W_sd=w_sd,
                    test=test,
                    ret_features=True,
                    only_max=True,
                )
                if sig:
                    score = float(np.nanmax(np.abs(ccg_counts))) if ccg_counts.size else np.nan
                    items.append(
                        {
                            "u1": int(a),
                            "u2": int(b),
                            "x": bins[: ccg_counts.size],
                            "y": ccg_counts,
                            "significant": True,
                            "score": score,
                        }
                    )
        if not items:
            return _meta_payload({"kind": "text", "title": "Significant CCG stack", "text": "No significant CCG pairs."}, requested_dp, dp)
        return _meta_payload({"kind": "corr_pairs", "title": "Significant CCG pairs", "items": items}, requested_dp, dp)

    if method_key == "gen_sfc":
        if len(units) < 2:
            return _meta_payload({"kind": "text", "title": "Functional correlation graph", "text": "Need at least two selected units."}, requested_dp, dp)
        out = corr.gen_sfc(
            dp,
            corr_type=str(_p("corr_type", "connections")),
            metric=str(_p("metric", "amp_z")),
            cbin=cbin,
            cwin=cwin,
            units=units,
            name="ui_tmp",
        )
        if len(out) >= 2:
            sfcm = np.asarray(out[1], dtype=float)
            if sfcm.ndim == 1:
                side = int(round(np.sqrt(float(sfcm.size))))
                if side * side == sfcm.size:
                    sfcm = sfcm.reshape(side, side)
                else:
                    sfcm = sfcm.reshape(1, -1)
            return _meta_payload({"kind": "image", "title": "Functional correlation matrix", "mat": sfcm}, requested_dp, dp)
        return _meta_payload({"kind": "text", "title": "Functional correlation graph", "text": "No matrix returned."}, requested_dp, dp)

    if method_key in {"cisi_numba_para", "cisi_numba"}:
        t1 = _train(corr, dp, u0).astype(np.float64)
        t2 = _train(corr, dp, u1).astype(np.float64)
        fn = corr.cisi_numba_para if method_key == "cisi_numba_para" else corr.cisi_numba
        vals = np.asarray(fn(t1, t2, int(psutil.virtual_memory().available)), dtype=float)
        vals = vals[np.isfinite(vals)]
        hist, edges = np.histogram(vals, bins=min(120, max(20, int(np.sqrt(max(vals.size, 1))))))
        return _meta_payload({"kind": "hist", "title": method_key, "x": edges[:-1], "y": hist.astype(float), "w": np.diff(edges)}, requested_dp, dp)

    if method_key == "cisi_chunk":
        t1 = _train(corr, dp, u0).astype(np.float64)
        t2 = _train(corr, dp, u1).astype(np.float64)
        n = 1
        s = max(1, min(t1.size, 1000))
        chunk = t1[:s]
        slc, mins, _, _ = corr.cisi_chunk(0, chunk, n, t2, 0, s)
        txt = f"slice={slc.start}:{slc.stop}, n={len(mins)}"
        return _meta_payload({"kind": "text", "title": "cisi_chunk summary", "text": txt}, requested_dp, dp)

    if method_key == "par_process":
        t1 = _train(corr, dp, u0).astype(np.float64)
        t2 = _train(corr, dp, u1).astype(np.float64)
        n = 1
        s = max(1, min(t1.size, 1000))
        chunk = t1[:s]
        out = corr.par_process(0, chunk, t2, n, 0)
        return _meta_payload({"kind": "text", "title": "par_process summary", "text": str(type(out))}, requested_dp, dp)

    if method_key == "get_cisi_parprocess":
        t1 = _train(corr, dp, u0)
        t2 = _train(corr, dp, u1)
        vals = np.asarray(corr.get_cisi_parprocess(t1, t2, direction=0), dtype=float)
        vals = vals[np.isfinite(vals)]
        hist, edges = np.histogram(vals, bins=min(120, max(20, int(np.sqrt(max(vals.size, 1))))))
        return _meta_payload({"kind": "hist", "title": "Cross-ISI (parprocess)", "x": edges[:-1], "y": hist.astype(float), "w": np.diff(edges)}, requested_dp, dp)

    if method_key == "pearson_corr_trn":
        trains = _train_list(corr, dp, units)
        if len(trains) < 2:
            return _meta_payload({"kind": "text", "title": "Pearson Corr (train-binned)", "text": "Need at least two selected units."}, requested_dp, dp)
        rec_len = int(np.load(Path(dp) / "spike_times.npy")[-1])
        b_ms = float(_p("bin_ms", max(1.0, cbin)))
        rows = [np.asarray(corr.binarize(t, b_ms, fs, rec_len), dtype=float) for t in trains]
        M = np.vstack(rows)
        mat = np.asarray(corr.pearson_corr(M), dtype=float)
        return _meta_payload({"kind": "image", "title": "Pearson Corr (train-binned)", "mat": mat,
                              "x_label": "Unit #", "y_label": "Unit #", "cbar_label": "Pearson r"}, requested_dp, dp)

    if method_key == "correlation_index":
        trains = _train_list(corr, dp, units)
        if len(trains) < 2:
            return _meta_payload({"kind": "text", "title": "Correlation index matrix", "text": "Need at least two selected units."}, requested_dp, dp)
        out = corr.correlation_index(trains, dt=float(_p("dt_ms", max(1.0, cbin))), dp=dp)
        arr = np.asarray(out, dtype=float)
        if arr.ndim < 2:
            return _meta_payload({"kind": "scalar", "title": "Correlation index", "value": float(arr.reshape(-1)[0])}, requested_dp, dp)
        return _meta_payload({"kind": "image", "title": "Correlation index matrix", "mat": arr,
                              "x_label": "Unit #", "y_label": "Unit #", "cbar_label": "Correlation index"}, requested_dp, dp)

    if method_key == "covariance":
        ccg_rate = np.asarray(corr.ccg_hz(dp, u0, u1, cbin, cwin, rate_corrected=True), dtype=float)
        mfr1 = float(corr.mfr(dp, u0)) if hasattr(corr, "mfr") else 1.0
        cov = np.asarray(corr.convert_ccg_to_covariance(ccg_rate, mfr_1=max(mfr1, 1e-9), cbin=cbin), dtype=float)
        x = _lag_axis(cbin, cwin, cov.size)
        return _meta_payload({"kind": "line", "title": "Covariance from CCG", "x": x, "y": cov}, requested_dp, dp)

    if method_key == "convert_ccg_to_covariance":
        ccg_rate = np.asarray(corr.ccg_hz(dp, u0, u1, cbin, cwin, rate_corrected=True), dtype=float)
        mfr1 = float(corr.mfr(dp, u0)) if hasattr(corr, "mfr") else 1.0
        cov = np.asarray(corr.convert_ccg_to_covariance(ccg_rate, mfr_1=max(mfr1, 1e-9), cbin=cbin), dtype=float)
        x = _lag_axis(cbin, cwin, cov.size)
        return _meta_payload({"kind": "line", "title": "CCG -> Covariance", "x": x, "y": cov}, requested_dp, dp)

    if method_key == "cofiring_tags":
        t1 = _train(corr, dp, u0)
        t2 = _train(corr, dp, u1)
        t_end = int(max(t1[-1] if t1.size else 0, t2[-1] if t2.size else 0))
        tags = np.asarray(corr.cofiring_tags(t1, t2, fs=30000, t_end=t_end, b=max(1.0, cbin)), dtype=float)
        return _meta_payload({"kind": "line", "title": "Cofiring tags over spikes", "x": np.arange(tags.size, dtype=float), "y": tags}, requested_dp, dp)

    if method_key in {"frac_pop_sync_old", "frac_pop_sync"}:
        trn_list = _train_list(corr, dp, units)
        if len(trn_list) < 2:
            return _meta_payload({"kind": "text", "title": method_key, "text": "Need at least two units selected."}, requested_dp, dp)
        t1 = trn_list[0]
        others = trn_list[1:]
        t_end = int(max([int(t[-1]) for t in trn_list if t.size] + [0]))
        if method_key == "frac_pop_sync_old":
            frac = np.asarray(corr.frac_pop_sync_old(t1, others, fs=30000, t_end=t_end, sync_win=2.0, b=max(1.0, cbin)), dtype=float)
            return _meta_payload({"kind": "line", "title": method_key, "x": np.arange(frac.size, dtype=float), "y": frac}, requested_dp, dp)
        else:
            out = corr.frac_pop_sync(t1, others, fs=30000, t_end=t_end, sync_win=0.5, firing_b=max(1.0, cbin))
            if isinstance(out, tuple) and len(out) >= 1:
                frac = np.asarray(out[0], dtype=float)
            else:
                frac = np.asarray(out, dtype=float)
            return _meta_payload({"kind": "line", "title": method_key, "x": np.arange(frac.size, dtype=float), "y": frac}, requested_dp, dp)

    if method_key == "fraction_pop_sync":
        out = corr.fraction_pop_sync(dp, u0, units[1:] if len(units) > 1 else [u0], sync_win=2.0)
        if isinstance(out, tuple) and len(out) >= 1:
            vals = np.asarray(out[0], dtype=float)
        else:
            vals = np.asarray(out, dtype=float)
        return _meta_payload({"kind": "line", "title": "Fraction population synchrony", "x": np.arange(vals.size, dtype=float), "y": vals}, requested_dp, dp)

    if method_key == "spike_time_tiling_coefficient":
        t1 = _train(corr, dp, u0)
        t2 = _train(corr, dp, u1)
        L = float(max(t1[-1] if t1.size else 0, t2[-1] if t2.size else 0) / float(fs))
        sttc = float(corr.spike_time_tiling_coefficient(t1, t2, L=max(L, 1e-6), dt=max(1.0, cbin), dp=dp))
        return _meta_payload({"kind": "scalar", "title": "Spike Time Tiling Coefficient", "value": sttc}, requested_dp, dp)

    return {
        "kind": "text",
        "title": "Method result",
        "text": f"Method `{method_key}` returned no dedicated visualization adapter.",
        "resolved_dp": dp,
        "requested_dp": requested_dp,
    }

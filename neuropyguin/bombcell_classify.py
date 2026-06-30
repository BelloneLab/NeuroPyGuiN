"""Bombcell region-specific cell-type classification.

Bombcell (Fabre et al.) classifies cell types from per-unit electrophysiological
properties (waveform duration, post-spike suppression, proportion of long ISIs,
firing rate) using region-specific thresholds, mirroring the MATLAB
``classifyCells.m``:

- cortex: Wide-spiking / Narrow-spiking
- striatum: MSN / FSI / TAN / UIN

Unlike the C4 classifier (cerebellar CNN ensemble, isolated env), this is a pure
numpy/threshold method and runs in the main app env. Computing the ephys
properties (~1 min for a full recording) is cached per datapath for the session.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Per-datapath cache of computed ephys properties: dp -> (ephys_properties, ephys_param)
_EPHYS_CACHE: Dict[str, tuple] = {}

REGION_CLASSES = {
    "cortex": ["Wide-spiking", "Narrow-spiking", "Unknown"],
    "striatum": ["MSN", "FSI", "TAN", "UIN", "Unknown"],
}


def _ensure_bombcell_on_path() -> None:
    for p in (_REPO_ROOT, _REPO_ROOT / "py_bombcell"):
        sp = str(p)
        if p.exists() and sp not in sys.path:
            sys.path.insert(0, sp)


def _empty_result(region: str, error: Optional[str]) -> Dict[str, object]:
    return {
        "units": [], "predicted_type": [], "class_names": REGION_CLASSES.get(region, ["Unknown"]),
        "region": region, "method": "bombcell",
        "metrics": {"waveform_duration_us": [], "post_spike_suppression_ms": [],
                    "prop_long_isi": [], "firing_rate_hz": []},
        "skipped_units": [], "error": error,
    }


def _compute_ephys(dp: str, progress_cb=None) -> tuple:
    """Compute (and cache) bombcell ephys properties for a datapath."""
    if dp in _EPHYS_CACHE:
        return _EPHYS_CACHE[dp]
    _ensure_bombcell_on_path()
    from bombcell.ephys_properties import run_all_ephys_properties  # type: ignore

    if progress_cb is not None:
        try:
            progress_cb("Computing bombcell ephys properties (~1 min)...")
        except Exception:
            pass
    ep, epar = run_all_ephys_properties(str(dp))
    _EPHYS_CACHE[dp] = (ep, epar)
    return ep, epar


def run_bombcell_classifier(
    dp: str,
    units: Sequence[int],
    *,
    region: str = "cortex",
    progress_cb=None,
) -> Dict[str, object]:
    """Classify ``units`` by cell type using bombcell's region-specific thresholds.

    Returns {units, predicted_type, class_names, region, method, metrics{...},
    skipped_units, error}. Metrics are aligned to ``units``. Computes ephys
    properties for the whole recording (cached per datapath) then maps the
    requested units. On failure ``error`` is set and the rest is empty.
    """
    region = str(region).lower().strip()
    if region not in REGION_CLASSES:
        return _empty_result(region, f"Unknown brain region '{region}'. Use 'cortex' or 'striatum'.")
    units = [int(u) for u in units]
    if not units:
        return _empty_result(region, "Select at least one unit to classify.")

    try:
        _ensure_bombcell_on_path()
        from bombcell.classification import classify_cortex_cells, classify_striatum_cells  # type: ignore

        ep, epar = _compute_ephys(dp, progress_cb=progress_cb)
        if not ep:
            return _empty_result(region, "Bombcell returned no ephys properties for this dataset.")

        if progress_cb is not None:
            try:
                progress_cb(f"Classifying ({region}) ...")
            except Exception:
                pass
        labels_all = (classify_cortex_cells(ep, epar) if region == "cortex"
                      else classify_striatum_cells(ep, epar))

        # Map by unit id (each property dict carries 'unit_id'); fall back to order.
        by_unit: Dict[int, dict] = {}
        for i, props in enumerate(ep):
            uid = props.get("unit_id", props.get("cluster_id", i))
            try:
                uid = int(uid)
            except Exception:
                uid = i
            by_unit[uid] = {"label": str(labels_all[i]), "props": props}

        out_units: List[int] = []
        pred: List[str] = []
        wf, pss, pli, fr = [], [], [], []
        skipped: List[int] = []
        for u in units:
            rec = by_unit.get(u)
            if rec is None:
                skipped.append(u)
                continue
            p = rec["props"]
            out_units.append(u)
            pred.append(rec["label"])
            wf.append(float(p.get("waveformDuration_peakTrough_us", np.nan)))
            pss.append(float(p.get("postSpikeSuppression_ms", np.nan)))
            pli.append(float(p.get("propLongISI", p.get("prop_long_isi", np.nan))))
            fr.append(float(p.get("firing_rate_mean", np.nan)))

        if not out_units:
            return _empty_result(region, "None of the selected units were found in the bombcell output.")

        return {
            "units": out_units,
            "predicted_type": pred,
            "class_names": REGION_CLASSES[region],
            "region": region,
            "method": "bombcell",
            "metrics": {
                "waveform_duration_us": wf,
                "post_spike_suppression_ms": pss,
                "prop_long_isi": pli,
                "firing_rate_hz": fr,
            },
            "skipped_units": skipped,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return _empty_result(region, f"{type(exc).__name__}: {exc}")

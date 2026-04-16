from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .bombcell_core import sync_phy_cluster_group
from .pybombcell_runtime import ensure_pybombcell_on_sys_path


def _best_meta_file(ks_folder: Path) -> Optional[Path]:
    # Try parent tree first for SpikeGLX ap meta.
    pats = ["*.ap.meta", "*.imec*.meta", "*.meta"]
    for pat in pats:
        found = list(ks_folder.parent.rglob(pat))
        if found:
            return sorted(found, key=lambda p: len(str(p)))[0]
    return None


def _unit_metrics_dataframe(quality_metrics: Dict, unit_ids: np.ndarray) -> pd.DataFrame:
    unit_ids = np.asarray(unit_ids, dtype=int)
    n_units = int(len(unit_ids))
    payload = {"cluster_id": unit_ids}
    for key, value in quality_metrics.items():
        try:
            value_len = len(value)
        except Exception:
            continue
        arr = np.asarray(value)
        if value_len != n_units or arr.ndim != 1:
            continue
        payload[str(key)] = arr
    return pd.DataFrame(payload)


def run_pybombcell_on_folder(ks_folder: str, save_plots: bool = True, force_recompute: bool = False) -> Dict:
    ensure_pybombcell_on_sys_path()
    # Force non-interactive backend for GUI app stability.
    import matplotlib

    matplotlib.use("Agg")

    from bombcell.default_parameters import get_default_parameters
    from bombcell.helper_functions import run_bombcell

    ks = Path(ks_folder)
    if not ks.exists():
        raise RuntimeError(f"Invalid Kilosort folder: {ks}")

    save_dir = ks / "bombcell"
    save_dir.mkdir(parents=True, exist_ok=True)
    out_csv = save_dir / "templates._bc_qMetrics.csv"
    labels_csv = ks / "bombcell_labels.csv"
    plots_dir = ks / "bombcell_plots"

    if (not force_recompute) and out_csv.exists():
        n_units = 0
        try:
            n_units = int(len(pd.read_csv(out_csv)))
        except Exception:
            n_units = 0
        return {
            "metrics_csv": str(out_csv),
            "labels_csv": str(labels_csv) if labels_csv.exists() else "",
            "plots_dir": str(plots_dir),
            "n_units": n_units,
            "cached": True,
        }

    meta = _best_meta_file(ks)
    param = get_default_parameters(
        kilosort_path=str(ks),
        raw_file=None,
        kilosort_version=4,
        meta_file=str(meta) if meta else None,
        gain_to_uV=None,
    )
    param["verbose"] = False
    param["plotGlobal"] = bool(save_plots)
    param["savePlots"] = bool(save_plots)
    param["plotsSaveDir"] = str(ks / "bombcell_plots")
    # Keep runtime stable and fast by default.
    param["extractRaw"] = False
    param["computeDistanceMetrics"] = False
    param["computeDrift"] = False

    quality_metrics, out_param, unit_type, unit_type_string = run_bombcell(
        str(ks), str(save_dir), param, save_figures=bool(save_plots), return_figures=False
    )

    # Save a flat CSV for app consumption.
    df = _unit_metrics_dataframe(quality_metrics, np.asarray(out_param["unique_templates"], dtype=int))
    df.to_csv(out_csv, index=False)

    labels = pd.DataFrame({"cluster_id": np.asarray(out_param["unique_templates"], dtype=int), "bombcell_label": unit_type_string})
    labels.to_csv(labels_csv, index=False)
    sync_result = sync_phy_cluster_group(ks, force=True)

    return {
        "metrics_csv": str(out_csv),
        "labels_csv": str(labels_csv),
        "plots_dir": str(plots_dir),
        "n_units": int(len(labels)),
        "cached": False,
        "phy_group_sync": sync_result,
    }

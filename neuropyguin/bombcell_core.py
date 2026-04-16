from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def is_threshold_disabled(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def bombcell_get_default_thresholds() -> dict:
    return {
        "noise": {
            "num_positive_peaks": {"min": None, "max": 2},
            "num_negative_peaks": {"min": None, "max": 1},
            "peak_to_trough_duration": {"min": 0.0001, "max": 0.00115},
            "waveform_baseline_flatness": {"min": None, "max": 0.5},
            "peak_after_to_trough_ratio": {"min": None, "max": 0.8},
            "exp_decay": {"min": 0.01, "max": 0.1},
        },
        "mua": {
            "amplitude_median": {"min": 30, "max": None, "abs": True},
            "snr": {"min": 5, "max": None},
            "amplitude_cutoff": {"min": None, "max": 0.2},
            "num_spikes": {"min": 300, "max": None},
            "rp_contamination": {"min": None, "max": 0.1},
            "presence_ratio": {"min": 0.7, "max": None},
            "drift_ptp": {"min": None, "max": 100},
        },
        "non-somatic": {
            "peak_before_to_trough_ratio": {"min": None, "max": 3},
            "peak_before_width": {"min": 0.00015, "max": None},
            "trough_width": {"min": 0.0002, "max": None},
            "peak_before_to_peak_after_ratio": {"min": None, "max": 3},
            "main_peak_to_trough_ratio": {"min": None, "max": 0.8},
        },
    }


def _label_by_thresholds(metrics: pd.DataFrame, thresholds: Dict[str, Dict], pass_label: str, fail_label: str) -> pd.Series:
    labels = pd.Series(pass_label, index=metrics.index, dtype=object)
    if metrics.empty:
        return labels

    fail_mask = np.zeros(len(metrics), dtype=bool)
    for metric, conf in thresholds.items():
        if metric not in metrics.columns:
            continue
        vals = pd.to_numeric(metrics[metric], errors="coerce").to_numpy(copy=True)
        if conf.get("abs", False):
            vals = np.abs(vals)

        metric_fail = np.isnan(vals)
        min_v = conf.get("min")
        max_v = conf.get("max")
        if not is_threshold_disabled(min_v):
            metric_fail |= vals < min_v
        if not is_threshold_disabled(max_v):
            metric_fail |= vals > max_v
        fail_mask |= metric_fail

    labels.iloc[fail_mask] = fail_label
    return labels


def bombcell_label_units_from_metrics(metrics: pd.DataFrame, thresholds: Optional[dict] = None) -> pd.DataFrame:
    thresholds = thresholds or bombcell_get_default_thresholds()

    labels = _label_by_thresholds(metrics, thresholds.get("noise", {}), "good", "noise")

    non_noise_idx = labels.index[labels != "noise"]
    if len(non_noise_idx) > 0:
        mua_labels = _label_by_thresholds(metrics.loc[non_noise_idx], thresholds.get("mua", {}), "good", "mua")
        labels.loc[non_noise_idx] = mua_labels.values

    non_s = thresholds.get("non-somatic", {})
    ratio_cols = [c for c in ["peak_before_to_trough_ratio", "peak_before_to_peak_after_ratio"] if c in metrics.columns]
    width_cols = [c for c in ["peak_before_width", "trough_width"] if c in metrics.columns]
    mpt_col = "main_peak_to_trough_ratio"

    ratio_large = pd.Series(False, index=metrics.index)
    width_narrow = pd.Series(False, index=metrics.index)
    main_peak_large = pd.Series(False, index=metrics.index)

    for c in ratio_cols:
        max_v = non_s.get(c, {}).get("max")
        if not is_threshold_disabled(max_v):
            vals = pd.to_numeric(metrics[c], errors="coerce")
            ratio_large |= vals > max_v
    for c in width_cols:
        min_v = non_s.get(c, {}).get("min")
        if not is_threshold_disabled(min_v):
            vals = pd.to_numeric(metrics[c], errors="coerce")
            width_narrow |= vals < min_v
    if mpt_col in metrics.columns:
        max_v = non_s.get(mpt_col, {}).get("max")
        if not is_threshold_disabled(max_v):
            vals = pd.to_numeric(metrics[mpt_col], errors="coerce")
            main_peak_large |= vals > max_v

    non_somatic = (ratio_large & width_narrow) | main_peak_large
    labels[(labels != "noise") & non_somatic] = "non_soma"

    return pd.DataFrame({"bombcell_label": labels}, index=metrics.index)


def run_bombcell_on_folder(folder: str) -> Dict:
    return run_bombcell_on_folder_with_thresholds(folder=folder, thresholds=None)


def _normalize_label_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = [str(c) for c in out.columns]
    if "cluster_id" in cols:
        out = out.set_index("cluster_id", drop=True)
    elif "unit_id" in cols:
        out = out.set_index("unit_id", drop=True)
    elif cols:
        c0 = cols[0]
        if c0.lower().startswith("unnamed") or c0.lower() in {"id", "cluster", "unit"}:
            out = out.set_index(c0, drop=True)
    try:
        idx = pd.to_numeric(out.index, errors="coerce")
        valid = ~pd.isna(idx)
        if valid.any():
            out = out.loc[valid]
            out.index = idx[valid].astype(int)
    except Exception:
        pass
    return out


def _labels_to_phy_groups(labels: pd.Series) -> pd.Series:
    out = labels.astype(str).str.strip().str.lower()
    remap = {
        "non-soma": "non_soma",
        "non soma": "non_soma",
        "non_somatic": "non_soma",
    }
    out = out.map(lambda v: remap.get(v, v))
    return out


def _read_label_source_for_phy(ks_folder: Path) -> Tuple[Optional[pd.Series], str]:
    bombcell_csv = ks_folder / "bombcell_labels.csv"
    if bombcell_csv.exists():
        try:
            df = _normalize_label_df(pd.read_csv(bombcell_csv))
            if "bombcell_label" in df.columns:
                return _labels_to_phy_groups(df["bombcell_label"]), "bombcell_labels.csv"
        except Exception:
            pass

    ks_label_tsv = ks_folder / "cluster_KSLabel.tsv"
    if ks_label_tsv.exists():
        try:
            df = _normalize_label_df(pd.read_csv(ks_label_tsv, sep="\t"))
            if "KSLabel" in df.columns:
                return _labels_to_phy_groups(df["KSLabel"]), "cluster_KSLabel.tsv"
        except Exception:
            pass

    return None, ""


def sync_phy_cluster_group(ks_folder: str | Path, force: bool = False) -> Dict[str, object]:
    folder = Path(ks_folder)
    source_labels, source_name = _read_label_source_for_phy(folder)
    if source_labels is None or source_labels.empty:
        return {
            "updated": False,
            "reason": "no_label_source",
            "path": str(folder / "cluster_group.tsv"),
            "source": source_name,
        }

    group_path = folder / "cluster_group.tsv"
    existing: Optional[pd.Series] = None
    if group_path.exists():
        try:
            current_df = _normalize_label_df(pd.read_csv(group_path, sep="\t"))
            if "group" in current_df.columns:
                existing = _labels_to_phy_groups(current_df["group"])
        except Exception:
            existing = None

    should_write = force or (existing is None)
    if existing is not None and not force:
        has_good = bool((existing == "good").any())
        all_noise = bool((existing == "noise").all()) if len(existing) else False
        source_has_non_noise = bool((source_labels != "noise").any())
        should_write = (not has_good) and all_noise and source_has_non_noise

    if not should_write:
        return {
            "updated": False,
            "reason": "existing_kept",
            "path": str(group_path),
            "source": source_name,
        }

    out = pd.DataFrame({"cluster_id": source_labels.index.astype(int), "group": source_labels.values})
    out = out.sort_values("cluster_id")
    out.to_csv(group_path, sep="\t", index=False)
    return {
        "updated": True,
        "reason": "written",
        "path": str(group_path),
        "source": source_name,
        "n_units": int(len(out)),
    }


def run_bombcell_on_folder_with_thresholds(folder: str, thresholds: Optional[dict] = None) -> Dict:
    ks_folder = Path(folder)
    metrics_path = ks_folder / "metrics.csv"
    if not metrics_path.exists():
        raise RuntimeError(f"metrics.csv not found in {ks_folder}")

    metrics = pd.read_csv(metrics_path)
    if "cluster_id" in metrics.columns:
        metrics = metrics.set_index("cluster_id", drop=True)
    elif "unit_id" in metrics.columns:
        metrics = metrics.set_index("unit_id", drop=True)

    labels = bombcell_label_units_from_metrics(metrics, thresholds=thresholds)
    out = ks_folder / "bombcell_labels.csv"
    labels.to_csv(out)
    sync_result = sync_phy_cluster_group(ks_folder, force=True)

    counts = labels["bombcell_label"].value_counts().to_dict()
    return {
        "output": str(out),
        "n_units": int(len(labels)),
        "counts": counts,
        "phy_group_sync": sync_result,
    }

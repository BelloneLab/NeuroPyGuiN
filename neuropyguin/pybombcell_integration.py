"""Integration layer between NeuroPyGuiN and the py_bombcell quality-metrics tool.

This module wraps py_bombcell to compute and persist Kilosort unit quality
metrics and labels (good / mua / noise / non-soma). It owns:

- the default settings schema and helpers to normalize / hash settings,
- reading and writing the BombCell manifest / metadata sidecar files,
- a fast path that reclassifies units from previously saved metrics, and
- a full-run path that invokes py_bombcell from scratch (optionally with raw
  waveform / SNR extraction), plus a Jupyter launcher for the BombCell GUI.

It performs no Qt work; callers (e.g. the curation tab) drive it from worker
threads.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .bombcell_core import sync_phy_cluster_group
from .processes import tracked_popen
from .pybombcell_runtime import ensure_pybombcell_on_sys_path

PYBOMBCELL_SETTINGS_SCHEMA: List[tuple[str, object]] = [
    ("removeDuplicateSpikes", False),
    ("duplicateSpikeWindow_s", 0.000034),
    ("saveSpikes_withoutDuplicates", True),
    ("recomputeDuplicateSpikes", False),
    ("detrendWaveform", True),
    ("detrendForUnitMatch", False),
    ("nRawSpikesToExtract", 100),
    ("decompress_data", False),
    ("extractRaw", True),
    ("probeType", 1),
    ("tauR_valuesMin", 1 / 1000),
    ("tauR_valuesMax", 2 / 1000),
    ("tauR_valuesStep", 0.5 / 1000),
    ("tauC", 0.1 / 1000),
    ("hillOrLlobetMethod", True),
    ("computeTimeChunks", False),
    ("deltaTimeChunk", 360),
    ("presenceRatioBinSize", 60),
    ("driftBinSize", 60),
    ("computeDrift", False),
    ("minThreshDetectPeaksTroughs", 0.2),
    ("normalizeSpDecay", True),
    ("spDecayLinFit", False),
    ("computeSpatialDecay", True),
    ("ephys_sample_rate", 30000),
    ("nChannels", 385),
    ("nSyncChannels", 1),
    ("computeDistanceMetrics", False),
    ("nChannelsIsoDist", 4),
    ("splitGoodAndMua_NonSomatic", True),
    ("maxNPeaks", 2),
    ("maxNTroughs", 1),
    ("minWvDuration", 100),
    ("maxWvDuration", 1250),
    ("minSpatialDecaySlope", -0.008),
    ("minSpatialDecaySlopeExp", 0.01),
    ("maxSpatialDecaySlopeExp", 0.1),
    ("maxWvBaselineFraction", 0.3),
    ("maxScndPeakToTroughRatio_noise", 0.8),
    ("minTroughToPeak2Ratio_nonSomatic", 5),
    ("minWidthFirstPeak_nonSomatic", 4),
    ("minWidthMainTrough_nonSomatic", 5),
    ("maxPeak1ToPeak2Ratio_nonSomatic", 3),
    ("maxMainPeakToTroughRatio_nonSomatic", 0.8),
    ("isoDmin", 20),
    ("lratioMax", 0.3),
    ("ss_min", np.nan),
    ("minAmplitude", 40),
    ("maxRPVviolations", 0.25),
    ("maxPercSpikesMissing", 25),
    ("minNumSpikes", 300),
    ("maxDrift", 100),
    ("minPresenceRatio", 0.7),
    ("minSNR", 2),
]


def pybombcell_default_settings() -> Dict[str, object]:
    """Return a fresh dict of default BombCell settings keyed by setting name."""
    return {key: value for key, value in PYBOMBCELL_SETTINGS_SCHEMA}


def pybombcell_setting_keys() -> List[str]:
    """Return the ordered list of recognized BombCell setting keys."""
    return [key for key, _value in PYBOMBCELL_SETTINGS_SCHEMA]


def _is_nan(value: object) -> bool:
    try:
        return bool(np.isnan(value))
    except Exception:
        return False


def _jsonable_setting_value(value: object) -> object:
    if _is_nan(value):
        return "__NaN__"
    if isinstance(value, np.generic):
        return value.item()
    return value


def normalize_pybombcell_settings(settings: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Merge user-supplied settings onto the defaults, keeping only known keys.

    Unknown keys in ``settings`` are ignored and missing keys fall back to their
    defaults, so the result always contains exactly the recognized setting keys.
    """
    defaults = pybombcell_default_settings()
    if not settings:
        return dict(defaults)
    out = dict(defaults)
    for key in pybombcell_setting_keys():
        if key in settings:
            out[key] = settings[key]
    return out


def pybombcell_settings_signature(settings: Optional[Dict[str, object]] = None) -> str:
    """Return a stable SHA-256 hash of the normalized settings.

    Used to decide whether previously saved metrics were computed with the same
    settings and can be reused.
    """
    normalized = normalize_pybombcell_settings(settings)
    payload = {key: _jsonable_setting_value(value) for key, value in normalized.items()}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def pybombcell_manifest_path(ks_folder: str | Path) -> Path:
    """Return the path to the legacy BombCell manifest JSON for a KS folder."""
    return Path(ks_folder) / "bombcell" / "pybombcell_manifest.json"


def pybombcell_metadata_path(ks_folder: str | Path) -> Path:
    """Return the path to the current BombCell metadata JSON for a KS folder."""
    return Path(ks_folder) / "bombcell" / "bombcell_metadata.json"


def _read_json_object(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def load_pybombcell_manifest(ks_folder: str | Path) -> Dict[str, object]:
    """Load the BombCell manifest, preferring the metadata file over the legacy one.

    Returns the first non-empty JSON object found, or an empty dict if neither
    sidecar exists or parses.
    """
    for path in (pybombcell_metadata_path(ks_folder), pybombcell_manifest_path(ks_folder)):
        data = _read_json_object(path)
        if data:
            return data
    return {}


def _normalize_label_name(label: object) -> str:
    raw = str(label or "").strip().lower().replace(",", "")
    raw = raw.replace("-", "_").replace(" ", "_")
    if not raw:
        return ""
    if raw.startswith("non_soma"):
        return "non_soma"
    if raw.startswith("somatic_good"):
        return "good"
    if raw.startswith("somatic_mua"):
        return "mua"
    remap = {
        "noise": "noise",
        "good": "good",
        "mua": "mua",
        "non_somatic": "non_soma",
        "nonsomatic": "non_soma",
        "non_soma_good": "non_soma",
        "non_soma_mua": "non_soma",
    }
    return remap.get(raw, raw)


def normalize_pybombcell_label(label: object) -> str:
    """Normalize a py_bombcell unit label to the GUI's four-label vocabulary."""
    return _normalize_label_name(label)


def _normalize_counts(counts: Optional[Dict[object, object]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for key, value in (counts or {}).items():
        label = _normalize_label_name(key)
        if not label:
            continue
        try:
            out[label] = out.get(label, 0) + int(value)
        except Exception:
            continue
    return out


def _restore_jsonable_setting_value(value: object) -> object:
    """Restore values serialized by _jsonable_setting_value."""
    if value == "__NaN__":
        return float("nan")
    return value


def decode_pybombcell_settings_payload(settings: Optional[Dict[str, object]]) -> Dict[str, object]:
    """Normalize py_bombcell settings read from a manifest JSON payload."""
    restored = {
        key: _restore_jsonable_setting_value(value)
        for key, value in (settings or {}).items()
    }
    return normalize_pybombcell_settings(restored)


def _write_pybombcell_manifest(
    ks_folder: Path,
    settings: Dict[str, object],
    *,
    metrics_csv: Path,
    labels_csv: Path,
    plots_dir: Path,
    counts: Dict[str, int],
    n_units: int,
    metrics_reused: bool = False,
    mode: str = "full_run",
) -> Path:
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "settings": {key: _jsonable_setting_value(value) for key, value in normalize_pybombcell_settings(settings).items()},
        "settings_signature": pybombcell_settings_signature(settings),
        "metrics_csv": str(metrics_csv),
        "labels_csv": str(labels_csv),
        "plots_dir": str(plots_dir),
        "counts": _normalize_counts(counts),
        "n_units": int(n_units),
        "metrics_reused": bool(metrics_reused),
        "mode": str(mode),
    }
    paths = [pybombcell_metadata_path(ks_folder), pybombcell_manifest_path(ks_folder)]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return paths[0]


def _best_meta_file(ks_folder: Path) -> Optional[Path]:
    pats = ["*.ap.meta", "*.imec*.meta", "*.meta"]
    for pat in pats:
        found = list(ks_folder.parent.rglob(pat))
        if found:
            return sorted(found, key=lambda p: len(str(p)))[0]
    return None


def _resolve_meta_file(ks_folder: Path) -> Optional[Path]:
    """Locate the SpikeGLX meta for a Kilosort folder.

    Prefer the .meta sitting next to the raw bin recorded in ``params.py``.
    This is robust for concatenated/joint sorts where the KS output folder
    lives in a different tree than the recording (so the legacy sibling search
    in :func:`_best_meta_file` would miss the combined .meta). Falls back to the
    nearby-folder search when params.py is absent or points nowhere.
    """
    ks_folder = Path(ks_folder)
    params_py = ks_folder / "params.py"
    if params_py.exists():
        try:
            from .preprocessing import find_meta_for_bin, parse_kilosort_params_dat_path

            dat_path = parse_kilosort_params_dat_path(params_py)
            if dat_path:
                meta = find_meta_for_bin(dat_path)
                if meta.exists():
                    return meta
        except Exception:
            pass
    return _best_meta_file(ks_folder)


def _resolve_raw_file(ks_folder: Path) -> Optional[Path]:
    """Resolve the raw .bin a Kilosort folder was sorted from (via params.py).

    Used for opt-in raw-waveform / SNR extraction. For a concatenated joint
    sort this returns the fused .bin, so raw waveforms are extracted across the
    full concatenated timeline using the joint spike times.
    """
    params_py = Path(ks_folder) / "params.py"
    if not params_py.exists():
        return None
    try:
        from .preprocessing import parse_kilosort_params_dat_path

        dat_path = parse_kilosort_params_dat_path(params_py)
    except Exception:
        dat_path = ""
    if dat_path and Path(dat_path).exists():
        return Path(dat_path)
    return None


def _label_counts(labels_csv: Path) -> Dict[str, int]:
    if not labels_csv.exists():
        return {}
    try:
        df = pd.read_csv(labels_csv)
    except Exception:
        return {}
    if "bombcell_label" not in df.columns:
        return {}
    counts = df["bombcell_label"].astype(str).map(_normalize_label_name).value_counts().to_dict()
    return _normalize_counts(counts)


def _coerce_cluster_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy indexed by integer cluster ids when such an id is present."""
    out = df.copy()
    cols = [str(c) for c in out.columns]
    for name in ("cluster_id", "phy_clusterID", "unit_id"):
        if name in cols:
            out = out.set_index(name, drop=True)
            break
    else:
        if cols:
            first = cols[0]
            if first.lower().startswith("unnamed") or first.lower() in {"id", "cluster", "unit"}:
                out = out.set_index(first, drop=True)

    idx = pd.to_numeric(out.index, errors="coerce")
    valid = ~pd.isna(idx)
    if valid.any():
        valid_mask = np.asarray(valid, dtype=bool)
        out = out.iloc[valid_mask]
        out.index = np.asarray(idx)[valid_mask].astype(int)
        out.index.name = "cluster_id"
    return out


def load_pybombcell_labels(ks_folder: str | Path) -> pd.DataFrame:
    """Load ``bombcell_labels.csv`` as a normalized label DataFrame.

    The returned frame is indexed by ``cluster_id`` and has one column,
    ``bombcell_label``. Missing files return an empty frame with that schema.
    """
    labels_csv = Path(ks_folder) / "bombcell_labels.csv"
    empty = pd.DataFrame(columns=["bombcell_label"])
    empty.index.name = "cluster_id"
    if not labels_csv.exists():
        return empty
    df = _coerce_cluster_index(pd.read_csv(labels_csv))
    label_column = ""
    for candidate in (
        "bombcell_label",
        "unit_type",
        "Bombcell_unit_type",
        "unitType",
        "label",
        "group",
    ):
        if candidate in df.columns:
            label_column = candidate
            break
    if not label_column:
        return empty
    labels = df[label_column].map(_normalize_label_name)
    labels = labels[labels.astype(str).str.len() > 0]
    out = pd.DataFrame({"bombcell_label": labels})
    out.index.name = "cluster_id"
    return out


def _normalized_unit_labels(labels: Iterable[object]) -> List[str]:
    return [_normalize_label_name(label) for label in labels]


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


def _saved_metrics_path(ks_folder: str | Path) -> Optional[Path]:
    save_dir = Path(ks_folder) / "bombcell"
    csv_path = save_dir / "templates._bc_qMetrics.csv"
    if csv_path.exists():
        return csv_path
    parquet_path = save_dir / "templates._bc_qMetrics.parquet"
    if parquet_path.exists():
        return parquet_path
    return None


def summarize_saved_pybombcell_results(
    ks_folder: str | Path,
    *,
    settings: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Summarize on-disk BombCell results for a KS folder without recomputing.

    Inspects saved metrics, labels, plots, and the manifest to report counts,
    unit totals, and whether the cached metrics can be reused for the given
    ``settings`` (via ``can_reuse`` / ``cache_reason``).
    """
    ks = Path(ks_folder)
    save_dir = ks / "bombcell"
    metrics_path = _saved_metrics_path(ks)
    labels_csv = ks / "bombcell_labels.csv"
    plots_dir = ks / "bombcell_plots"
    manifest = load_pybombcell_manifest(ks)
    counts = _label_counts(labels_csv)
    if not counts:
        counts = _normalize_counts(manifest.get("counts") if isinstance(manifest, dict) else {})
    n_units = int(sum(counts.values()))
    if n_units == 0:
        try:
            n_units = int(manifest.get("n_units") or 0)
        except Exception:
            n_units = 0
    if n_units == 0 and metrics_path is not None:
        try:
            if metrics_path.suffix == ".parquet":
                n_units = int(len(pd.read_parquet(metrics_path)))
            else:
                n_units = int(len(pd.read_csv(metrics_path)))
        except Exception:
            n_units = 0

    settings_signature = pybombcell_settings_signature(settings)
    manifest_signature = str(manifest.get("settings_signature") or "")
    default_signature = pybombcell_settings_signature(pybombcell_default_settings())

    # Decide whether the saved metrics can be reused as-is. With no metrics on
    # disk we cannot reuse anything. With a recorded signature we reuse only on
    # an exact match. Legacy results predate signature tracking, so we assume
    # they used the defaults and reuse only when the requested settings are the
    # defaults too.
    if metrics_path is None:
        can_reuse = False
        cache_reason = "missing_metrics"
    elif manifest_signature:
        can_reuse = manifest_signature == settings_signature
        cache_reason = "matching_signature" if can_reuse else "settings_changed"
    else:
        can_reuse = settings_signature == default_signature
        cache_reason = "legacy_default_assumed" if can_reuse else "legacy_unknown_settings"

    return {
        "metrics_csv": str(metrics_path) if metrics_path is not None else str(save_dir / "templates._bc_qMetrics.csv"),
        "labels_csv": str(labels_csv) if labels_csv.exists() else "",
        "plots_dir": str(plots_dir),
        "manifest_path": str(pybombcell_metadata_path(ks)),
        "metadata_path": str(pybombcell_metadata_path(ks)),
        "legacy_manifest_path": str(pybombcell_manifest_path(ks)),
        "manifest": manifest,
        "counts": counts,
        "n_units": n_units,
        "settings_signature": settings_signature,
        "manifest_signature": manifest_signature,
        "has_metrics": metrics_path is not None,
        "has_labels": labels_csv.exists(),
        "has_plots": plots_dir.exists(),
        "can_reuse": can_reuse,
        "cache_reason": cache_reason,
        "metrics_reused": bool(manifest.get("metrics_reused", False)) if isinstance(manifest, dict) else False,
        "mode": str(manifest.get("mode") or "") if isinstance(manifest, dict) else "",
    }


def _load_saved_metrics_dataframe(ks_folder: Path) -> pd.DataFrame:
    save_dir = ks_folder / "bombcell"
    csv_path = save_dir / "templates._bc_qMetrics.csv"
    parquet_path = save_dir / "templates._bc_qMetrics.parquet"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    raise FileNotFoundError(f"Missing saved py_bombcell metrics in {save_dir}")


def _load_saved_param_dict(ks_folder: Path) -> Dict[str, object]:
    param_path = ks_folder / "bombcell" / "_bc_parameters._bc_qMetrics.parquet"
    if not param_path.exists():
        return {}
    try:
        param_df = pd.read_parquet(param_path)
    except Exception:
        return {}
    if param_df.empty:
        return {}
    return dict(param_df.iloc[0].to_dict())


def _quality_metrics_dict_from_df(metrics_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for column in metrics_df.columns:
        if str(column).lower().startswith("unnamed"):
            continue
        out[str(column)] = metrics_df[column].to_numpy()
    return out


def _numeric_quality_metrics_dict_from_df(metrics_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for column in metrics_df.columns:
        name = str(column)
        if name.lower().startswith("unnamed"):
            continue
        try:
            out[name] = pd.to_numeric(metrics_df[column], errors="coerce").to_numpy(dtype=float)
        except Exception:
            continue
    return out


def classify_pybombcell_metrics(
    metrics_df: pd.DataFrame,
    settings: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """Classify saved py_bombcell metrics with BombCell's own threshold logic.

    This is the lightweight path used by the live QC panel. It does not
    recompute metrics or raw waveforms; it only applies ``get_quality_unit_type``
    to an already loaded ``templates._bc_qMetrics`` table.
    """
    if metrics_df.empty:
        out = pd.DataFrame(columns=["bombcell_label"])
        out.index.name = "cluster_id"
        return out

    metrics = _coerce_cluster_index(metrics_df)
    quality_metrics = _numeric_quality_metrics_dict_from_df(metrics)
    normalized_settings = normalize_pybombcell_settings(settings)
    n_units = int(len(metrics))

    required = [
        "nPeaks",
        "nTroughs",
        "waveformDuration_peakTrough",
        "waveformBaselineFlatness",
        "scndPeakToTroughRatio",
        "troughToPeak2Ratio",
        "mainPeak_before_width",
        "mainTrough_width",
        "peak1ToPeak2Ratio",
        "mainPeakToTroughRatio",
        "percentageSpikesMissing_gaussian",
        "nSpikes",
        "fractionRPVs_estimatedTauR",
        "presenceRatio",
    ]
    if normalized_settings.get("computeSpatialDecay", False):
        required.append("spatialDecaySlope")
    if normalized_settings.get("computeDrift", False):
        required.append("maxDriftEstimate")
    if normalized_settings.get("computeDistanceMetrics", False):
        required.extend(["isolationDistance", "Lratio"])

    missing = [name for name in required if name not in quality_metrics]
    if missing:
        raise RuntimeError(
            "Saved py_bombcell metrics are missing required columns: "
            + ", ".join(missing)
        )

    for optional in ("rawAmplitude", "signalToNoiseRatio", "maxDriftEstimate", "isolationDistance", "Lratio"):
        quality_metrics.setdefault(optional, np.full(n_units, np.nan, dtype=float))

    ensure_pybombcell_on_sys_path()
    from bombcell.quality_metrics import get_quality_unit_type

    _unit_type, unit_type_string = get_quality_unit_type(normalized_settings, quality_metrics)
    labels = _normalized_unit_labels(unit_type_string)
    out = pd.DataFrame({"bombcell_label": labels}, index=metrics.index.astype(int))
    out.index.name = "cluster_id"
    return out


def save_pybombcell_labels(
    ks_folder: str | Path,
    labels_df: pd.DataFrame,
    *,
    settings: Optional[Dict[str, object]] = None,
    mode: str = "live_threshold_preview",
) -> Dict[str, object]:
    """Write the displayed py_bombcell labels and sync ``cluster_group.tsv``."""
    ks = Path(ks_folder)
    if labels_df.empty or "bombcell_label" not in labels_df.columns:
        raise RuntimeError("No py_bombcell labels are available to save.")

    labels = _coerce_cluster_index(labels_df)
    if "bombcell_label" not in labels.columns:
        raise RuntimeError("No py_bombcell label column is available to save.")
    labels = pd.DataFrame(
        {"bombcell_label": labels["bombcell_label"].map(_normalize_label_name)},
        index=labels.index.astype(int),
    )
    labels.index.name = "cluster_id"
    labels = labels.sort_index()

    labels_csv = ks / "bombcell_labels.csv"
    labels.reset_index().to_csv(labels_csv, index=False)
    counts = _normalize_counts(labels["bombcell_label"].value_counts().to_dict())
    sync_result = sync_phy_cluster_group(ks, force=True)

    metrics_path = _saved_metrics_path(ks) or (ks / "bombcell" / "templates._bc_qMetrics.csv")
    plots_dir = ks / "bombcell_plots"
    manifest_path = _write_pybombcell_manifest(
        ks,
        normalize_pybombcell_settings(settings),
        metrics_csv=metrics_path,
        labels_csv=labels_csv,
        plots_dir=plots_dir,
        counts=counts,
        n_units=len(labels),
        metrics_reused=True,
        mode=mode,
    )
    return {
        "labels_csv": str(labels_csv),
        "metrics_csv": str(metrics_path),
        "manifest_path": str(manifest_path),
        "n_units": int(len(labels)),
        "counts": counts,
        "phy_group_sync": sync_result,
        "mode": mode,
    }


def _persist_saved_metrics_dataframe(ks_folder: Path, metrics_df: pd.DataFrame) -> None:
    save_dir = ks_folder / "bombcell"
    csv_path = save_dir / "templates._bc_qMetrics.csv"
    parquet_path = save_dir / "templates._bc_qMetrics.parquet"
    if csv_path.exists() or not parquet_path.exists():
        metrics_df.to_csv(csv_path, index=False)
    if parquet_path.exists():
        try:
            metrics_df.to_parquet(parquet_path, index=False)
        except Exception:
            pass


def _coerce_metric_ids(metrics_df: pd.DataFrame) -> Optional[np.ndarray]:
    for column in ("phy_clusterID", "cluster_id"):
        if column not in metrics_df.columns:
            continue
        values = pd.to_numeric(metrics_df[column], errors="coerce")
        if values.isna().any():
            continue
        return values.to_numpy(dtype=int)
    return None


def _ensure_saved_metrics_maxchannels(
    metrics_df: pd.DataFrame,
    template_waveforms: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    ensure_pybombcell_on_sys_path()
    from bombcell.quality_metrics import get_waveform_peak_channel

    full_max_channels = np.asarray(get_waveform_peak_channel(template_waveforms), dtype=int)
    if full_max_channels.ndim != 1 or full_max_channels.size == 0:
        raise RuntimeError("Unable to reconstruct BombCell maxChannels from template waveforms.")

    if "maxChannels" in metrics_df.columns and not metrics_df["maxChannels"].isna().all():
        return metrics_df, full_max_channels

    metric_ids = _coerce_metric_ids(metrics_df)
    if metric_ids is not None:
        if np.any(metric_ids < 0):
            raise RuntimeError("Saved BombCell metric IDs contain negative values; cannot reconstruct maxChannels.")
        if metric_ids.size == len(metrics_df) and metric_ids.max(initial=-1) < full_max_channels.size:
            rebuilt = full_max_channels[metric_ids]
            updated = metrics_df.copy()
            updated["maxChannels"] = rebuilt
            return updated, full_max_channels

    if len(metrics_df) == len(full_max_channels):
        updated = metrics_df.copy()
        updated["maxChannels"] = full_max_channels
        return updated, full_max_channels

    raise RuntimeError(
        "Saved BombCell metrics are missing maxChannels and could not be aligned to template waveforms."
    )


def _prepare_param_for_saved_metrics(ks_folder: Path, settings: Dict[str, object], metrics_df: pd.DataFrame) -> Dict[str, object]:
    ensure_pybombcell_on_sys_path()
    from bombcell.default_parameters import get_default_parameters

    param = _load_saved_param_dict(ks_folder)
    if not param:
        meta = _resolve_meta_file(ks_folder)
        param = get_default_parameters(
            kilosort_path=str(ks_folder),
            raw_file=None,
            kilosort_version=4,
            meta_file=str(meta) if meta else None,
            gain_to_uV=None,
        )
    param = dict(param)
    param["ephysKilosortPath"] = str(ks_folder)
    for key, value in settings.items():
        param[key] = value

    unique_templates: Optional[np.ndarray] = None
    if "unique_templates" in param and param["unique_templates"] is not None:
        try:
            unique_templates = np.asarray(param["unique_templates"], dtype=int)
        except Exception:
            unique_templates = None
    if unique_templates is None or len(unique_templates) != len(metrics_df):
        if "cluster_id" in metrics_df.columns:
            unique_templates = pd.to_numeric(metrics_df["cluster_id"], errors="coerce").dropna().to_numpy(dtype=int)
        elif "phy_clusterID" in metrics_df.columns:
            unique_templates = pd.to_numeric(metrics_df["phy_clusterID"], errors="coerce").dropna().to_numpy(dtype=int)
        else:
            unique_templates = np.arange(len(metrics_df), dtype=int)
    param["unique_templates"] = np.asarray(unique_templates, dtype=int)
    return param


def _refresh_pybombcell_outputs_from_saved_metrics(
    ks_folder: str | Path,
    settings: Dict[str, object],
    *,
    save_plots: bool,
) -> Dict[str, object]:
    ks = Path(ks_folder)
    save_dir = ks / "bombcell"
    plots_dir = ks / "bombcell_plots"
    metrics_path = _saved_metrics_path(ks)
    if metrics_path is None:
        raise FileNotFoundError(f"Missing saved py_bombcell metrics in {save_dir}")
    metrics_df = _load_saved_metrics_dataframe(ks)
    param = _prepare_param_for_saved_metrics(ks, settings, metrics_df)

    ensure_pybombcell_on_sys_path()
    import matplotlib

    matplotlib.use("Agg")

    from bombcell.loading_utils import load_ephys_data
    from bombcell.plot_functions import plot_summary_data
    from bombcell.quality_metrics import get_quality_unit_type

    if save_plots and plots_dir.exists():
        try:
            shutil.rmtree(plots_dir)
        except Exception:
            pass

    param["verbose"] = False
    param["plotGlobal"] = bool(save_plots)
    param["savePlots"] = bool(save_plots)
    param["plotsSaveDir"] = str(plots_dir)

    _spike_times, _spike_clusters, template_waveforms, _template_amplitudes, _pc_features, _pc_features_idx, _channel_positions = load_ephys_data(str(ks))
    metrics_df, full_max_channels = _ensure_saved_metrics_maxchannels(metrics_df, template_waveforms)
    _persist_saved_metrics_dataframe(ks, metrics_df)

    quality_metrics = _quality_metrics_dict_from_df(metrics_df)
    if "phy_clusterID" not in quality_metrics:
        quality_metrics["phy_clusterID"] = np.asarray(param["unique_templates"], dtype=int)

    unit_type, unit_type_string = get_quality_unit_type(param, quality_metrics)
    if save_plots:
        plot_quality_metrics = dict(quality_metrics)
        plot_quality_metrics["maxChannels"] = full_max_channels
        plot_summary_data(
            plot_quality_metrics,
            template_waveforms,
            unit_type,
            unit_type_string,
            param,
            return_figures=False,
        )

    normalized_labels = _normalized_unit_labels(unit_type_string)
    labels_csv = ks / "bombcell_labels.csv"
    labels = pd.DataFrame(
        {
            "cluster_id": np.asarray(param["unique_templates"], dtype=int),
            "bombcell_label": normalized_labels,
        }
    )
    labels.to_csv(labels_csv, index=False)
    counts = _normalize_counts(labels["bombcell_label"].value_counts().to_dict())
    sync_result = sync_phy_cluster_group(ks, force=True)
    manifest_path = _write_pybombcell_manifest(
        ks,
        settings,
        metrics_csv=metrics_path,
        labels_csv=labels_csv,
        plots_dir=plots_dir,
        counts=counts,
        n_units=len(labels),
        metrics_reused=True,
        mode="reclassified_from_saved_metrics",
    )

    return {
        "metrics_csv": str(metrics_path),
        "labels_csv": str(labels_csv),
        "plots_dir": str(plots_dir),
        "manifest_path": str(manifest_path),
        "n_units": int(len(labels)),
        "counts": counts,
        "settings_signature": pybombcell_settings_signature(settings),
        "cached": False,
        "metrics_reused": True,
        "cache_reason": "reused_saved_metrics",
        "phy_group_sync": sync_result,
    }


def _bombcell_gui_notebook_path(ks_folder: str | Path) -> Path:
    return Path(ks_folder) / "bombcell" / "for_GUI" / "open_bombcell_gui.ipynb"


def _write_bombcell_gui_notebook(ks_folder: str | Path) -> Path:
    ks = Path(ks_folder)
    notebook_path = _bombcell_gui_notebook_path(ks)
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]
    code = "\n".join(
        [
            "from pathlib import Path",
            "import sys",
            "import pandas as pd",
            f"repo_root = Path({repo_root.as_posix()!r})",
            "py_bombcell_root = repo_root / 'py_bombcell'",
            "if str(py_bombcell_root) not in sys.path:",
            "    sys.path.insert(0, str(py_bombcell_root))",
            "from bombcell.unit_quality_gui import unit_quality_gui",
            f"ks_dir = Path({ks.as_posix()!r})",
            "save_path = ks_dir / 'bombcell'",
            "param_path = save_path / '_bc_parameters._bc_qMetrics.parquet'",
            "metrics_csv = save_path / 'templates._bc_qMetrics.csv'",
            "metrics_parquet = save_path / 'templates._bc_qMetrics.parquet'",
            "try:",
            "    param = pd.read_parquet(param_path).iloc[0].to_dict() if param_path.exists() else {}",
            "except Exception:",
            "    param = {}",
            "if metrics_csv.exists():",
            "    quality_metrics = pd.read_csv(metrics_csv)",
            "elif metrics_parquet.exists():",
            "    quality_metrics = pd.read_parquet(metrics_parquet)",
            "else:",
            "    raise FileNotFoundError(f'Missing BombCell metrics in {save_path}')",
            "param['ephysKilosortPath'] = str(ks_dir)",
            "gui = unit_quality_gui(ks_dir=str(ks_dir), quality_metrics=quality_metrics, param=param, save_path=str(save_path))",
            "gui",
        ]
    )
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# BombCell GUI\n",
                    "This notebook launches the py_bombcell ipywidgets GUI for the selected Kilosort folder.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in code.splitlines()],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    return notebook_path


def launch_pybombcell_gui(ks_folder: str | Path) -> Dict[str, object]:
    """Write a launcher notebook and open the py_bombcell GUI in Jupyter.

    Requires saved metrics to exist for ``ks_folder``. Tries several Jupyter
    launchers in turn and returns the one that started, raising RuntimeError if
    none is available.
    """
    ks = Path(ks_folder)
    summary = summarize_saved_pybombcell_results(ks)
    if not summary.get("has_metrics", False):
        raise RuntimeError(f"No saved py_bombcell metrics found for {ks}")

    notebook_path = _write_bombcell_gui_notebook(ks)

    # Only offer launchers that actually exist in this environment. Running
    # "python -m jupyterlab" when jupyterlab is not installed still spawns a
    # process (python exists) that dies immediately, so the GUI silently never
    # opens. Probing first lets us fail with a clear, actionable message.
    def _module_available(name: str) -> bool:
        try:
            import importlib.util
            return importlib.util.find_spec(name) is not None
        except Exception:
            return False

    commands: List[List[str]] = []
    if _module_available("jupyterlab"):
        commands.append([sys.executable, "-m", "jupyterlab", str(notebook_path)])
    if _module_available("notebook"):
        commands.append([sys.executable, "-m", "notebook", str(notebook_path)])
    for exe in ("jupyter-lab", "jupyter-notebook"):
        if shutil.which(exe):
            commands.append([exe, str(notebook_path)])
    if shutil.which("jupyter"):
        commands.append(["jupyter", "lab", str(notebook_path)])
        commands.append(["jupyter", "notebook", str(notebook_path)])

    if not commands:
        raise RuntimeError(
            "Jupyter is not installed in the app's Python environment, so the BombCell "
            "GUI cannot open. Install it with:\n"
            f'    "{sys.executable}" -m pip install jupyterlab'
        )

    # Capture the Jupyter server output to a log next to the notebook so a failed
    # launch is diagnosable instead of vanishing into DEVNULL.
    log_path = notebook_path.parent / "bombcell_gui.log"
    creationflags = 0
    if sys.platform.startswith("win"):
        creationflags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    last_error = "No Jupyter launcher could be started."
    for cmd in commands:
        log_handle = None
        try:
            log_handle = open(log_path, "w", encoding="utf-8")
            tracked_popen(
                cmd,
                cwd=str(notebook_path.parent),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                creationflags=creationflags,
            )
            return {
                "launcher": " ".join(cmd[:3]) if len(cmd) > 2 else " ".join(cmd),
                "notebook_path": str(notebook_path),
                "log_path": str(log_path),
            }
        except FileNotFoundError:
            continue
        except Exception as exc:
            last_error = str(exc)
        finally:
            # The detached child keeps its own inherited handle; close ours so the
            # log file is not held open by the app process.
            if log_handle is not None:
                try:
                    log_handle.close()
                except Exception:
                    pass
    raise RuntimeError(f"Unable to launch BombCell GUI notebook: {last_error}")


def run_pybombcell_on_folder(
    ks_folder: str,
    save_plots: bool = True,
    force_recompute: bool = False,
    settings: Optional[Dict[str, object]] = None,
    extract_raw: bool = False,
) -> Dict:
    """Run (or reuse) BombCell quality metrics for a single Kilosort folder.

    When cached metrics match the settings and no raw extraction is requested,
    units are reclassified from the saved metrics; otherwise py_bombcell is run
    from scratch. Writes metrics, labels, plots, and a manifest, syncs the phy
    cluster groups, and returns a result dict describing the outputs.
    """
    normalized_settings = normalize_pybombcell_settings(settings)
    ks = Path(ks_folder)
    if not ks.exists():
        raise RuntimeError(f"Invalid Kilosort folder: {ks}")

    # Opt-in raw-waveform / SNR extraction. When a raw file is found we always
    # recompute, because the cached metrics were template-only and reusing them
    # would silently skip the requested raw waveforms / SNR.
    raw_file = _resolve_raw_file(ks) if extract_raw else None

    summary = summarize_saved_pybombcell_results(ks, settings=normalized_settings)
    if (not force_recompute) and raw_file is None and summary.get("has_metrics", False):
        return _refresh_pybombcell_outputs_from_saved_metrics(ks, normalized_settings, save_plots=save_plots)

    ensure_pybombcell_on_sys_path()
    import matplotlib

    matplotlib.use("Agg")

    from bombcell.default_parameters import get_default_parameters
    from bombcell.helper_functions import run_bombcell

    save_dir = ks / "bombcell"
    save_dir.mkdir(parents=True, exist_ok=True)
    out_csv = save_dir / "templates._bc_qMetrics.csv"
    labels_csv = ks / "bombcell_labels.csv"
    plots_dir = ks / "bombcell_plots"

    meta = _resolve_meta_file(ks)
    param = get_default_parameters(
        kilosort_path=str(ks),
        raw_file=str(raw_file) if raw_file else None,
        kilosort_version=4,
        meta_file=str(meta) if meta else None,
        gain_to_uV=None,
    )
    param["verbose"] = False
    param["plotGlobal"] = bool(save_plots)
    param["savePlots"] = bool(save_plots)
    param["plotsSaveDir"] = str(plots_dir)
    for key, value in normalized_settings.items():
        param[key] = value

    quality_metrics, out_param, _unit_type, unit_type_string = run_bombcell(
        str(ks),
        str(save_dir),
        param,
        save_figures=bool(save_plots),
        return_figures=False,
    )

    df = _unit_metrics_dataframe(quality_metrics, np.asarray(out_param["unique_templates"], dtype=int))
    df.to_csv(out_csv, index=False)

    labels = pd.DataFrame(
        {
            "cluster_id": np.asarray(out_param["unique_templates"], dtype=int),
            "bombcell_label": _normalized_unit_labels(unit_type_string),
        }
    )
    labels.to_csv(labels_csv, index=False)
    counts = _normalize_counts(labels["bombcell_label"].value_counts().to_dict())
    sync_result = sync_phy_cluster_group(ks, force=True)
    manifest_path = _write_pybombcell_manifest(
        ks,
        normalized_settings,
        metrics_csv=out_csv,
        labels_csv=labels_csv,
        plots_dir=plots_dir,
        counts=counts,
        n_units=len(labels),
        metrics_reused=False,
        mode="full_run",
    )

    return {
        "metrics_csv": str(out_csv),
        "labels_csv": str(labels_csv),
        "plots_dir": str(plots_dir),
        "manifest_path": str(manifest_path),
        "n_units": int(len(labels)),
        "counts": counts,
        "settings_signature": pybombcell_settings_signature(normalized_settings),
        "cached": False,
        "metrics_reused": False,
        "cache_reason": "reran",
        "raw_extracted": bool(raw_file is not None),
        "phy_group_sync": sync_result,
    }


def run_pybombcell_on_folders(
    folders: Iterable[str | Path],
    *,
    save_plots: bool = True,
    force_recompute: bool = False,
    settings: Optional[Dict[str, object]] = None,
    extract_raw: bool = False,
) -> Dict[str, object]:
    """Run BombCell over many Kilosort folders, de-duplicating the input list.

    Folders are processed in first-seen order (case-insensitive de-dup). Failures
    are captured per folder rather than aborting the batch. Returns the per-folder
    results plus an aggregate summary (run/reuse/fail counts and label totals).
    """
    normalized_settings = normalize_pybombcell_settings(settings)
    seen: set[str] = set()
    ordered_folders: List[str] = []
    for folder in folders:
        raw = str(folder).strip()
        if not raw:
            continue
        key = raw.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered_folders.append(raw)

    results: List[Dict[str, object]] = []
    summary = {
        "total": len(ordered_folders),
        "reran": 0,
        "reused_metrics": 0,
        "cached": 0,
        "failed": 0,
        "good": 0,
        "noise": 0,
        "mua": 0,
        "non_soma": 0,
    }

    for folder in ordered_folders:
        try:
            result = run_pybombcell_on_folder(
                folder,
                save_plots=save_plots,
                force_recompute=force_recompute,
                settings=normalized_settings,
                extract_raw=extract_raw,
            )
        except Exception as exc:
            results.append({"folder": folder, "ok": False, "error": str(exc)})
            summary["failed"] += 1
            continue

        results.append({"folder": folder, "ok": True, "result": result})
        if result.get("cached", False):
            summary["cached"] += 1
        elif result.get("metrics_reused", False):
            summary["reused_metrics"] += 1
        else:
            summary["reran"] += 1
        counts = result.get("counts", {})
        for key in ("good", "noise", "mua", "non_soma"):
            summary[key] += int(counts.get(key, 0))

    return {
        "results": results,
        "summary": summary,
        "settings_signature": pybombcell_settings_signature(normalized_settings),
    }

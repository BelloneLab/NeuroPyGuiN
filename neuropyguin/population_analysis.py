"""Validated population summaries and time-aware continuous behavior decoding.

Matrices use rows=time and columns=units. Embeddings describe the complete
selected interval; predictive evaluation independently fits scaling and ridge
regression inside each held-out temporal fold. No embedding fitted on test data
is used for that evaluation. All spike times and behavioral clocks are seconds.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import importlib
import importlib.metadata
import json
import sys

import numpy as np
import pandas as pd

# Conservative defaults keep interactive work bounded and reproducible.
MAX_MATRIX_CELLS = 5_000_000
MAX_TIME_BINS = 100_000
DEFAULT_SEED = 0
NEMBA_SOURCE = "https://github.com/Andrianarivelo/NEMBA"


@dataclass(frozen=True)
class PopulationConfig:
    """User-visible analysis settings, also saved verbatim in the manifest."""
    start_s: float = 0.0
    stop_s: float = 60.0
    bin_s: float = 0.1
    method: str = "pca"
    components: int = 3
    standardize: bool = True
    behavior_path: str = ""
    behavior_column: str = ""
    max_behavior_gap_s: float = 1.0
    cv_gap_s: float = 1.0
    permutations: int = 99
    seed: int = DEFAULT_SEED
    nemba_path: str = ""


def bin_population(dataset, units, config: PopulationConfig):
    """Count spikes in complete, half-open bins without changing source arrays.

    Incomplete trailing bins are omitted and the effective end is recorded.
    Explicit allocation limits prevent a small bin setting exhausting memory.
    """
    units = np.asarray(list(dict.fromkeys(int(u) for u in units)), dtype=np.int64)
    if units.size < 2:
        raise ValueError("Choose at least two units for population analysis.")
    if not np.isin(units, np.asarray(dataset.units)).all():
        raise ValueError("The selection contains units absent from this dataset.")
    start, stop, width = config.start_s, config.stop_s, config.bin_s
    if not np.isfinite([start, stop, width]).all() or start < 0 or stop <= start or width <= 0:
        raise ValueError("Use a finite interval with 0 <= start < end and a positive bin width.")
    n_bins = int(np.floor((stop - start) / width + 1e-9))
    if n_bins < 10:
        raise ValueError("The selected interval must contain at least 10 complete time bins.")
    if n_bins > MAX_TIME_BINS or n_bins * units.size > MAX_MATRIX_CELLS:
        raise ValueError("This matrix is too large. Increase bin width, shorten the interval, or select fewer units.")
    edges = start + np.arange(n_bins + 1, dtype=float) * width
    counts = np.zeros((n_bins, len(units)), dtype=np.float64)
    for col, unit in enumerate(units):
        times = np.asarray(dataset.unit_spike_times_s(int(unit)), dtype=float)
        # Histogram's final bin is closed by default; explicitly enforce [start,end).
        times = times[(times >= edges[0]) & (times < edges[-1]) & np.isfinite(times)]
        counts[:, col] = np.histogram(times, bins=edges)[0]
    return units, (edges[:-1] + edges[1:]) / 2, counts


def align_continuous_behavior(path, column, times, max_gap_s=1.0):
    """Interpolate a numeric behavior on the neural clock, preserving missingness.

    CSV must contain a strictly increasing ``time`` or ``time_s`` column in
    seconds on the SAME clock as spikes. There is no extrapolation, interpolation
    across NaNs, or interpolation across gaps exceeding ``max_gap_s``.
    """
    table = pd.read_csv(path)
    time_key = "time_s" if "time_s" in table else "time"
    if time_key not in table or column not in table or column == time_key:
        raise ValueError("Behavior CSV needs time_s (or time) and a separate numeric target column.")
    t = pd.to_numeric(table[time_key], errors="raise").to_numpy(dtype=float)
    y = pd.to_numeric(table[column], errors="raise").to_numpy(dtype=float)
    if len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Behavior timestamps must be finite, unique, and strictly increasing.")
    if not np.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("Maximum behavior gap must be positive and finite.")
    times = np.asarray(times, float)
    right = np.searchsorted(t, times, side="left")
    left = np.clip(right - 1, 0, len(t) - 1)
    right = np.clip(right, 0, len(t) - 1)
    exact = t[right] == times
    valid = ((times >= t[0]) & (times <= t[-1]) & np.isfinite(y[left])
             & np.isfinite(y[right]) & ((t[right] - t[left]) <= max_gap_s))
    valid |= exact & np.isfinite(y[right])
    aligned = np.full(times.size, np.nan)
    interpolated = np.interp(times, t, y)
    aligned[valid] = interpolated[valid]
    return aligned


def temporal_decode(rates, target, *, gap_bins=10, permutations=99, seed=0):
    """Evaluate ridge decoding against a train-mean baseline on five time blocks.

    Scaling is fitted ONLY on training bins. Test blocks are purged on both
    sides. Circular-shift nulls preserve target autocorrelation under the
    stationarity assumption, which is recorded and exposed in the interface.
    Scores are descriptive within this recording, not across-animal inference.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler

    X, y = np.asarray(rates, float), np.asarray(target, float)
    if X.ndim != 2 or y.shape != (len(X),) or not np.isfinite(X).all():
        raise ValueError("Decoding needs a finite time-by-unit matrix and one target per bin.")
    if gap_bins < 0 or permutations < 0:
        raise ValueError("CV gap and permutation count cannot be negative.")
    if np.isfinite(y).sum() < 60 or np.nanstd(y) <= 1e-12:
        raise ValueError("Decoding needs at least 60 covered bins and a variable continuous target.")
    blocks = np.array_split(np.arange(len(y)), 5)

    def evaluate(values, details=False):
        """Fit the complete preprocessing and model pipeline for every fold."""
        pred = np.full(len(values), np.nan)
        baseline = np.full(len(values), np.nan)
        rows = []
        for fold, block in enumerate(blocks):
            train = np.isfinite(values)
            train[max(0, block[0] - gap_bins):min(len(y), block[-1] + gap_bins + 1)] = False
            test = block[np.isfinite(values[block])]
            if train.sum() < 20 or len(test) < 5:
                continue
            scaler = StandardScaler().fit(X[train])
            model = Ridge(alpha=1.0).fit(scaler.transform(X[train]), values[train])
            pred[test] = model.predict(scaler.transform(X[test]))
            baseline[test] = np.mean(values[train])
            rows.append({"fold": fold + 1, "n_train": int(train.sum()), "n_test": len(test),
                         "r2": float(r2_score(values[test], pred[test])),
                         "baseline_r2": float(r2_score(values[test], baseline[test]))})
        valid = np.isfinite(pred)
        if valid.sum() < 30:
            raise ValueError("Too few bins remain after temporal purging. Reduce the gap or extend the interval.")
        score = float(r2_score(values[valid], pred[valid]))
        return score, pred, baseline, rows

    score, predictions, baseline, rows = evaluate(y, True)
    rng = np.random.default_rng(seed)
    min_shift = max(gap_bins + 1, len(y) // 10)
    possible = np.arange(min_shift, len(y) - min_shift + 1)
    if permutations and not possible.size:
        raise ValueError("Interval is too short for a shift null with this purge gap.")
    # Unique shifts avoid pretending that repeated offsets are independent null draws.
    shifts = rng.choice(possible, size=min(int(permutations), len(possible)), replace=False)
    null_scores, valid_shifts = [], []
    if np.isfinite(y).all() and len(shifts):
        # The same training features serve every shifted target. Multi-output
        # ridge shares its matrix factorization, avoiding hundreds of repeated
        # solves. Small batches bound temporary memory for long recordings.
        for offset in range(0, len(shifts), 32):
            batch_shifts = shifts[offset:offset + 32]
            targets = np.column_stack([np.roll(y, int(shift)) for shift in batch_shifts])
            batch_predictions = np.full_like(targets, np.nan)
            for block in blocks:
                train = np.ones(len(y), dtype=bool)
                train[max(0, block[0] - gap_bins):min(len(y), block[-1] + gap_bins + 1)] = False
                if train.sum() < 20 or len(block) < 5:
                    continue
                scaler = StandardScaler().fit(X[train])
                model = Ridge(alpha=1.0).fit(scaler.transform(X[train]), targets[train])
                batch_predictions[block] = model.predict(scaler.transform(X[block]))
            keep = np.isfinite(batch_predictions[:, 0])
            null_scores.extend(np.atleast_1d(r2_score(targets[keep], batch_predictions[keep], multioutput="raw_values")))
            valid_shifts.extend(batch_shifts)
    else:
        for shift in shifts:
            try:
                shifted_score = evaluate(np.roll(y, int(shift)))[0]
            except ValueError:
                # Missing coverage may make a shifted target unevaluable. Dropping
                # such draws would condition the null on coverage, so withhold p.
                null_scores = []
                valid_shifts = []
                break
            null_scores.append(shifted_score)
            valid_shifts.append(shift)
    null = np.asarray(null_scores, float)
    shifts = np.asarray(valid_shifts, int)
    valid = np.isfinite(predictions)
    return {"predictions": predictions, "baseline": baseline, "folds": rows,
            "r2": score, "baseline_r2": float(r2_score(y[valid], baseline[valid])),
            "p_shift": float((1 + np.sum(null >= score)) / (len(null) + 1)) if len(null) else None,
            "null_scores": null, "null_shifts_bins": shifts,
            "n_evaluated": int(valid.sum()),
            "null_note": "Circular-shift test assumes stationarity." if len(null) else "Null test disabled or unavailable because shifted missing coverage prevents valid evaluation."}


def nemba_embedding(counts, times, config):
    """Call NEMBA's verified public Embedder.fit_matrix API lazily.

    A separately selected interpreter can run this module, avoiding changes to
    the sorting environment's Torch stack. An optional source checkout can be
    supplied explicitly. NEMBA never receives or modifies original spike files.
    """
    if config.nemba_path:
        source = Path(config.nemba_path).expanduser().resolve()
        if not (source / "embedders_backbones" / "__init__.py").is_file():
            raise ValueError("NEMBA source must contain embedders_backbones/__init__.py.")
        sys.path.insert(0, str(source))
    try:
        module = importlib.import_module("embedders_backbones")
    except ImportError as exc:
        raise RuntimeError("NEMBA is unavailable in this Python environment. Choose a NEMBA Python interpreter and/or source checkout in Advanced settings. See README Population + behavior setup.") from exc
    backbone = config.method.removeprefix("nemba:")
    X = counts.T.astype(np.float32)
    X = (X - X.mean(axis=1, keepdims=True)) / np.maximum(X.std(axis=1, keepdims=True), 1e-8)
    embedder = module.Embedder(backbone, bin_size=config.bin_s,
                              embedding_dim=config.components, n_states=3,
                              smooth_sigma=0, seed=config.seed, speed="fast",
                              device="cpu", n_threads=2, log=print)
    embedder.fit_matrix(X, times=times, counts=counts.T)
    Z = np.asarray(embedder.Z, float)
    if Z.shape[0] != len(times) or Z.ndim != 2 or not np.isfinite(Z).all():
        raise ValueError("NEMBA returned an invalid embedding for this time axis.")
    try:
        distribution = importlib.metadata.distribution("embedders_backbones")
        version = distribution.version
        direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    except importlib.metadata.PackageNotFoundError:
        version, direct_url = str(getattr(module, "__version__", "unknown")), {}
    return Z, {"nemba_source": NEMBA_SOURCE, "nemba_package": str(Path(module.__file__).resolve()),
               "nemba_version": version, "nemba_install_provenance": direct_url,
               "nemba_backbone": backbone, "nemba_smooth_sigma": 0,
               "nemba_states_note": "Fixed three-state model is internal only; states are not interpreted here."}


def analyze_population(dataset, units, config):
    """Compute one reproducible population result suitable for worker processes."""
    from sklearn.decomposition import PCA, FactorAnalysis

    units, times, counts = bin_population(dataset, units, config)
    rates = counts / config.bin_s
    varying = rates.std(axis=0) > 1e-10
    if varying.sum() < 2:
        raise ValueError("At least two units need variable spike counts in this interval.")
    dimensions = min(int(config.components), int(varying.sum()), len(times) - 1)
    if dimensions < 2:
        raise ValueError("Choose at least two embedding dimensions.")
    X = rates[:, varying]
    if config.standardize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    explained = np.array([])
    metadata = {"config": asdict(config), "source_dataset": str(dataset.ks_folder),
                "sample_rate_hz": float(dataset.sample_rate) if hasattr(dataset, "sample_rate") else None,
                "selected_units": units.tolist(), "excluded_constant_units": units[~varying].tolist(),
                "effective_stop_s": float(times[-1] + config.bin_s / 2),
                "analysis_note": "Embeddings are descriptive fits to this interval. Decoding uses unsmoothed raw rates, with fold-local scaling, five contiguous test blocks and purged training bins.",
                "null_assumption": "Circular-shift inference assumes approximately stationary behavior and population statistics. Inspect drift and repeat across independent recordings.",
                "behavior_clock": "Behavior time_s/time must already share the spike clock; no clock synchronization is inferred."}
    if config.method == "pca":
        model = PCA(n_components=dimensions, svd_solver="full")
        embedding = model.fit_transform(X)
        explained = model.explained_variance_ratio_
    elif config.method == "fa":
        embedding = FactorAnalysis(n_components=dimensions, random_state=config.seed).fit_transform(X)
    elif config.method.startswith("nemba:"):
        embedding, details = nemba_embedding(counts[:, varying], times, replace(config, components=dimensions))
        metadata.update(details)
    else:
        raise ValueError(f"Unknown embedding method: {config.method}")
    metadata["embedding_dimensions_used"] = int(embedding.shape[1])
    metadata["source_files"] = {}
    source_paths = [Path(dataset.ks_folder) / name for name in ("spike_times.npy", "spike_clusters.npy", "params.py")]
    source_paths.extend(Path(dataset.ks_folder).glob("*.meta"))
    source_paths.extend(Path(p) for p in getattr(dataset, "source_metadata_files", []))
    source_paths.extend([Path(config.behavior_path)] if config.behavior_path else [])
    for source in source_paths:
        if source.is_file():
            stat = source.stat()
            metadata["source_files"][str(source.resolve())] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    metadata["software_versions"] = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "pandas")}
    result = {"times": times, "units": units, "counts": counts, "rates": rates,
              "embedding": embedding, "explained_variance": explained, "metadata": metadata}
    if config.behavior_path:
        target = align_continuous_behavior(config.behavior_path, config.behavior_column,
                                           times, config.max_behavior_gap_s)
        result["behavior"] = target
        metadata["behavior_coverage"] = float(np.isfinite(target).mean())
        if np.isfinite(target).sum() < 2:
            raise ValueError("Behavior does not overlap this interval. Check seconds, clocks and maximum gap.")
        try:
            result["decoding"] = temporal_decode(rates[:, varying], target,
                                                 gap_bins=int(np.ceil(config.cv_gap_s / config.bin_s)),
                                                 permutations=config.permutations, seed=config.seed)
        except ValueError as exc:
            metadata["decoding_unavailable"] = str(exc)
    return result


def save_result(result, path):
    """Write only plain arrays and JSON, never executable pickle data."""
    arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    metadata = dict(result["metadata"])
    if "decoding" in result:
        metadata["decoding"] = {}
        for key, value in result["decoding"].items():
            if isinstance(value, np.ndarray):
                arrays["decoding_" + key] = value
            else:
                metadata["decoding"][key] = value
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, allow_nan=False))
    np.savez_compressed(path, **arrays)


def load_result(path):
    """Read a worker result without allowing pickled objects."""
    with np.load(path, allow_pickle=False) as archive:
        result = {k: archive[k] for k in archive.files if k != "metadata_json"}
        result["metadata"] = json.loads(str(archive["metadata_json"]))
    if "decoding" in result["metadata"]:
        result["decoding"] = result["metadata"].pop("decoding")
        for key in list(result):
            if key.startswith("decoding_"):
                result["decoding"][key[9:]] = result.pop(key)
    return result


def result_tables(result):
    """Return all plotted values and evaluation tables for CSV export."""
    frame = pd.DataFrame({"time_s": result["times"]})
    for col in range(result["embedding"].shape[1]):
        frame[f"component_{col + 1}"] = result["embedding"][:, col]
    if "behavior" in result:
        frame["behavior"] = result["behavior"]
    tables = [("embedding", frame),
              ("rates_hz", pd.DataFrame(result["rates"], index=result["times"],
                                       columns=[f"unit_{u}" for u in result["units"]]).rename_axis("time_s").reset_index())]
    if result["explained_variance"].size:
        tables.append(("explained_variance", pd.DataFrame({"component": np.arange(len(result["explained_variance"])) + 1,
                                                          "variance_fraction": result["explained_variance"]})))
    if "decoding" in result:
        dec = result["decoding"]
        tables.append(("decoding_summary", pd.DataFrame([{key: dec[key] for key in ("r2", "baseline_r2", "p_shift", "n_evaluated", "null_note")}])) )
        tables.extend([("decoding_folds", pd.DataFrame(dec["folds"])),
                       ("decoding_predictions", pd.DataFrame({"time_s": result["times"],
                          "observed": result["behavior"], "predicted": dec["predictions"], "train_mean": dec["baseline"]})),
                       ("decoding_null", pd.DataFrame({"shift_bins": dec["null_shifts_bins"], "r2": dec["null_scores"]}))])
    return tables

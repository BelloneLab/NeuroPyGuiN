"""``neuropyguin postproc ...`` - everything the Post Processing tab can do.

Dataset inspection, unit listing with labels, HDF5 unit export, condition PSTHs,
correlograms (native and npyx), population network analysis, the C4 and BombCell
cell-type classifiers, event-CSV inspection, and figure rendering to PNG/PDF.

All analyses run through :class:`~neuropyguin.postproc_engine.NeuropixelsDataset`,
the same object the tab loads, so results match the GUI exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ._common import (
    CommandError,
    Reporter,
    format_table,
    parse_int_list,
    require_file,
    resolve_ks_folder,
    write_json_file,
)


# ---------------------------------------------------------------------------
# Dataset, labels and unit selection
# ---------------------------------------------------------------------------


def _load_dataset(path: str):
    """Load a :class:`NeuropixelsDataset` from a (possibly nested) folder path."""
    from ..postproc_engine import NeuropixelsDataset

    folder = resolve_ks_folder(path)
    try:
        return NeuropixelsDataset.load(str(folder))
    except Exception as exc:  # noqa: BLE001 - reported as a clean CLI error
        raise CommandError(f"Failed to load dataset from {folder}: {exc}") from exc


def _read_label_sources(folder: Path) -> Dict[str, Any]:
    """Read the three unit-label sources the GUI understands.

    Mirrors ``PostProcessingTab._read_label_sources``: BombCell labels, the phy
    ``cluster_group.tsv``, and the Kilosort ``cluster_KSLabel.tsv``. Each frame is
    re-indexed by integer cluster id. The logic is duplicated here (rather than
    imported from the tab) so the CLI never has to import the Qt widget stack.
    """
    import pandas as pd

    from ..bombcell_core import _normalize_label_df  # shared index normalisation

    sources: Dict[str, Any] = {}
    candidates = [
        ("Bombcell", folder / "bombcell_labels.csv", {}),
        ("Phy", folder / "cluster_group.tsv", {"sep": "\t"}),
        ("KSLabel", folder / "cluster_KSLabel.tsv", {"sep": "\t"}),
    ]
    for name, path, kwargs in candidates:
        if not path.exists():
            continue
        try:
            sources[name] = _normalize_label_df(pd.read_csv(path, **kwargs))
        except Exception:
            continue  # an unreadable sidecar simply means that source is absent
    return sources


def _is_good_label(value: object) -> bool:
    """Return True for the label spellings that denote a usable unit."""
    text = str(value).strip().lower()
    if not text or text == "nan":
        return False
    normalized = text.replace("-", "_").replace(" ", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized in {"good", "non_soma", "non_soma_good", "nonsoma", "nonsomagood"}


def _unit_label(sources: Dict[str, Any], unit: int, source: str) -> tuple[str, str]:
    """Return ``(label, source_name)`` for one unit under the chosen source.

    ``source="Auto"`` walks BombCell, then Phy, then KSLabel, taking the first
    source that has a row for this unit, which is the priority the GUI uses.
    """
    order = ["Bombcell", "Phy", "KSLabel"] if source == "Auto" else [source]
    for name in order:
        frame = sources.get(name)
        if frame is None or frame.empty or unit not in frame.index:
            continue
        row = frame.loc[unit]
        if hasattr(row, "iloc") and getattr(row, "ndim", 1) > 1:
            row = row.iloc[0]
        for key in ("bombcell_label", "group", "KSLabel", "label", "kslabel"):
            if key in row.index:
                return str(row[key]), name
        return str(row.iloc[0]), name
    return "", ""


def _select_units(dataset, sources: Dict[str, Any], args) -> List[int]:
    """Resolve ``--units`` / ``--good-only`` / ``--limit`` into a unit id list."""
    all_units = [int(u) for u in dataset.units.tolist()]
    requested = parse_int_list(getattr(args, "units", None))
    if requested:
        unknown = [u for u in requested if u not in set(all_units)]
        if unknown:
            raise CommandError(f"Units not present in the dataset: {unknown}")
        units = requested
    else:
        units = list(all_units)

    if getattr(args, "good_only", False):
        source = getattr(args, "good_source", "Auto") or "Auto"
        if not sources:
            # No label file at all: the GUI treats every unit as good.
            pass
        else:
            units = [u for u in units if _is_good_label(_unit_label(sources, u, source)[0])]

    limit = int(getattr(args, "limit", 0) or 0)
    if limit > 0:
        units = units[:limit]
    if not units:
        raise CommandError("No units matched the selection.")
    return units


def _figure_output(path: str | None, default_name: str, folder: Path) -> Path:
    """Resolve a figure output path, defaulting to ``<ks_folder>/<default_name>``."""
    if path:
        out = Path(path).expanduser()
    else:
        out = folder / default_name
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _save_figure(fig, out_path: Path, dpi: int = 150) -> Path:
    """Save a matplotlib figure and close it, returning the written path."""
    import matplotlib.pyplot as plt

    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _use_agg() -> None:
    """Force matplotlib's headless backend before any figure is created."""
    import matplotlib

    matplotlib.use("Agg", force=False)


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


def cmd_info(args, reporter: Reporter) -> int:
    """Summarise a sorted dataset: units, spikes, duration, geometry, labels."""
    import numpy as np

    dataset = _load_dataset(args.ks_folder)
    sources = _read_label_sources(dataset.ks_folder)
    n_spikes = int(np.asarray(dataset.spike_times).size)
    duration = float(np.max(dataset.spike_times)) / float(dataset.sample_rate) if n_spikes else 0.0

    label_counts: Dict[str, Dict[str, int]] = {}
    for name, frame in sources.items():
        column = next(
            (c for c in ("bombcell_label", "group", "KSLabel") if c in frame.columns),
            None,
        )
        if column is not None:
            label_counts[name] = {
                str(k): int(v) for k, v in frame[column].astype(str).value_counts().items()
            }

    payload = {
        "ks_folder": str(dataset.ks_folder),
        "n_units": int(len(dataset.units)),
        "n_spikes": n_spikes,
        "sample_rate_hz": float(dataset.sample_rate),
        "duration_s": round(duration, 3),
        "n_channels": int(dataset.n_channels),
        "bit_uV": float(dataset.bit_uV),
        "ap_bin_path": str(dataset.ap_bin_path) if dataset.ap_bin_path else "",
        "has_templates": dataset.templates is not None,
        "label_sources": sorted(sources),
        "label_counts": label_counts,
    }
    reporter.emit(payload)
    return 0


def cmd_units(args, reporter: Reporter) -> int:
    """List units with spike counts, mean rate, and their labels."""
    import numpy as np

    dataset = _load_dataset(args.ks_folder)
    sources = _read_label_sources(dataset.ks_folder)
    units = _select_units(dataset, sources, args)

    clusters = np.asarray(dataset.spike_clusters)
    times = np.asarray(dataset.spike_times, dtype=float)
    duration = float(times.max()) / float(dataset.sample_rate) if times.size else 0.0

    rows: List[Dict[str, Any]] = []
    for unit in units:
        mask = clusters == unit
        count = int(np.count_nonzero(mask))
        label, source = _unit_label(sources, unit, args.good_source or "Auto")
        rows.append(
            {
                "unit": unit,
                "n_spikes": count,
                "rate_hz": round(count / duration, 3) if duration > 0 else 0.0,
                "label": label,
                "source": source,
            }
        )

    if args.output:
        import pandas as pd

        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        reporter.log(f"Wrote {out_path}")

    reporter.emit(
        {"ks_folder": str(dataset.ks_folder), "count": len(rows), "units": rows},
        text=format_table(rows, ["unit", "n_spikes", "rate_hz", "label", "source"]),
    )
    return 0


def cmd_events(args, reporter: Reporter) -> int:
    """Inspect an event CSV: detected shape, time column, and usable labels."""
    from ..postproc_events import inspect_event_csv, load_event_times

    path = require_file(args.csv, "event CSV")
    info = inspect_event_csv(path)
    frame = info.pop("dataframe")

    payload = dict(info)
    payload["n_rows"] = int(len(frame))
    payload["columns"] = [str(c) for c in frame.columns]

    if args.label is not None:
        times = load_event_times(
            path,
            selected_label=args.label,
            frame_rate=args.frame_rate,
            alignment=args.alignment,
            min_bout_s=args.min_bout_s,
        )
        payload["selected_label"] = args.label
        payload["n_events"] = int(times.size)
        payload["first_events_s"] = [round(float(v), 4) for v in times.head(10).tolist()]

    reporter.emit(payload)
    return 0


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def cmd_export_units(args, reporter: Reporter) -> int:
    """Export selected units to the structured NeuroPyGuiN HDF5 format."""
    import pandas as pd

    from ..postproc_engine import export_units_h5

    dataset = _load_dataset(args.ks_folder)
    sources = _read_label_sources(dataset.ks_folder)
    units = _select_units(dataset, sources, args)

    metrics_df = pd.DataFrame()
    metrics_path = dataset.ks_folder / "metrics.csv"
    if metrics_path.exists():
        try:
            metrics_df = pd.read_csv(metrics_path)
            if "cluster_id" in metrics_df.columns:
                metrics_df = metrics_df.set_index("cluster_id", drop=True)
        except Exception as exc:  # noqa: BLE001
            reporter.warn(f"Could not read metrics.csv: {exc}")

    good_source = args.good_source or "Auto"
    good_units = [u for u in units if _is_good_label(_unit_label(sources, u, good_source)[0])]

    suffix = "good_units" if args.good_only else "all_units"
    out_path = Path(
        args.output or (dataset.ks_folder / f"{dataset.ks_folder.name}_{suffix}.h5")
    ).expanduser()
    if out_path.suffix.lower() not in {".h5", ".hdf5"}:
        out_path = out_path.with_suffix(".h5")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    reporter.log(f"Exporting {len(units)} unit(s) to {out_path}...")
    last = {"pct": -1}

    def progress(pct: int) -> None:
        """Log only whole-percent changes so the terminal stays readable."""
        if pct != last["pct"]:
            last["pct"] = pct
            reporter.log(f"  {pct}%")

    summary = export_units_h5(
        dataset,
        out_path,
        units,
        labels_df=sources.get("Bombcell"),
        metrics_df=metrics_df if not metrics_df.empty else None,
        label_sources=sources,
        good_units=good_units,
        good_source=good_source,
        export_mode="good_only" if args.good_only else "all",
        progress_callback=progress,
    )
    payload = dict(summary)
    payload["output"] = str(out_path)
    reporter.emit(payload)
    return 0


def cmd_export_figures(args, reporter: Reporter) -> int:
    """Render one waveform + ACG card per unit as PNGs plus a combined PDF."""
    _use_agg()
    # Imported lazily: this pulls the Qt widget stack in through the tab module,
    # which only this command needs.
    from ..tabs.postprocessing_tab import _export_good_unit_figures

    dataset = _load_dataset(args.ks_folder)
    sources = _read_label_sources(dataset.ks_folder)
    units = _select_units(dataset, sources, args)
    out_dir = Path(args.output or (dataset.ks_folder / "unit_figures")).expanduser()

    reporter.log(f"Rendering {len(units)} figure(s) to {out_dir} (reads the AP binary per unit)...")

    def progress(done: int, total: int) -> None:
        """Report every tenth unit so long batches show movement."""
        if done % 10 == 0 or done == total:
            reporter.log(f"  {done}/{total}")

    result = _export_good_unit_figures(
        dataset, units, str(out_dir), dark=args.dark, progress_cb=progress
    )
    reporter.emit(result)
    return 1 if result.get("error") and not result.get("n") else 0


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------


def cmd_psth(args, reporter: Reporter) -> int:
    """Compute a condition PSTH from one or more event CSVs.

    Each ``--events NAME=PATH[:LABEL]`` adds one condition. The result mirrors the
    structure the GUI caches, so the same figure renderer can draw it.
    """
    import numpy as np

    from ..postproc_events import load_event_times

    _use_agg()
    dataset = _load_dataset(args.ks_folder)
    sources = _read_label_sources(dataset.ks_folder)
    units = _select_units(dataset, sources, args)

    conditions: List[Dict[str, Any]] = []
    t_ref = np.array([], dtype=float)

    for spec in args.events:
        name, _, rest = spec.partition("=")
        if not rest:
            name, rest = Path(spec).stem, spec
        csv_path, _, label = rest.partition("::")
        path = require_file(csv_path, "event CSV")
        times = load_event_times(
            path,
            selected_label=label or None,
            frame_rate=args.frame_rate,
            alignment=args.alignment,
            min_bout_s=args.min_bout_s,
        ).to_numpy(dtype=float)
        if times.size == 0:
            reporter.warn(f"No usable events in {path} (label={label or 'all'}); skipping.")
            continue

        unit_ids: List[int] = []
        trial_mats: List[Any] = []
        for unit in units:
            t_ms, trial_mat = dataset.psth_trials(int(unit), times, args.pre, args.post, args.bin_ms)
            if t_ms.size == 0 or trial_mat.size == 0:
                continue
            if t_ref.size == 0:
                t_ref = np.asarray(t_ms, dtype=float)
            unit_ids.append(int(unit))
            trial_mats.append(np.asarray(trial_mat, dtype=float))
        if not trial_mats:
            reporter.warn(f"Could not build a PSTH for condition {name}; skipping.")
            continue

        conditions.append(
            {
                "condition": name,
                "selected_label": label,
                "source_csv": str(path),
                "unit_ids": unit_ids,
                "unit_trial_mats": trial_mats,
                "trial_count": int(times.size),
            }
        )

    if not conditions or t_ref.size == 0:
        raise CommandError("No valid conditions produced a PSTH.")

    results = {"t_ms": t_ref, "conditions": conditions, "units": units}

    written: Dict[str, str] = {}
    if args.output:
        # One row per (condition, unit) with the trial-averaged rate per bin.
        import pandas as pd

        frames = []
        for cond in conditions:
            for unit, mat in zip(cond["unit_ids"], cond["unit_trial_mats"]):
                mean_rate = np.nanmean(mat, axis=0)
                frames.append(
                    pd.DataFrame(
                        {
                            "condition": cond["condition"],
                            "unit": unit,
                            "t_ms": t_ref,
                            "rate_hz": mean_rate,
                        }
                    )
                )
        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)
        written["csv"] = str(out_path)
        reporter.log(f"Wrote {out_path}")

    if args.figure is not None:
        from ..unit_figures import condition_psth_figure

        fig = condition_psth_figure(
            results, mode=args.mode, baseline=args.baseline, dark=args.dark
        )
        out = _figure_output(args.figure, "condition_psth.png", dataset.ks_folder)
        written["figure"] = str(_save_figure(fig, out))
        reporter.log(f"Wrote {out}")

    summary = {
        "ks_folder": str(dataset.ks_folder),
        "n_units": len(units),
        "bins": int(t_ref.size),
        "window_s": [-args.pre, args.post],
        "bin_ms": args.bin_ms,
        "conditions": [
            {
                "condition": c["condition"],
                "trials": c["trial_count"],
                "units": len(c["unit_ids"]),
                "source_csv": c["source_csv"],
            }
            for c in conditions
        ],
        "written": written,
    }
    reporter.emit(summary)
    return 0


def cmd_correlogram(args, reporter: Reporter) -> int:
    """Compute auto- and cross-correlograms for the selected units."""
    import numpy as np

    _use_agg()
    dataset = _load_dataset(args.ks_folder)
    sources = _read_label_sources(dataset.ks_folder)
    units = _select_units(dataset, sources, args)
    if len(units) > args.max_units:
        reporter.warn(f"Limiting to the first {args.max_units} units (use --max-units to raise).")
        units = units[: args.max_units]

    pairs: List[Dict[str, Any]] = []
    arrays: Dict[str, Any] = {}
    for i, unit_a in enumerate(units):
        targets = units[i:] if args.mode == "cross" else [unit_a]
        for unit_b in targets:
            centers, counts = dataset.correlogram(
                unit_a, unit_b, bin_ms=args.bin_ms, win_ms=args.win_ms, remove_zero=(unit_a == unit_b)
            )
            key = f"{unit_a}_{unit_b}"
            arrays[f"counts_{key}"] = np.asarray(counts)
            pairs.append(
                {
                    "unit_a": unit_a,
                    "unit_b": unit_b,
                    "kind": "acg" if unit_a == unit_b else "ccg",
                    "peak": float(np.nanmax(counts)) if counts.size else 0.0,
                }
            )
    if pairs:
        arrays["lags_ms"] = np.asarray(centers)

    written = {}
    if args.output:
        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, units=np.asarray(units), **arrays)
        written["npz"] = str(out_path)
        reporter.log(f"Wrote {out_path}")

    if args.figure is not None:
        # The npyx grid renderers take a datapath and compute their own counts,
        # so they get the resolved analysis path rather than the dataset object.
        from ..npyx_corr_bridge import resolve_analysis_datapath
        from ..npyx_figures import acg_grid_figure, ccg_grid_figure

        datapath = resolve_analysis_datapath(str(dataset.ks_folder))
        figure_kwargs = dict(
            cbin=args.bin_ms, cwin=args.win_ms, fs=float(dataset.sample_rate), dark=args.dark
        )
        if args.mode == "cross" and len(units) > 1:
            fig = ccg_grid_figure(datapath, units, **figure_kwargs)
        else:
            fig = acg_grid_figure(datapath, units, **figure_kwargs)
        out = _figure_output(args.figure, f"correlogram_{args.mode}.png", dataset.ks_folder)
        written["figure"] = str(_save_figure(fig, out))
        reporter.log(f"Wrote {out}")

    reporter.emit(
        {
            "ks_folder": str(dataset.ks_folder),
            "mode": args.mode,
            "bin_ms": args.bin_ms,
            "win_ms": args.win_ms,
            "pairs": pairs,
            "written": written,
        },
        text=format_table(pairs, ["kind", "unit_a", "unit_b", "peak"]),
    )
    return 0


def cmd_npyx(args, reporter: Reporter) -> int:
    """Run one of the advanced npyx correlation methods on the selected units."""
    from ..npyx_corr_bridge import method_metadata, method_options, resolve_analysis_datapath, run_method

    if args.list_methods:
        rows = [{"method": key, "description": label} for key, label in method_options()]
        reporter.emit({"methods": rows}, text=format_table(rows, ["method", "description"]))
        return 0
    if not args.method:
        raise CommandError("Pass --method (or --list-methods to see the options).")

    dataset = _load_dataset(args.ks_folder)
    sources = _read_label_sources(dataset.ks_folder)
    units = _select_units(dataset, sources, args)
    datapath = resolve_analysis_datapath(str(dataset.ks_folder))

    params: Dict[str, Any] = {}
    for item in args.param or []:
        if "=" not in item:
            raise CommandError(f"--param expects name=value, got {item!r}")
        import json

        name, _, raw = item.partition("=")
        try:
            params[name.strip()] = json.loads(raw)
        except ValueError:
            params[name.strip()] = raw

    reporter.log(f"Running npyx method '{args.method}' on {len(units)} unit(s) at {datapath}")
    payload = run_method(
        args.method, datapath, units, bin_ms=args.bin_ms, win_ms=args.win_ms, params=params
    )

    if args.output:
        import numpy as np

        arrays = {k: np.asarray(v) for k, v in payload.items() if hasattr(v, "__len__") and not isinstance(v, (str, dict))}
        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **arrays)
        reporter.log(f"Wrote {out_path}")

    meta = method_metadata(args.method)
    reporter.emit(
        {
            "method": args.method,
            "description": meta.get("label", args.method),
            "datapath": datapath,
            "units": units,
            "keys": sorted(payload),
            "error": payload.get("error"),
        }
    )
    return 1 if payload.get("error") else 0


def cmd_network(args, reporter: Reporter) -> int:
    """Run the population network analysis (correlations, coupling, connections)."""
    import numpy as np

    _use_agg()
    dataset = _load_dataset(args.ks_folder)
    sources = _read_label_sources(dataset.ks_folder)
    units = _select_units(dataset, sources, args)

    reporter.log(f"Analysing {len(units)} unit(s)...")
    results = dataset.network_analysis(
        units,
        bin_ms=args.bin_ms,
        compute_connections=not args.no_connections,
        conn_bin_ms=args.conn_bin_ms,
        conn_win_ms=args.conn_win_ms,
        conn_z=args.conn_z,
        max_conn_units=args.max_conn_units,
    )

    written: Dict[str, str] = {}
    if args.output:
        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            key: np.asarray(value)
            for key, value in results.items()
            if isinstance(value, np.ndarray) or isinstance(value, (list, tuple))
        }
        np.savez_compressed(out_path, **{k: v for k, v in arrays.items() if v is not None})
        written["npz"] = str(out_path)
        reporter.log(f"Wrote {out_path}")

    if args.figure is not None:
        from ..unit_figures import network_figure

        out = _figure_output(args.figure, "network.png", dataset.ks_folder)
        written["figure"] = str(_save_figure(network_figure(results, dark=args.dark), out))
        reporter.log(f"Wrote {out}")

    corr = np.asarray(results.get("corr_matrix"))
    coupling = np.asarray(results.get("population_coupling"))
    reporter.emit(
        {
            "ks_folder": str(dataset.ks_folder),
            "n_units": len(units),
            "corr_bin_ms": results.get("corr_bin_ms"),
            "mean_abs_correlation": float(np.nanmean(np.abs(corr))) if corr.size else 0.0,
            "max_population_coupling": float(np.nanmax(coupling)) if coupling.size else 0.0,
            "written": written,
        }
    )
    return 0


def cmd_c4(args, reporter: Reporter) -> int:
    """Classify units with the C4 cell-type classifier (isolated environment)."""
    _use_agg()
    from ..c4_runner import run_c4_classifier
    from ..npyx_corr_bridge import resolve_analysis_datapath

    dataset = _load_dataset(args.ks_folder)
    sources = _read_label_sources(dataset.ks_folder)
    units = _select_units(dataset, sources, args)
    datapath = resolve_analysis_datapath(str(dataset.ks_folder))

    results = run_c4_classifier(
        datapath,
        units,
        quality=args.quality,
        threshold=args.threshold,
        device=args.device,
        progress_cb=reporter.log,
        timeout_s=args.timeout,
    )
    if results.get("error"):
        raise CommandError(str(results["error"]))

    rows = [
        {"unit": int(u), "predicted": str(p), "confidence": round(float(c), 3)}
        for u, p, c in zip(results["units"], results["predicted_type"], results["confidence"])
    ]
    written: Dict[str, str] = {}
    if args.output:
        written["json"] = str(write_json_file(args.output, {"units": rows, "class_names": results.get("class_names")}))
        reporter.log(f"Wrote {written['json']}")
    if args.figure is not None:
        from ..unit_figures import c4_figure

        out = _figure_output(args.figure, "c4_celltypes.png", dataset.ks_folder)
        written["figure"] = str(_save_figure(c4_figure(results, dark=args.dark), out))
        reporter.log(f"Wrote {out}")

    reporter.emit(
        {
            "ks_folder": str(dataset.ks_folder),
            "model_type": results.get("model_type"),
            "class_names": results.get("class_names"),
            "skipped_units": results.get("skipped_units"),
            "predictions": rows,
            "written": written,
        },
        text=format_table(rows, ["unit", "predicted", "confidence"]),
    )
    return 0


def cmd_celltype(args, reporter: Reporter) -> int:
    """Classify units by cell type using BombCell's region-specific thresholds."""
    _use_agg()
    from ..bombcell_classify import run_bombcell_classifier
    from ..npyx_corr_bridge import resolve_analysis_datapath

    dataset = _load_dataset(args.ks_folder)
    sources = _read_label_sources(dataset.ks_folder)
    units = _select_units(dataset, sources, args)
    datapath = resolve_analysis_datapath(str(dataset.ks_folder))

    results = run_bombcell_classifier(datapath, units, region=args.region, progress_cb=reporter.log)
    if results.get("error"):
        raise CommandError(str(results["error"]))

    rows = [
        {"unit": int(u), "predicted": str(p)}
        for u, p in zip(results["units"], results["predicted_type"])
    ]
    written: Dict[str, str] = {}
    if args.output:
        written["json"] = str(write_json_file(args.output, {"region": args.region, "units": rows}))
        reporter.log(f"Wrote {written['json']}")
    if args.figure is not None:
        from ..unit_figures import bombcell_celltype_figure

        out = _figure_output(args.figure, f"celltypes_{args.region}.png", dataset.ks_folder)
        written["figure"] = str(_save_figure(bombcell_celltype_figure(results, dark=args.dark), out))
        reporter.log(f"Wrote {out}")

    reporter.emit(
        {
            "ks_folder": str(dataset.ks_folder),
            "region": results.get("region"),
            "class_names": results.get("class_names"),
            "skipped_units": results.get("skipped_units"),
            "predictions": rows,
            "written": written,
        },
        text=format_table(rows, ["unit", "predicted"]),
    )
    return 0


def cmd_figure(args, reporter: Reporter) -> int:
    """Render one of the standard analysis figures to an image file."""
    _use_agg()
    from ..unit_figures import raw_explorer_figure, unit_basics_figure, unit_waveform_acg_figure

    dataset = _load_dataset(args.ks_folder)
    sources = _read_label_sources(dataset.ks_folder)
    units = _select_units(dataset, sources, args)

    if args.kind == "basics":
        fig = unit_basics_figure(
            dataset,
            units,
            window_start_s=args.t0,
            window_s=args.duration,
            acg_bin_ms=args.acg_bin_ms,
            acg_win_ms=args.acg_win_ms,
            isi_max_ms=args.isi_max_ms,
            dark=args.dark,
        )
        default_name = f"unit_{units[0]}_basics.png"
    elif args.kind == "raw":
        fig = raw_explorer_figure(
            dataset,
            t0_s=args.t0,
            dur_s=args.duration,
            n_channels=args.channels,
            hp_hz=args.highpass,
            lp_hz=args.lowpass,
            overlay_units=units,
            dark=args.dark,
        )
        default_name = "raw_explorer.png"
    else:  # waveform-acg
        fig = unit_waveform_acg_figure(
            dataset, units[0], acg_bin_ms=args.acg_bin_ms, acg_win_ms=args.acg_win_ms, dark=args.dark
        )
        default_name = f"unit_{units[0]}_waveform_acg.png"

    out = _figure_output(args.output, default_name, dataset.ks_folder)
    _save_figure(fig, out, dpi=args.dpi)
    reporter.emit({"figure": str(out), "kind": args.kind, "units": units}, text=[str(out)])
    return 0


# ---------------------------------------------------------------------------
# Parser wiring
# ---------------------------------------------------------------------------


def _add_unit_selection(parser, *, default_limit: int = 0) -> None:
    """Attach the shared unit-selection options to a subcommand parser."""
    group = parser.add_argument_group("unit selection")
    group.add_argument("--units", help="Unit ids, e.g. '3,7,10-14'. Default: every unit.")
    group.add_argument("--good-only", action="store_true", help="Keep only units labelled good.")
    group.add_argument(
        "--good-source",
        choices=["Auto", "Bombcell", "Phy", "KSLabel"],
        default="Auto",
        help="Label source used to decide what counts as good (default: Auto).",
    )
    group.add_argument("--limit", type=int, default=default_limit, help="Stop after this many units (0 = no limit).")


def _add_figure_options(parser) -> None:
    """Attach the shared figure-rendering options to a subcommand parser."""
    parser.add_argument(
        "--figure",
        nargs="?",
        const="",
        metavar="PATH",
        help="Also render a figure. Give a path or omit it to use a default name.",
    )
    parser.add_argument("--dark", action="store_true", help="Render the figure on a dark background.")


def register(subparsers) -> None:
    """Attach the ``postproc`` command group to the top level parser."""
    group = subparsers.add_parser(
        "postproc",
        help="Analyse and export sorted datasets (PSTH, correlograms, network, exports).",
        description="Post Processing tab equivalents: dataset inspection, HDF5 export, PSTHs, "
        "correlograms, network analysis, and the cell-type classifiers.",
    )
    commands = group.add_subparsers(dest="command", required=True)

    p = commands.add_parser("info", help="Summarise a sorted dataset.")
    p.add_argument("ks_folder", help="Kilosort output folder (or a parent of it).")
    p.set_defaults(func=cmd_info)

    p = commands.add_parser("units", help="List units with spike counts and labels.")
    p.add_argument("ks_folder", help="Kilosort output folder.")
    p.add_argument("--output", metavar="PATH", help="Write the full table to a CSV file.")
    _add_unit_selection(p, default_limit=50)
    p.set_defaults(func=cmd_units)

    p = commands.add_parser("events", help="Inspect an event CSV before using it for a PSTH.")
    p.add_argument("csv", help="Event or behaviour CSV file.")
    p.add_argument("--label", help="Also load the times for this label / behaviour column.")
    p.add_argument("--frame-rate", type=float, help="Frame rate for behaviour matrices.")
    p.add_argument("--alignment", default="Onset (rising)", help="Bout alignment for behaviour matrices.")
    p.add_argument("--min-bout-s", type=float, default=0.0, help="Minimum bout duration (s).")
    p.set_defaults(func=cmd_events)

    p = commands.add_parser("export-units", help="Export units to the NeuroPyGuiN HDF5 format.")
    p.add_argument("ks_folder", help="Kilosort output folder.")
    p.add_argument("-o", "--output", metavar="PATH", help="Target .h5 file.")
    _add_unit_selection(p)
    p.set_defaults(func=cmd_export_units)

    p = commands.add_parser("export-figures", help="Render one waveform + ACG card per unit (PNG + PDF).")
    p.add_argument("ks_folder", help="Kilosort output folder.")
    p.add_argument("-o", "--output", metavar="DIR", help="Output folder (default: <ks_folder>/unit_figures).")
    p.add_argument("--dark", action="store_true", help="Render on a dark background.")
    _add_unit_selection(p)
    p.set_defaults(func=cmd_export_figures)

    p = commands.add_parser("psth", help="Compute a condition PSTH from event CSVs.")
    p.add_argument("ks_folder", help="Kilosort output folder.")
    p.add_argument(
        "--events",
        action="append",
        required=True,
        metavar="NAME=PATH[::LABEL]",
        help="One condition per flag. Repeatable.",
    )
    p.add_argument("--pre", type=float, default=1.0, help="Seconds before the event (default: 1.0).")
    p.add_argument("--post", type=float, default=2.0, help="Seconds after the event (default: 2.0).")
    p.add_argument("--bin-ms", type=float, default=20.0, help="PSTH bin width in ms (default: 20).")
    p.add_argument("--mode", choices=["average", "per_unit"], default="average", help="Figure layout.")
    p.add_argument("--baseline", action="store_true", help="Subtract the pre-event baseline.")
    p.add_argument("--frame-rate", type=float, help="Frame rate for behaviour matrices.")
    p.add_argument("--alignment", default="Onset (rising)", help="Bout alignment for behaviour matrices.")
    p.add_argument("--min-bout-s", type=float, default=0.0, help="Minimum bout duration (s).")
    p.add_argument("-o", "--output", metavar="PATH", help="Write the binned rates to a CSV file.")
    _add_figure_options(p)
    _add_unit_selection(p)
    p.set_defaults(func=cmd_psth)

    p = commands.add_parser("correlogram", help="Compute auto- and cross-correlograms.")
    p.add_argument("ks_folder", help="Kilosort output folder.")
    p.add_argument("--mode", choices=["auto", "cross"], default="auto", help="ACGs only, or all pairs.")
    p.add_argument("--bin-ms", type=float, default=0.5, help="Correlogram bin width in ms.")
    p.add_argument("--win-ms", type=float, default=100.0, help="Correlogram half-window in ms.")
    p.add_argument("--max-units", type=int, default=16, help="Safety cap on the number of units (default: 16).")
    p.add_argument("-o", "--output", metavar="PATH", help="Write the raw counts to an .npz file.")
    _add_figure_options(p)
    _add_unit_selection(p)
    p.set_defaults(func=cmd_correlogram)

    p = commands.add_parser("npyx", help="Run an advanced npyx correlation method.")
    p.add_argument("ks_folder", nargs="?", help="Kilosort output folder.")
    p.add_argument("--method", help="Method key (see --list-methods).")
    p.add_argument("--list-methods", action="store_true", help="List the available methods and exit.")
    p.add_argument("--bin-ms", type=float, default=0.5, help="Bin width in ms.")
    p.add_argument("--win-ms", type=float, default=100.0, help="Half-window in ms.")
    p.add_argument("--param", action="append", metavar="NAME=VALUE", help="Method parameter. Repeatable.")
    p.add_argument("-o", "--output", metavar="PATH", help="Write the numeric results to an .npz file.")
    _add_unit_selection(p)
    p.set_defaults(func=cmd_npyx)

    p = commands.add_parser("network", help="Population network analysis for the selected units.")
    p.add_argument("ks_folder", help="Kilosort output folder.")
    p.add_argument("--bin-ms", type=float, default=25.0, help="Correlation bin width in ms.")
    p.add_argument("--no-connections", action="store_true", help="Skip the CCG-based connection matrix.")
    p.add_argument("--conn-bin-ms", type=float, default=0.5, help="Connection CCG bin width in ms.")
    p.add_argument("--conn-win-ms", type=float, default=50.0, help="Connection CCG half-window in ms.")
    p.add_argument("--conn-z", type=float, default=5.0, help="Connection significance z-threshold.")
    p.add_argument("--max-conn-units", type=int, default=16, help="Cap on units used for connections.")
    p.add_argument("-o", "--output", metavar="PATH", help="Write the matrices to an .npz file.")
    _add_figure_options(p)
    _add_unit_selection(p)
    p.set_defaults(func=cmd_network)

    p = commands.add_parser("c4", help="Classify units with the C4 cell-type classifier.")
    p.add_argument("ks_folder", help="Kilosort output folder.")
    p.add_argument("--quality", default="good", help="Unit quality passed to C4 (default: good).")
    p.add_argument("--threshold", type=float, default=2.0, help="Confidence-ratio threshold (default: 2.0).")
    p.add_argument("--device", default="cpu", help="Torch device for C4 (default: cpu).")
    p.add_argument("--timeout", type=float, default=1800.0, help="Subprocess timeout in seconds.")
    p.add_argument("-o", "--output", metavar="PATH", help="Write the predictions to a JSON file.")
    _add_figure_options(p)
    _add_unit_selection(p)
    p.set_defaults(func=cmd_c4)

    p = commands.add_parser("celltype", help="Classify units with BombCell's region thresholds.")
    p.add_argument("ks_folder", help="Kilosort output folder.")
    p.add_argument("--region", choices=["cortex", "striatum"], default="cortex", help="Brain region ruleset.")
    p.add_argument("-o", "--output", metavar="PATH", help="Write the predictions to a JSON file.")
    _add_figure_options(p)
    _add_unit_selection(p)
    p.set_defaults(func=cmd_celltype)

    p = commands.add_parser("figure", help="Render a single analysis figure to an image.")
    p.add_argument("ks_folder", help="Kilosort output folder.")
    p.add_argument("--kind", choices=["basics", "raw", "waveform-acg"], default="basics", help="Figure type.")
    p.add_argument("-o", "--output", metavar="PATH", help="Output image path.")
    p.add_argument("--t0", type=float, default=0.0, help="Window start in seconds.")
    p.add_argument("--duration", type=float, default=1.0, help="Window length in seconds.")
    p.add_argument("--channels", type=int, default=32, help="Channels to draw for the raw explorer.")
    p.add_argument("--highpass", type=float, default=300.0, help="Raw explorer high-pass corner (Hz).")
    p.add_argument("--lowpass", type=float, default=0.0, help="Raw explorer low-pass corner (Hz, 0 = off).")
    p.add_argument("--acg-bin-ms", type=float, default=1.0, help="ACG bin width in ms.")
    p.add_argument("--acg-win-ms", type=float, default=100.0, help="ACG half-window in ms.")
    p.add_argument("--isi-max-ms", type=float, default=200.0, help="ISI histogram range in ms.")
    p.add_argument("--dpi", type=int, default=150, help="Output resolution (default: 150).")
    p.add_argument("--dark", action="store_true", help="Render on a dark background.")
    _add_unit_selection(p, default_limit=8)
    p.set_defaults(func=cmd_figure)

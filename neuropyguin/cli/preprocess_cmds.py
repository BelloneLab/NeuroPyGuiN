"""``neuropyguin preprocess ...`` - everything the Preprocessing tab can do.

Discovery and validation of SpikeGLX AP binaries, the full
CatGT -> Kilosort -> postprocessing -> TPrime -> py_bombcell pipeline, multi
session concatenation for joint sorting, splitting a joint sort back into per
session folders, and the CatGT/TPrime command-string builders.

The pipeline itself is not reimplemented here. ``preprocess run`` instantiates
the very same :class:`~neuropyguin.workers.EcephysPipelineWorker` the GUI uses
and executes it synchronously on the calling thread, so a CLI run and a GUI run
take byte-identical code paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ._common import (
    CommandError,
    Reporter,
    ensure_qt_core_app,
    format_table,
    require_dir,
    resolve_ks_folder,
    write_json_file,
)
from .pipeline_config import (
    PIPELINE_STEPS,
    STEP_LABELS,
    build_pipeline_config,
    config_as_dict,
    enabled_steps,
    validate_config_for_run,
)


# ---------------------------------------------------------------------------
# Discovery and validation
# ---------------------------------------------------------------------------


def _discover_bins(paths: List[str], reporter: Reporter) -> List[str]:
    """Expand files and folders into a sorted list of SpikeGLX AP binaries."""
    from ..preprocessing import discover_bin_files

    expanded = [str(Path(p).expanduser()) for p in paths]
    missing = [p for p in expanded if not Path(p).exists()]
    for path in missing:
        reporter.warn(f"Path does not exist: {path}")
    found = discover_bin_files([p for p in expanded if Path(p).exists()])
    if not found:
        raise CommandError(
            "No SpikeGLX AP binaries (*.imecN.ap.bin) found in the given paths."
        )
    return found


def cmd_discover(args, reporter: Reporter) -> int:
    """List the AP binaries under the given paths with their parsed run identity."""
    from ..preprocessing import (
        is_catgt_processed_bin,
        is_concatenated_run_bin,
        parse_spikeglx_bin_name,
        validate_spikeglx_ap_bin,
    )

    bins = _discover_bins(args.paths, reporter)
    rows: List[Dict[str, Any]] = []
    for bin_file in bins:
        parsed = parse_spikeglx_bin_name(bin_file)
        ok, reason = validate_spikeglx_ap_bin(bin_file)
        if args.valid_only and not ok:
            continue
        rows.append(
            {
                "run_name": parsed["run_name"],
                "gate": parsed["gate_string"],
                "trigger": parsed["trigger_string"],
                "probe": parsed["probe_string"],
                "catgt_processed": is_catgt_processed_bin(bin_file),
                "concatenated": is_concatenated_run_bin(bin_file),
                "valid": ok,
                "reason": reason,
                "bin_file": bin_file,
            }
        )

    payload = {"count": len(rows), "recordings": rows}
    reporter.emit(
        payload,
        text=format_table(rows, ["run_name", "gate", "trigger", "probe", "valid", "bin_file"]),
    )
    return 0


def cmd_validate(args, reporter: Reporter) -> int:
    """Check that each AP binary has a readable meta and real AP channels."""
    from ..preprocessing import validate_spikeglx_ap_bin

    bins = _discover_bins(args.paths, reporter)
    rows = []
    for bin_file in bins:
        ok, reason = validate_spikeglx_ap_bin(bin_file)
        rows.append({"bin_file": bin_file, "valid": ok, "reason": reason or "ok"})

    n_bad = sum(1 for row in rows if not row["valid"])
    reporter.emit(
        {"count": len(rows), "invalid": n_bad, "recordings": rows},
        text=format_table(rows, ["valid", "reason", "bin_file"]),
    )
    return 1 if n_bad else 0


def cmd_runs(args, reporter: Reporter) -> int:
    """List completed Kilosort sorts found anywhere under a root folder."""
    from ..preprocessing import discover_completed_runs

    root = require_dir(args.root, "root folder")
    entries = discover_completed_runs(root)
    reporter.emit(
        {"count": len(entries), "runs": entries},
        text=format_table(entries, ["run_name", "finished_at", "ks_folder"]),
    )
    return 0


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _build_jobs(bins: List[str], cfg, reporter: Reporter) -> List[Dict[str, str]]:
    """Turn validated AP binaries into the job dicts the worker expects.

    Gate/trigger/probe come from each file name (exactly as the queue does), and
    the configured values act as the fallback for names that do not follow the
    SpikeGLX pattern.
    """
    from ..preprocessing import parse_spikeglx_bin_name, validate_spikeglx_ap_bin

    jobs: List[Dict[str, str]] = []
    for bin_file in bins:
        ok, reason = validate_spikeglx_ap_bin(bin_file)
        if not ok:
            reporter.warn(f"Skipping {bin_file}: {reason}")
            continue
        parsed = parse_spikeglx_bin_name(bin_file)
        jobs.append(
            {
                "name": parsed["run_name"],
                "bin_file": bin_file,
                "workdir": str(Path(bin_file).parent),
                "gate_string": parsed["gate_string"] or cfg.gate_string,
                "trigger_string": parsed["trigger_string"] or cfg.trigger_string,
                "probe_string": parsed["probe_string"] or cfg.probe_string,
            }
        )
    if not jobs:
        raise CommandError("No processable recordings remain after validation.")
    return jobs


def _plan_rows(jobs: List[Dict[str, str]], cfg) -> List[Dict[str, Any]]:
    """Predict where each job will write, without touching the filesystem."""
    from ..preprocessing import (
        default_pipeline_ks_output_dir,
        default_pipeline_raw_output_layout,
        is_catgt_processed_bin,
    )

    ks_map = {"2.0": "ks2", "2.5": "ks25", "3.0": "ks3", "4": "ks4"}
    ks_tag = ks_map.get(cfg.ks_ver, "ks4")
    rows: List[Dict[str, Any]] = []
    for job in jobs:
        extracted_root, ks_folder = default_pipeline_raw_output_layout(
            job["bin_file"],
            cfg.output_root,
            ks_tag,
            job["probe_string"],
            run_name=job["name"],
            mirror_raw_hierarchy=cfg.mirror_raw_hierarchy_output,
            layout=cfg.output_layout,
        )
        if is_catgt_processed_bin(job["bin_file"]):
            # A CatGT output stays where it is; only the sorter folder moves.
            ks_folder = default_pipeline_ks_output_dir(
                job["bin_file"],
                ks_tag,
                job["probe_string"],
                output_root=cfg.output_root,
                run_name=job["name"],
                mirror_raw_hierarchy=cfg.mirror_raw_hierarchy_output,
                layout=cfg.output_layout,
            )
        rows.append(
            {
                "run_name": job["name"],
                "bin_file": job["bin_file"],
                "extracted_data_root": str(extracted_root),
                "ks_folder": str(ks_folder),
            }
        )
    return rows


def cmd_plan(args, reporter: Reporter) -> int:
    """Show the resolved configuration and the output paths without running."""
    cfg = build_pipeline_config(args, reporter)
    bins = _discover_bins(args.paths, reporter)
    jobs = _build_jobs(bins, cfg, reporter)
    rows = _plan_rows(jobs, cfg)
    problems = validate_config_for_run(cfg)

    payload = {
        "steps": enabled_steps(cfg),
        "output_layout": cfg.output_layout,
        "output_root": cfg.output_root,
        "problems": problems,
        "jobs": rows,
        "config": config_as_dict(cfg),
    }
    text = [
        f"Steps        : {', '.join(enabled_steps(cfg)) or '(none)'}",
        f"Kilosort     : {cfg.ks_ver}",
        f"Output root  : {cfg.output_root}",
        f"Output layout: {cfg.output_layout}",
        "",
        *format_table(rows, ["run_name", "extracted_data_root", "ks_folder"]),
    ]
    if problems:
        text += ["", "Problems:"] + [f"  - {p}" for p in problems]
    reporter.emit(payload, text=text)
    return 1 if problems else 0


def cmd_run(args, reporter: Reporter) -> int:
    """Run the ecephys pipeline over one or more recordings, synchronously.

    Each recording is handed to a real :class:`EcephysPipelineWorker`, whose Qt
    signals are wired to the reporter so the CLI streams the same log the GUI
    shows. Per-job results are collected and summarised at the end; the exit
    status is non-zero when any job failed.
    """
    from ..workers import EcephysPipelineWorker

    ensure_qt_core_app()
    cfg = build_pipeline_config(args, reporter)
    bins = _discover_bins(args.paths, reporter)
    jobs = _build_jobs(bins, cfg, reporter)

    problems = validate_config_for_run(cfg)
    if problems and not args.force:
        raise CommandError(
            "Configuration cannot run:\n"
            + "\n".join(f"  - {p}" for p in problems)
            + "\nPass --force to attempt it anyway."
        )
    for problem in problems:
        reporter.warn(problem)

    reporter.log(f"Steps: {', '.join(enabled_steps(cfg)) or '(none)'}")
    reporter.log(f"Output root: {cfg.output_root} (layout: {cfg.output_layout})")
    reporter.log(f"Queued {len(jobs)} recording(s).")

    if args.save_config:
        saved = write_json_file(args.save_config, config_as_dict(cfg))
        reporter.log(f"Saved resolved configuration: {saved}")

    results: List[Dict[str, Any]] = []
    for index, job in enumerate(jobs, start=1):
        reporter.log("")
        reporter.log(f"=== [{index}/{len(jobs)}] {job['name']} ===")
        reporter.log(f"    {job['bin_file']}")

        outcome: Dict[str, Any] = {"job": job["name"], "ok": False}
        errors: List[str] = []

        worker = EcephysPipelineWorker(job, cfg)
        worker.signals.log.connect(reporter.log)
        worker.signals.error.connect(lambda message: errors.append(str(message)))
        worker.signals.error.connect(lambda message: reporter.log(str(message)))
        worker.signals.stepStarted.connect(
            lambda _key, label: reporter.log(f"--> {label}")
        )
        worker.signals.stepFinished.connect(
            lambda key, ok: reporter.log(f"<-- {key}: {'ok' if ok else 'FAILED'}")
        )
        worker.signals.finished.connect(outcome.update)

        # QRunnable.run() is an ordinary method: calling it here executes the
        # pipeline on this thread instead of a pool thread, which is exactly what
        # a blocking CLI wants.
        worker.run()

        entry = dict(outcome)
        entry["bin_file"] = job["bin_file"]
        if errors:
            entry["errors"] = errors
        results.append(entry)

        if not entry.get("ok") and args.stop_on_error:
            reporter.warn("Stopping after the first failure (--stop-on-error).")
            break

    n_ok = sum(1 for entry in results if entry.get("ok"))
    n_failed = len(results) - n_ok
    payload = {
        "total": len(results),
        "succeeded": n_ok,
        "failed": n_failed,
        "results": results,
        "config": config_as_dict(cfg),
    }
    summary_rows = [
        {
            "run_name": entry.get("job"),
            "status": "ok" if entry.get("ok") else "failed",
            "ks_folder": entry.get("ks_folder", ""),
            "warnings": len(entry.get("warnings") or []),
        }
        for entry in results
    ]
    reporter.emit(
        payload,
        text=["", f"Finished: {n_ok} ok, {n_failed} failed", ""]
        + format_table(summary_rows, ["status", "run_name", "warnings", "ks_folder"]),
    )
    return 1 if n_failed else 0


# ---------------------------------------------------------------------------
# Concatenation and splitting
# ---------------------------------------------------------------------------


def cmd_concat(args, reporter: Reporter) -> int:
    """Fuse several AP recordings into one binary for a joint spike sort.

    Writes the combined ``.bin``, a merged ``.meta``, the per-session split-info
    map used later by ``preprocess split``, and a human readable manifest.
    """
    from ..preprocessing import (
        OUTPUT_LAYOUT_MIRROR,
        build_concat_run_name,
        concatenate_ap_session,
        default_concat_run_layout,
        find_meta_for_bin,
        mirrored_concat_base_dir,
        normalize_output_layout,
        parse_spikeglx_bin_name,
        validate_concat_inputs,
    )

    bins = _discover_bins(args.paths, reporter)
    if len(bins) < 2:
        raise CommandError("Concatenation needs at least two AP recordings.")

    metas = [str(find_meta_for_bin(b)) for b in bins]
    missing = [b for b, m in zip(bins, metas) if not Path(m).exists()]
    if missing:
        raise CommandError("No .meta file found next to:\n" + "\n".join(f"  {m}" for m in missing))

    ok, reason, _info = validate_concat_inputs(metas)
    if not ok:
        raise CommandError(f"Recordings cannot be concatenated: {reason}")

    run_names = [str(parse_spikeglx_bin_name(b).get("run_name") or Path(b).stem) for b in bins]
    combined_name = args.run_name or build_concat_run_name(run_names)

    if args.target:
        target_bin = Path(args.target).expanduser()
        layout_paths = {"bin": target_bin}
    else:
        if not args.output_root:
            raise CommandError("Pass --output-root (or --target) to say where the fused run goes.")
        layout_mode = normalize_output_layout(args.output_layout or "mirror")
        base = mirrored_concat_base_dir(
            bins[0], args.output_root, combined_name, layout=layout_mode,
        )
        # Only the mirrored layout adds the extra per-session 'spike_sorting' level.
        base_dir = base / "spike_sorting" if layout_mode == OUTPUT_LAYOUT_MIRROR else base
        probe = str(parse_spikeglx_bin_name(bins[0]).get("probe_string") or "0")
        layout_paths = default_concat_run_layout(base_dir, combined_name, probe)
        target_bin = Path(layout_paths["bin"])

    reporter.log(f"Combined run name: {combined_name}")
    reporter.log(f"Target binary    : {target_bin}")
    for name, path in zip(run_names, bins):
        reporter.log(f"  + {name}: {path}")

    if args.dry_run:
        reporter.emit(
            {
                "dry_run": True,
                "run_name": combined_name,
                "target_bin": str(target_bin),
                "sources": bins,
            }
        )
        return 0

    last_percent = {"value": -1}

    def progress(percent: int) -> None:
        """Log progress only when the whole-percent value actually changes."""
        if percent != last_percent["value"]:
            last_percent["value"] = percent
            reporter.log(f"  concatenating... {percent}%")

    result = concatenate_ap_session(
        bins,
        metas,
        target_bin,
        svd_clean=not args.no_svd_clean,
        n_svd_components=args.svd_components,
        batch_seconds=args.batch_seconds,
        progress_cb=progress,
        log_cb=reporter.log,
    )
    result["run_name"] = combined_name
    result["sources"] = bins
    reporter.emit(result)
    return 0


def cmd_split(args, reporter: Reporter) -> int:
    """Split a joint (concatenated) sort back into per-session phy folders."""
    from ..preprocessing import split_concatenated_sort

    ks_folder = resolve_ks_folder(args.ks_folder)
    manifest = split_concatenated_sort(
        ks_folder,
        output_dir=args.output_dir,
        splitinfo_path=args.splitinfo,
        copy_events=not args.no_events,
        event_search_roots=[args.event_root] if args.event_root else None,
        log_cb=reporter.log,
    )
    sessions = manifest.get("sessions") or []
    reporter.emit(
        manifest,
        text=[
            f"Split {manifest.get('n_sessions', 0)} session(s) from {ks_folder}",
            f"Output root: {manifest.get('output_root')}",
            "",
            *format_table(sessions, ["index", "run_name", "n_spikes", "n_clusters", "output_dir"]),
        ],
    )
    return 0


# ---------------------------------------------------------------------------
# Command-string builders
# ---------------------------------------------------------------------------


def cmd_build_catgt(args, reporter: Reporter) -> int:
    """Build a CatGT flag string from readable options (the GUI's Build dialog)."""
    from ..string_builders import CatGTCommandSpec, build_catgt_command_string, parse_catgt_command_string

    spec = parse_catgt_command_string(args.base or "") if args.base else CatGTCommandSpec()
    if args.probe_folders is not None:
        spec.use_probe_folders = args.probe_folders
    if args.out_probe_folders is not None:
        spec.use_output_probe_folders = args.out_probe_folders
    if args.allow_missing_probes is not None:
        spec.allow_missing_probes = args.allow_missing_probes
    if args.allow_missing_trials is not None:
        spec.allow_missing_trials = args.allow_missing_trials
    if args.no_auto_sync is not None:
        spec.disable_auto_sync = args.no_auto_sync
    if args.ap_filter is not None:
        spec.use_ap_filter = args.ap_filter
    if args.filter_type:
        spec.ap_filter_type = args.filter_type
    if args.filter_order is not None:
        spec.ap_filter_order = args.filter_order
    if args.highpass is not None:
        spec.ap_filter_highpass_hz = args.highpass
    if args.lowpass is not None:
        spec.ap_filter_lowpass_hz = args.lowpass
    if args.gfix is not None:
        spec.use_gfix = args.gfix
    if args.gfix_amp is not None:
        spec.gfix_amp_mv = args.gfix_amp
    if args.gfix_slope is not None:
        spec.gfix_slope_mv_per_sample = args.gfix_slope
    if args.gfix_noise is not None:
        spec.gfix_noise_mv = args.gfix_noise
    if args.extra:
        spec.extra_flags = args.extra

    command = build_catgt_command_string(spec)
    if args.save:
        _persist_setting("preproc/catgt_cmd_string", command, reporter)
    reporter.emit({"catgt_cmd_string": command}, text=[command])
    return 0


def cmd_build_tprime(args, reporter: Reporter) -> int:
    """Build the TPrime reference stream and event-extractor strings.

    ``--xa 0:1.0:0.0`` style specifications describe one extractor each; the
    ``--ni-analog-preset`` shortcut reproduces the GUI preset that emits rising
    (and optionally falling) NI analog extractors for a channel range.
    """
    from ..string_builders import (
        TPrimeExtractorSpec,
        build_tostream_sync_params,
        build_tprime_extractor_string,
        parse_channel_spec,
        parse_tprime_extractor_string,
    )

    specs: List[TPrimeExtractorSpec] = []
    extras = ""
    if args.base:
        specs, extras = parse_tprime_extractor_string(args.base)

    def add_spec(mode: str, raw: str) -> None:
        """Parse ``word[:value_a[:value_b[:debounce_ms[:label]]]]`` into a spec."""
        parts = str(raw).split(":")
        try:
            word = int(parts[0])
        except (IndexError, ValueError) as exc:
            raise CommandError(f"--{mode} expects word[:a[:b[:ms[:label]]]], got {raw!r}") from exc
        value_a = float(parts[1]) if len(parts) > 1 and parts[1] else 0.0
        value_b = float(parts[2]) if len(parts) > 2 and parts[2] else 0.0
        debounce = float(parts[3]) if len(parts) > 3 and parts[3] else 0.0
        label = parts[4] if len(parts) > 4 else ""
        specs.append(
            TPrimeExtractorSpec(
                mode=mode,
                stream_kind=args.stream_kind,
                stream_index=args.stream_index,
                word=word,
                value_a=value_a,
                value_b=value_b,
                debounce_ms=debounce,
                label=label,
            )
        )

    for mode, values in (
        ("xd", args.xd or []),
        ("xid", args.xid or []),
        ("xa", args.xa or []),
        ("xia", args.xia or []),
    ):
        for value in values:
            add_spec(mode, value)

    if args.ni_analog_preset:
        channels = parse_channel_spec(args.ni_analog_preset)
        if not channels:
            raise CommandError("--ni-analog-preset needs at least one channel, e.g. '0-2,4'.")
        for channel in channels:
            common = dict(
                stream_kind="ni",
                stream_index=0,
                word=int(channel),
                value_a=args.preset_th1,
                value_b=args.preset_th2,
                debounce_ms=args.preset_pulse_ms,
            )
            specs.append(TPrimeExtractorSpec(mode="xa", **common))
            if not args.preset_no_falling:
                specs.append(TPrimeExtractorSpec(mode="xia", **common))

    extractors = build_tprime_extractor_string(specs, args.extra or extras)
    tostream = build_tostream_sync_params(args.tostream_kind, args.tostream_index)

    if args.save:
        _persist_setting("preproc/ni_extract_string", extractors, reporter)
        _persist_setting("preproc/tostream_sync_params", tostream, reporter)

    reporter.emit(
        {"tostream_sync_params": tostream, "ni_extract_string": extractors},
        text=[f"toStream      : {tostream}", f"Extractors    : {extractors}"],
    )
    return 0


def _persist_setting(key: str, value: str, reporter: Reporter) -> None:
    """Write one value into the shared settings store used by the GUI."""
    from ._common import open_settings

    settings = open_settings()
    settings.setValue(key, value)
    settings.sync()
    reporter.log(f"Saved to settings: {key}")


def cmd_steps(args, reporter: Reporter) -> int:
    """List the pipeline stage names accepted by ``--steps``."""
    rows = [{"step": step, "description": STEP_LABELS[step]} for step in PIPELINE_STEPS]
    reporter.emit({"steps": rows}, text=format_table(rows, ["step", "description"]))
    return 0


def cmd_show_config(args, reporter: Reporter) -> int:
    """Print the pipeline configuration that a run would use right now."""
    cfg = build_pipeline_config(args, reporter)
    payload = config_as_dict(cfg)
    payload["enabled_steps"] = enabled_steps(cfg)
    payload["problems"] = validate_config_for_run(cfg)
    if args.save_config:
        saved = write_json_file(args.save_config, config_as_dict(cfg))
        reporter.log(f"Wrote {saved}")
    reporter.emit(payload)
    return 0


# ---------------------------------------------------------------------------
# Parser wiring
# ---------------------------------------------------------------------------


def register(subparsers) -> None:
    """Attach the ``preprocess`` command group to the top level parser."""
    from .pipeline_config import add_pipeline_arguments

    group = subparsers.add_parser(
        "preprocess",
        help="Discover recordings and run the CatGT / Kilosort pipeline.",
        description="Preprocessing tab equivalents: discovery, the spike-sorting pipeline, "
        "concatenation, joint-sort splitting, and the CatGT/TPrime string builders.",
    )
    commands = group.add_subparsers(dest="command", required=True)

    p = commands.add_parser("discover", help="List SpikeGLX AP binaries under the given paths.")
    p.add_argument("paths", nargs="+", help="Files or folders to search recursively.")
    p.add_argument("--valid-only", action="store_true", help="Hide recordings that fail validation.")
    p.set_defaults(func=cmd_discover)

    p = commands.add_parser("validate", help="Check AP binaries for a usable meta and AP channels.")
    p.add_argument("paths", nargs="+", help="Files or folders to search recursively.")
    p.set_defaults(func=cmd_validate)

    p = commands.add_parser("runs", help="List completed Kilosort sorts under a folder.")
    p.add_argument("root", help="Folder to scan recursively for params.py.")
    p.set_defaults(func=cmd_runs)

    p = commands.add_parser("steps", help="List the pipeline stages that --steps accepts.")
    p.set_defaults(func=cmd_steps)

    p = commands.add_parser(
        "plan",
        help="Resolve the configuration and show where each run would be written.",
    )
    p.add_argument("paths", nargs="+", help="Files or folders holding the recordings.")
    add_pipeline_arguments(p)
    p.set_defaults(func=cmd_plan)

    p = commands.add_parser(
        "run",
        help="Run the full ecephys pipeline over one or more recordings.",
        description="Runs CatGT, Kilosort, postprocessing, TPrime and py_bombcell exactly as the "
        "Preprocessing tab does, streaming the same log to the terminal.",
    )
    p.add_argument("paths", nargs="+", help="Files or folders holding the recordings.")
    p.add_argument("--stop-on-error", action="store_true", help="Abort the batch after the first failure.")
    p.add_argument("--force", action="store_true", help="Run even when the configuration looks invalid.")
    p.add_argument("--save-config", metavar="PATH", help="Write the resolved configuration to a JSON file.")
    add_pipeline_arguments(p)
    p.set_defaults(func=cmd_run)

    p = commands.add_parser(
        "show-config",
        help="Print the resolved pipeline configuration without running anything.",
    )
    p.add_argument("--save-config", metavar="PATH", help="Also write the configuration to a JSON file.")
    add_pipeline_arguments(p)
    p.set_defaults(func=cmd_show_config)

    p = commands.add_parser(
        "concat",
        help="Concatenate several AP recordings into one binary for joint sorting.",
    )
    p.add_argument("paths", nargs="+", help="Two or more recordings (files or folders).")
    p.add_argument("--output-root", help="Root under which the fused run is created.")
    p.add_argument("--output-layout", choices=["mirror", "run_folder", "exact"], help="Placement under the output root.")
    p.add_argument("--target", help="Exact path for the fused .bin (overrides the layout).")
    p.add_argument("--run-name", help="Combined run name (default: derived from the sources).")
    p.add_argument("--no-svd-clean", action="store_true", help="Skip shared-component SVD denoising.")
    p.add_argument("--svd-components", type=int, default=5, help="Number of SVD components to remove (default: 5).")
    p.add_argument("--batch-seconds", type=float, default=0.5, help="Streaming batch size in seconds (default: 0.5).")
    p.add_argument("--dry-run", action="store_true", help="Show the plan without writing anything.")
    p.set_defaults(func=cmd_concat)

    p = commands.add_parser(
        "split",
        help="Split a joint (concatenated) sort into per-session phy folders.",
    )
    p.add_argument("ks_folder", help="Kilosort output folder of the joint sort.")
    p.add_argument("--output-dir", help="Where session folders are written (default: <ks_folder>/sessions).")
    p.add_argument("--splitinfo", help="Explicit *.ap.splitinfo.json path.")
    p.add_argument("--event-root", help="Extra root to search for per-session NI/TPrime event files.")
    p.add_argument("--no-events", action="store_true", help="Do not copy per-session event files.")
    p.set_defaults(func=cmd_split)

    p = commands.add_parser("build-catgt", help="Build a CatGT flag string from readable options.")
    p.add_argument("--base", help="Existing CatGT string to start from.")
    p.add_argument("--probe-folders", action="store_true", default=None, help="Add -prb_fld.")
    p.add_argument("--no-probe-folders", dest="probe_folders", action="store_false", help="Remove -prb_fld.")
    p.add_argument("--out-probe-folders", action="store_true", default=None, help="Add -out_prb_fld.")
    p.add_argument("--no-out-probe-folders", dest="out_probe_folders", action="store_false", help="Remove -out_prb_fld.")
    p.add_argument("--allow-missing-probes", action="store_true", default=None, help="Add -prb_miss_ok.")
    p.add_argument("--allow-missing-trials", action="store_true", default=None, help="Add -t_miss_ok.")
    p.add_argument("--no-auto-sync", action="store_true", default=None, help="Add -no_auto_sync.")
    p.add_argument("--ap-filter", action="store_true", default=None, help="Enable -apfilter.")
    p.add_argument("--no-ap-filter", dest="ap_filter", action="store_false", help="Disable -apfilter.")
    p.add_argument("--filter-type", choices=["butter", "biquad"], help="AP filter type.")
    p.add_argument("--filter-order", type=int, help="AP filter order.")
    p.add_argument("--highpass", type=float, help="AP filter high-pass corner (Hz).")
    p.add_argument("--lowpass", type=float, help="AP filter low-pass corner (Hz).")
    p.add_argument("--gfix", action="store_true", default=None, help="Enable -gfix artifact suppression.")
    p.add_argument("--no-gfix", dest="gfix", action="store_false", help="Disable -gfix.")
    p.add_argument("--gfix-amp", type=float, help="gfix amplitude (mV).")
    p.add_argument("--gfix-slope", type=float, help="gfix slope (mV/sample).")
    p.add_argument("--gfix-noise", type=float, help="gfix noise (mV).")
    p.add_argument("--extra", help="Extra raw CatGT flags to append.")
    p.add_argument("--save", action="store_true", help="Store the result in the shared settings.")
    p.set_defaults(func=cmd_build_catgt)

    p = commands.add_parser("build-tprime", help="Build TPrime reference-stream and extractor strings.")
    p.add_argument("--base", help="Existing extractor string to start from.")
    p.add_argument("--tostream-kind", choices=["imec", "ni", "obx"], default="imec", help="Reference stream type.")
    p.add_argument("--tostream-index", type=int, default=0, help="Reference stream index.")
    p.add_argument("--stream-kind", choices=["ni", "obx", "imec"], default="ni", help="Stream for --xd/--xa rows.")
    p.add_argument("--stream-index", type=int, default=0, help="Stream index for --xd/--xa rows.")
    p.add_argument("--xd", action="append", metavar="WORD[:BIT[::MS[:LABEL]]]", help="Digital rising extractor. Repeatable.")
    p.add_argument("--xid", action="append", metavar="WORD[:BIT[::MS[:LABEL]]]", help="Digital falling extractor. Repeatable.")
    p.add_argument("--xa", action="append", metavar="WORD[:TH1[:TH2[:MS[:LABEL]]]]", help="Analog rising extractor. Repeatable.")
    p.add_argument("--xia", action="append", metavar="WORD[:TH1[:TH2[:MS[:LABEL]]]]", help="Analog falling extractor. Repeatable.")
    p.add_argument("--ni-analog-preset", metavar="CHANNELS", help="NI analog preset for a channel spec such as '0-2,4'.")
    p.add_argument("--preset-th1", type=float, default=1.0, help="Preset threshold 1 (V).")
    p.add_argument("--preset-th2", type=float, default=0.0, help="Preset threshold 2 (V).")
    p.add_argument("--preset-pulse-ms", type=float, default=0.0, help="Preset pulse duration (ms).")
    p.add_argument("--preset-no-falling", action="store_true", help="Preset emits rising edges only.")
    p.add_argument("--extra", help="Extra raw extractor flags to append.")
    p.add_argument("--save", action="store_true", help="Store the result in the shared settings.")
    p.set_defaults(func=cmd_build_tprime)

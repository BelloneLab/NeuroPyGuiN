"""``neuropyguin curate ...`` - everything the Curation tab can do.

Launching phy on a sorted folder (with the bundled plugins installed and the
cluster groups synced first), running py_bombcell quality metrics over one or
many folders, reading back saved results, the lightweight threshold-based
labeller, and opening the BombCell review GUI.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from ._common import (
    CommandError,
    Reporter,
    format_table,
    load_json_file,
    open_in_file_manager,
    resolve_ks_folder,
    write_json_file,
)


def _resolve_folders(paths: List[str], reporter: Reporter) -> List[Path]:
    """Resolve every input path to a Kilosort output folder, skipping bad ones."""
    folders: List[Path] = []
    for raw in paths:
        try:
            folders.append(resolve_ks_folder(raw))
        except CommandError as exc:
            reporter.warn(str(exc))
    if not folders:
        raise CommandError("None of the given paths resolved to a Kilosort output folder.")
    return folders


# ---------------------------------------------------------------------------
# phy
# ---------------------------------------------------------------------------


def cmd_phy(args, reporter: Reporter) -> int:
    """Launch phy's template GUI on a curated folder.

    Reproduces the Curation tab's launch sequence: install the NeuroPyGuiN phy
    plugins (short-ISI split and the gamepad curation plugin), conservatively
    sync ``cluster_group.tsv`` from the best available label source, then start
    ``phy template-gui`` with the environment phy needs.
    """
    import subprocess

    from ..bombcell_core import sync_phy_cluster_group
    from ..phy_integration import ensure_phy_short_isi_plugin
    from ..phy_launch import phy_child_environment, resolve_phy_executable

    folder = resolve_ks_folder(args.ks_folder)
    params = folder / "params.py"
    if not params.exists():
        raise CommandError(f"Missing params.py in {folder}")

    if not args.no_plugin:
        try:
            status = ensure_phy_short_isi_plugin()
            if status.get("plugin_updated") or status.get("config_updated"):
                reporter.log("Phy plugin ready: added the 'Split short ISI' context-menu action.")
            if status.get("gamepad_plugin_updated"):
                reporter.log("Phy gamepad plugin ready: controller curation enabled.")
        except Exception as exc:  # noqa: BLE001 - plugin problems must not block phy
            reporter.warn(f"Could not install the NeuroPyGuiN phy plugin: {exc}")

    if not args.no_sync:
        sync = sync_phy_cluster_group(folder, force=False)
        if sync.get("updated"):
            reporter.log(
                f"Updated cluster_group.tsv from {sync.get('source', 'labels')} "
                f"({sync.get('n_units', 0)} units)."
            )

    program = resolve_phy_executable(os.environ)
    command = [program, "template-gui", str(params)]
    child_env = phy_child_environment(os.environ, program)
    reporter.log("Launching: " + " ".join(command))

    if args.print_command:
        reporter.emit({"program": program, "args": command[1:], "cwd": str(folder)})
        return 0

    try:
        process = subprocess.Popen(command, cwd=str(folder), env=child_env)
    except OSError as exc:
        raise CommandError(f"Failed to start phy ({program}): {exc}") from exc

    if args.wait:
        code = process.wait()
        reporter.emit({"pid": process.pid, "exit_code": code, "ks_folder": str(folder)})
        return int(code)

    reporter.emit(
        {"pid": process.pid, "ks_folder": str(folder), "detached": True},
        text=[f"phy started (pid {process.pid}) on {folder}"],
    )
    return 0


# ---------------------------------------------------------------------------
# py_bombcell
# ---------------------------------------------------------------------------


def _bombcell_settings(args) -> Dict[str, Any] | None:
    """Load py_bombcell settings from ``--settings`` and/or ``--set name=value``."""
    from ..pybombcell_integration import normalize_pybombcell_settings, pybombcell_setting_keys

    if not args.settings and not args.set:
        return None

    import json

    values: Dict[str, Any] = {}
    if args.settings:
        values.update(load_json_file(args.settings))
    for item in args.set or []:
        if "=" not in item:
            raise CommandError(f"--set expects name=value, got {item!r}")
        name, _, raw = item.partition("=")
        try:
            values[name.strip()] = json.loads(raw)
        except ValueError:
            values[name.strip()] = raw

    known = set(pybombcell_setting_keys())
    unknown = sorted(set(values) - known)
    if unknown:
        raise CommandError(
            "Unknown py_bombcell settings: "
            + ", ".join(unknown)
            + ". Run 'neuropyguin curate bombcell-settings' to list valid keys."
        )
    return normalize_pybombcell_settings(values)


def cmd_bombcell(args, reporter: Reporter) -> int:
    """Run (or reuse) py_bombcell quality metrics over one or more folders."""
    from ..pybombcell_integration import run_pybombcell_on_folders

    folders = _resolve_folders(args.ks_folders, reporter)
    settings = _bombcell_settings(args)

    reporter.log(f"Running py_bombcell on {len(folders)} folder(s).")
    if args.extract_raw:
        reporter.log("Raw waveform extraction is on; metrics will be recomputed.")

    payload = run_pybombcell_on_folders(
        [str(f) for f in folders],
        save_plots=not args.no_plots,
        force_recompute=args.force,
        settings=settings,
        extract_raw=args.extract_raw,
    )

    results = payload.get("results") or []
    rows = [
        {
            "status": "ok" if not entry.get("error") else "failed",
            "folder": entry.get("ks_folder", ""),
            "units": entry.get("n_units", ""),
            "good": (entry.get("counts") or {}).get("good", ""),
            "detail": entry.get("error", entry.get("mode", "")),
        }
        for entry in results
    ]
    reporter.emit(payload, text=format_table(rows, ["status", "units", "good", "folder", "detail"]))

    summary = payload.get("summary") or {}
    return 1 if int(summary.get("failed", 0) or 0) else 0


def cmd_bombcell_settings(args, reporter: Reporter) -> int:
    """Print the default py_bombcell parameters and the keys ``--set`` accepts."""
    from ..pybombcell_integration import pybombcell_default_settings

    defaults = pybombcell_default_settings()
    if args.save:
        reporter.log(f"Wrote {write_json_file(args.save, defaults)}")
    reporter.emit(defaults)
    return 0


def cmd_bombcell_summary(args, reporter: Reporter) -> int:
    """Report the py_bombcell results already saved next to a Kilosort folder."""
    from ..pybombcell_integration import summarize_saved_pybombcell_results

    folder = resolve_ks_folder(args.ks_folder)
    summary = summarize_saved_pybombcell_results(folder)
    if args.brief:
        summary.pop("manifest", None)
    reporter.emit(summary)
    return 0


def cmd_bombcell_labels(args, reporter: Reporter) -> int:
    """Print the saved py_bombcell unit labels as a table or CSV."""
    from ..pybombcell_integration import load_pybombcell_labels

    folder = resolve_ks_folder(args.ks_folder)
    frame = load_pybombcell_labels(folder)
    if frame.empty:
        raise CommandError(f"No py_bombcell labels found in {folder}. Run 'curate bombcell' first.")

    if args.output:
        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_path)
        reporter.log(f"Wrote {out_path}")

    label_column = next((c for c in frame.columns if "label" in str(c).lower()), frame.columns[0])
    counts = frame[label_column].astype(str).value_counts().to_dict()
    rows = [
        {"unit": int(idx), "label": str(row[label_column])}
        for idx, row in frame.head(args.limit).iterrows()
    ]
    reporter.emit(
        {"ks_folder": str(folder), "n_units": int(len(frame)), "counts": counts, "units": rows},
        text=[
            f"{len(frame)} unit(s) in {folder}",
            "counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
            "",
            *format_table(rows, ["unit", "label"]),
        ],
    )
    return 0


def cmd_bombcell_gui(args, reporter: Reporter) -> int:
    """Open the interactive BombCell review GUI for a Kilosort folder."""
    from ..pybombcell_integration import launch_pybombcell_gui

    folder = resolve_ks_folder(args.ks_folder)
    result = launch_pybombcell_gui(folder)
    reporter.emit(result)
    return 0 if not result.get("error") else 1


def cmd_figures(args, reporter: Reporter) -> int:
    """Locate (and optionally open) the py_bombcell figure folder."""
    folder = resolve_ks_folder(args.ks_folder)
    plots_dir = folder / "bombcell" / "plots"
    if not plots_dir.is_dir():
        raise CommandError(f"No py_bombcell plots found at {plots_dir}. Run 'curate bombcell' first.")
    images = sorted(p.name for p in plots_dir.iterdir() if p.is_file())
    if args.open:
        open_in_file_manager(plots_dir)
    reporter.emit(
        {"plots_dir": str(plots_dir), "count": len(images), "files": images},
        text=[str(plots_dir), *[f"  {name}" for name in images]],
    )
    return 0


# ---------------------------------------------------------------------------
# Lightweight threshold labelling (bombcell_core)
# ---------------------------------------------------------------------------


def cmd_label(args, reporter: Reporter) -> int:
    """Label units good / noise / mua / non_soma from an existing metrics.csv.

    This is the fast threshold-based labeller (``bombcell_core``), not the full
    py_bombcell run. It writes ``bombcell_labels.csv`` and force-syncs the phy
    ``cluster_group.tsv`` so the labels show up immediately in phy.
    """
    from ..bombcell_core import run_bombcell_on_folder_with_thresholds

    folder = resolve_ks_folder(args.ks_folder)
    thresholds = load_json_file(args.thresholds) if args.thresholds else None
    result = run_bombcell_on_folder_with_thresholds(str(folder), thresholds=thresholds)
    reporter.emit(
        result,
        text=[
            f"Labelled {result['n_units']} unit(s) -> {result['output']}",
            "counts: " + ", ".join(f"{k}={v}" for k, v in sorted(result["counts"].items())),
        ],
    )
    return 0


def cmd_thresholds(args, reporter: Reporter) -> int:
    """Print the default thresholds used by ``curate label``."""
    from ..bombcell_core import bombcell_get_default_thresholds

    thresholds = bombcell_get_default_thresholds()
    if args.save:
        reporter.log(f"Wrote {write_json_file(args.save, thresholds)}")
    reporter.emit(thresholds)
    return 0


def cmd_sync_phy(args, reporter: Reporter) -> int:
    """Write a phy ``cluster_group.tsv`` from the best available label source."""
    from ..bombcell_core import sync_phy_cluster_group

    folder = resolve_ks_folder(args.ks_folder)
    result = sync_phy_cluster_group(folder, force=args.force)
    reporter.emit(result)
    return 0


def cmd_plugin(args, reporter: Reporter) -> int:
    """Install or refresh the bundled phy plugins without launching phy."""
    from ..phy_integration import ensure_phy_short_isi_plugin

    status = ensure_phy_short_isi_plugin(Path(args.phy_home).expanduser() if args.phy_home else None)
    reporter.emit(status)
    return 0


# ---------------------------------------------------------------------------
# Parser wiring
# ---------------------------------------------------------------------------


def register(subparsers) -> None:
    """Attach the ``curate`` command group to the top level parser."""
    group = subparsers.add_parser(
        "curate",
        help="Phy, py_bombcell and unit labelling for sorted folders.",
        description="Curation tab equivalents: launch phy, run py_bombcell quality metrics, "
        "read saved labels, and apply the threshold-based labeller.",
    )
    commands = group.add_subparsers(dest="command", required=True)

    p = commands.add_parser("phy", help="Launch phy's template GUI on a sorted folder.")
    p.add_argument("ks_folder", help="Kilosort output folder (or a parent of it).")
    p.add_argument("--wait", action="store_true", help="Block until phy exits and forward its status.")
    p.add_argument("--no-plugin", action="store_true", help="Do not install the NeuroPyGuiN phy plugins.")
    p.add_argument("--no-sync", action="store_true", help="Do not refresh cluster_group.tsv first.")
    p.add_argument("--print-command", action="store_true", help="Print the phy command instead of running it.")
    p.set_defaults(func=cmd_phy)

    p = commands.add_parser("bombcell", help="Run py_bombcell quality metrics on one or more folders.")
    p.add_argument("ks_folders", nargs="+", help="One or more Kilosort output folders.")
    p.add_argument("--force", action="store_true", help="Recompute even when cached metrics match.")
    p.add_argument("--extract-raw", action="store_true", help="Extract raw waveforms for SNR (slower).")
    p.add_argument("--no-plots", action="store_true", help="Skip figure generation.")
    p.add_argument("--settings", metavar="PATH", help="JSON file with py_bombcell parameter overrides.")
    p.add_argument("--set", action="append", metavar="NAME=VALUE", help="Single parameter override. Repeatable.")
    p.set_defaults(func=cmd_bombcell)

    p = commands.add_parser("bombcell-settings", help="Show the default py_bombcell parameters.")
    p.add_argument("--save", metavar="PATH", help="Write the defaults to a JSON file.")
    p.set_defaults(func=cmd_bombcell_settings)

    p = commands.add_parser("bombcell-summary", help="Report saved py_bombcell results for a folder.")
    p.add_argument("ks_folder", help="Kilosort output folder.")
    p.add_argument("--brief", action="store_true", help="Omit the full manifest from the output.")
    p.set_defaults(func=cmd_bombcell_summary)

    p = commands.add_parser("bombcell-gui", help="Open the interactive BombCell review GUI.")
    p.add_argument("ks_folder", help="Kilosort output folder.")
    p.set_defaults(func=cmd_bombcell_gui)

    p = commands.add_parser("labels", help="Print the saved py_bombcell unit labels.")
    p.add_argument("ks_folder", help="Kilosort output folder.")
    p.add_argument("--limit", type=int, default=25, help="Rows to show in text mode (default: 25).")
    p.add_argument("--output", metavar="PATH", help="Write the full label table to a CSV file.")
    p.set_defaults(func=cmd_bombcell_labels)

    p = commands.add_parser("figures", help="Locate the py_bombcell figure folder.")
    p.add_argument("ks_folder", help="Kilosort output folder.")
    p.add_argument("--open", action="store_true", help="Also open the folder in the file manager.")
    p.set_defaults(func=cmd_figures)

    p = commands.add_parser("label", help="Label units from metrics.csv using min/max thresholds.")
    p.add_argument("ks_folder", help="Kilosort output folder containing metrics.csv.")
    p.add_argument("--thresholds", metavar="PATH", help="JSON file with threshold overrides.")
    p.set_defaults(func=cmd_label)

    p = commands.add_parser("thresholds", help="Show the default labelling thresholds.")
    p.add_argument("--save", metavar="PATH", help="Write the thresholds to a JSON file.")
    p.set_defaults(func=cmd_thresholds)

    p = commands.add_parser("sync-phy", help="Write cluster_group.tsv from the best label source.")
    p.add_argument("ks_folder", help="Kilosort output folder.")
    p.add_argument("--force", action="store_true", help="Overwrite an existing cluster_group.tsv.")
    p.set_defaults(func=cmd_sync_phy)

    p = commands.add_parser("install-phy-plugin", help="Install or refresh the bundled phy plugins.")
    p.add_argument("--phy-home", help="Override the phy home directory.")
    p.set_defaults(func=cmd_plugin)

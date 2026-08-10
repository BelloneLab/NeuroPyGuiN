"""``neuropyguin doctor|tools|config|gui|version`` - the application-level commands.

These cover the parts of the window that are not one of the four workflow tabs:
the Help > Run Diagnostics self-check, the missing-tool installer, the Settings
dialog and the File > Save/Load Settings and Clear History actions, plus a way
to start the GUI itself.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

from ._common import (
    CommandError,
    PROJECT_ROOT,
    Reporter,
    format_table,
    open_settings,
    require_file,
    write_json_file,
)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def cmd_doctor(args, reporter: Reporter) -> int:
    """Run the dependency self-check and report every result.

    Exits non-zero when a blocking failure is present, so the command works as a
    CI or provisioning gate. ``--install-missing`` chains straight into the tool
    installer for whatever the diagnostics flagged as installable.
    """
    from .. import doctor as doctor_module

    settings = open_settings()

    def read(key: str, default: object) -> object:
        """QSettings-style reader so the checks see the GUI's configured folders."""
        return settings.value(key, default)

    def progress(done: int, total: int, label: str) -> None:
        reporter.log(f"[{done}/{total}] {label}")

    results = doctor_module.run_diagnostics(
        settings_get=read, progress=None if args.quiet_progress else progress
    )

    rows = [
        {
            "status": getattr(result, "status", ""),
            "category": getattr(result, "category", ""),
            "check": getattr(result, "label", ""),
            "detail": getattr(result, "detail", ""),
            "fix": getattr(result, "fix", ""),
        }
        for result in results
    ]
    if args.failed_only:
        rows = [row for row in rows if str(row["status"]).lower() not in {"ok", "pass"}]

    summary = doctor_module.summarize(results)
    headline = doctor_module.headline(results)
    blocking = doctor_module.has_blocking_failures(results)

    if args.output:
        reporter.log(f"Wrote {write_json_file(args.output, {'summary': summary, 'checks': rows})}")

    reporter.emit(
        {"headline": headline, "summary": summary, "blocking_failures": blocking, "checks": rows},
        text=[doctor_module.report_text(results), "", headline],
    )

    if args.install_missing:
        # Every non-OK row that knows how to repair itself contributes its keys.
        installable = [
            key
            for result in results
            if getattr(result, "status", "") != doctor_module.OK
            for key in (getattr(result, "install_keys", None) or [])
        ]
        if installable:
            reporter.log("")
            reporter.log("Installing what the diagnostics flagged...")
            return _install_tools(sorted(set(installable)), reporter, force=False)
        reporter.warn("Nothing the diagnostics reported can be installed automatically.")

    return 1 if blocking else 0


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def _configured_tool_paths() -> Dict[str, str]:
    """Return the tool folders currently configured, defaults filling the gaps."""
    from ..tool_installer import default_tool_paths

    settings = open_settings()
    defaults = default_tool_paths(PROJECT_ROOT)
    keys = {"catgt": "preproc/catgt_path", "tprime": "preproc/tprime_path",
            "cwaves": "preproc/cwaves_path", "kilosort": "preproc/ks4_repo_path"}
    paths: Dict[str, str] = {}
    for tool_key, settings_key in keys.items():
        stored = str(settings.value(settings_key, "") or "").strip()
        paths[tool_key] = stored or defaults.get(tool_key, "")
    return paths


def _persist_tool_paths(installed: Dict[str, str], reporter: Reporter) -> None:
    """Write freshly installed tool folders back into the shared settings store."""
    settings_keys = {
        "catgt": "preproc/catgt_path",
        "tprime": "preproc/tprime_path",
        "cwaves": "preproc/cwaves_path",
        "kilosort": "preproc/ks4_repo_path",
        "iblapps": "histology/iblapps_path",
    }
    settings = open_settings()
    saved: List[str] = []
    for key, path in installed.items():
        settings_key = settings_keys.get(key)
        if settings_key and path:
            settings.setValue(settings_key, str(path))
            saved.append(key)
    if saved:
        settings.sync()
        reporter.log(f"Saved tool paths to settings: {', '.join(sorted(saved))}")


def cmd_tools_list(args, reporter: Reporter) -> int:
    """Show each external tool, where it is configured, and whether it is usable."""
    from ..tool_installer import (
        NATIVE_TOOLS,
        detected_os,
        installed_kilosort_path,
        installed_kilosort_version,
        native_tool_is_installed,
        tool_display_name,
    )

    os_name = detected_os()
    paths = _configured_tool_paths()
    rows: List[Dict[str, Any]] = []
    for tool in NATIVE_TOOLS:
        folder = paths.get(tool.key, "")
        rows.append(
            {
                "tool": tool_display_name(tool.key),
                "key": tool.key,
                "installed": native_tool_is_installed(tool, folder, os_name),
                "path": folder,
            }
        )
    ks_path = installed_kilosort_path()
    rows.append(
        {
            "tool": "Kilosort4",
            "key": "kilosort",
            "installed": ks_path is not None,
            "path": str(ks_path or paths.get("kilosort", "")),
            "version": installed_kilosort_version(),
        }
    )

    missing = [row["key"] for row in rows if not row["installed"]]
    reporter.emit(
        {"platform": os_name, "missing": missing, "tools": rows},
        text=format_table(rows, ["installed", "tool", "key", "path"]),
    )
    return 0


def _install_tools(keys: List[str], reporter: Reporter, *, force: bool) -> int:
    """Run the installer for ``keys`` and persist the resulting paths."""
    from ..tool_installer import install_missing_tools

    def report(message: str) -> None:
        reporter.log(message)

    last = {"label": "", "pct": -1}

    def progress(label: str, fraction: float) -> None:
        """Log a coarse percentage, skipping repeats so downloads stay readable."""
        pct = -1 if fraction is None else int(round(float(fraction) * 100))
        if label == last["label"] and pct == last["pct"]:
            return
        last["label"], last["pct"] = label, pct
        reporter.log(f"  {label}: {pct}%" if pct >= 0 else f"  {label}...")

    try:
        installed = install_missing_tools(
            PROJECT_ROOT,
            _configured_tool_paths(),
            requested=keys or None,
            report=report,
            progress=progress,
            force=force,
        )
    except Exception as exc:  # noqa: BLE001
        raise CommandError(f"Tool installation failed: {exc}") from exc

    _persist_tool_paths(installed, reporter)
    reporter.emit(
        {"installed": installed, "count": len(installed)},
        text=[f"Installed {len(installed)} item(s):"] + [f"  {k}: {v}" for k, v in installed.items()],
    )
    return 0


def cmd_tools_install(args, reporter: Reporter) -> int:
    """Download and install the missing external tools (CatGT, TPrime, C_Waves, ...)."""
    from ..tool_installer import missing_tools

    keys = list(args.tools or [])
    if not keys:
        keys = missing_tools(_configured_tool_paths())
        if not keys and not args.force:
            reporter.emit({"installed": {}, "count": 0}, text=["Nothing is missing."])
            return 0
        reporter.log(f"Missing: {', '.join(keys) or '(none)'}")
    return _install_tools(keys, reporter, force=args.force)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def cmd_config_list(args, reporter: Reporter) -> int:
    """List the stored settings, optionally filtered by key prefix."""
    settings = open_settings()
    rows = []
    for key in sorted(settings.allKeys()):
        if args.prefix and not str(key).startswith(args.prefix):
            continue
        value = settings.value(key)
        text = str(value)
        if len(text) > args.max_width:
            text = text[: args.max_width - 3] + "..."
        rows.append({"key": str(key), "value": text})
    reporter.emit(
        {"count": len(rows), "settings": rows, "location": settings.fileName()},
        text=format_table(rows, ["key", "value"]),
    )
    return 0


def cmd_config_get(args, reporter: Reporter) -> int:
    """Print the value of one setting."""
    settings = open_settings()
    if not settings.contains(args.key):
        raise CommandError(f"Setting not found: {args.key}")
    value = settings.value(args.key)
    reporter.emit({"key": args.key, "value": value}, text=[str(value)])
    return 0


def cmd_config_set(args, reporter: Reporter) -> int:
    """Set one setting. Values are parsed as JSON when possible, else stored as text."""
    import json

    settings = open_settings()
    try:
        value: Any = json.loads(args.value)
    except ValueError:
        value = args.value
    settings.setValue(args.key, value)
    settings.sync()
    reporter.emit({"key": args.key, "value": value}, text=[f"{args.key} = {value}"])
    return 0


def cmd_config_unset(args, reporter: Reporter) -> int:
    """Remove one setting (or a whole prefix group with ``--group``)."""
    settings = open_settings()
    if args.group:
        removed = [k for k in settings.allKeys() if str(k).startswith(args.key)]
        for key in removed:
            settings.remove(key)
    else:
        if not settings.contains(args.key):
            raise CommandError(f"Setting not found: {args.key}")
        removed = [args.key]
        settings.remove(args.key)
    settings.sync()
    reporter.emit({"removed": removed, "count": len(removed)})
    return 0


def cmd_config_export(args, reporter: Reporter) -> int:
    """Write every setting to an INI file (the GUI's Save Settings to File)."""
    settings = open_settings()
    out_path = Path(args.path).expanduser()
    if out_path.suffix.lower() != ".ini":
        out_path = out_path.with_suffix(".ini")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    target = open_settings(out_path)
    target.clear()
    for key in settings.allKeys():
        target.setValue(key, settings.value(key))
    target.sync()
    reporter.emit({"path": str(out_path), "count": len(settings.allKeys())}, text=[f"Wrote {out_path}"])
    return 0


def cmd_config_import(args, reporter: Reporter) -> int:
    """Replace the stored settings from an INI file (the GUI's Load Settings)."""
    in_path = require_file(args.path, "settings file")
    source = open_settings(in_path)
    keys = list(source.allKeys())
    if not keys:
        raise CommandError(f"No settings found in {in_path}")
    if not args.yes:
        raise CommandError(
            f"This replaces all current settings with {len(keys)} value(s) from {in_path}. "
            "Re-run with --yes to confirm."
        )
    target = open_settings()
    target.clear()
    for key in keys:
        target.setValue(key, source.value(key))
    target.sync()
    reporter.emit({"path": str(in_path), "count": len(keys)}, text=[f"Loaded {len(keys)} setting(s)"])
    return 0


def cmd_config_clear_history(args, reporter: Reporter) -> int:
    """Clear folder history, recents, and completed-run history (File menu action)."""
    keys = [
        "paths/last_folder",
        "paths/last_file_dir",
        "recent_files",
        "recent_folders",
        "preproc/completed_runs_history_json",
        "curation/phy_folder",
        "curation/bomb_folder",
        "quality/last_folder",
        "post/last_folder",
    ]
    if not args.yes:
        raise CommandError("This clears saved folders and recents. Re-run with --yes to confirm.")
    settings = open_settings()
    removed = [key for key in keys if settings.contains(key)]
    for key in keys:
        settings.remove(key)
    settings.sync()
    reporter.emit({"removed": removed, "count": len(removed)})
    return 0


def cmd_config_path(args, reporter: Reporter) -> int:
    """Print where the settings are stored on this machine."""
    settings = open_settings()
    reporter.emit({"location": settings.fileName()}, text=[settings.fileName()])
    return 0


# ---------------------------------------------------------------------------
# GUI and version
# ---------------------------------------------------------------------------


def cmd_gui(args, reporter: Reporter) -> int:
    """Start the graphical application (identical to running ``main.py`` bare)."""
    try:
        from ..app import main as app_main
    except Exception as exc:  # noqa: BLE001
        raise CommandError(
            f"Could not load the GUI: {exc}. "
            "This usually means PySide6 is missing from the active Python environment."
        ) from exc
    return int(app_main() or 0)


def cmd_version(args, reporter: Reporter) -> int:
    """Report the application, Python, and Qt versions."""
    payload: Dict[str, Any] = {
        "application": "NeuroPyGuiN",
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "project_root": str(PROJECT_ROOT),
    }
    try:
        from PySide6 import QtCore

        payload["qt"] = QtCore.qVersion()
        payload["pyside6"] = QtCore.__version__
    except Exception:
        payload["qt"] = "(PySide6 not importable)"
    reporter.emit(payload)
    return 0


# ---------------------------------------------------------------------------
# Parser wiring
# ---------------------------------------------------------------------------


def register(subparsers) -> None:
    """Attach the application-level commands to the top level parser."""
    p = subparsers.add_parser(
        "doctor",
        help="Run the dependency self-check (Help > Run Diagnostics).",
        description="Checks every Python package, bundled toolbox, GPU runtime and external "
        "tool the application needs. Exits non-zero on a blocking failure.",
    )
    p.add_argument("--failed-only", action="store_true", help="Only report checks that are not OK.")
    p.add_argument("--quiet-progress", action="store_true", help="Do not print per-check progress.")
    p.add_argument("--install-missing", action="store_true", help="Install what the check flags as installable.")
    p.add_argument("-o", "--output", metavar="PATH", help="Write the full report to a JSON file.")
    p.set_defaults(func=cmd_doctor)

    group = subparsers.add_parser("tools", help="Inspect and install the external tools.")
    commands = group.add_subparsers(dest="command", required=True)

    sub = commands.add_parser("list", help="Show configured tool folders and their status.")
    sub.set_defaults(func=cmd_tools_list)

    sub = commands.add_parser("install", help="Download and install missing external tools.")
    sub.add_argument("tools", nargs="*", help="Tool keys to install (default: everything missing).")
    sub.add_argument("--force", action="store_true", help="Reinstall even when already present.")
    sub.set_defaults(func=cmd_tools_install)

    group = subparsers.add_parser("config", help="Read and write the shared application settings.")
    commands = group.add_subparsers(dest="command", required=True)

    sub = commands.add_parser("list", help="List stored settings.")
    sub.add_argument("--prefix", help="Only show keys starting with this prefix.")
    sub.add_argument("--max-width", type=int, default=80, help="Truncate long values (default: 80).")
    sub.set_defaults(func=cmd_config_list)

    sub = commands.add_parser("get", help="Print one setting.")
    sub.add_argument("key", help="Setting key, e.g. preproc/output_root.")
    sub.set_defaults(func=cmd_config_get)

    sub = commands.add_parser("set", help="Set one setting.")
    sub.add_argument("key", help="Setting key.")
    sub.add_argument("value", help="Value. Parsed as JSON when possible, otherwise stored as text.")
    sub.set_defaults(func=cmd_config_set)

    sub = commands.add_parser("unset", help="Remove one setting or a whole group.")
    sub.add_argument("key", help="Setting key, or a prefix when --group is given.")
    sub.add_argument("--group", action="store_true", help="Treat the key as a prefix and remove every match.")
    sub.set_defaults(func=cmd_config_unset)

    sub = commands.add_parser("export", help="Write all settings to an INI file.")
    sub.add_argument("path", help="Target .ini file.")
    sub.set_defaults(func=cmd_config_export)

    sub = commands.add_parser("import", help="Replace all settings from an INI file.")
    sub.add_argument("path", help="Source .ini file.")
    sub.add_argument("--yes", action="store_true", help="Confirm replacing the current settings.")
    sub.set_defaults(func=cmd_config_import)

    sub = commands.add_parser("clear-history", help="Clear folder history, recents and run history.")
    sub.add_argument("--yes", action="store_true", help="Confirm clearing.")
    sub.set_defaults(func=cmd_config_clear_history)

    sub = commands.add_parser("path", help="Print where the settings are stored.")
    sub.set_defaults(func=cmd_config_path)

    p = subparsers.add_parser("gui", help="Start the graphical application.")
    p.set_defaults(func=cmd_gui)

    p = subparsers.add_parser("version", help="Print the application, Python and Qt versions.")
    p.set_defaults(func=cmd_version)

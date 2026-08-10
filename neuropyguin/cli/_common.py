"""Shared plumbing for the NeuroPyGuiN command line interface.

Everything the individual command modules need but that is not specific to a
single workflow lives here:

* :class:`Reporter`   structured output (human readable text or machine JSON),
* :class:`CommandError` the one exception the dispatcher turns into exit code 2,
* QSettings access, so the CLI inherits whatever the GUI was configured with,
* small path / parsing helpers reused by several command groups.

Nothing heavy is imported at module import time. Qt, numpy, pandas and the
processing engines are pulled in lazily inside the functions that need them, so
``neuropyguin --help`` stays fast and still works when an optional scientific
dependency is missing from the environment.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence


#: Organisation / application names used by the GUI's ``QSettings`` store. The
#: CLI reads and writes the very same store so both front ends stay in sync.
SETTINGS_ORG = "NeuroPyGuiN"
SETTINGS_APP = "NeuroPyGuiN"

#: Repository root (the folder holding ``main.py``), used to locate ``tools/``.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CommandError(RuntimeError):
    """A user-facing failure that should end the command with a clean message.

    Raising this (instead of letting an arbitrary exception escape) prints a
    single ``error: ...`` line rather than a traceback. ``exit_code`` lets a
    command choose a specific status; the default of 2 distinguishes a runtime
    failure from argparse's own usage error (also 2) only by the message text,
    which is intentional: both mean "the command did not do its job".
    """

    def __init__(self, message: str, exit_code: int = 2) -> None:
        super().__init__(message)
        self.exit_code = int(exit_code)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    """Convert ``value`` into something :func:`json.dumps` accepts.

    Handles the types the engines hand back that JSON does not know about:
    ``Path``, numpy scalars and arrays, sets, and anything else that has to fall
    back to ``str``. Containers are converted recursively.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    # numpy is optional at this level, so probe by duck typing rather than import.
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _jsonable(tolist())
        except Exception:
            pass
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except Exception:
            pass
    return str(value)


def _render_text(payload: Any, indent: int = 0) -> List[str]:
    """Render a JSON-able payload as aligned ``key: value`` text lines.

    Scalars print on one line, mappings print one key per line with nested
    values indented, and lists of mappings print as indented blocks separated by
    a ``-`` bullet. This keeps the default (non-JSON) output readable without
    every command having to write its own formatter.
    """
    pad = "  " * indent
    lines: List[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if isinstance(value, (Mapping, list)) and value:
                lines.append(f"{pad}{key}:")
                lines.extend(_render_text(value, indent + 1))
            elif isinstance(value, (Mapping, list)):
                lines.append(f"{pad}{key}: (empty)")
            else:
                lines.append(f"{pad}{key}: {value}")
    elif isinstance(payload, list):
        for entry in payload:
            if isinstance(entry, (Mapping, list)):
                block = _render_text(entry, indent + 1)
                if block:
                    block[0] = f"{pad}- {block[0].lstrip()}"
                    lines.extend(block)
            else:
                lines.append(f"{pad}- {entry}")
    else:
        lines.append(f"{pad}{payload}")
    return lines


class Reporter:
    """Routes progress messages and final results to the right stream.

    In text mode (the default) progress and results both go to stdout. In
    ``--json`` mode progress is diverted to stderr and stdout carries nothing but
    the single JSON document produced by :meth:`emit`, so the command can be
    piped straight into ``jq`` or read by another program.
    """

    def __init__(self, as_json: bool = False, quiet: bool = False) -> None:
        self.as_json = bool(as_json)
        self.quiet = bool(quiet)
        self._emitted = False

    @property
    def _log_stream(self):
        """Stream that progress lines go to (stderr in JSON mode)."""
        return sys.stderr if self.as_json else sys.stdout

    def log(self, message: str) -> None:
        """Print one progress line unless ``--quiet`` was requested."""
        if self.quiet:
            return
        print(str(message), file=self._log_stream, flush=True)

    def warn(self, message: str) -> None:
        """Print a warning. Warnings survive ``--quiet`` and always hit stderr."""
        print(f"warning: {message}", file=sys.stderr, flush=True)

    def emit(self, payload: Any, *, text: Sequence[str] | None = None) -> None:
        """Write the command's final result.

        ``payload`` is the structured result (printed verbatim as JSON with
        ``--json``). ``text`` optionally overrides the human rendering with
        pre-formatted lines when the generic key/value layout would read poorly.
        """
        self._emitted = True
        if self.as_json:
            print(json.dumps(_jsonable(payload), indent=2), flush=True)
            return
        if self.quiet:
            return
        lines = list(text) if text is not None else _render_text(_jsonable(payload))
        for line in lines:
            print(line, flush=True)


# ---------------------------------------------------------------------------
# Qt / settings bridge
# ---------------------------------------------------------------------------


def ensure_qt_core_app():
    """Create a headless ``QCoreApplication`` if the process has none yet.

    The processing workers are ``QRunnable`` objects that report through Qt
    signals. Those work without an event loop when the worker is executed
    in-process, but Qt still wants a living application object for its object
    tree and settings paths. A ``QCoreApplication`` (not ``QApplication``) keeps
    this display-free, so the CLI runs fine over SSH or in a container.
    """
    from PySide6 import QtCore

    app = QtCore.QCoreApplication.instance()
    if app is None:
        # Qt keeps a reference internally; the local name only avoids GC here.
        app = QtCore.QCoreApplication(sys.argv[:1])
        app.setOrganizationName(SETTINGS_ORG)
        app.setApplicationName(SETTINGS_APP)
    return app


def open_settings(path: str | Path | None = None):
    """Open the GUI's settings store, or an INI file when ``path`` is given.

    With no argument this is exactly the store the GUI writes to, which is what
    makes ``neuropyguin preprocess run`` inherit the tool paths, CatGT flags and
    stage selection the user configured in the window.
    """
    from PySide6 import QtCore

    ensure_qt_core_app()
    if path is not None:
        return QtCore.QSettings(str(path), QtCore.QSettings.IniFormat)
    return QtCore.QSettings(SETTINGS_ORG, SETTINGS_APP)


def setting_str(settings, key: str, default: str = "") -> str:
    """Read a string setting, falling back to ``default`` when unset or blank."""
    value = settings.value(key, None)
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def setting_bool(settings, key: str, default: bool) -> bool:
    """Read a boolean setting using Qt's own type coercion."""
    return bool(settings.value(key, default, type=bool))


def setting_float(settings, key: str, default: float) -> float:
    """Read a float setting, tolerating strings and malformed values."""
    try:
        return float(settings.value(key, default))
    except (TypeError, ValueError):
        return float(default)


# ---------------------------------------------------------------------------
# Parsing / path helpers
# ---------------------------------------------------------------------------


def parse_csv_list(raw: str | None) -> List[str]:
    """Split a comma-separated option value into stripped, non-empty tokens."""
    if not raw:
        return []
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def parse_int_list(raw: str | None) -> List[int]:
    """Parse ``"1,3,5-8"`` into ``[1, 3, 5, 6, 7, 8]``, de-duplicated in order.

    Used wherever a command takes a unit or channel selection. Ranges may be
    given in either direction; ``5-8`` and ``8-5`` produce the same list.
    """
    values: List[int] = []
    for part in parse_csv_list(raw):
        if "-" in part[1:]:  # skip index 0 so negative numbers still parse
            left, _, right = part.partition("-")
            try:
                start, stop = int(left), int(right)
            except ValueError as exc:
                raise CommandError(f"Invalid range in selection: {part!r}") from exc
            if stop < start:
                start, stop = stop, start
            candidates = range(start, stop + 1)
        else:
            try:
                candidates = [int(part)]
            except ValueError as exc:
                raise CommandError(f"Invalid integer in selection: {part!r}") from exc
        for value in candidates:
            if value not in values:
                values.append(value)
    return values


def load_json_file(path: str | Path) -> Dict[str, Any]:
    """Read a JSON object from disk, raising :class:`CommandError` on problems."""
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise CommandError(f"JSON file not found: {file_path}")
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CommandError(f"Could not read JSON from {file_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise CommandError(f"Expected a JSON object in {file_path}, got {type(data).__name__}")
    return data


def write_json_file(path: str | Path, payload: Any) -> Path:
    """Write ``payload`` to ``path`` as indented UTF-8 JSON and return the path."""
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8")
    return out_path


def require_dir(path: str | Path, what: str = "folder") -> Path:
    """Return ``path`` as an existing directory or raise :class:`CommandError`."""
    resolved = Path(path).expanduser()
    if not resolved.is_dir():
        raise CommandError(f"{what.capitalize()} not found: {resolved}")
    return resolved


def require_file(path: str | Path, what: str = "file") -> Path:
    """Return ``path`` as an existing file or raise :class:`CommandError`."""
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise CommandError(f"{what.capitalize()} not found: {resolved}")
    return resolved


def resolve_ks_folder(path: str | Path, *, strict: bool = True) -> Path:
    """Resolve a user-supplied path to an actual Kilosort output folder.

    Accepts the folder itself, a parent of it, or a path to a file inside it,
    and reuses the same search the GUI performs (``ks_output_resolver``) so that
    ``postproc info <session>`` finds ``.../imec0_ks4`` without the user having
    to spell it out. With ``strict`` the result must contain a real Kilosort
    output (``spike_times.npy`` and friends).
    """
    from ..ks_output_resolver import has_kilosort_output, resolve_kilosort_output_dir

    candidate = Path(path).expanduser()
    resolved = resolve_kilosort_output_dir(candidate)
    if strict and not has_kilosort_output(resolved):
        raise CommandError(
            f"No Kilosort output found at or below {candidate}. "
            "Expected a folder containing spike_times.npy and params.py."
        )
    return resolved


def open_in_file_manager(path: str | Path) -> None:
    """Reveal ``path`` in the platform file manager (Explorer, Finder, xdg-open)."""
    import subprocess

    target = Path(path).expanduser()
    if sys.platform.startswith("win"):
        os.startfile(str(target))  # type: ignore[attr-defined]  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(target)])
    else:
        subprocess.Popen(["xdg-open", str(target)])


def iter_existing(paths: Iterable[str | Path]) -> List[Path]:
    """Return the subset of ``paths`` that exist on disk, preserving order."""
    return [Path(p).expanduser() for p in paths if Path(p).expanduser().exists()]


def format_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> List[str]:
    """Render ``rows`` as a fixed-width text table with a header underline.

    Only ``columns`` are shown, in the given order. Missing keys render as an
    empty cell so partially populated result dicts do not need padding first.
    """
    if not rows:
        return ["(no rows)"]
    widths = {col: len(col) for col in columns}
    cells: List[List[str]] = []
    for row in rows:
        line = []
        for col in columns:
            text = "" if row.get(col) is None else str(row.get(col))
            widths[col] = max(widths[col], len(text))
            line.append(text)
        cells.append(line)
    header = "  ".join(col.ljust(widths[col]) for col in columns)
    rule = "  ".join("-" * widths[col] for col in columns)
    body = ["  ".join(cell.ljust(widths[col]) for cell, col in zip(line, columns)) for line in cells]
    return [header, rule, *body]


def run_with_reporter(reporter: Reporter, label: str, fn: Callable[[], Any]) -> Any:
    """Run ``fn`` announcing ``label`` first and turning failures into CommandError.

    Keeps the "say what you are about to do, then blame the right step if it
    breaks" pattern in one place instead of repeating try/except in every
    command body.
    """
    reporter.log(label)
    try:
        return fn()
    except CommandError:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced to the user as a clean error
        raise CommandError(f"{label} failed: {exc}") from exc

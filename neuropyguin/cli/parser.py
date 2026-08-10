"""Top level argument parser and dispatcher for the NeuroPyGuiN CLI.

Builds the ``neuropyguin <group> <command>`` tree from the per-group ``register``
functions, then dispatches to the ``func`` each subcommand set on its namespace.
Every command receives ``(args, reporter)`` and returns a process exit code.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Sequence

from ._common import CommandError, Reporter


EPILOG = """\
examples:
  neuropyguin doctor
  neuropyguin preprocess discover D:/rawData/mouse01
  neuropyguin preprocess run D:/rawData/mouse01 --steps catgt,kilosort,quality_metrics
  neuropyguin curate bombcell D:/processed/run1/imec0_ks4 --extract-raw
  neuropyguin curate phy D:/processed/run1/imec0_ks4
  neuropyguin postproc export-units D:/processed/run1/imec0_ks4 --good-only -o units.h5
  neuropyguin postproc psth D:/processed/run1/imec0_ks4 --events reward=events.csv --figure
  neuropyguin histology pipeline D:/histology/mouse01 --extract --ks D:/processed/run1/imec0_ks4

Run 'neuropyguin <group> --help' for the commands in a group, and
'neuropyguin <group> <command> --help' for its options.
Running 'neuropyguin' or 'main.py' with no arguments starts the GUI.
"""


def build_parser() -> argparse.ArgumentParser:
    """Construct the full command tree."""
    parser = argparse.ArgumentParser(
        prog="neuropyguin",
        description="NeuroPyGuiN command line interface: every workflow the GUI offers, "
        "driven from a terminal or a script.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit the result as JSON on stdout (progress goes to stderr).",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output.")
    parser.add_argument(
        "--traceback",
        action="store_true",
        help="Show the full Python traceback instead of a one-line error.",
    )

    subparsers = parser.add_subparsers(dest="group", metavar="<group>")

    # Imported here rather than at module import time so that a broken optional
    # dependency in one group cannot stop the whole CLI from starting.
    from . import curate_cmds, histology_cmds, postproc_cmds, preprocess_cmds, system_cmds

    preprocess_cmds.register(subparsers)
    curate_cmds.register(subparsers)
    postproc_cmds.register(subparsers)
    histology_cmds.register(subparsers)
    system_cmds.register(subparsers)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and run the selected command.

    With no arguments the GUI is started, so ``python main.py`` keeps behaving
    exactly as it always has. Returns the process exit code: 0 on success, 1 when
    a command reports a functional failure (a failed job, a blocking diagnostic),
    2 for a usage or runtime error, and 130 on Ctrl+C.
    """
    args_list: List[str] = list(sys.argv[1:] if argv is None else argv)

    parser = build_parser()
    if not args_list:
        # Bare invocation: launch the window, matching the historical behaviour.
        from .system_cmds import cmd_gui

        return cmd_gui(parser.parse_args(["gui"]), Reporter())

    args = parser.parse_args(args_list)
    if not getattr(args, "group", None):
        parser.print_help()
        return 2

    handler = getattr(args, "func", None)
    if handler is None:
        # A group was given without a command; argparse already requires one for
        # every group, so this only fires for a group we forgot to wire up.
        parser.parse_args([args.group, "--help"])
        return 2

    reporter = Reporter(as_json=bool(args.as_json), quiet=bool(args.quiet))
    try:
        return int(handler(args, reporter) or 0)
    except CommandError as exc:
        if args.traceback:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - the CLI's last line of defence
        if args.traceback:
            raise
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Re-run with --traceback for the full stack trace.", file=sys.stderr)
        return 2

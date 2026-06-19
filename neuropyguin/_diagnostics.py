"""Native-crash diagnostics for NeuroPyGuiN.

Hard crashes from Qt/pyqtgraph rendering or OpenCV/native code do not raise a
catchable Python exception, so they leave nothing in the app log: the window just
vanishes. :func:`faulthandler.enable` installs OS-level handlers (SIGSEGV,
SIGABRT, access violations on Windows) that dump the Python stack of every thread
at the moment of the fault. We point it at a persistent file and stamp a build
header, so after a crash we can see exactly where it happened and confirm which
copy of the code was running.

Send the file printed at startup (default ``~/neuropyguin_crash.log``) when a
crash occurs.
"""

from __future__ import annotations

import datetime
import faulthandler
import os
import sys
from pathlib import Path

_INSTALLED = False
_LOG_FILE = None  # keep a module-level reference so the handle is not GC'd/closed


def crash_log_path() -> Path:
    env = os.environ.get("NPG_CRASH_LOG")
    if env:
        return Path(env)
    return Path.home() / "neuropyguin_crash.log"


def _write_build_header(fh) -> None:
    fh.write(
        f"\n==== NeuroPyGuiN session {datetime.datetime.now().isoformat()} "
        f"pid={os.getpid()} ====\n"
    )
    try:
        import neuropyguin
        from neuropyguin.histology import alignment as _al

        fh.write(f"  python:  {sys.executable}\n")
        fh.write(f"  package: {neuropyguin.__file__}\n")
        fh.write(f"  auto_align_isolated: {hasattr(_al, 'auto_align_isolated')}\n")
        fh.write(
            f"  warp_atlas.use_cv2: "
            f"{'use_cv2' in getattr(_al.warp_atlas, '__code__').co_varnames}\n"
        )
    except Exception as exc:  # pragma: no cover - best effort
        fh.write(f"  (build info unavailable: {exc})\n")
    fh.flush()


def breadcrumb(msg: str) -> None:
    """Record the last GUI/render step so a native crash log shows where we were.

    The access violation is in async Qt paint code with no Python frame, so the
    faulthandler stack only shows ``app.exec()``. The last breadcrumb written
    before the crash identifies which widget operation was being painted.
    """
    fh = _LOG_FILE
    if fh is None:
        return
    try:
        fh.write(f"  . {datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} {msg}\n")
        fh.flush()
    except Exception:
        pass


def install_crash_logging() -> Path:
    """Enable faulthandler to a persistent log file. Idempotent. Returns the path."""
    global _INSTALLED, _LOG_FILE
    path = crash_log_path()
    if _INSTALLED:
        return path
    _INSTALLED = True
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _LOG_FILE = open(path, "a", encoding="utf-8", buffering=1)
        _write_build_header(_LOG_FILE)
        faulthandler.enable(file=_LOG_FILE, all_threads=True)
    except Exception:
        # Fall back to stderr so we still get *something* when launched in a console.
        try:
            faulthandler.enable(all_threads=True)
        except Exception:
            pass
    return path

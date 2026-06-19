"""Dispatch IBL-stack work to a capable interpreter as a subprocess.

In the unified ``neuropygui`` environment, ``sys.executable`` already has the IBL
stack, so it is the default interpreter for everything. The IBL alignment GUI is
launched as a *separate process* regardless, because it imports PyQt5 and must
never share a process with NeuroPyGuiN's PySide6.

Two entry points:

* :func:`run_bridge`  -- run a :mod:`ibl_bridge` subcommand synchronously
                         (xyz_picks / channels / extract_alf / all). Use from a
                         worker thread.
* :func:`launch_ibl_gui` -- launch the unmodified
                            ``atlaselectrophysiology/ephys_atlas_gui.py`` offline,
                            pointed at a histology folder.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

#: Repo root (so ``-m neuropyguin.histology.ibl_bridge`` resolves in the child).
_REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_IBLAPPS_PATH = r"D:\int-brain-lab\iblapps"


def resolve_ibl_python(configured: Optional[str] = None) -> str:
    """Return the interpreter to use for IBL work.

    Order: explicit setting -> the running interpreter if it has iblatlas ->
    a couple of well-known conda envs -> the running interpreter as a last resort.
    """
    if configured and Path(configured).exists():
        return configured
    if _has_iblatlas(sys.executable):
        return sys.executable
    for name in ("neuropygui", "iblenv"):
        for base in (Path(os.environ.get("CONDA_PREFIX", "")).parent,
                     Path.home() / ".conda" / "envs",
                     Path(r"C:\ProgramData\anaconda3\envs")):
            cand = base / name / "python.exe"
            if cand.exists() and _has_iblatlas(str(cand)):
                return str(cand)
    return sys.executable


def _has_iblatlas(python_exe: str) -> bool:
    try:
        r = subprocess.run(
            [python_exe, "-c", "import iblatlas"],
            capture_output=True, timeout=60,
        )
        return r.returncode == 0
    except Exception:
        return False


def _child_env(iblapps_path: Optional[str]) -> dict:
    env = os.environ.copy()
    extra = [str(_REPO_ROOT)]
    if iblapps_path:
        extra.append(str(iblapps_path))
    env["PYTHONPATH"] = os.pathsep.join(extra + [env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return env


def run_bridge(
    args: List[str],
    ibl_python: Optional[str] = None,
    iblapps_path: Optional[str] = None,
    log: Optional[Callable[[str], None]] = None,
    timeout: Optional[float] = None,
) -> Tuple[int, str]:
    """Run ``ibl_bridge <args...>`` synchronously. Returns ``(returncode, output)``."""
    python = resolve_ibl_python(ibl_python)
    cmd = [python, "-m", "neuropyguin.histology.ibl_bridge", *args]
    if log:
        log(f"$ {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, cwd=str(_REPO_ROOT), env=_child_env(iblapps_path),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    lines: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line.rstrip("\n"))
        if log:
            log(line.rstrip("\n"))
    proc.wait(timeout=timeout)
    return proc.returncode, "\n".join(lines)


def launch_ibl_gui(
    hist_folder: str | Path,
    ibl_python: Optional[str] = None,
    iblapps_path: Optional[str] = None,
    log: Optional[Callable[[str], None]] = None,
    auto_load: bool = True,
) -> subprocess.Popen:
    """Launch the IBL ephys-alignment GUI (offline) as a subprocess.

    With ``auto_load`` (default) the GUI opens straight on ``hist_folder`` and
    presses "Get Data" itself, via :mod:`ibl_gui_launcher`. Otherwise the stock
    GUI opens with a manual folder picker. The child's stdout/stderr is redirected
    to ``<hist_folder>/ibl_gui.log`` so a startup failure is visible afterwards.
    Returns the running ``Popen`` (non-blocking).
    """
    python = resolve_ibl_python(ibl_python)
    apps = Path(iblapps_path or DEFAULT_IBLAPPS_PATH)
    gui_script = apps / "atlaselectrophysiology" / "ephys_atlas_gui.py"
    if not gui_script.exists():
        raise FileNotFoundError(f"IBL GUI not found: {gui_script}")
    hist_folder = Path(hist_folder)

    if auto_load:
        cmd = [python, "-m", "neuropyguin.histology.ibl_gui_launcher", str(hist_folder)]
    else:
        cmd = [python, str(gui_script), "-o", "True"]

    logfile = hist_folder / "ibl_gui.log"
    try:
        out = open(logfile, "w", encoding="utf-8", errors="ignore")
    except OSError:
        out, logfile = None, None  # fall back to inheriting the parent's streams

    if log:
        target = f"auto-loading {hist_folder.name}" if auto_load else f"select folder: {hist_folder}"
        log(f"Launching IBL alignment GUI ({target}).")
        log(f"$ {' '.join(cmd)}")
        if logfile is not None:
            log(f"GUI output -> {logfile}")
    return subprocess.Popen(
        cmd, cwd=str(apps), env=_child_env(str(apps)),
        stdout=out, stderr=(subprocess.STDOUT if out is not None else None), text=True,
    )

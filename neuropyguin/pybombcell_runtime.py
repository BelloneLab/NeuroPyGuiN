"""Locate the bundled ``py_bombcell`` repository and expose it to Python.

These helpers find the vendored ``py_bombcell`` checkout relative to this file,
make it importable (either by inserting it onto ``sys.path`` or by building a
``PYTHONPATH`` for a subprocess), and verify that a candidate folder really is a
BombCell repository.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict


def workspace_root() -> Path:
    """Return the workspace directory two levels above this file."""
    return Path(__file__).resolve().parents[2]


def app_root() -> Path:
    """Return the application directory one level above this file."""
    return Path(__file__).resolve().parents[1]


def _is_pybombcell_repo(path: Path) -> bool:
    """Return True if ``path`` looks like a BombCell repository checkout."""
    # A valid checkout contains an importable ``bombcell`` package.
    return (path / "bombcell" / "__init__.py").exists()


def resolve_pybombcell_repo() -> Path:
    """Return the best-guess path to the bundled ``py_bombcell`` repository.

    Prefers the copy bundled inside the app folder, falling back to a legacy
    location. If neither candidate is a valid repository, the bundled candidate
    is returned anyway so callers can surface a clear error.
    """
    root = workspace_root()
    app = app_root()
    candidates = [
        app / "py_bombcell",  # bundled first, folder-name agnostic
        root / "NeuroPyGuiN" / "py_bombcell",  # legacy fallback
    ]
    for c in candidates:
        if _is_pybombcell_repo(c):
            return c
    return candidates[0]


def ensure_pybombcell_on_sys_path() -> Path:
    """Insert the bundled ``py_bombcell`` repo onto ``sys.path`` and return it.

    Raises ``RuntimeError`` if the bundled repository cannot be found.
    """
    repo = resolve_pybombcell_repo()
    if not repo.exists() or not _is_pybombcell_repo(repo):
        raise RuntimeError(f"Bundled py_bombcell not found in app folder: {repo}")
    p = str(repo)
    if p not in sys.path:
        sys.path.insert(0, p)
    return repo


def pybombcell_subprocess_env() -> Dict[str, str]:
    """Return a copy of ``os.environ`` with ``py_bombcell`` on ``PYTHONPATH``.

    The repository path is prepended to any existing ``PYTHONPATH`` so a child
    process can import the bundled BombCell package.
    """
    env = os.environ.copy()
    repo = ensure_pybombcell_on_sys_path()
    py_path = env.get("PYTHONPATH", "")
    repo_s = str(repo)
    env["PYTHONPATH"] = f"{repo_s}{os.pathsep}{py_path}" if py_path else repo_s
    return env

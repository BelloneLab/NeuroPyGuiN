from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def app_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_pybombcell_repo(path: Path) -> bool:
    if (path / "bombcell" / "__init__.py").exists():
        return True
    return False


def resolve_pybombcell_repo() -> Path:
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
    repo = resolve_pybombcell_repo()
    if not repo.exists() or not _is_pybombcell_repo(repo):
        raise RuntimeError(f"Bundled py_bombcell not found in app folder: {repo}")
    p = str(repo)
    if p not in sys.path:
        sys.path.insert(0, p)
    return repo


def pybombcell_subprocess_env() -> Dict[str, str]:
    env = os.environ.copy()
    repo = ensure_pybombcell_on_sys_path()
    py_path = env.get("PYTHONPATH", "")
    repo_s = str(repo)
    env["PYTHONPATH"] = f"{repo_s}{os.pathsep}{py_path}" if py_path else repo_s
    return env

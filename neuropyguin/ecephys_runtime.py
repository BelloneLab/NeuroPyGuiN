from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def app_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_ecephys_repo(path: Path) -> bool:
    # repo root layout
    if (path / "ecephys_spike_sorting" / "scripts" / "create_input_json.py").exists():
        return True
    # package-only layout
    if (path / "scripts" / "create_input_json.py").exists() and (path / "__init__.py").exists():
        return True
    return False


def _repo_pythonpath_root(path: Path) -> Path:
    # If caller passed package root, PYTHONPATH must be its parent.
    if (path / "__init__.py").exists() and (path / "scripts" / "create_input_json.py").exists():
        return path.parent
    return path


def resolve_ecephys_repo() -> Path:
    root = workspace_root()
    app = app_root()
    candidates = [
        app / "ecephys_spike_sorting",  # bundled first, folder-name agnostic
        app / "ecephys_spike_sorting" / "ecephys_spike_sorting",  # bundled package-only fallback
        root / "NeuroPyGuiN" / "ecephys_spike_sorting",  # legacy fallback
        root / "NeuroPyGuiN" / "ecephys_spike_sorting" / "ecephys_spike_sorting",  # legacy package fallback
    ]
    for c in candidates:
        if _is_ecephys_repo(c):
            return c
    return candidates[0]


def ensure_ecephys_on_sys_path() -> Path:
    repo = resolve_ecephys_repo()
    if not repo.exists() or not _is_ecephys_repo(repo):
        raise RuntimeError(
            f"Bundled ecephys_spike_sorting not found in app folder: {repo}"
        )
    p = str(_repo_pythonpath_root(repo))
    if p not in sys.path:
        sys.path.insert(0, p)
    return repo


def ecephys_subprocess_env() -> Dict[str, str]:
    env = os.environ.copy()
    repo = ensure_ecephys_on_sys_path()
    py_path = env.get("PYTHONPATH", "")
    repo_s = str(_repo_pythonpath_root(repo))
    env["PYTHONPATH"] = f"{repo_s}{os.pathsep}{py_path}" if py_path else repo_s
    # Ecephys modules run without an interactive display. Kilosort imports
    # matplotlib and generates diagnostic PNGs; when Qt is installed,
    # matplotlib otherwise selects QtAgg and connects this child process to the
    # desktop X11/ICE session. A broken session-manager pipe can then make the
    # interpreter exit with code 1 after Kilosort has successfully saved every
    # result. Agg renders the same files without creating a GUI connection.
    env["MPLBACKEND"] = "Agg"
    # MKL 2026.1.0 in this conda env is paired with llvm-openmp 22.1.8, whose
    # libiomp5md.dll does not export __atomic_load / __atomic_compare_exchange.
    # mkl_intel_thread.3.dll delay-loads both of them, so the first threaded
    # LAPACK call (Kilosort's TruncatedSVD inside spikedetect.extract_wPCA_wTEMP)
    # makes the Windows delay-load helper terminate the process with
    # STATUS_ENTRYPOINT_NOT_FOUND (0xC06D007F, surfacing as exit code
    # 3228369023) and no Python traceback. Routing MKL through TBB, which is
    # already installed, avoids libiomp5md entirely. An explicit user setting wins.
    if not env.get("MKL_THREADING_LAYER"):
        env["MKL_THREADING_LAYER"] = "TBB"
    return env


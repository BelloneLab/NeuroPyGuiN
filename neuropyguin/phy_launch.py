"""Helpers for launching phy from NeuroPyGuiN.

NeuroPyGuiN itself is a PySide6 application, while phy 2 is a PyQt5
application. The child process must therefore get a clean Qt binding selection
and, on Windows, the DLL search path for the environment that owns phy.exe.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, Mapping


PHY_EXECUTABLE_ENV = "NEUROPYGUIN_PHY_EXE"

PHY_ENV_OVERRIDES = {
    "PYQTGRAPH_QT_LIB": "PyQt5",
    "QT_API": "pyqt5",
    "SDL_VIDEODRIVER": "dummy",
    "SDL_AUDIODRIVER": "dummy",
    "PYGAME_HIDE_SUPPORT_PROMPT": "1",
}


def _scripts_dir_name() -> str:
    return "Scripts" if os.name == "nt" else "bin"


def _phy_exe_name() -> str:
    return "phy.exe" if os.name == "nt" else "phy"


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        try:
            key = str(path.resolve()).lower() if os.name == "nt" else str(path.resolve())
        except OSError:
            key = str(path).lower() if os.name == "nt" else str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _active_environment_candidates(env: Mapping[str, str]) -> list[Path]:
    """Return phy candidates in the currently running Python or conda env."""
    exe_name = _phy_exe_name()
    scripts_dir = _scripts_dir_name()
    candidates: list[Path] = []

    python_path = Path(sys.executable)
    candidates.append(python_path.parent / scripts_dir / exe_name)
    if python_path.parent.name.lower() == scripts_dir.lower():
        candidates.append(python_path.parent / exe_name)

    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / scripts_dir / exe_name)

    return _dedupe_paths(candidates)


def _conda_env_search_roots(env: Mapping[str, str]) -> list[Path]:
    roots: list[Path] = []

    conda_exe = env.get("CONDA_EXE") or shutil.which("conda", path=env.get("PATH"))
    if conda_exe:
        exe_path = Path(conda_exe)
        for parent in exe_path.parents:
            if (parent / "envs").exists() or (parent / "conda-meta").exists():
                roots.append(parent / "envs")
                break

    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        prefix = Path(conda_prefix)
        roots.append(prefix.parent)
        if prefix.parent.name.lower() == "envs":
            roots.append(prefix.parent)

    roots.append(Path.home() / ".conda" / "envs")
    if os.name == "nt":
        program_data = Path(env.get("ProgramData", r"C:\ProgramData"))
        roots.append(program_data / "anaconda3" / "envs")
        roots.append(program_data / "miniconda3" / "envs")

    return _dedupe_paths(roots)


def _conda_phy_candidates(env: Mapping[str, str]) -> list[Path]:
    exe_name = _phy_exe_name()
    scripts_dir = _scripts_dir_name()
    found: list[tuple[tuple[int, str], Path]] = []
    priority_names = {
        "phy2": 0,
        "phy": 1,
        "neuropygui": 2,
    }

    for root in _conda_env_search_roots(env):
        if not root.exists():
            continue
        try:
            children = list(root.iterdir())
        except OSError:
            continue
        for child in children:
            exe = child / scripts_dir / exe_name
            if not exe.exists():
                continue
            name = child.name.lower()
            priority = priority_names.get(name, 10 if "phy" in name else 20)
            found.append(((priority, name), exe))

    return _dedupe_paths(path for _sort_key, path in sorted(found, key=lambda item: item[0]))


def resolve_phy_executable(env: Mapping[str, str] | None = None) -> str:
    """Return the phy executable NeuroPyGuiN should launch.

    Order: explicit ``NEUROPYGUIN_PHY_EXE`` -> active Python or conda env ->
    discovered conda phy envs -> PATH lookup -> the bare ``phy`` command.
    """
    env = dict(os.environ if env is None else env)

    configured = env.get(PHY_EXECUTABLE_ENV, "").strip().strip('"')
    if configured:
        return configured

    for candidate in [*_active_environment_candidates(env), *_conda_phy_candidates(env)]:
        if candidate.exists():
            return str(candidate)

    found = shutil.which("phy", path=env.get("PATH"))
    return found or "phy"


def _conda_root_for_executable(program: str | os.PathLike[str]) -> Path | None:
    try:
        path = Path(program).resolve()
    except OSError:
        return None
    if path.parent.name.lower() not in {"scripts", "bin"}:
        return None
    root = path.parent.parent
    if (root / "conda-meta").exists():
        return root
    return None


def _prepend_path_entries(env: dict[str, str], entries: Iterable[Path]) -> None:
    existing = env.get("PATH", "")
    parts = [str(entry) for entry in entries if entry.exists()]
    if not parts:
        return
    env["PATH"] = os.pathsep.join([*parts, existing]) if existing else os.pathsep.join(parts)


def phy_child_environment(
    base_env: Mapping[str, str] | None = None,
    phy_executable: str | os.PathLike[str] | None = None,
) -> dict[str, str]:
    """Return a process environment suitable for a phy child process."""
    env = dict(os.environ if base_env is None else base_env)
    env.update(PHY_ENV_OVERRIDES)

    if phy_executable is not None:
        conda_root = _conda_root_for_executable(phy_executable)
        if conda_root is not None:
            if os.name == "nt":
                _prepend_path_entries(
                    env,
                    [
                        conda_root,
                        conda_root / "Scripts",
                        conda_root / "Library" / "bin",
                        conda_root / "Library" / "usr" / "bin",
                    ],
                )
                site_packages: Path | None = conda_root / "Lib" / "site-packages"
            else:
                _prepend_path_entries(env, [conda_root / "bin"])
                site_packages = next(iter(sorted(conda_root.glob("lib/python*/site-packages"))), None)
            # Do NOT hard-isolate phy from the user site-packages. Several user phy
            # plugins (mahalanobis, UMAP, ...) import packages that live only in the
            # user site (seaborn, umap-learn, numba), so setting PYTHONNOUSERSITE=1
            # made phy's plugin discovery crash with ModuleNotFoundError. Instead, put
            # the phy env's own site-packages first on PYTHONPATH so its tested
            # numpy/scipy/scikit-learn still take precedence, while the user site stays
            # available as a fallback for the plugin-only dependencies.
            env.pop("PYTHONNOUSERSITE", None)
            if site_packages is not None and site_packages.exists():
                existing_pp = env.get("PYTHONPATH", "")
                parts = [str(site_packages), existing_pp] if existing_pp else [str(site_packages)]
                env["PYTHONPATH"] = os.pathsep.join(parts)

    return env

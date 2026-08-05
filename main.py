from __future__ import annotations

import sys
import os
import subprocess
import traceback
from pathlib import Path


def _neuropyguin_env_python() -> Path | None:
    """Return a sibling conda env Python that should have the supported Qt stack."""
    current = Path(sys.executable).resolve()
    candidates = []
    if current.parent.name.lower() == "anaconda3":
        candidates.append(current.parent / "envs" / "neuropyguin" / "python.exe")
    if current.parent.parent.name.lower() == "envs":
        candidates.append(current.parent.parent / "neuropyguin" / "python.exe")
    candidates.extend(
        [
            Path.home() / "anaconda3" / "envs" / "neuropyguin" / "python.exe",
            Path.home() / "miniconda3" / "envs" / "neuropyguin" / "python.exe",
            Path(r"C:\ProgramData\anaconda3\envs\neuropyguin\python.exe"),
        ]
    )
    for candidate in candidates:
        try:
            if candidate.exists() and candidate.resolve() != current:
                return candidate
        except Exception:
            continue
    return None


def _maybe_relaunch_in_supported_env() -> int | None:
    """Rerun under the supported conda env when launched from a broken base env."""
    if os.environ.get("NEUROPYGUIN_NO_ENV_RELAUNCH") == "1":
        return None
    env_python = _neuropyguin_env_python()
    if env_python is None:
        return None
    print(
        f"Relaunching NeuroPyGuiN with the supported conda env: {env_python}",
        file=sys.stderr,
    )
    env = os.environ.copy()
    env["NEUROPYGUIN_NO_ENV_RELAUNCH"] = "1"
    args = [str(env_python), str(Path(__file__).resolve()), *sys.argv[1:]]
    return subprocess.call(args, cwd=str(Path(__file__).resolve().parent), env=env)


def _entry() -> int:
    try:
        from neuropyguin._diagnostics import install_crash_logging
        crash_log = install_crash_logging()
        print(f"NeuroPyGuiN crash log: {crash_log}", file=sys.stderr)
    except Exception:
        pass
    try:
        from neuropyguin.app import main
    except Exception as exc:
        msg = str(exc)
        if "PySide6" in msg or "QtCore" in msg or "DLL load failed" in msg:
            relaunched = _maybe_relaunch_in_supported_env()
            if relaunched is not None:
                return int(relaunched)
            print("Failed to import PySide6/Qt runtime.", file=sys.stderr)
            print(f"Python executable: {sys.executable}", file=sys.stderr)
            print("This usually means you are launching with a different Python than the one where PySide6 is installed.", file=sys.stderr)
            print("Use the intended conda env, for example:", file=sys.stderr)
            print("  conda activate neuropyguin", file=sys.stderr)
            print("  python main.py", file=sys.stderr)
            return 2
        traceback.print_exc()
        return 1
    return int(main())


if __name__ == "__main__":
    raise SystemExit(_entry())

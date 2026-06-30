"""Dispatch the NeuroPyxels C4 cell-type classifier into its isolated env.

C4 predicts cerebellar cell types (GoC, MLI, MFB, PkC_ss, PkC_cs) from each unit's
3D autocorrelogram + waveform with a Laplace-calibrated CNN ensemble. The Laplace
stack requires torch >= 2.6, which conflicts with the app env's CUDA torch 2.5.1
(kilosort), so C4 runs in a dedicated ``npyx_c4`` environment via subprocess - the
same isolation pattern used for phy (phy2) and the IBL GUI.

``run_c4_classifier`` writes the request to a temp JSON, runs
``neuropyguin.c4_bridge`` with the npyx_c4 interpreter, and reads the result back.
The model (~2.7 GB) is cached under ``~/.npyx_c4_resources`` after first use.

NOTE: the model is cerebellum-trained, so on other regions (e.g. VTA) the labels
are indicative cell-type shapes, not ground-truth identities.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Candidate locations for the isolated C4 interpreter (override with NPYX_C4_PYTHON).
_C4_PYTHON_CANDIDATES = [
    os.environ.get("NPYX_C4_PYTHON", ""),
    r"C:/Users/bellone/.conda/envs/npyx_c4/python.exe",
    str(Path.home() / ".conda" / "envs" / "npyx_c4" / "python.exe"),
    r"C:/ProgramData/anaconda3/envs/npyx_c4/python.exe",
]


def find_c4_python() -> Optional[str]:
    """Return the path to the isolated C4 interpreter, or None if not set up."""
    for cand in _C4_PYTHON_CANDIDATES:
        if cand and Path(cand).exists():
            return cand
    return None


def c4_class_names() -> List[str]:
    """Return the C4 base-model cell-type class names (importable in the main env)."""
    try:
        repo = str(_REPO_ROOT)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from npyx.datasets import CORRESPONDENCE_NO_GRC  # type: ignore

        keys = sorted(k for k in CORRESPONDENCE_NO_GRC if int(k) >= 0)
        return [str(CORRESPONDENCE_NO_GRC[k]) for k in keys]
    except Exception:
        return ["GoC", "MLI", "MFB", "PkC_ss", "PkC_cs"]


def _empty_result(error: Optional[str]) -> Dict[str, object]:
    names = c4_class_names()
    return {
        "units": [], "predicted_type": [], "confidence": np.array([]),
        "confidence_ratio": np.array([]), "model_votes": np.array([]),
        "probabilities": np.zeros((0, len(names)), dtype=float),
        "class_names": names, "skipped_units": [], "model_type": "base", "error": error,
    }


def run_c4_classifier(
    dp: str,
    units: Sequence[int],
    *,
    quality: str = "good",
    threshold: float = 2.0,
    device: str = "cpu",
    progress_cb=None,
    timeout_s: float = 1800.0,
) -> Dict[str, object]:
    """Classify ``units`` with C4 (in the isolated env) and return a render-ready dict.

    Keys: ``units``, ``predicted_type``, ``confidence``, ``confidence_ratio``,
    ``model_votes``, ``probabilities`` (n_units x n_classes ndarray), ``class_names``,
    ``skipped_units``, ``model_type``, ``error`` (str or None). On any failure
    ``error`` is set and the arrays are empty.
    """
    def _say(msg: str) -> None:
        if progress_cb is not None:
            try:
                progress_cb(msg)
            except Exception:
                pass

    units = [int(u) for u in units]
    if not units:
        return _empty_result("Select at least one unit to classify.")

    c4_python = find_c4_python()
    if c4_python is None:
        return _empty_result(
            "C4 environment not found. Create it once with:\n"
            "  conda create -p <...>/.conda/envs/npyx_c4 python=3.10 -y\n"
            "  <npyx_c4>/python -m pip install \"npyx[c4]\"\n"
            "or set NPYX_C4_PYTHON to its python.exe."
        )

    with tempfile.TemporaryDirectory(prefix="c4_") as tmp:
        in_path = Path(tmp) / "c4_in.json"
        out_path = Path(tmp) / "c4_out.json"
        in_path.write_text(json.dumps({
            "dp": str(dp), "units": units, "threshold": float(threshold), "device": str(device),
        }), encoding="utf-8")

        _say("Launching C4 in its isolated environment...")
        cmd = [c4_python, "-W", "ignore", "-m", "neuropyguin.c4_bridge", str(in_path), str(out_path)]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        try:
            proc = subprocess.Popen(
                cmd, cwd=str(_REPO_ROOT), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
        except Exception as exc:  # noqa: BLE001
            return _empty_result(f"Failed to launch C4 subprocess: {exc}")

        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                if line.startswith("[c4]"):
                    _say(line[4:].strip())
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            return _empty_result(f"C4 timed out after {int(timeout_s)} s.")
        except Exception as exc:  # noqa: BLE001
            return _empty_result(f"C4 subprocess error: {exc}")

        if not out_path.exists():
            return _empty_result(f"C4 produced no output (exit code {rc}). Check the C4 environment.")
        try:
            res = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            return _empty_result(f"Could not read C4 output: {exc}")

    if res.get("error"):
        out = _empty_result(str(res["error"]))
        out["skipped_units"] = res.get("skipped_units", [])
        return out

    return {
        "units": [int(u) for u in res.get("units", [])],
        "predicted_type": [str(t) for t in res.get("predicted_type", [])],
        "confidence": np.asarray(res.get("confidence", []), dtype=float),
        "confidence_ratio": np.asarray(res.get("confidence_ratio", []), dtype=float),
        "model_votes": np.asarray(res.get("model_votes", []), dtype=float),
        "probabilities": np.asarray(res.get("probabilities", []), dtype=float),
        "class_names": [str(c) for c in res.get("class_names", c4_class_names())],
        "skipped_units": [int(u) for u in res.get("skipped_units", [])],
        "model_type": str(res.get("model_type", "base")),
        "n_models": int(res.get("n_models", 0) or 0),
        "error": None,
    }

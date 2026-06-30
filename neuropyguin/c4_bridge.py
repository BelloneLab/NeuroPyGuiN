"""C4 cell-type classifier worker, run inside the isolated ``npyx_c4`` env.

C4's Laplace-calibrated ensemble requires the ``laplace`` package, which pulls a
torch >= 2.6 that is incompatible with the main app env's CUDA torch 2.5.1
(kilosort). So, exactly like phy (phy2 env) and the IBL GUI, C4 runs in its own
environment via subprocess. This script is launched by ``neuropyguin.c4_runner``
with the npyx_c4 interpreter:

    python -m neuropyguin.c4_bridge <input.json> <output.json>

It reads {"dp", "units", "threshold", "device"} from input.json, runs the
in-memory prediction against the vendored npyx, and writes the results dict to
output.json. All progress is printed to stdout so the dispatcher can stream it.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path


def _ensure_vendored_npyx() -> None:
    """Prefer the repo's vendored npyx (matches the cached calibrated models)."""
    repo_root = Path(__file__).resolve().parents[1]
    if (repo_root / "npyx").exists() and str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _stage_raw_files(dp: Path):
    """Hardlink the .ap.bin and copy the .ap.meta into ``dp`` (KS4 leaves them in the
    parent). Returns the list of created files so we can remove them afterwards."""
    created = []
    parent = dp.parent
    for pattern, link in (("*.ap.bin", True), ("*.ap.meta", False)):
        if any(dp.glob(pattern)):
            continue
        srcs = sorted(parent.glob(pattern))
        if not srcs:
            continue
        src, dest = srcs[0], dp / srcs[0].name
        try:
            if link:
                try:
                    os.link(str(src), str(dest))
                except OSError:
                    os.symlink(str(src), str(dest))
            else:
                shutil.copy2(str(src), str(dest))
            created.append(dest)
        except Exception:
            pass
    return created


def run(dp: str, units, threshold: float = 2.0, device: str = "cpu") -> dict:
    _ensure_vendored_npyx()
    import numpy as np
    import torch
    import torch.utils.data as torch_data

    from npyx.c4.predict_cell_types import (
        prepare_dataset_from_binary,
        load_precalibrated_ensemble,
        ensemble_predict,
        format_predictions,
    )
    from npyx.c4.run_deep_classifier import CustomDataset
    from npyx.datasets import CORRESPONDENCE_NO_GRC, LABELLING_NO_GRC

    keys = sorted(k for k in CORRESPONDENCE_NO_GRC if int(k) >= 0)
    class_names = [str(CORRESPONDENCE_NO_GRC[k]) for k in keys]
    out = {
        "units": [], "predicted_type": [], "confidence": [], "confidence_ratio": [],
        "model_votes": [], "probabilities": [], "class_names": class_names,
        "skipped_units": [], "model_type": "base", "n_models": 0, "error": None,
    }

    dp_path = Path(dp)
    units = [int(u) for u in units]
    if not units:
        out["error"] = "Select at least one unit to classify."
        return out

    models_dir = Path.home() / ".npyx_c4_resources" / "models" / "base" / "calibrated_models"
    if not models_dir.exists():
        out["error"] = (
            "C4 ensemble not found. It downloads (~2.7 GB) on first use; run "
            "npyx.c4.run_cell_types_classifier once with network access."
        )
        return out

    staged = _stage_raw_files(dp_path)
    try:
        print(f"[c4] extracting features for {len(units)} unit(s)...", flush=True)
        prepared = prepare_dataset_from_binary(str(dp_path), units)
    finally:
        for p in staged:
            try:
                p.unlink()
            except Exception:
                pass

    if isinstance(prepared, tuple):
        dataset, bad_units = prepared[0], (prepared[1] if len(prepared) > 1 else [])
    else:
        dataset, bad_units = prepared, []
    bad_set = {int(b) for b in (bad_units or [])}
    good_units = [u for u in units if u not in bad_set]
    out["skipped_units"] = sorted(bad_set)
    if len(dataset) == 0 or not good_units:
        out["error"] = "No selected units passed C4 quality checks (need >=3 min, low RPV, enough spikes)."
        return out

    iterator = torch_data.DataLoader(
        CustomDataset(dataset, np.zeros(len(dataset)), spikes_list=None, layer=None),
        batch_size=len(dataset),
    )
    print("[c4] loading ensemble (can take ~30-60 s)...", flush=True)
    ensemble = load_precalibrated_ensemble(str(models_dir), fast=False)
    print("[c4] running inference...", flush=True)
    raw_probs = ensemble_predict(
        ensemble, iterator, device=torch.device(device),
        method="raw", enforce_layer=False, labelling=LABELLING_NO_GRC,
    )
    preds, mean_conf, _delta, n_votes, conf_ratio = format_predictions(raw_probs)
    preds = np.asarray(preds).reshape(-1)
    mean_conf = np.asarray(mean_conf, dtype=float).reshape(-1)
    conf_ratio = np.asarray(conf_ratio, dtype=float).reshape(-1)
    n_votes = np.asarray(n_votes).reshape(-1)
    probs = np.asarray(raw_probs, dtype=float)
    prob_mat = probs.mean(axis=2) if probs.ndim == 3 else probs

    labels = [
        (str(CORRESPONDENCE_NO_GRC[int(p)]) if float(cr) >= float(threshold) else "unlabelled")
        for p, cr in zip(preds, conf_ratio)
    ]
    out.update({
        "units": [int(x) for x in good_units[: len(preds)]],
        "predicted_type": labels,
        "confidence": [float(x) for x in mean_conf],
        "confidence_ratio": [float(x) for x in conf_ratio],
        "model_votes": [float(x) for x in n_votes],
        "probabilities": [[float(v) for v in row] for row in prob_mat],
        "n_models": int(probs.shape[2]) if probs.ndim == 3 else 0,
    })
    print(f"[c4] done: {len(good_units)} classified, {len(bad_set)} skipped.", flush=True)
    return out


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: python -m neuropyguin.c4_bridge <input.json> <output.json>", file=sys.stderr)
        return 2
    in_path, out_path = sys.argv[1], sys.argv[2]
    with open(in_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    try:
        result = run(
            cfg["dp"], cfg.get("units", []),
            threshold=float(cfg.get("threshold", 2.0)),
            device=str(cfg.get("device", "cpu")),
        )
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        result = {"error": f"{type(exc).__name__}: {exc}"}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

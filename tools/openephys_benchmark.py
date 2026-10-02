"""End-to-end benchmark: Open Ephys staging vs native SpikeGLX on the same real data.

Three arms are sorted with the *real* ``EcephysPipelineWorker`` (CatGT -> Kilosort4
-> postprocessing -> C_Waves -> quality metrics -> py_bombcell):

* ``sglx_a``  the SpikeGLX chunk (reference),
* ``sglx_b``  the same SpikeGLX chunk sorted a second time (Kilosort4 is not
  bit-deterministic on GPU, so A-vs-B agreement is the noise floor),
* ``sglx_dither`` the SpikeGLX chunk plus +-1 count random dither on the AP
  channels. Kilosort4 turns out to be deterministic (A == B), so this tiny
  perturbation, comparable to Open Ephys' different int16 quantization, is the
  meaningful noise floor for unit agreement,
* ``oe``      the same chunk converted to a format-faithful Open Ephys recording
  (sync channel -> TTL events, data rescaled to 0.195 uV/bit, electrode
  positions in settings.xml, synchronized timestamps with an offset and clock
  drift injected) and staged by :mod:`neuropyguin.openephys`.

If staging is correct, OE-vs-A must look like B-vs-A: same units, same uV
amplitudes, same depths, and TTL edges on the same clock.

Usage::

    python tools/openephys_benchmark.py prepare --source <real .ap.bin> --out <dir>
    python tools/openephys_benchmark.py run --out <dir> [--arms sglx_a sglx_b oe]
    python tools/openephys_benchmark.py analyze --out <dir>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tests")]

FS = 30000
RUN = "bench"


# ---------------------------------------------------------------------------
# Prepare: cut a real chunk and build the SpikeGLX and Open Ephys inputs
# ---------------------------------------------------------------------------


def _read_meta(path: Path) -> dict:
    meta = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            meta[k] = v
    return meta


def prepare(source: Path, out: Path, start_s: float, dur_s: float, drift: float, offset_s: float) -> None:
    """Write the SpikeGLX reference chunk and its Open Ephys twin under ``out``."""
    from openephys_fake import FakeProbe, make_openephys_session

    meta_src = Path(str(source).replace(".ap.bin", ".ap.meta"))
    meta = _read_meta(meta_src)
    n_ch = int(meta["nSavedChans"])
    raw = np.memmap(source, dtype=np.int16, mode="r").reshape(-1, n_ch)
    s0, s1 = int(start_s * FS), int((start_s + dur_s) * FS)
    chunk = np.ascontiguousarray(raw[s0:s1])
    n = chunk.shape[0]

    # --- SpikeGLX reference: SpikeGLX run-folder naming, meta patched for the cut.
    sg_dir = out / "inputs" / "sglx" / "rawData" / "bench_mouse" / f"{RUN}_g0" / f"{RUN}_g0_imec0"
    sg_dir.mkdir(parents=True, exist_ok=True)
    sg_bin = sg_dir / f"{RUN}_g0_t0.imec0.ap.bin"
    chunk.tofile(sg_bin)
    lines = []
    for line in meta_src.read_text(encoding="utf-8", errors="ignore").splitlines():
        key = line.split("=", 1)[0]
        if key == "fileSizeBytes":
            line = f"fileSizeBytes={chunk.nbytes}"
        elif key == "fileTimeSecs":
            line = f"fileTimeSecs={n / FS:.6f}"
        elif key == "firstSample":
            line = f"firstSample={int(meta['firstSample']) + s0}"
        elif key in {"fileSHA1", "fileName"}:
            continue
        lines.append(line)
    sg_bin.with_suffix(".meta").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- Dithered SpikeGLX control: +-1 count on AP channels, SYNC untouched.
    dt_dir = out / "inputs" / "sglx_dither" / "rawData" / "bench_mouse" / f"{RUN}_g0" / f"{RUN}_g0_imec0"
    dt_dir.mkdir(parents=True, exist_ok=True)
    dithered = chunk.copy()
    dither = np.random.default_rng(1).integers(-1, 2, size=(n, n_ch - 1), dtype=np.int16)
    dithered[:, : n_ch - 1] = np.clip(chunk[:, : n_ch - 1].astype(np.int32) + dither, -32768, 32767).astype(np.int16)
    dithered.tofile(dt_dir / sg_bin.name)
    del dithered, dither
    shutil.copy2(sg_bin.with_suffix(".meta"), dt_dir / sg_bin.with_suffix(".meta").name)

    # --- Open Ephys twin: drop SY, rescale counts to Open Ephys' 0.195 uV/bit.
    uv_sglx = 1e6 * (float(meta["imAiRangeMax"]) - float(meta["imAiRangeMin"])) / float(meta["imChan0apGain"]) / (
        2 * int(meta["imMaxInt"])
    )
    bit_volts = 0.1949999928474426
    scaled = np.rint(chunk[:, :384].astype(np.float32) * (uv_sglx / bit_volts))
    clipped = float(np.mean(np.abs(scaled) > 32767))
    oe_data = np.clip(scaled, -32768, 32767).astype(np.int16)
    del scaled

    geom = meta["~snsGeomMap"].split(")(")
    header = geom[0].lstrip("(").split(",")
    pitch = float(header[2])
    entries = [e.strip("()").split(":") for e in geom[1:]]
    shank = np.array([int(e[0]) for e in entries][:384])
    x_local = np.array([float(e[1]) for e in entries][:384])
    y = np.array([float(e[2]) for e in entries][:384])

    sync = chunk[:, n_ch - 1]
    bit = (sync >> 6) & 1
    edges = np.flatnonzero(np.diff(bit) != 0) + 1
    rising = edges[bit[edges] == 1]
    width = int(np.median(np.diff(edges))) if len(edges) > 1 else 15000

    oe_root = out / "inputs" / "openephys" / "rawData" / "bench_mouse"
    probe = FakeProbe(
        "ProbeA",
        meta.get("imDatPrb_pn", "NP2014"),
        oe_data,
        x=x_local + shank * pitch,  # Open Ephys stores x across shanks
        y=y,
        shank=shank,
        with_lfp=False,
        ttl_samples=rising.tolist(),
        ttl_width=width,
    )
    make_openephys_session(
        oe_root,
        [probe],
        first_sample=int(meta["firstSample"]) + s0,
        ts_offset_s=offset_s,
        clock_scale=1.0 + drift,
        session_name="2026-06-23_17-08-38",
    )
    info = {
        "source": str(source),
        "start_s": start_s,
        "dur_s": dur_s,
        "n_samples": n,
        "uv_per_bit_sglx": uv_sglx,
        "bit_volts_oe": bit_volts,
        "scale": uv_sglx / bit_volts,
        "clipped_fraction": clipped,
        "sync_rising_edges": int(len(rising)),
        "injected_drift": drift,
        "injected_offset_s": offset_s,
        "probe": meta.get("imDatPrb_pn"),
    }
    (out / "prepare.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


# ---------------------------------------------------------------------------
# Run: the real pipeline worker, one arm at a time
# ---------------------------------------------------------------------------


def _pipeline_cfg(output_root: Path, json_root: Path):
    from neuropyguin.workers import EcephysPipelineConfig

    tools = REPO / "tools"
    import kilosort

    return EcephysPipelineConfig(
        output_root=str(output_root),
        json_root=str(json_root),
        mirror_raw_hierarchy_output=True,
        save_catgt_ap_bin=False,
        run_catgt=True,
        run_catgt_extract_only=False,
        run_tprime=False,
        run_kilosort=True,
        run_kilosort_postproc=True,
        run_noise_templates=False,
        run_mean_waveforms=True,
        run_quality_metrics=True,
        run_pybombcell=True,
        ks_ver="4",
        gate_string="0",
        trigger_string="0,0",
        probe_string="current",
        region_name="default",
        ni_extract_string="",
        # Imec SYNC extraction (word 384, bit 6) for the SpikeGLX arms; the worker
        # drops extractors automatically for the Open Ephys arm.
        catgt_cmd_string="-prb_fld -out_prb_fld -apfilter=butter,12,300,10000 -gfix=0.4,0.10,0.02 -xd=2,0,384,6,500",
        catgt_output_streams="ap",
        catgt_lf_lowpass_hz=300.0,
        catgt_lf_downsample=12,
        sync_period=1.0,
        tostream_sync_params="imec0",
        ks_th="[8,9]",
        qm_isi_thresh=0.002,
        catgt_car_mode="gbldmx",
        catgt_loccar_min_um=40.0,
        catgt_loccar_max_um=160.0,
        ks4_duplicate_spike_ms=0.25,
        ks4_min_template_size_um=10.0,
        c_waves_snr_um=160.0,
        # The user's saved advanced params include n_chan_bin=385; keeping it here
        # also exercises the guard that must ignore it for 384-channel inputs.
        ks4_advanced_params={"n_chan_bin": 385, "batch_size": 60000, "nblocks": 4},
        catgt_path=str(tools / "CatGT-linux"),
        tprime_path=str(tools / "TPrime-linux"),
        cwaves_path=str(tools / "C_Waves-linux"),
        ks4_repo_path=str(Path(kilosort.__file__).parent),
        kilosort_output_tmp=str(REPO / "kilosort_datatemp"),
        output_layout="mirror",
    )


def run_arm(out: Path, arm: str) -> dict:
    from PySide6 import QtCore

    from neuropyguin.openephys import discover_openephys_recordings, stage_openephys_recording
    from neuropyguin.preprocessing import parse_spikeglx_bin_name
    from neuropyguin.workers import EcephysPipelineWorker

    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    arm_root = out / "runs" / arm
    if arm_root.exists():
        shutil.rmtree(arm_root)
    arm_root.mkdir(parents=True)
    if arm == "oe":
        recs, skipped = discover_openephys_recordings([out / "inputs" / "openephys"])
        assert len(recs) == 1, (recs, skipped)
        staged = stage_openephys_recording(recs[0], arm_root / "processed")
        bin_file = staged.ap_bin
        print(f"[oe] staged {bin_file} ({staged.link_mode}) warnings={staged.warnings}")
    else:
        src = "sglx_dither" if arm == "sglx_dither" else "sglx"
        bin_file = next((out / "inputs" / src).rglob("*.ap.bin"))
    parsed = parse_spikeglx_bin_name(str(bin_file))
    job = {
        "name": parsed["run_name"],
        "bin_file": str(bin_file),
        "workdir": str(bin_file.parent),
        "gate_string": parsed["gate_string"],
        "trigger_string": parsed["trigger_string"],
        "probe_string": parsed["probe_string"],
    }
    cfg = _pipeline_cfg(arm_root / "processed", arm_root / "json")
    worker = EcephysPipelineWorker(job, cfg)
    result: dict = {}
    log_path = arm_root / "pipeline.log"
    with log_path.open("w") as logf:
        def log(msg: str) -> None:
            logf.write(msg + "\n")
            logf.flush()

        worker.signals.log.connect(log)
        worker.signals.error.connect(lambda m: log("ERROR " + m))
        worker.signals.stepStarted.connect(lambda k, l: print(f"[{arm}] >> {l}", flush=True))
        worker.signals.stepFinished.connect(lambda k, ok: print(f"[{arm}] << {k} ok={ok}", flush=True))
        worker.signals.finished.connect(result.update)
        t0 = time.time()
        worker.run()
        app.processEvents()
        result["seconds"] = time.time() - t0
    result["bin_file"] = str(bin_file)
    (arm_root / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[{arm}] finished ok={result.get('ok')} in {result['seconds']:.0f}s -> {result.get('ks_folder')}")
    return result


# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------


def _load_sort(ks: Path) -> dict:
    import pandas as pd

    st = np.load(ks / "spike_times.npy").ravel().astype(np.int64)
    sc = np.load(ks / "spike_clusters.npy").ravel().astype(np.int64)
    templates = np.load(ks / "templates.npy")
    pos = np.load(ks / "channel_positions.npy")
    tpl_ids = np.arange(templates.shape[0])
    ptp = templates.max(axis=1) - templates.min(axis=1)
    peak = ptp.argmax(axis=1)
    ks_label = pd.read_csv(ks / "cluster_KSLabel.tsv", sep="\t").set_index("cluster_id")["KSLabel"]
    # Clusters after postprocessing map 1:1 to templates when no merges happened;
    # fall back to the dominant template per cluster otherwise.
    spike_templates = np.load(ks / "spike_templates.npy").ravel()
    units = {}
    for cid in np.unique(sc):
        mask = sc == cid
        tid = int(np.bincount(spike_templates[mask]).argmax())
        units[int(cid)] = {
            "times": np.sort(st[mask]),
            "x": float(pos[peak[tid], 0]),
            "y": float(pos[peak[tid], 1]),
            "ks_good": ks_label.get(int(cid), "") == "good",
        }
    amp = {}
    wm = ks / "waveform_metrics.csv"
    if wm.exists():
        df = pd.read_csv(wm)
        amp = dict(zip(df["cluster_id"].astype(int), df["amplitude"].astype(float)))
    bc_good = set()
    for cand in (ks / "cluster_bc_unitType.tsv", ks / "bombcell" / "cluster_bc_unitType.tsv"):
        if cand.exists():
            df = pd.read_csv(cand, sep="\t")
            col = [c for c in df.columns if c != "cluster_id"][0]
            bc_good = set(df.loc[df[col].astype(str).str.upper() == "GOOD", "cluster_id"].astype(int))
            break
    for cid, u in units.items():
        u["amp_uv"] = amp.get(cid, np.nan)
        u["bc_good"] = cid in bc_good
    return units


def _agreement(a: np.ndarray, b: np.ndarray, tol: int = 12) -> float:
    """SpikeInterface-style agreement: matches / (na + nb - matches), +-0.4 ms."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    idx = np.searchsorted(b, a)
    left = np.abs(a - b[np.clip(idx - 1, 0, len(b) - 1)])
    right = np.abs(b[np.clip(idx, 0, len(b) - 1)] - a)
    matches = int(np.sum(np.minimum(left, right) <= tol))
    return matches / (len(a) + len(b) - matches)


def match_sorts(ref: dict, other: dict, radius_um: float = 100.0):
    """Hungarian matching on agreement; only units within ``radius_um`` are compared."""
    from scipy.optimize import linear_sum_assignment

    r_ids, o_ids = list(ref), list(other)
    M = np.zeros((len(r_ids), len(o_ids)))
    o_xy = np.array([[other[o]["x"], other[o]["y"]] for o in o_ids])
    for i, rid in enumerate(r_ids):
        ru = ref[rid]
        near = np.flatnonzero(np.hypot(o_xy[:, 0] - ru["x"], o_xy[:, 1] - ru["y"]) <= radius_um)
        for j in near:
            M[i, j] = _agreement(ru["times"], other[o_ids[j]]["times"])
    rows, cols = linear_sum_assignment(-M)
    pairs = [(r_ids[i], o_ids[j], M[i, j]) for i, j in zip(rows, cols) if M[i, j] > 0]
    best = M.max(axis=1) if M.size else np.zeros(len(r_ids))
    return pairs, best


def _lin_ccc(x: np.ndarray, y: np.ndarray) -> float:
    mx, my = x.mean(), y.mean()
    return float(2 * np.cov(x, y, bias=True)[0, 1] / (x.var() + y.var() + (mx - my) ** 2))


def analyze(out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy import stats

    info = json.loads((out / "prepare.json").read_text())
    arms = ("sglx_a", "sglx_b", "sglx_dither", "oe")
    res = {arm: json.loads((out / "runs" / arm / "result.json").read_text()) for arm in arms}
    sorts = {arm: _load_sort(Path(r["ks_folder"])) for arm, r in res.items()}
    ref = sorts["sglx_a"]
    comps = {}
    for arm in ("sglx_b", "sglx_dither", "oe"):
        pairs, best = match_sorts(ref, sorts[arm])
        comps[arm] = {"pairs": pairs, "best": best}

    # --- TTL / sync timing: CatGT imec SY extraction vs Open Ephys events CSV.
    xd = sorted(Path(res["sglx_a"]["ks_folder"]).parent.glob("*.ap.xd_384_6_500.txt"))
    sglx_edges = np.loadtxt(xd[0]) if xd else np.array([])
    oe_csv = next(Path(res["oe"]["ks_folder"]).parent.glob("openephys_events.csv"), None)
    oe_edges = np.array([])
    if oe_csv is not None:
        ev = pd.read_csv(oe_csv)
        oe_edges = ev.loc[ev.edge == "rising", "time_s"].to_numpy()
    k = min(len(sglx_edges), len(oe_edges))
    sync_err_us = (oe_edges[:k] - sglx_edges[:k]) * 1e6

    # --- Summary table.
    rows = []
    for arm, units in sorts.items():
        rows.append(
            {
                "arm": arm,
                "units": len(units),
                "ks_good": sum(u["ks_good"] for u in units.values()),
                "bombcell_good": sum(u["bc_good"] for u in units.values()),
                "runtime_s": round(res[arm]["seconds"], 1),
            }
        )
    summary = pd.DataFrame(rows)
    stats_out: dict = {"summary": rows, "prepare": info}
    for arm, c in comps.items():
        ag = np.array([p[2] for p in c["pairs"]])
        matched = ag >= 0.5
        amp_pairs = np.array(
            [(ref[r]["amp_uv"], sorts[arm][o]["amp_uv"]) for r, o, a in c["pairs"] if a >= 0.5]
        )
        amp_pairs = amp_pairs[np.isfinite(amp_pairs).all(axis=1)] if amp_pairs.size else amp_pairs.reshape(0, 2)
        dy = np.array([sorts[arm][o]["y"] - ref[r]["y"] for r, o, a in c["pairs"] if a >= 0.5])
        entry = {
            "matched_fraction_of_ref": float(np.mean(c["best"] >= 0.5)),
            "matched_good_fraction_of_ref_good": float(
                np.mean([b >= 0.5 for b, rid in zip(c["best"], ref) if ref[rid]["ks_good"]] or [np.nan])
            ),
            "median_agreement_matched": float(np.median(ag[matched])) if matched.any() else float("nan"),
            "n_matched": int(matched.sum()),
            "depth_abs_error_um_median": float(np.median(np.abs(dy))) if dy.size else float("nan"),
        }
        if len(amp_pairs) > 2:
            log_ratio = np.log(amp_pairs[:, 1] / amp_pairs[:, 0])
            w = stats.wilcoxon(log_ratio)
            entry.update(
                {
                    "amp_ccc": _lin_ccc(amp_pairs[:, 0], amp_pairs[:, 1]),
                    "amp_median_ratio": float(np.exp(np.median(log_ratio))),
                    "amp_wilcoxon_p": float(w.pvalue),
                    "amp_pearson_r": float(stats.pearsonr(amp_pairs[:, 0], amp_pairs[:, 1])[0]),
                }
            )
        stats_out[arm] = entry
    # Is OE-vs-A agreement different from the dither noise floor? (unpaired, two-sided;
    # A-vs-B is not used: Kilosort4 is deterministic, so that floor is trivially perfect.)
    mw = stats.mannwhitneyu(comps["oe"]["best"], comps["sglx_dither"]["best"], alternative="two-sided")
    stats_out["agreement_mannwhitney_oe_vs_dither"] = {"U": float(mw.statistic), "p": float(mw.pvalue)}
    # Paired view: per reference unit, is it recovered (>=0.5) by dither / by OE? McNemar exact.
    rec_d = comps["sglx_dither"]["best"] >= 0.5
    rec_o = comps["oe"]["best"] >= 0.5
    b01, b10 = int(np.sum(rec_d & ~rec_o)), int(np.sum(~rec_d & rec_o))
    mcn = stats.binomtest(min(b01, b10), b01 + b10, 0.5).pvalue if (b01 + b10) else 1.0
    stats_out["recovery_mcnemar_dither_vs_oe"] = {"only_dither": b01, "only_oe": b10, "p": float(mcn)}
    if k:
        stats_out["sync"] = {
            "n_edges": int(k),
            "median_abs_error_us": float(np.median(np.abs(sync_err_us))),
            "max_abs_error_us": float(np.max(np.abs(sync_err_us))),
        }
    (out / "benchmark_stats.json").write_text(json.dumps(stats_out, indent=2, default=float))
    summary.to_csv(out / "benchmark_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(json.dumps({k2: v for k2, v in stats_out.items() if k2 not in {"summary", "prepare"}}, indent=2, default=float))

    # --- Figure.
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    c_ref, c_b, c_oe = "#4C72B0", "#8C8C8C", "#DD8452"
    fig, ax = plt.subplots(2, 3, figsize=(13, 7.6), constrained_layout=True)
    fig.suptitle(
        f"Open Ephys staging vs native SpikeGLX - {info['probe']}, {info['dur_s']:.0f} s real recording "
        f"(OE twin: 0.195 uV/bit, +{info['injected_offset_s']} s offset, {info['injected_drift'] * 1e6:.0f} ppm drift)",
        fontsize=10.5,
    )
    # (a) unit counts
    a = ax[0, 0]
    labels = ["all units", "KS good", "BombCell good"]
    keys = ["units", "ks_good", "bombcell_good"]
    w = 0.2
    c_d = "#55A868"
    for i, (arm, col, name) in enumerate((("sglx_a", c_ref, "SpikeGLX A"), ("sglx_b", c_b, "SpikeGLX B (rerun)"),
                                          ("sglx_dither", c_d, "SpikeGLX +-1 count dither"), ("oe", c_oe, "Open Ephys"))):
        vals = [int(summary.loc[summary.arm == arm, kk].iloc[0]) for kk in keys]
        bars = a.bar(np.arange(3) + (i - 1.5) * w, vals, w, color=col, label=name)
        a.bar_label(bars, fontsize=7, padding=1)
    a.set_xticks(range(3), labels)
    a.set_ylabel("count")
    a.set_title("a  Units per arm", loc="left", fontweight="bold")
    a.legend(frameon=False, fontsize=7.5)
    # (b) best agreement per reference unit
    a = ax[0, 1]
    bins = np.linspace(0, 1, 21)
    a.hist(comps["sglx_dither"]["best"], bins, color=c_d, alpha=0.6, label="SpikeGLX +-1 dither vs A")
    a.hist(comps["oe"]["best"], bins, histtype="step", lw=2, color=c_oe, label="Open Ephys vs A")
    a.axvline(0.5, ls="--", color="k", lw=0.8)
    s_b, s_d, s_o = stats_out["sglx_b"], stats_out["sglx_dither"], stats_out["oe"]
    mc = stats_out["recovery_mcnemar_dither_vs_oe"]
    a.text(
        0.02, 0.97,
        f"recovered (>=0.5): dither {s_d['matched_fraction_of_ref']:.1%} | OE {s_o['matched_fraction_of_ref']:.1%}\n"
        f"rerun B {s_b['matched_fraction_of_ref']:.0%} (KS4 deterministic)\n"
        f"Mann-Whitney p = {mw.pvalue:.3g}; McNemar p = {mc['p']:.3g}",
        transform=a.transAxes, va="top", fontsize=7.5,
    )
    a.set_xlabel("best spike-train agreement with SpikeGLX A")
    a.set_ylabel("reference units")
    a.set_title("b  Unit recovery vs perturbation floor", loc="left", fontweight="bold")
    a.legend(frameon=False, fontsize=7.5, loc="center left")

    # (c, d) amplitude in uV
    for a, arm, col, title in ((ax[0, 2], "oe", c_oe, "c  Amplitude: Open Ephys vs A"), (ax[1, 0], "sglx_dither", c_d, "d  Amplitude: dither vs A")):
        pairs = np.array([(ref[r]["amp_uv"], sorts[arm][o]["amp_uv"]) for r, o, g in comps[arm]["pairs"] if g >= 0.5])
        if pairs.size:
            pairs = pairs[np.isfinite(pairs).all(axis=1)]
            a.scatter(pairs[:, 0], pairs[:, 1], s=9, color=col, alpha=0.7, edgecolor="none")
            lim = [0, float(np.nanmax(pairs)) * 1.05]
            a.plot(lim, lim, "k--", lw=0.8)
            a.set_xlim(lim); a.set_ylim(lim)
            e = stats_out[arm]
            a.text(
                0.03, 0.97,
                f"n = {len(pairs)}\nLin CCC = {e.get('amp_ccc', float('nan')):.3f}\n"
                f"median ratio = {e.get('amp_median_ratio', float('nan')):.3f}\n"
                f"Wilcoxon(log ratio) p = {e.get('amp_wilcoxon_p', float('nan')):.3g}",
                transform=a.transAxes, va="top", fontsize=7.5,
            )
        a.set_xlabel("SpikeGLX A amplitude (uV)")
        a.set_ylabel(f"{'Open Ephys' if arm == 'oe' else 'dithered SpikeGLX'} amplitude (uV)")
        a.set_title(title, loc="left", fontweight="bold")
        a.set_aspect("equal", adjustable="box")

    # (e) depth error
    a = ax[1, 1]
    for arm, col, name in (("sglx_dither", c_d, "dither"), ("oe", c_oe, "Open Ephys")):
        dy = np.array([sorts[arm][o]["y"] - ref[r]["y"] for r, o, g in comps[arm]["pairs"] if g >= 0.5])
        a.hist(dy, bins=np.arange(-62.5, 63, 15), alpha=0.6 if arm == "sglx_dither" else 1.0,
               histtype="bar" if arm == "sglx_dither" else "step", lw=2, color=col,
               label=f"{name}: median |dy| = {np.median(np.abs(dy)) if dy.size else float('nan'):.0f} um")
    a.set_xlabel("peak-channel depth difference vs A (um)")
    a.set_ylabel("matched units")
    a.set_title("e  Geometry: unit depth", loc="left", fontweight="bold")
    a.legend(frameon=False, fontsize=7.5)

    # (f) sync timing
    a = ax[1, 2]
    if k:
        a.plot(sglx_edges[:k], sync_err_us, "o", ms=3, color=c_oe)
        a.axhline(0, color="k", lw=0.8)
        a.axhspan(-1e6 / FS / 2, 1e6 / FS / 2, color="0.9", zorder=0, label="+-0.5 sample")
        st = stats_out["sync"]
        a.text(0.03, 0.97, f"{k} sync edges\nmedian |err| = {st['median_abs_error_us']:.2f} us\n"
               f"max |err| = {st['max_abs_error_us']:.2f} us", transform=a.transAxes, va="top", fontsize=7.5)
        a.legend(frameon=False, fontsize=7.5, loc="lower right")
    a.set_xlabel("time in recording (s)")
    a.set_ylabel("OE TTL - CatGT SY edge (us)")
    a.set_title("f  Event timing on the AP clock", loc="left", fontweight="bold")

    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"openephys_benchmark.{ext}", dpi=200)
    print("figures:", out / "openephys_benchmark.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--start", type=float, default=200.0)
    p.add_argument("--dur", type=float, default=120.0)
    p.add_argument("--drift", type=float, default=50e-6)
    p.add_argument("--offset", type=float, default=37.2)
    p = sub.add_parser("run")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arms", nargs="+", default=["sglx_a", "sglx_b", "sglx_dither", "oe"])
    p = sub.add_parser("analyze")
    p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args.source, args.out, args.start, args.dur, args.drift, args.offset)
    elif args.cmd == "run":
        for arm in args.arms:
            run_arm(args.out, arm)
    else:
        analyze(args.out)


if __name__ == "__main__":
    main()

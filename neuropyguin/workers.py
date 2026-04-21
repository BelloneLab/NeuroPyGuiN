from __future__ import annotations

import ast
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PySide6 import QtCore
from .ecephys_runtime import ecephys_subprocess_env, ensure_ecephys_on_sys_path
from .ks_output_resolver import archive_output_dir, find_kilosort_output_dir, has_kilosort_output


def _safe_emit(signal, *args) -> None:
    try:
        signal.emit(*args)
    except RuntimeError:
        # Receiver/source may already be deleted during shutdown/tab switches.
        pass


@dataclass
class PipelineStep:
    name: str
    enabled: bool
    command_template: str


class WorkerSignals(QtCore.QObject):
    log = QtCore.Signal(str)
    progress = QtCore.Signal(int)
    finished = QtCore.Signal(dict)
    error = QtCore.Signal(str)
    stepStarted = QtCore.Signal(str, str)
    stepProgress = QtCore.Signal(str, int)
    stepFinished = QtCore.Signal(str, bool)


class PipelineWorker(QtCore.QRunnable):
    def __init__(self, job: Dict[str, str], steps: List[PipelineStep], placeholders: Dict[str, str]) -> None:
        super().__init__()
        self.job = job
        self.steps = steps
        self.placeholders = placeholders
        self.signals = WorkerSignals()

    @QtCore.Slot()
    def run(self) -> None:
        completed = 0
        enabled_steps = [s for s in self.steps if s.enabled]
        total = max(len(enabled_steps), 1)

        for step in self.steps:
            if not step.enabled:
                continue
            cmd = step.command_template.format(**self.placeholders)
            _safe_emit(self.signals.log, f"[{self.job['name']}] {step.name}: {cmd}")
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=self.job["workdir"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    shell=True,
                    bufsize=1,
                )
            except Exception as exc:
                _safe_emit(self.signals.error, f"{self.job['name']} failed to start {step.name}: {exc}")
                _safe_emit(self.signals.finished, {"job": self.job["name"], "ok": False})
                return

            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    _safe_emit(self.signals.log, line)

            rc = proc.wait()
            if rc != 0:
                _safe_emit(self.signals.error, f"{self.job['name']} {step.name} failed (exit={rc})")
                _safe_emit(self.signals.finished, {"job": self.job["name"], "ok": False})
                return

            completed += 1
            _safe_emit(self.signals.progress, int(completed * 100 / total))

        _safe_emit(self.signals.finished, {"job": self.job["name"], "ok": True})


class FunctionWorker(QtCore.QRunnable):
    def __init__(self, fn, *args, **kwargs) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @QtCore.Slot()
    def run(self) -> None:
        try:
            out = self.fn(*self.args, **self.kwargs)
        except Exception as exc:
            _safe_emit(self.signals.error, str(exc))
            _safe_emit(self.signals.finished, {"ok": False})
            return
        _safe_emit(self.signals.finished, {"ok": True, "result": out})


@dataclass
class EcephysPipelineConfig:
    output_root: str
    json_root: str
    mirror_raw_hierarchy_output: bool
    save_catgt_ap_bin: bool
    run_catgt: bool
    run_catgt_extract_only: bool
    run_tprime: bool
    run_kilosort: bool
    run_kilosort_postproc: bool
    run_noise_templates: bool
    run_mean_waveforms: bool
    run_quality_metrics: bool
    run_pybombcell: bool
    ks_ver: str
    gate_string: str
    trigger_string: str
    probe_string: str
    region_name: str
    ni_extract_string: str
    catgt_cmd_string: str
    sync_period: float
    tostream_sync_params: str
    ks_th: str
    qm_isi_thresh: float
    catgt_car_mode: str
    catgt_loccar_min_um: float
    catgt_loccar_max_um: float
    ks4_duplicate_spike_ms: float
    ks4_min_template_size_um: float
    c_waves_snr_um: float
    ks4_advanced_params: Dict[str, object]
    catgt_path: str
    tprime_path: str
    cwaves_path: str
    ks4_repo_path: str
    kilosort_output_tmp: str


class EcephysPipelineWorker(QtCore.QRunnable):
    def __init__(self, job: Dict[str, str], cfg: EcephysPipelineConfig) -> None:
        super().__init__()
        self.job = job
        self.cfg = cfg
        self.signals = WorkerSignals()
        self._active_step_key: str | None = None
        self._step_progress_cache: Dict[str, int] = {}

    def _begin_step(self, step_key: str, label: str) -> None:
        self._active_step_key = step_key
        self._step_progress_cache.pop(step_key, None)
        _safe_emit(self.signals.stepStarted, step_key, label)

    def _finish_step(self, step_key: str, ok: bool) -> None:
        if ok:
            _safe_emit(self.signals.stepProgress, step_key, 100)
        _safe_emit(self.signals.stepFinished, step_key, ok)
        if self._active_step_key == step_key:
            self._active_step_key = None

    def _emit_step_progress_from_line(self, line: str) -> None:
        step_key = self._active_step_key
        if not step_key:
            return
        matches = re.findall(r"(\d{1,3})%", str(line))
        if not matches:
            return
        try:
            percent = int(matches[-1])
        except Exception:
            return
        percent = max(0, min(100, percent))
        if self._step_progress_cache.get(step_key) == percent:
            return
        self._step_progress_cache[step_key] = percent
        _safe_emit(self.signals.stepProgress, step_key, percent)

    def _run_module(self, module_name: str, input_json: Path, output_json: Path, cwd: str) -> List[str]:
        cmd: Sequence[str] = (
            sys.executable,
            "-W",
            "ignore",
            "-m",
            f"ecephys_spike_sorting.modules.{module_name}",
            "--input_json",
            str(input_json),
            "--output_json",
            str(output_json),
        )
        lines: List[str] = []
        _safe_emit(self.signals.log, f"[{self.job['name']}] Running {module_name}")
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=ecephys_subprocess_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                lines.append(line)
                _safe_emit(self.signals.log, line)
                self._emit_step_progress_from_line(line)
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"{module_name} exited with code {rc}")
        return lines

    @staticmethod
    def _apply_ks4_overrides(input_json: Path, overrides: Dict[str, object]) -> None:
        if not overrides or not input_json.exists():
            return
        import json

        with input_json.open("r", encoding="utf-8") as f:
            data = json.load(f)
        helper = data.setdefault("ks4_helper_params", {})
        params = helper.setdefault("ks4_params", {})
        for k, v in overrides.items():
            if v is None:
                continue
            params[str(k)] = v
        with input_json.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def _meta_for_bin(bin_path: Path) -> Path:
        candidate = Path(str(bin_path).replace(".ap.bin", ".ap.meta"))
        if candidate.exists():
            return candidate
        return bin_path.with_suffix(".meta")

    @staticmethod
    def _trial_triggers(trigger_string: str) -> List[str]:
        vals: List[str] = []
        base = str(trigger_string).strip()
        if base:
            vals.append(base)
        if "," in base:
            first = base.split(",", 1)[0].strip()
            if first:
                vals.append(first)
        else:
            vals.append(f"{base},{base}" if base else "0,0")
        vals.append("0,0")
        vals.append("0")
        out: List[str] = []
        for v in vals:
            if v and v not in out:
                out.append(v)
        return out

    @staticmethod
    def _trial_run_names(run_name: str, bin_file: Path) -> List[str]:
        vals: List[str] = [run_name]
        stem = bin_file.name
        if stem.lower().endswith(".bin"):
            stem = stem[:-4]
        m = re.match(r"(?P<run>.+)_g\d+_t\d+\.imec\d+\.ap$", stem, re.IGNORECASE)
        if m:
            vals.append(m.group("run"))
        for p in [bin_file.parent, bin_file.parent.parent]:
            name = p.name
            if name:
                vals.append(re.sub(r"_g\d+$", "", name))
        out: List[str] = []
        for v in vals:
            vv = str(v).strip()
            if vv and vv not in out:
                out.append(vv)
        return out

    @staticmethod
    def _parse_bin_parts(bin_file: Path) -> Dict[str, str]:
        stem = bin_file.name
        if stem.lower().endswith(".bin"):
            stem = stem[:-4]
        m = re.match(r"(?P<run>.+)_g(?P<gate>\d+)_t(?P<trig>\d+)\.imec(?P<probe>\d+)\.ap$", stem, re.IGNORECASE)
        if not m:
            return {}
        return {
            "run": m.group("run"),
            "gate": m.group("gate"),
            "trig": m.group("trig"),
            "probe": m.group("probe"),
        }

    @staticmethod
    def _trial_dirs(bin_file: Path) -> List[Path]:
        vals = [bin_file.parent]
        cur = bin_file.parent
        for _ in range(5):
            parent = cur.parent
            if parent == cur:
                break
            vals.append(parent)
            cur = parent
        out: List[Path] = []
        for v in vals:
            if v.exists() and v.is_dir() and v not in out:
                out.append(v)
        return out

    @staticmethod
    def _meta_exists_for_trial(npx_directory: str, run_name: str, gate_string: str, trigger_string: str, probe_string: str) -> bool:
        g = str(gate_string).strip()
        t = str(trigger_string).strip().split(",", 1)[0].strip()
        p = str(probe_string).strip()
        if not g or not t or not p or not run_name:
            return False
        base = Path(npx_directory)
        meta = base / f"{run_name}_g{g}" / f"{run_name}_g{g}_imec{p}" / f"{run_name}_g{g}_t{t}.imec{p}.ap.meta"
        return meta.exists()

    def _build_catgt_trials(self, run_name: str, bin_file: Path, gate_string: str, trigger_string: str, probe_string: str) -> List[Dict[str, str]]:
        dirs = self._trial_dirs(bin_file)
        parts = self._parse_bin_parts(bin_file)
        # Prefer SpikeGLX-standard root folder first: .../<session_root>
        pref_dirs: List[Path] = []
        for d in [bin_file.parent.parent.parent, bin_file.parent.parent, bin_file.parent]:
            if d.exists() and d.is_dir() and d not in pref_dirs:
                pref_dirs.append(d)
        for d in dirs:
            if d not in pref_dirs:
                pref_dirs.append(d)
        dirs = pref_dirs

        runs = self._trial_run_names(run_name, bin_file)
        if parts.get("run"):
            runs = [parts["run"]] + [r for r in runs if r != parts["run"]]

        triggers = self._trial_triggers(trigger_string)
        if parts.get("trig"):
            t_exact = parts["trig"]
            for t in [f"{t_exact},{t_exact}", t_exact]:
                if t not in triggers:
                    triggers.insert(0, t)

        gates = [str(gate_string).strip(), "0"]
        if parts.get("gate") and parts["gate"] not in gates:
            gates.insert(0, parts["gate"])
        gates = [g for i, g in enumerate(gates) if g and g not in gates[:i]]

        probe_vals = [str(probe_string).strip()]
        if parts.get("probe") and parts["probe"] not in probe_vals:
            probe_vals.insert(0, parts["probe"])
        probe_vals = [p for i, p in enumerate(probe_vals) if p and p not in probe_vals[:i]]

        existing_trials: List[Dict[str, str]] = []
        fallback_trials: List[Dict[str, str]] = []
        for d in dirs:
            for rn in runs:
                for gg in gates:
                    for tr in triggers:
                        for pp in probe_vals:
                            trial = {
                                "npx_directory": str(d),
                                "catgt_run_name": rn,
                                "gate_string": gg,
                                "trigger_string": tr,
                                "probe_string": pp,
                            }
                            if self._meta_exists_for_trial(str(d), rn, gg, tr, pp):
                                existing_trials.append(trial)
                            else:
                                fallback_trials.append(trial)
                            if len(existing_trials) >= 10:
                                return existing_trials[:10]
        if existing_trials:
            return existing_trials[:10]
        return fallback_trials[:10]

    @staticmethod
    def _find_recent_catgt_ap(job_out: Path, trial_start_time: float, probe_string: str) -> Path | None:
        patterns = [f"*.imec{probe_string}.ap.bin", "*.ap.bin"]
        matches: List[Path] = []
        for pat in patterns:
            for p in job_out.rglob(pat):
                if "catgt_" not in str(p).lower():
                    continue
                try:
                    if p.stat().st_mtime >= trial_start_time - 1.0:
                        matches.append(p)
                except Exception:
                    continue
        if not matches:
            return None
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0]

    @staticmethod
    def _find_recent_text_outputs(root: Path, trial_start_time: float) -> List[Path]:
        matches: List[Path] = []
        for p in root.rglob("*.txt"):
            try:
                if p.stat().st_mtime >= trial_start_time - 1.0:
                    matches.append(p)
            except Exception:
                continue
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return matches

    @staticmethod
    def _catgt_run_dir(job_out: Path, run_name: str, gate_string: str, trigger_string: str) -> Path:
        gate = str(gate_string).strip().split(",", 1)[0].strip() or "0"
        base_run_name = str(run_name).strip()
        trigger = str(trigger_string).strip().lower()
        if trigger == "cat" or base_run_name.lower().startswith("catgt_"):
            run_dir_name = f"{base_run_name}_g{gate}"
        else:
            run_dir_name = f"catgt_{base_run_name}_g{gate}"
        return job_out / run_dir_name

    def _run_catgt_extract_only_from_raw(
        self,
        create_input_json_fn,
        run_name: str,
        bin_file: Path,
        input_meta: Path,
        job_out: Path,
        ks_tag: str,
        gate_string: str,
        trigger_string: str,
        probe_string: str,
        json_root: Path,
        catgt_cmd_string: str,
        catgt_stream_string: str,
    ) -> Dict[str, str]:
        trials = self._build_catgt_trials(run_name, bin_file, gate_string, trigger_string, probe_string)
        if not trials:
            raise RuntimeError("No valid CatGT extract-only trial combinations found.")
        catgt_dir = self._normalize_tool_dir(self.cfg.catgt_path, ("CatGT",))
        tprime_dir = self._normalize_tool_dir(self.cfg.tprime_path, ("TPrime",))
        cwaves_dir = self._normalize_tool_dir(self.cfg.cwaves_path, ("C_Waves", "C_Waves_win", "C_Waves-win"))
        last_reason = "unknown failure"
        for idx, trial in enumerate(trials, start=1):
            trial_in = json_root / f"{run_name}_catgt-extract-input-trial{idx}.json"
            trial_out = json_root / f"{run_name}_catgt-extract-output-trial{idx}.json"
            _safe_emit(
                self.signals.log,
                f"[{self.job['name']}] CatGT extract-only trial {idx}/10 "
                f"dir={trial['npx_directory']} run={trial['catgt_run_name']} g={trial['gate_string']} t={trial['trigger_string']}",
            )
            create_input_json_fn(
                str(trial_in),
                npx_directory=trial["npx_directory"],
                continuous_file=str(bin_file),
                input_meta_path=str(input_meta),
                extracted_data_directory=str(job_out),
                # CatGT-only passes do not use the Kilosort output directory, so keep this
                # on an existing folder to avoid creating a stray placeholder sorter folder.
                kilosort_output_directory=str(job_out),
                catGT_run_name=trial["catgt_run_name"],
                gate_string=trial["gate_string"],
                trigger_string=trial["trigger_string"],
                probe_string=trial["probe_string"],
                catGT_stream_string=catgt_stream_string,
                catGT_cmd_string=catgt_cmd_string,
                catGT_car_mode="none",
                catGT_loccar_min_um=self.cfg.catgt_loccar_min_um,
                catGT_loccar_max_um=self.cfg.catgt_loccar_max_um,
                ks_ver=self.cfg.ks_ver,
                ks_output_tag=ks_tag,
                ks4_duplicate_spike_ms=self.cfg.ks4_duplicate_spike_ms,
                ks4_min_template_size_um=self.cfg.ks4_min_template_size_um,
                c_Waves_snr_um=self.cfg.c_waves_snr_um,
                external_catgt_path=str(catgt_dir),
                external_tprime_path=str(tprime_dir),
                external_cwaves_path=str(cwaves_dir),
                external_ks4_repo_path=self.cfg.ks4_repo_path,
                external_kilosort_output_tmp=self.cfg.kilosort_output_tmp,
            )
            trial_start = time.time()
            lines = self._run_module("catGT_helper", trial_in, trial_out, self.job["workdir"])
            catgt_run_dir = self._catgt_run_dir(
                job_out,
                trial["catgt_run_name"],
                trial["gate_string"],
                trial["trigger_string"],
            )
            if catgt_run_dir.is_dir():
                fresh_txt = self._find_recent_text_outputs(catgt_run_dir, trial_start)
                if fresh_txt:
                    preview = ", ".join(p.name for p in fresh_txt[:4])
                    extra = "" if len(fresh_txt) <= 4 else f" (+{len(fresh_txt) - 4} more)"
                    _safe_emit(
                        self.signals.log,
                        f"[{self.job['name']}] Extract-only CatGT outputs: {preview}{extra}",
                    )
                else:
                    _safe_emit(
                        self.signals.log,
                        f"[{self.job['name']}] Extract-only CatGT completed in {catgt_run_dir}",
                    )
                return {
                    "catgt_run_dir": str(catgt_run_dir),
                    "source_run_name": trial["catgt_run_name"],
                    "gate_string": trial["gate_string"],
                    "trigger_string": trial["trigger_string"],
                    "probe_string": trial["probe_string"],
                }
            last_reason = self._catgt_failure_reason(lines, job_out)
            _safe_emit(self.signals.log, f"[{self.job['name']}] CatGT extract-only trial {idx} failed: {last_reason}")
        raise RuntimeError(f"CatGT extract-only failed after {len(trials)} trial(s): {last_reason}")

    @staticmethod
    def _catgt_failure_reason(module_lines: List[str], job_out: Path) -> str:
        merged = "\n".join(module_lines)
        fail_patterns = [
            "meta file not found",
            "error",
            "failed",
            "cannot",
            "no such file",
        ]
        lower = merged.lower()
        for pat in fail_patterns:
            if pat in lower:
                return pat
        logs = sorted(job_out.rglob("*_CatGT.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if logs:
            try:
                txt = logs[0].read_text(encoding="utf-8", errors="ignore").lower()
                for pat in fail_patterns:
                    if pat in txt:
                        return f"{pat} (in {logs[0].name})"
            except Exception:
                pass
        return "no CatGT output AP file produced"

    def _resolve_existing_ks_folder(self, requested: Path, job_out: Path, processing_bin: Path, ks_tag: str, probe_string: str) -> Path | None:
        search_roots = [
            job_out,
            job_out.parent,
            Path(self.cfg.output_root),
            processing_bin.parent,
            processing_bin.parent.parent,
            processing_bin.parent.parent.parent,
        ]
        return find_kilosort_output_dir(
            requested,
            ks_tag=ks_tag,
            probe_string=probe_string,
            extra_roots=search_roots,
            max_depth=4,
        )

    @staticmethod
    def _normalize_tool_dir(raw_path: str, executable_stems: Sequence[str]) -> Path:
        text = str(raw_path).strip().strip('"').strip("'")
        if not text:
            return Path()
        path = Path(text).expanduser()
        if path.is_dir():
            return path
        if path.is_file():
            return path.parent
        for stem in executable_stems:
            candidates = [path / stem, path / f"{stem}.exe", path / f"{stem}.bat", path / f"{stem}.cmd"]
            if any(candidate.exists() for candidate in candidates):
                return path
        return path

    @staticmethod
    def _parse_phy_dat_path(params_path: Path) -> object | None:
        if not params_path.exists():
            return None
        text = params_path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"^dat_path\s*=\s*(.+)$", text, flags=re.MULTILINE)
        if not match:
            return None
        raw = match.group(1).strip()
        try:
            return ast.literal_eval(raw)
        except Exception:
            return raw.strip().strip("'").strip('"')

    @classmethod
    def _resolve_processing_bin_for_ks_folder(cls, ks_folder: Path, fallback_bin: Path) -> Path | None:
        raw_value = cls._parse_phy_dat_path(ks_folder / "params.py")
        raw_candidates: List[str] = []
        if isinstance(raw_value, (list, tuple)):
            raw_candidates.extend(str(value).strip() for value in raw_value if str(value).strip())
        elif raw_value is not None and str(raw_value).strip():
            raw_candidates.append(str(raw_value).strip())

        basenames: List[str] = []
        for candidate in raw_candidates:
            path = Path(candidate).expanduser()
            if not path.is_absolute():
                path = (ks_folder / path).resolve()
            if path.exists():
                return path.resolve()
            if path.name and path.name not in basenames:
                basenames.append(path.name)

        if fallback_bin.name and fallback_bin.name not in basenames:
            basenames.append(fallback_bin.name)

        sibling_bins = sorted(ks_folder.parent.glob("*.ap.bin"))
        for name in basenames:
            direct = ks_folder.parent / name
            if direct.exists():
                return direct.resolve()
        if len(sibling_bins) == 1:
            return sibling_bins[0].resolve()

        search_roots = [ks_folder.parent, ks_folder.parent.parent, ks_folder.parent.parent.parent]
        for root in search_roots:
            if not root.exists():
                continue
            for name in basenames:
                matches = sorted(root.rglob(name))
                if matches:
                    return matches[0].resolve()

        if sibling_bins:
            return sibling_bins[0].resolve()
        return None

    def _run_catgt_with_retries(
        self,
        create_input_json_fn,
        run_name: str,
        bin_file: Path,
        input_meta: Path,
        job_out: Path,
        ks_tag: str,
        gate_string: str,
        trigger_string: str,
        probe_string: str,
        json_root: Path,
        catgt_cmd_string: str,
        catgt_stream_string: str,
    ) -> Tuple[Path, Path]:
        trials = self._build_catgt_trials(run_name, bin_file, gate_string, trigger_string, probe_string)
        if not trials:
            raise RuntimeError("No valid CatGT trial combinations found.")
        catgt_dir = self._normalize_tool_dir(self.cfg.catgt_path, ("CatGT",))
        tprime_dir = self._normalize_tool_dir(self.cfg.tprime_path, ("TPrime",))
        cwaves_dir = self._normalize_tool_dir(self.cfg.cwaves_path, ("C_Waves", "C_Waves_win", "C_Waves-win"))
        last_reason = "unknown failure"
        for idx, trial in enumerate(trials, start=1):
            trial_in = json_root / f"{run_name}_catgt-input-trial{idx}.json"
            trial_out = json_root / f"{run_name}_catgt-output-trial{idx}.json"
            _safe_emit(
                self.signals.log,
                f"[{self.job['name']}] CatGT trial {idx}/10 "
                f"dir={trial['npx_directory']} run={trial['catgt_run_name']} g={trial['gate_string']} t={trial['trigger_string']}"
            )
            create_input_json_fn(
                str(trial_in),
                npx_directory=trial["npx_directory"],
                continuous_file=str(bin_file),
                input_meta_path=str(input_meta),
                extracted_data_directory=str(job_out),
                # CatGT itself does not need a sorter output directory; reusing the existing
                # extracted-data root avoids creating a misleading bare "ks4" folder.
                kilosort_output_directory=str(job_out),
                catGT_run_name=trial["catgt_run_name"],
                gate_string=trial["gate_string"],
                trigger_string=trial["trigger_string"],
                probe_string=trial["probe_string"],
                catGT_stream_string=catgt_stream_string,
                catGT_cmd_string=catgt_cmd_string,
                catGT_car_mode=self.cfg.catgt_car_mode,
                catGT_loccar_min_um=self.cfg.catgt_loccar_min_um,
                catGT_loccar_max_um=self.cfg.catgt_loccar_max_um,
                ks_ver=self.cfg.ks_ver,
                ks_output_tag=ks_tag,
                ks4_duplicate_spike_ms=self.cfg.ks4_duplicate_spike_ms,
                ks4_min_template_size_um=self.cfg.ks4_min_template_size_um,
                c_Waves_snr_um=self.cfg.c_waves_snr_um,
                external_catgt_path=str(catgt_dir),
                external_tprime_path=str(tprime_dir),
                external_cwaves_path=str(cwaves_dir),
                external_ks4_repo_path=self.cfg.ks4_repo_path,
                external_kilosort_output_tmp=self.cfg.kilosort_output_tmp,
            )
            trial_start = time.time()
            lines = self._run_module("catGT_helper", trial_in, trial_out, self.job["workdir"])
            catgt_ap = self._find_recent_catgt_ap(job_out, trial_start, probe_string)
            if catgt_ap is not None and catgt_ap.exists():
                meta_path = self._meta_for_bin(catgt_ap)
                if not meta_path.exists():
                    last_reason = f"CatGT produced AP file but missing meta: {meta_path}"
                    _safe_emit(self.signals.log, f"[{self.job['name']}] CatGT trial {idx} failed: {last_reason}")
                    continue
                _safe_emit(self.signals.log, f"[{self.job['name']}] CatGT succeeded on trial {idx}: {catgt_ap}")
                return catgt_ap, meta_path
            last_reason = self._catgt_failure_reason(lines, job_out)
            _safe_emit(self.signals.log, f"[{self.job['name']}] CatGT trial {idx} failed: {last_reason}")
        raise RuntimeError(f"CatGT failed after {len(trials)} trial(s): {last_reason}")

    def _run_catgt_extract_only(
        self,
        create_input_json_fn,
        processing_bin: Path,
        processing_meta: Path,
        extracted_data_root: Path,
        ks_tag: str,
        session_run_name: str,
        gate_string: str,
        probe_string: str,
        json_root: Path,
        catgt_cmd_string: str,
        catgt_stream_string: str,
        catgt_context: Dict[str, str],
    ) -> None:
        catgt_dir = self._normalize_tool_dir(self.cfg.catgt_path, ("CatGT",))
        tprime_dir = self._normalize_tool_dir(self.cfg.tprime_path, ("TPrime",))
        cwaves_dir = self._normalize_tool_dir(self.cfg.cwaves_path, ("C_Waves", "C_Waves_win", "C_Waves-win"))
        extract_in = json_root / f"{session_run_name}_catgt-extract-only-input.json"
        extract_out = json_root / f"{session_run_name}_catgt-extract-only-output.json"
        _safe_emit(
            self.signals.log,
            f"[{self.job['name']}] Running CatGT extract-only pass on {catgt_context['catgt_run_dir']}",
        )
        create_input_json_fn(
            str(extract_in),
            npx_directory=catgt_context["catgt_dest"],
            continuous_file=str(processing_bin),
            input_meta_path=str(processing_meta),
            extracted_data_directory=str(extracted_data_root),
            # Extract-only reruns should not materialize a placeholder sorter folder.
            kilosort_output_directory=str(processing_bin.parent),
            catGT_run_name=catgt_context["catgt_run_name"],
            gate_string=gate_string,
            trigger_string="cat",
            probe_string=probe_string,
            catGT_stream_string=catgt_stream_string,
            catGT_cmd_string=catgt_cmd_string,
            catGT_car_mode="none",
            catGT_loccar_min_um=self.cfg.catgt_loccar_min_um,
            catGT_loccar_max_um=self.cfg.catgt_loccar_max_um,
            ks_ver=self.cfg.ks_ver,
            ks_output_tag=ks_tag,
            ks4_duplicate_spike_ms=self.cfg.ks4_duplicate_spike_ms,
            ks4_min_template_size_um=self.cfg.ks4_min_template_size_um,
            c_Waves_snr_um=self.cfg.c_waves_snr_um,
            external_catgt_path=str(catgt_dir),
            external_tprime_path=str(tprime_dir),
            external_cwaves_path=str(cwaves_dir),
            external_ks4_repo_path=self.cfg.ks4_repo_path,
            external_kilosort_output_tmp=self.cfg.kilosort_output_tmp,
        )
        trial_start = time.time()
        self._run_module("catGT_helper", extract_in, extract_out, self.job["workdir"])
        fresh_txt = self._find_recent_text_outputs(Path(catgt_context["catgt_run_dir"]), trial_start)
        if not fresh_txt:
            raise RuntimeError("CatGT extract-only pass finished without producing any new text outputs.")
        preview = ", ".join(p.name for p in fresh_txt[:4])
        extra = "" if len(fresh_txt) <= 4 else f" (+{len(fresh_txt) - 4} more)"
        _safe_emit(self.signals.log, f"[{self.job['name']}] Extract-only CatGT outputs: {preview}{extra}")

    def _verify_ni_extractor_outputs(
        self,
        *,
        catgt_run_dir: Path,
        source_run_name: str,
        gate_string: str,
        catgt_cmd_string: str,
    ) -> None:
        from .preprocessing import expected_ni_catgt_output_patterns, has_ni_catgt_extractors

        if not has_ni_catgt_extractors(catgt_cmd_string):
            return
        expected = expected_ni_catgt_output_patterns(catgt_cmd_string, source_run_name, gate_string)
        if not expected:
            return

        found: List[str] = []
        missing: List[str] = []
        for pattern in expected:
            matches = sorted(catgt_run_dir.glob(pattern))
            if matches:
                found.append(matches[0].name)
            else:
                missing.append(Path(pattern).name)

        if missing:
            raise RuntimeError(
                "CatGT finished but some expected NI extractor files are missing in "
                f"{catgt_run_dir}: {', '.join(missing)}"
            )

        preview = ", ".join(found[:6])
        extra = "" if len(found) <= 6 else f" (+{len(found) - 6} more)"
        _safe_emit(self.signals.log, f"[{self.job['name']}] Verified NI extractor outputs: {preview}{extra}")

    def _apply_extractor_labels(
        self,
        *,
        catgt_run_dir: Path,
        source_run_name: str,
        gate_string: str,
        ni_extract_string: str,
    ) -> None:
        from .preprocessing import extractor_label_rename_map

        rename_map = extractor_label_rename_map(ni_extract_string, source_run_name, gate_string)
        if not rename_map:
            return
        copied: List[str] = []
        for original_pattern, labelled_name in rename_map.items():
            matches = sorted(catgt_run_dir.glob(original_pattern))
            for src in matches:
                dest = src.parent / labelled_name
                if dest.exists():
                    continue
                try:
                    import shutil
                    shutil.copy2(str(src), str(dest))
                    copied.append(f"{src.name} -> {dest.name}")
                except Exception as exc:
                    _safe_emit(
                        self.signals.log,
                        f"[{self.job['name']}] Warning: could not copy labelled extractor {src.name}: {exc}",
                    )
        if copied:
            _safe_emit(
                self.signals.log,
                f"[{self.job['name']}] Labelled extractor copies: {', '.join(copied)}",
            )

    @QtCore.Slot()
    def run(self) -> None:
        repo = ensure_ecephys_on_sys_path()
        try:
            from ecephys_spike_sorting.scripts.create_input_json import createInputJson
            from .preprocessing import (
                catgt_extract_command_string,
                catgt_extract_stream_selection,
                catgt_stream_string,
                default_kilosort_output_name,
                default_pipeline_output_dir,
                default_pipeline_ks_output_dir,
                default_pipeline_raw_output_layout,
                expected_ni_catgt_output_patterns,
                extractor_label_rename_map,
                is_catgt_processed_bin,
                has_ni_catgt_extractors,
                merge_extractors_into_catgt_command,
                parse_catgt_processed_bin_context,
                resolve_labelled_output_context,
                parse_spikeglx_bin_name,
                validate_spikeglx_ap_bin,
            )
        except Exception as exc:
            _safe_emit(self.signals.error, f"Failed importing ecephys_spike_sorting: {exc}")
            _safe_emit(self.signals.error, f"Expected local repo at: {repo}")
            _safe_emit(self.signals.finished, {"job": self.job["name"], "ok": False})
            return

        try:
            output_root = Path(self.cfg.output_root)
            output_root.mkdir(parents=True, exist_ok=True)
            json_root = Path(self.cfg.json_root)
            json_root.mkdir(parents=True, exist_ok=True)

            run_name = self.job["name"]
            bin_file = Path(self.job["bin_file"])
            ok_ap, reason_ap = validate_spikeglx_ap_bin(str(bin_file))
            if not ok_ap:
                raise RuntimeError(f"Input not processable for ecephys spike sorting: {reason_ap}")
            gate_string = self.job.get("gate_string", self.cfg.gate_string)
            trigger_string = self.job.get("trigger_string", self.cfg.trigger_string)
            probe_string = self.job.get("probe_string", self.cfg.probe_string)

            input_meta = Path(str(bin_file).replace(".ap.bin", ".ap.meta"))
            if not input_meta.exists():
                input_meta = bin_file.with_suffix(".meta")
            if not input_meta.exists():
                raise RuntimeError(f"Missing meta file for {bin_file}: expected {input_meta}")

            ks_map = {"2.0": "ks2", "2.5": "ks25", "3.0": "ks3", "4": "ks4"}
            ks_tag = ks_map.get(self.cfg.ks_ver, "ks4")
            catgt_processed_input = is_catgt_processed_bin(str(bin_file))
            catgt_context = parse_catgt_processed_bin_context(str(bin_file)) if catgt_processed_input else {}
            if catgt_context:
                run_name = catgt_context.get("source_run_name") or run_name
                gate_string = catgt_context.get("gate_string") or gate_string
                trigger_string = catgt_context.get("trigger_string") or trigger_string
                probe_string = catgt_context.get("probe_string") or probe_string

            run_catgt_extract_only = self.cfg.run_catgt_extract_only
            run_catgt_effective = self.cfg.run_catgt and not run_catgt_extract_only and not catgt_processed_input
            catgt_dir = self._normalize_tool_dir(self.cfg.catgt_path, ("CatGT",))
            tprime_dir = self._normalize_tool_dir(self.cfg.tprime_path, ("TPrime",))
            cwaves_dir = self._normalize_tool_dir(self.cfg.cwaves_path, ("C_Waves", "C_Waves_win", "C_Waves-win"))
            effective_catgt_cmd = merge_extractors_into_catgt_command(
                self.cfg.catgt_cmd_string,
                self.cfg.ni_extract_string,
            )
            if effective_catgt_cmd != self.cfg.catgt_cmd_string:
                _safe_emit(
                    self.signals.log,
                    f"[{self.job['name']}] Merged extractor field into CatGT command: {effective_catgt_cmd}",
                )
            full_catgt_stream = catgt_stream_string(effective_catgt_cmd)
            extract_only_catgt_cmd = catgt_extract_command_string(
                effective_catgt_cmd,
                save_ap_bin=False,
            )
            extract_only_stream = catgt_extract_stream_selection(
                effective_catgt_cmd,
                self.cfg.ni_extract_string,
                save_ap_bin=False,
            )
            extract_only_save_ap_cmd = catgt_extract_command_string(
                effective_catgt_cmd,
                save_ap_bin=self.cfg.save_catgt_ap_bin,
            )
            extract_only_save_ap_stream = catgt_extract_stream_selection(
                effective_catgt_cmd,
                self.cfg.ni_extract_string,
                save_ap_bin=self.cfg.save_catgt_ap_bin,
            )
            if has_ni_catgt_extractors(effective_catgt_cmd):
                expected_ni = expected_ni_catgt_output_patterns(effective_catgt_cmd, run_name, gate_string)
                if expected_ni:
                    _safe_emit(self.signals.log, f"[{self.job['name']}] CatGT NI extraction enabled; expected NI files: {', '.join(Path(p).name for p in expected_ni)}")

            if (run_catgt_effective or run_catgt_extract_only) and not catgt_dir.is_dir():
                raise RuntimeError(
                    "CatGT executable dir is invalid. Set it to the folder containing CatGT "
                    f"(current value: {self.cfg.catgt_path})"
                )
            if self.cfg.run_tprime and not tprime_dir.is_dir():
                raise RuntimeError(
                    "TPrime executable dir is invalid. Set it to the folder containing TPrime "
                    f"(current value: {self.cfg.tprime_path})"
                )
            if (self.cfg.run_kilosort_postproc or self.cfg.run_mean_waveforms) and not cwaves_dir.is_dir():
                raise RuntimeError(
                    "C_Waves executable dir is invalid. Set it to the folder containing C_Waves "
                    f"(current value: {self.cfg.cwaves_path})"
                )

            if catgt_processed_input:
                extracted_data_root = Path(catgt_context.get("catgt_dest") or bin_file.parent)
                extracted_data_root.mkdir(parents=True, exist_ok=True)
                ks_folder = default_pipeline_ks_output_dir(
                    str(bin_file),
                    ks_tag,
                    probe_string,
                    output_root=output_root,
                    run_name=run_name,
                    mirror_raw_hierarchy=self.cfg.mirror_raw_hierarchy_output,
                )
                if not self.cfg.run_kilosort:
                    resolved_local_ks = self._resolve_existing_ks_folder(
                        ks_folder,
                        extracted_data_root,
                        bin_file,
                        ks_tag,
                        probe_string,
                    )
                    if resolved_local_ks is not None:
                        ks_folder = resolved_local_ks
                if self.cfg.run_catgt and not run_catgt_extract_only:
                    _safe_emit(
                        self.signals.log,
                        f"[{self.job['name']}] Input already appears CatGT-processed; skipping CatGT and using local KS area near {bin_file.parent}.",
                    )
            else:
                extracted_data_root, ks_folder = default_pipeline_raw_output_layout(
                    str(bin_file),
                    output_root,
                    ks_tag,
                    probe_string,
                    run_name=run_name,
                    mirror_raw_hierarchy=self.cfg.mirror_raw_hierarchy_output,
                )
                extracted_data_root.mkdir(parents=True, exist_ok=True)
                if self.cfg.mirror_raw_hierarchy_output:
                    _safe_emit(
                        self.signals.log,
                        f"[{self.job['name']}] Mirrored raw hierarchy output root: {extracted_data_root}",
                    )

            module_steps: List[Tuple[str, str, str]] = []
            if self.cfg.run_kilosort:
                module_steps.append(
                    (
                        "kilosort",
                        "Kilosort",
                        "ks4_helper" if self.cfg.ks_ver == "4" else "kilosort_helper",
                    )
                )
            if self.cfg.run_kilosort_postproc:
                module_steps.append(("kilosort_postproc", "Kilosort Postprocessing", "kilosort_postprocessing"))
            if self.cfg.run_noise_templates:
                module_steps.append(("noise_templates", "Noise Templates", "noise_templates"))
            if self.cfg.run_mean_waveforms:
                module_steps.append(("mean_waveforms", "Mean Waveforms", "mean_waveforms"))
            if self.cfg.run_quality_metrics:
                module_steps.append(("quality_metrics", "Quality Metrics", "quality_metrics"))

            done = 0
            total = max(len(module_steps) + int(run_catgt_effective or run_catgt_extract_only) + int(self.cfg.run_tprime) + int(self.cfg.run_pybombcell), 1)

            def execute_step(step_key: str, step_label: str, fn) -> None:
                nonlocal done
                self._begin_step(step_key, step_label)
                try:
                    fn()
                except Exception:
                    self._finish_step(step_key, False)
                    raise
                self._finish_step(step_key, True)
                done += 1
                _safe_emit(self.signals.progress, int(done * 100 / total))

            module_in = json_root / f"{run_name}_modules-input.json"
            module_out = json_root / f"{run_name}_modules-output.json"
            tprime_in = json_root / f"{run_name}_tprime-input.json"
            tprime_out = json_root / f"{run_name}_tprime-output.json"

            processing_bin = bin_file
            processing_meta = input_meta

            if run_catgt_effective:
                def _run_catgt_step() -> None:
                    nonlocal processing_bin, processing_meta, catgt_context, ks_folder
                    _safe_emit(self.signals.log, f"[{self.job['name']}] CatGT stream selection: {full_catgt_stream}")
                    processing_bin, processing_meta = self._run_catgt_with_retries(
                        create_input_json_fn=createInputJson,
                        run_name=run_name,
                        bin_file=bin_file,
                        input_meta=input_meta,
                        job_out=extracted_data_root,
                        ks_tag=ks_tag,
                        gate_string=gate_string,
                        trigger_string=trigger_string,
                        probe_string=probe_string,
                        json_root=json_root,
                        catgt_cmd_string=effective_catgt_cmd,
                        catgt_stream_string=full_catgt_stream,
                    )
                    catgt_output_context = parse_catgt_processed_bin_context(str(processing_bin))
                    if catgt_output_context:
                        catgt_context = catgt_output_context
                        self._verify_ni_extractor_outputs(
                            catgt_run_dir=Path(catgt_output_context["catgt_run_dir"]),
                            source_run_name=catgt_output_context["source_run_name"],
                            gate_string=catgt_output_context["gate_string"],
                            catgt_cmd_string=effective_catgt_cmd,
                        )
                        self._apply_extractor_labels(
                            catgt_run_dir=Path(catgt_output_context["catgt_run_dir"]),
                            source_run_name=catgt_output_context["source_run_name"],
                            gate_string=catgt_output_context["gate_string"],
                            ni_extract_string=self.cfg.ni_extract_string,
                        )
                    ks_folder = default_pipeline_ks_output_dir(
                        str(processing_bin),
                        ks_tag,
                        probe_string,
                        output_root=output_root,
                        run_name=run_name,
                        store_next_to_bin=True,
                    )
                    _safe_emit(
                        self.signals.log,
                        f"[{self.job['name']}] Using KS folder next to CatGT output: {ks_folder}",
                    )

                execute_step("catgt", "CatGT", _run_catgt_step)
            elif run_catgt_extract_only:
                def _run_catgt_extract_only_step() -> None:
                    nonlocal catgt_context, processing_bin, processing_meta, ks_folder
                    active_extract_stream = extract_only_save_ap_stream if (self.cfg.save_catgt_ap_bin and not catgt_processed_input) else extract_only_stream
                    _safe_emit(self.signals.log, f"[{self.job['name']}] CatGT extract-only stream selection: {active_extract_stream}")
                    if catgt_processed_input:
                        if "-ni" in extract_only_stream.split():
                            raise RuntimeError(
                                "CatGT NI extract-only requires the raw session input, not a *_tcat.imecX.ap.bin file. "
                                "The NI stream lives at the run root and CatGT output folders usually contain only the "
                                "generated *.nidq.x*/bf* text files, not a reusable *_tcat.nidq.bin. Queue the raw "
                                "*_gX_tY.imecX.ap.bin file for NI extraction or TPrime alignment."
                            )
                        if not catgt_context:
                            raise RuntimeError("CatGT extract-only mode could not resolve the existing CatGT output layout.")
                        self._run_catgt_extract_only(
                            create_input_json_fn=createInputJson,
                            processing_bin=processing_bin,
                            processing_meta=processing_meta,
                            extracted_data_root=extracted_data_root,
                            ks_tag=ks_tag,
                            session_run_name=run_name,
                            gate_string=gate_string,
                            probe_string=probe_string,
                            json_root=json_root,
                            catgt_cmd_string=extract_only_catgt_cmd,
                            catgt_stream_string=extract_only_stream,
                            catgt_context=catgt_context,
                        )
                        self._verify_ni_extractor_outputs(
                            catgt_run_dir=Path(catgt_context["catgt_run_dir"]),
                            source_run_name=catgt_context["source_run_name"],
                            gate_string=catgt_context["gate_string"],
                            catgt_cmd_string=effective_catgt_cmd,
                        )
                        self._apply_extractor_labels(
                            catgt_run_dir=Path(catgt_context["catgt_run_dir"]),
                            source_run_name=catgt_context["source_run_name"],
                            gate_string=catgt_context["gate_string"],
                            ni_extract_string=self.cfg.ni_extract_string,
                        )
                    else:
                        if self.cfg.save_catgt_ap_bin:
                            _safe_emit(
                                self.signals.log,
                                f"[{self.job['name']}] Extract-only AP save enabled; running full CatGT AP processing so a *_tcat.imec{probe_string}.ap.bin is kept in the CatGT folder.",
                            )
                            processing_bin, processing_meta = self._run_catgt_with_retries(
                                create_input_json_fn=createInputJson,
                                run_name=run_name,
                                bin_file=bin_file,
                                input_meta=input_meta,
                                job_out=extracted_data_root,
                                ks_tag=ks_tag,
                                gate_string=gate_string,
                                trigger_string=trigger_string,
                                probe_string=probe_string,
                                json_root=json_root,
                                catgt_cmd_string=extract_only_save_ap_cmd,
                                catgt_stream_string=extract_only_save_ap_stream,
                            )
                            catgt_output_context = parse_catgt_processed_bin_context(str(processing_bin))
                            if catgt_output_context:
                                catgt_context = catgt_output_context
                                self._verify_ni_extractor_outputs(
                                    catgt_run_dir=Path(catgt_output_context["catgt_run_dir"]),
                                    source_run_name=catgt_output_context["source_run_name"],
                                    gate_string=catgt_output_context["gate_string"],
                                    catgt_cmd_string=effective_catgt_cmd,
                                )
                                self._apply_extractor_labels(
                                    catgt_run_dir=Path(catgt_output_context["catgt_run_dir"]),
                                    source_run_name=catgt_output_context["source_run_name"],
                                    gate_string=catgt_output_context["gate_string"],
                                    ni_extract_string=self.cfg.ni_extract_string,
                                )
                            ks_folder = default_pipeline_ks_output_dir(
                                str(processing_bin),
                                ks_tag,
                                probe_string,
                                output_root=output_root,
                                run_name=run_name,
                                store_next_to_bin=True,
                            )
                            _safe_emit(
                                self.signals.log,
                                f"[{self.job['name']}] Saved CatGT AP output: {processing_bin}",
                            )
                        else:
                            _safe_emit(
                                self.signals.log,
                                f"[{self.job['name']}] Running CatGT extract-only from raw AP input: {bin_file}",
                            )
                            extract_context = self._run_catgt_extract_only_from_raw(
                                create_input_json_fn=createInputJson,
                                run_name=run_name,
                                bin_file=bin_file,
                                input_meta=input_meta,
                                job_out=extracted_data_root,
                                ks_tag=ks_tag,
                                gate_string=gate_string,
                                trigger_string=trigger_string,
                                probe_string=probe_string,
                                json_root=json_root,
                                catgt_cmd_string=extract_only_catgt_cmd,
                                catgt_stream_string=extract_only_stream,
                            )
                            if extract_context:
                                catgt_context = extract_context
                                self._verify_ni_extractor_outputs(
                                    catgt_run_dir=Path(extract_context["catgt_run_dir"]),
                                    source_run_name=extract_context["source_run_name"],
                                    gate_string=extract_context["gate_string"],
                                    catgt_cmd_string=effective_catgt_cmd,
                                )
                                self._apply_extractor_labels(
                                    catgt_run_dir=Path(extract_context["catgt_run_dir"]),
                                    source_run_name=extract_context["source_run_name"],
                                    gate_string=extract_context["gate_string"],
                                    ni_extract_string=self.cfg.ni_extract_string,
                                )

                execute_step("catgt_extract_only", "CatGT extract-only", _run_catgt_extract_only_step)

            if self.cfg.run_kilosort:
                archived_ks = archive_output_dir(ks_folder)
                if archived_ks is not None:
                    _safe_emit(
                        self.signals.log,
                        f"[{self.job['name']}] Archived existing KS folder {ks_folder} -> {archived_ks}",
                    )

            needs_existing_ks = (
                self.cfg.run_kilosort_postproc
                or self.cfg.run_noise_templates
                or self.cfg.run_mean_waveforms
                or self.cfg.run_quality_metrics
                or self.cfg.run_tprime
                or self.cfg.run_pybombcell
            ) and not self.cfg.run_kilosort
            if needs_existing_ks:
                requested_ks_folder = ks_folder
                # When mirrored hierarchy is active the default flat ks_folder
                # (output_root/run_name/ks_tag) is far from the real location
                # inside extracted_data_root.  Use the mirrored root as the
                # resolver hint so parent-walking starts in the right area.
                resolver_hint = (
                    extracted_data_root / default_kilosort_output_name(ks_tag, probe_string)
                    if self.cfg.mirror_raw_hierarchy_output
                    else ks_folder
                )
                resolved_ks = self._resolve_existing_ks_folder(
                    resolver_hint,
                    extracted_data_root,
                    processing_bin,
                    ks_tag,
                    probe_string,
                )
                if resolved_ks is None or not has_kilosort_output(resolved_ks):
                    raise RuntimeError(
                        f"No valid {ks_tag} output folder found for downstream processing near "
                        f"{extracted_data_root.parent} or {processing_bin.parent}."
                    )
                ks_folder = resolved_ks
                if ks_folder != requested_ks_folder:
                    _safe_emit(self.signals.log, f"[{self.job['name']}] Resolved existing KS folder: {ks_folder}")
                resolved_processing_bin = self._resolve_processing_bin_for_ks_folder(ks_folder, processing_bin)
                if resolved_processing_bin is not None:
                    expected_identity = parse_spikeglx_bin_name(str(processing_bin))
                    resolved_identity = parse_spikeglx_bin_name(str(resolved_processing_bin))
                    expected_run = str(expected_identity.get("run_name") or run_name)
                    expected_gate = str(expected_identity.get("gate_string") or gate_string)
                    expected_probe = str(expected_identity.get("probe_string") or probe_string)
                    resolved_run = str(resolved_identity.get("run_name") or "")
                    resolved_gate = str(resolved_identity.get("gate_string") or "")
                    resolved_probe = str(resolved_identity.get("probe_string") or "")
                    if (
                        (resolved_run and resolved_run != expected_run)
                        or (resolved_gate and resolved_gate != expected_gate)
                        or (resolved_probe and resolved_probe != expected_probe)
                    ):
                        raise RuntimeError(
                            f"Resolved existing {ks_tag} folder {ks_folder} belongs to "
                            f"{resolved_run or '?'} g{resolved_gate or '?'} imec{resolved_probe or '?'} "
                            f"instead of the queued recording {expected_run} g{expected_gate} imec{expected_probe}."
                        )
                if resolved_processing_bin is not None and resolved_processing_bin != processing_bin:
                    processing_bin = resolved_processing_bin
                    processing_meta = self._meta_for_bin(processing_bin)
                    _safe_emit(
                        self.signals.log,
                        f"[{self.job['name']}] Using AP bin from resolved KS folder: {processing_bin}",
                    )

            createInputJson(
                str(module_in),
                npx_directory=str(processing_bin.parent),
                continuous_file=str(processing_bin),
                input_meta_path=str(processing_meta),
                extracted_data_directory=str(extracted_data_root),
                kilosort_output_directory=str(ks_folder),
                catGT_run_name=run_name,
                gate_string=gate_string,
                trigger_string=trigger_string,
                probe_string=probe_string,
                ks_ver=self.cfg.ks_ver,
                ks_Th=self.cfg.ks_th,
                qm_isi_thresh=self.cfg.qm_isi_thresh,
                ks_output_tag=ks_tag,
                catGT_car_mode=self.cfg.catgt_car_mode,
                catGT_loccar_min_um=self.cfg.catgt_loccar_min_um,
                catGT_loccar_max_um=self.cfg.catgt_loccar_max_um,
                ks4_duplicate_spike_ms=self.cfg.ks4_duplicate_spike_ms,
                ks4_min_template_size_um=self.cfg.ks4_min_template_size_um,
                c_Waves_snr_um=self.cfg.c_waves_snr_um,
                external_catgt_path=str(catgt_dir),
                external_tprime_path=str(tprime_dir),
                external_cwaves_path=str(cwaves_dir),
                external_ks4_repo_path=self.cfg.ks4_repo_path,
                external_kilosort_output_tmp=self.cfg.kilosort_output_tmp,
            )
            if self.cfg.ks_ver == "4":
                self._apply_ks4_overrides(module_in, self.cfg.ks4_advanced_params)

            for step_key, step_label, module_name in module_steps:
                execute_step(
                    step_key,
                    step_label,
                    lambda module_name=module_name: self._run_module(module_name, module_in, module_out, self.job["workdir"]),
                )

            if self.cfg.run_tprime:
                def _run_tprime_step() -> None:
                    createInputJson(
                        str(tprime_in),
                        npx_directory=str(processing_bin.parent),
                        continuous_file=str(processing_bin),
                        input_meta_path=str(processing_meta),
                        extracted_data_directory=str(extracted_data_root),
                        kilosort_output_directory=str(ks_folder),
                        catGT_run_name=run_name,
                        gate_string=gate_string,
                        trigger_string=trigger_string,
                        probe_string=probe_string,
                        tPrime_ni_ex_list=re.sub(r"\[[^\]]*\]", "", self.cfg.ni_extract_string),
                        sync_period=self.cfg.sync_period,
                        toStream_sync_params=self.cfg.tostream_sync_params,
                        ks_output_tag=ks_tag,
                        external_catgt_path=str(catgt_dir),
                        external_tprime_path=str(tprime_dir),
                        external_cwaves_path=str(cwaves_dir),
                        external_ks4_repo_path=self.cfg.ks4_repo_path,
                        external_kilosort_output_tmp=self.cfg.kilosort_output_tmp,
                    )
                    self._run_module("tPrime_helper", tprime_in, tprime_out, self.job["workdir"])
                    tprime_context = resolve_labelled_output_context(str(processing_bin), catgt_context)
                    if tprime_context:
                        self._apply_extractor_labels(
                            catgt_run_dir=Path(tprime_context["catgt_run_dir"]),
                            source_run_name=tprime_context["source_run_name"],
                            gate_string=tprime_context["gate_string"],
                            ni_extract_string=self.cfg.ni_extract_string,
                        )

                execute_step("tprime", "TPrime", _run_tprime_step)

            if self.cfg.run_pybombcell:
                from .pybombcell_integration import run_pybombcell_on_folder

                def _run_pybombcell_step() -> None:
                    ks_folder_str = str(ks_folder)
                    _safe_emit(self.signals.log, f"[{self.job['name']}] Running py_bombcell on {ks_folder_str}")
                    payload = run_pybombcell_on_folder(ks_folder_str, save_plots=True)
                    _safe_emit(
                        self.signals.log,
                        f"[{self.job['name']}] py_bombcell done: units={payload.get('n_units', 'NA')} "
                        f"metrics={payload.get('metrics_csv', '')}"
                    )

                execute_step("pybombcell", "py_bombcell", _run_pybombcell_step)

            _safe_emit(self.signals.finished, {"job": self.job["name"], "ok": True, "ks_folder": str(ks_folder.resolve())})
        except Exception as exc:
            tb = traceback.format_exc()
            if self._active_step_key:
                self._finish_step(self._active_step_key, False)
            _safe_emit(self.signals.error, f"{self.job['name']} failed: {exc}")
            _safe_emit(self.signals.error, tb)
            _safe_emit(self.signals.finished, {"job": self.job["name"], "ok": False})


def ensure_job_dirs(output_root: Path, run_name: str) -> Dict[str, Path]:
    job_root = output_root / run_name
    job_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "root": job_root,
        "catgt_input": job_root / "catgt-input.json",
        "catgt_output": job_root / "catgt-output.json",
        "tprime_input": job_root / "tprime-input.json",
        "tprime_output": job_root / "tprime-output.json",
        "ks_input": job_root / "ks4-input.json",
        "ks_output": job_root / "ks4-output.json",
    }
    return paths

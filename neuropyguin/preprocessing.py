from __future__ import annotations

import json
import math
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple


def discover_bin_files(paths: List[str]) -> List[str]:
    found: List[Path] = []
    ap_pat = re.compile(r".*\.imec\d+\.ap\.bin$", re.IGNORECASE)
    for p in paths:
        path = Path(p)
        if path.is_file() and ap_pat.match(path.name):
            found.append(path)
        elif path.is_dir():
            for fp in path.rglob("*.bin"):
                if ap_pat.match(fp.name):
                    found.append(fp)
    unique = sorted({str(p.resolve()) for p in found})
    return unique


def build_run_name(bin_file: str) -> str:
    p = Path(bin_file)
    stem = p.name
    if stem.lower().endswith(".bin"):
        stem = stem[:-4]
    m = re.match(r"(?P<run>.+)_g(?P<gate>\d+)_t(?P<trig>\d+)\.imec(?P<probe>\d+)\.ap$", stem, re.IGNORECASE)
    if m:
        return m.group("run").replace(" ", "_")
    m = re.match(r"(?P<run>.+)_g(?P<gate>\d+)_tcat\.imec(?P<probe>\d+)\.ap$", stem, re.IGNORECASE)
    if m:
        return m.group("run").replace(" ", "_")
    return p.stem.replace(" ", "_")


def parse_spikeglx_bin_name(bin_file: str) -> Dict[str, str]:
    p = Path(bin_file)
    name = p.name[:-4] if p.name.lower().endswith(".bin") else p.name
    m = re.match(r"(?P<run>.+)_g(?P<gate>\d+)_t(?P<trig>\d+)\.imec(?P<probe>\d+)\.ap$", name, re.IGNORECASE)
    if not m:
        m = re.match(r"(?P<run>.+)_g(?P<gate>\d+)_tcat\.imec(?P<probe>\d+)\.ap$", name, re.IGNORECASE)
    if not m:
        return {
            "run_name": build_run_name(bin_file),
            "gate_string": "0",
            "trigger_string": "0,0",
            "probe_string": "0",
        }
    trig = m.groupdict().get("trig", "cat")
    return {
        "run_name": m.group("run").replace(" ", "_"),
        "gate_string": m.group("gate"),
        "trigger_string": f"{trig},{trig}" if trig.isdigit() else trig,
        "probe_string": m.group("probe"),
    }


def _split_catgt_flags(raw: str) -> List[str]:
    return [part for part in re.split(r"\s+", str(raw).strip()) if part]


def _is_catgt_extractor_flag(token: str) -> bool:
    clean = re.sub(r"\[[^\]]*\]$", "", str(token).strip())
    return bool(re.fullmatch(r"-(xd|xid|xa|xia|bf)=(.+)", clean, flags=re.IGNORECASE))


def strip_catgt_extractor_flags(catgt_command: str) -> str:
    return " ".join(token for token in _split_catgt_flags(catgt_command) if not _is_catgt_extractor_flag(token))


def merge_extractors_into_catgt_command(catgt_command: str, extractor_string: str) -> str:
    base = strip_catgt_extractor_flags(catgt_command)
    clean_extractors = re.sub(r"\[[^\]]*\]", "", str(extractor_string).strip())
    parts = [part for part in [base.strip(), clean_extractors] if part]
    return " ".join(parts)


def _is_ni_catgt_extractor_flag(token: str) -> bool:
    clean = re.sub(r"\[[^\]]*\]$", "", str(token).strip())
    match = re.fullmatch(r"-(xd|xid|xa|xia|bf)=(.+)", clean, flags=re.IGNORECASE)
    if not match:
        return False
    values = [value.strip() for value in match.group(2).split(",")]
    return bool(values) and values[0] == "0"


def strip_ni_catgt_extractor_flags(catgt_command: str) -> str:
    """Drop NI-stream (``js=0``) extractor flags and any bare ``-ni`` selector.

    Used when a run has no nidq stream so CatGT is never asked to read a
    non-existent NI file. CatGT aborts immediately (``Meta file not found
    ...nidq.meta``) if an ``-ni`` extractor is requested but the stream does
    not exist, so probe-only / concatenated runs must have these removed.
    Extractors on other streams (AP ``js=2``, OneBox ``js=1``) are preserved.
    """
    kept: List[str] = []
    for token in _split_catgt_flags(catgt_command):
        if token.lower() == "-ni":
            continue
        if _is_ni_catgt_extractor_flag(token):
            continue
        kept.append(token)
    return " ".join(kept)


def _catgt_extractor_streams(catgt_command: str) -> List[str]:
    streams: List[str] = []
    stream_map = {"0": "ni", "1": "obx", "2": "ap"}
    for token in _split_catgt_flags(catgt_command):
        clean = re.sub(r"\[[^\]]*\]$", "", str(token).strip())
        match = re.fullmatch(r"-(xd|xid|xa|xia|bf)=(.+)", clean, flags=re.IGNORECASE)
        if not match:
            continue
        values = [value.strip() for value in match.group(2).split(",")]
        if not values:
            continue
        stream_name = stream_map.get(values[0])
        if stream_name and stream_name not in streams:
            streams.append(stream_name)
    return streams


def has_ni_catgt_extractors(catgt_command: str) -> bool:
    return any(_is_ni_catgt_extractor_flag(token) for token in _split_catgt_flags(catgt_command))


def _fmt_catgt_output_token(value: str) -> str:
    text = str(value).strip()
    if re.fullmatch(r"-?\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text.replace(".", "p")


def expected_ni_catgt_output_patterns(catgt_command: str, run_name: str, gate_string: str) -> List[str]:
    base = f"{str(run_name).strip()}_g{str(gate_string).strip()}_tcat.nidq."
    patterns: List[str] = []
    for token in _split_catgt_flags(catgt_command):
        clean = re.sub(r"\[[^\]]*\]$", "", str(token).strip())
        match = re.fullmatch(r"-(xd|xid|xa|xia|bf)=(.+)", clean, flags=re.IGNORECASE)
        if not match:
            continue
        mode = match.group(1).lower()
        values = [value.strip() for value in match.group(2).split(",")]
        if not values or values[0] != "0":
            continue
        if mode in {"xd", "xid"} and len(values) >= 5:
            word = "*" if values[2] == "-1" else values[2]
            patterns.append(base + f"{mode}_{word}_{values[3]}_{_fmt_catgt_output_token(values[4])}.txt")
        elif mode in {"xa", "xia"} and len(values) >= 6:
            word = "*" if values[2] == "-1" else values[2]
            patterns.append(base + f"{mode}_{word}_{_fmt_catgt_output_token(values[5])}.txt")
        elif mode == "bf" and len(values) >= 6:
            word = "*" if values[2] == "-1" else values[2]
            start_bit = values[3]
            n_bits = values[4]
            patterns.append(base + f"bfv_{word}_{start_bit}_{n_bits}.txt")
            patterns.append(base + f"bft_{word}_{start_bit}_{n_bits}.txt")
    out: List[str] = []
    for pattern in patterns:
        if pattern not in out:
            out.append(pattern)
    return out


def extractor_label_rename_map(
    ni_extract_string: str, run_name: str, gate_string: str,
) -> Dict[str, str]:
    """Return {original_pattern: label_prefixed_name} for extractors that carry a ``[label]``."""
    base = f"{str(run_name).strip()}_g{str(gate_string).strip()}_tcat.nidq."
    mapping: Dict[str, str] = {}
    for token in _split_catgt_flags(ni_extract_string):
        label = ""
        label_match = re.search(r"\[([^\]]*)\]$", token)
        if label_match:
            label = label_match.group(1).strip()
            token = token[: label_match.start()]
        if not label:
            continue
        match = re.fullmatch(r"-(xd|xid|xa|xia)=(.+)", token, flags=re.IGNORECASE)
        if not match:
            continue
        mode = match.group(1).lower()
        values = [v.strip() for v in match.group(2).split(",")]
        if not values or values[0] != "0":
            continue
        if mode in {"xd", "xid"} and len(values) >= 5:
            word = values[2]
            suffix = f"{mode}_{word}_{values[3]}_{_fmt_catgt_output_token(values[4])}.txt"
        elif mode in {"xa", "xia"} and len(values) >= 6:
            word = values[2]
            suffix = f"{mode}_{word}_{_fmt_catgt_output_token(values[5])}.txt"
        else:
            continue
        original = base + suffix
        safe_label = re.sub(r"[^\w\-.]", "_", label)
        mapping[original] = f"{safe_label}_{base}{suffix}"
        # TPrime adjusted file: stem.adj.txt
        adj_suffix = suffix.replace(".txt", ".adj.txt")
        adj_original = base + adj_suffix
        mapping[adj_original] = f"{safe_label}_{base}{adj_suffix}"
    return mapping


def catgt_stream_string(catgt_command: str, ni_extract_string: str = "", include_ap: bool = True) -> str:
    tokens = _split_catgt_flags(catgt_command)
    parts: List[str] = []
    if include_ap:
        parts.append("-ap")
    if (
        any(token.lower() == "-ni" for token in tokens)
        or any(_is_ni_catgt_extractor_flag(token) for token in tokens)
        or bool(str(ni_extract_string).strip())
    ):
        parts.append("-ni")
    out: List[str] = []
    for part in parts:
        if part not in out:
            out.append(part)
    return " ".join(out)


def catgt_extract_only_stream_string(catgt_command: str, ni_extract_string: str = "") -> str:
    tokens = [token.lower() for token in _split_catgt_flags(catgt_command)]
    streams = _catgt_extractor_streams(catgt_command)
    parts: List[str] = []
    if "-ap" in tokens or "ap" in streams:
        parts.append("-ap")
    if "-ni" in tokens or "ni" in streams or bool(str(ni_extract_string).strip()):
        parts.append("-ni")
    if "-obx" in tokens or "obx" in streams:
        parts.append("-obx")
    out: List[str] = []
    for part in parts:
        if part not in out:
            out.append(part)
    return " ".join(out)


def catgt_extract_command_string(catgt_command: str, *, save_ap_bin: bool = False) -> str:
    if save_ap_bin:
        return str(catgt_command).strip()
    return catgt_extract_only_flags(catgt_command)


def catgt_extract_stream_selection(catgt_command: str, ni_extract_string: str = "", *, save_ap_bin: bool = False) -> str:
    stream = catgt_extract_only_stream_string(catgt_command, ni_extract_string)
    if not save_ap_bin:
        return stream
    parts = [part for part in str(stream).split() if part.strip()]
    lowered = [part.lower() for part in parts]
    if "-ap" not in lowered:
        parts.insert(0, "-ap")
    out: List[str] = []
    for part in parts:
        if part not in out:
            out.append(part)
    return " ".join(out)


def catgt_extract_only_flags(catgt_command: str) -> str:
    keep_exact = {"-prb_fld", "-out_prb_fld", "-prb_miss_ok", "-t_miss_ok", "-no_auto_sync"}
    parts: List[str] = []
    for token in _split_catgt_flags(catgt_command):
        lower = token.lower()
        if lower in keep_exact or _is_catgt_extractor_flag(token):
            parts.append(token)
    if not any(token.lower() == "-prb_fld" for token in parts):
        parts.insert(0, "-prb_fld")
    if not any(token.lower() == "-no_tshift" for token in parts):
        parts.append("-no_tshift")
    return " ".join(parts)


def is_catgt_processed_bin(bin_file: str) -> bool:
    name = Path(bin_file).name.lower()
    return "catgt" in name or "tcat" in name


def _is_kilosort_output_dir_name(name: str) -> bool:
    return bool(re.fullmatch(r"(?:imec\d+_ks\d+|ks(?:2|25|3|4))", str(name).strip(), flags=re.IGNORECASE))


def parse_kilosort_params_dat_path(params_file: str | Path) -> str:
    path = Path(params_file)
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    match = re.search(r"(?m)^\s*dat_path\s*=\s*(?:r|R)?['\"]([^'\"]+)['\"]", text)
    if not match:
        return ""
    raw = match.group(1).strip()
    if not raw:
        return ""
    dat_path = Path(raw.replace("/", "\\")).expanduser()
    if not dat_path.is_absolute():
        dat_path = (path.parent / dat_path).resolve()
    else:
        dat_path = dat_path.resolve()
    return str(dat_path)


def infer_completed_run_name(ks_folder: str | Path) -> str:
    folder = Path(ks_folder)
    for parent in [folder.parent, *folder.parents]:
        match = re.fullmatch(r"catgt_(?P<run>.+)_g\d+", parent.name, flags=re.IGNORECASE)
        if match:
            return match.group("run").replace(" ", "_")
        match = re.fullmatch(r"(?P<run>.+)_g\d+_imec\d+", parent.name, flags=re.IGNORECASE)
        if match:
            return match.group("run").replace(" ", "_")
    return folder.parent.name.replace(" ", "_")


def discover_completed_runs(root_path: str | Path) -> List[Dict[str, str]]:
    root = Path(root_path).expanduser()
    if not root.is_dir():
        return []

    entries: List[Dict[str, str]] = []
    seen: set[str] = set()
    for params_file in sorted(root.rglob("params.py"), key=lambda p: p.stat().st_mtime, reverse=True):
        ks_folder = params_file.parent
        if not _is_kilosort_output_dir_name(ks_folder.name):
            continue
        ks_folder_str = str(ks_folder.resolve())
        if ks_folder_str in seen:
            continue
        seen.add(ks_folder_str)
        bin_file = parse_kilosort_params_dat_path(params_file)
        run_name = ""
        if bin_file:
            run_name = str(parse_spikeglx_bin_name(bin_file).get("run_name") or "").strip()
        if not run_name:
            run_name = infer_completed_run_name(ks_folder)
        finished_at = datetime.fromtimestamp(params_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        entries.append(
            {
                "run_name": run_name,
                "ks_folder": ks_folder_str,
                "bin_file": bin_file,
                "label": f"{run_name} | {ks_folder_str}",
                "finished_at": finished_at,
                "params_file": str(params_file.resolve()),
                "source_root": str(root.resolve()),
            }
        )
    return entries


def completed_run_target_folders(entries: Iterable[Dict[str, object]]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for entry in entries:
        raw = str((entry or {}).get("ks_folder") or "").strip()
        if not raw:
            continue
        key = raw.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(raw)
    return out


def default_kilosort_output_name(ks_tag: str, probe_string: str) -> str:
    probe = str(probe_string).strip()
    tag = str(ks_tag).strip()
    return f"imec{probe}_{tag}" if probe else tag


def default_local_ks_output_dir(bin_file: str, ks_tag: str, probe_string: str) -> Path:
    return Path(bin_file).resolve().parent / default_kilosort_output_name(ks_tag, probe_string)


def _session_root_for_spikeglx_bin(bin_file: str) -> Path:
    path = Path(bin_file).resolve()
    parents = list(path.parents)
    if len(parents) >= 3:
        return parents[2]
    if parents:
        return parents[-1]
    return path.parent


def _relative_session_parts_from_raw_hierarchy(bin_file: str) -> Tuple[str, ...]:
    session_root = _session_root_for_spikeglx_bin(bin_file)
    lowered = [part.lower() for part in session_root.parts]
    try:
        raw_idx = lowered.index("rawdata")
    except ValueError:
        return ()
    return tuple(session_root.parts[raw_idx + 1 :])


def default_pipeline_output_dir(
    bin_file: str,
    output_root: str | Path,
    *,
    run_name: str,
    mirror_raw_hierarchy: bool = False,
) -> Path:
    root = Path(output_root).expanduser()
    if not mirror_raw_hierarchy:
        return root / str(run_name).strip()
    relative_session = _relative_session_parts_from_raw_hierarchy(bin_file)
    if not relative_session:
        return root / str(run_name).strip()
    return root.joinpath(*relative_session, "spike_sorting")


def default_pipeline_raw_output_layout(
    bin_file: str,
    output_root: str | Path,
    ks_tag: str,
    probe_string: str,
    *,
    run_name: str,
    mirror_raw_hierarchy: bool = False,
) -> Tuple[Path, Path]:
    extracted_data_root = default_pipeline_output_dir(
        bin_file,
        output_root,
        run_name=run_name,
        mirror_raw_hierarchy=mirror_raw_hierarchy,
    )
    ks_folder = extracted_data_root / default_kilosort_output_name(ks_tag, probe_string)
    return extracted_data_root, ks_folder


def parse_catgt_processed_bin_context(bin_file: str) -> Dict[str, str]:
    path = Path(bin_file).resolve()
    probe_match = re.search(r"\.imec(?P<probe>\d+)\.ap\.bin$", path.name, flags=re.IGNORECASE)
    probe_string = probe_match.group("probe") if probe_match else ""
    catgt_dir = next((parent for parent in path.parents if parent.name.lower().startswith("catgt_")), None)
    if catgt_dir is None:
        return {}
    match = re.fullmatch(r"catgt_(?P<run>.+)_g(?P<gate>\d+)", catgt_dir.name, flags=re.IGNORECASE)
    if match:
        source_run_name = match.group("run")
        gate_string = match.group("gate")
    else:
        source_run_name = re.sub(r"(?i)^catgt_", "", catgt_dir.name)
        source_run_name = re.sub(r"_g\d+$", "", source_run_name)
        gate_string = "0"
    return {
        "catgt_dest": str(catgt_dir.parent),
        "catgt_run_dir": str(catgt_dir),
        "catgt_run_name": f"catgt_{source_run_name}",
        "source_run_name": source_run_name,
        "gate_string": gate_string,
        "trigger_string": "cat",
        "probe_string": probe_string,
    }


def resolve_labelled_output_context(processing_bin: str, fallback_context: Dict[str, str] | None = None) -> Dict[str, str]:
    resolved = parse_catgt_processed_bin_context(processing_bin)
    if resolved:
        return resolved
    return dict(fallback_context or {})


def default_pipeline_ks_output_dir(
    bin_file: str,
    ks_tag: str,
    probe_string: str,
    *,
    output_root: str | Path,
    run_name: str,
    store_next_to_bin: bool = False,
    mirror_raw_hierarchy: bool = False,
) -> Path:
    if store_next_to_bin or is_catgt_processed_bin(bin_file):
        return default_local_ks_output_dir(bin_file, ks_tag, probe_string)
    _, ks_folder = default_pipeline_raw_output_layout(
        bin_file,
        output_root,
        ks_tag,
        probe_string,
        run_name=run_name,
        mirror_raw_hierarchy=mirror_raw_hierarchy,
    )
    return ks_folder


def write_step_json(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def default_json_payload(bin_file: str, output_dir: str, run_name: str) -> Dict:
    return {
        "run_name": run_name,
        "bin_file": bin_file,
        "input_dir": str(Path(bin_file).parent),
        "output_dir": output_dir,
        "notes": "Generated by NeuroPyGuiN. Extend fields to match your CatGT/TPrime/KS4 module requirements.",
    }


def find_meta_for_bin(bin_file: str) -> Path:
    p = Path(bin_file)
    candidate = Path(str(p).replace(".ap.bin", ".ap.meta"))
    if candidate.exists():
        return candidate
    fallback = p.with_suffix(".meta")
    return fallback


def _read_meta_keyvals(meta_path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not meta_path.exists():
        return out
    for line in meta_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def validate_spikeglx_ap_bin(bin_file: str) -> Tuple[bool, str]:
    meta_path = find_meta_for_bin(bin_file)
    if not meta_path.exists():
        return False, f"Missing meta file next to bin ({meta_path.name})."
    meta = _read_meta_keyvals(meta_path)
    if not meta:
        return False, f"Meta file unreadable or empty: {meta_path}"

    # AP data must be present (not calibration-only sync recordings).
    ap_count = -1
    if "snsApLfSy" in meta:
        parts = [p.strip() for p in meta["snsApLfSy"].split(",")]
        if parts:
            try:
                ap_count = int(parts[0])
            except Exception:
                ap_count = -1

    try:
        n_saved = int(meta.get("nSavedChans", "-1"))
    except Exception:
        n_saved = -1

    chan_map = meta.get("~snsChanMap", "")
    only_sync = ("SY0" in chan_map) and ("AP" not in chan_map.upper())

    if ap_count == 0 or n_saved <= 1 or only_sync:
        return (
            False,
            "File has no AP channels (likely calibration/sync-only recording). "
            f"snsApLfSy={meta.get('snsApLfSy', 'NA')} nSavedChans={meta.get('nSavedChans', 'NA')}",
        )
    return True, ""


# ---------------------------------------------------------------------------
# Multi-session concatenation (joint spike sorting across recordings)
# ---------------------------------------------------------------------------
#
# Fusing several SpikeGLX AP binaries into a single file lets Kilosort detect
# and label the same units across multiple sessions in one sorting pass.  The
# core routine streams each file batch by batch (so memory stays flat even for
# multi-hour recordings), optionally removes the leading shared SVD components
# of the AP block as a denoising step, and writes one combined .bin plus a
# matching .meta and a split-info JSON that records the per-session sample
# boundaries needed to map sorted spikes back onto each original recording.


def spikeglx_meta_channel_info(meta: Dict[str, str]) -> Dict[str, float]:
    """Return channel counts and sample rate parsed from a SpikeGLX meta dict."""
    try:
        n_saved = int(float(meta.get("nSavedChans", "0") or 0))
    except Exception:
        n_saved = 0
    ap = lf = sy = 0
    raw = meta.get("snsApLfSy", "")
    if raw:
        parts = [p.strip() for p in raw.split(",")]
        try:
            ap = int(parts[0]); lf = int(parts[1]); sy = int(parts[2])
        except Exception:
            ap = lf = sy = 0
    sample_rate = 0.0
    for key in ("imSampRate", "niSampRate", "sampRate"):
        value = meta.get(key)
        if value:
            try:
                sample_rate = float(value)
                break
            except Exception:
                continue
    if n_saved <= 0 and (ap or lf or sy):
        n_saved = ap + lf + sy
    return {
        "n_saved": float(n_saved),
        "ap": float(ap),
        "lf": float(lf),
        "sy": float(sy),
        "sample_rate": float(sample_rate),
    }


def validate_concat_inputs(meta_files: Sequence[str]) -> Tuple[bool, str, Dict[str, float]]:
    """Confirm a set of recordings can be concatenated (same channels + rate)."""
    metas = [Path(m) for m in meta_files]
    if len(metas) < 2:
        return False, "Select at least two recordings to concatenate.", {}
    infos: List[Dict[str, float]] = []
    for meta_path in metas:
        meta = _read_meta_keyvals(meta_path)
        if not meta:
            return False, f"Meta file unreadable or empty: {meta_path}", {}
        info = spikeglx_meta_channel_info(meta)
        if info["n_saved"] <= 0:
            return False, f"Could not read channel count from {meta_path.name}", {}
        infos.append(info)
    ref = infos[0]
    for meta_path, info in zip(metas[1:], infos[1:]):
        if int(info["n_saved"]) != int(ref["n_saved"]):
            return (
                False,
                f"Channel-count mismatch: {meta_path.name} has {int(info['n_saved'])} "
                f"channels but the first recording has {int(ref['n_saved'])}.",
                {},
            )
        if ref["sample_rate"] and info["sample_rate"] and abs(info["sample_rate"] - ref["sample_rate"]) > 1e-3:
            return (
                False,
                f"Sample-rate mismatch: {meta_path.name} is {info['sample_rate']} Hz "
                f"but the first recording is {ref['sample_rate']} Hz.",
                {},
            )
    return True, "", ref


def build_concat_run_name(
    run_names: Sequence[str],
    *,
    prefix: str = "concat",
    max_runs: int = 4,
    max_len: int = 100,
) -> str:
    """Build a filesystem-safe combined run name from the source run names."""
    cleaned: List[str] = []
    for name in run_names:
        text = re.sub(r"\s+", "_", str(name).strip())
        text = re.sub(r"[^\w\-.]", "_", text)
        text = text.strip("_")
        if text:
            cleaned.append(text)
    if not cleaned:
        return prefix
    if len(cleaned) > max_runs:
        body = f"{cleaned[0]}__and{len(cleaned) - 1}more"
    else:
        body = "__".join(cleaned)
    combined = f"{prefix}_{body}"
    if len(combined) > max_len:
        combined = f"{prefix}_{cleaned[0]}__and{len(cleaned) - 1}more"
    return combined


def default_concat_run_layout(
    output_dir: str | Path,
    combined_run_name: str,
    probe_string: str,
    *,
    gate: str = "0",
    trigger: str = "0",
) -> Dict[str, Path]:
    """Return SpikeGLX-standard paths for a concatenated run.

    The nested ``<run>_g0/<run>_g0_imec<p>/<run>_g0_t0.imec<p>.ap.bin`` layout
    keeps the fused file compatible with the rest of the pipeline (CatGT trial
    discovery, KS output placement, completed-run parsing) exactly as if it was
    an ordinary SpikeGLX recording.
    """
    base = Path(output_dir).expanduser()
    g = str(gate).strip() or "0"
    t = str(trigger).strip() or "0"
    p = str(probe_string).strip() or "0"
    run_g = f"{combined_run_name}_g{g}"
    probe_dir_name = f"{run_g}_imec{p}"
    probe_folder = base / run_g / probe_dir_name
    stem = f"{run_g}_t{t}.imec{p}.ap"
    return {
        "run_folder": base / run_g,
        "probe_folder": probe_folder,
        "bin": probe_folder / f"{stem}.bin",
        "meta": probe_folder / f"{stem}.meta",
    }


def _concat_meta_path_for_bin(target_bin: Path) -> Path:
    candidate = Path(str(target_bin).replace(".ap.bin", ".ap.meta"))
    if str(candidate) != str(target_bin):
        return candidate
    return target_bin.with_suffix(".meta")


def _concat_splitinfo_path_for_bin(target_bin: Path) -> Path:
    candidate = Path(str(target_bin).replace(".ap.bin", ".ap.splitinfo.json"))
    if str(candidate) != str(target_bin):
        return candidate
    return target_bin.with_suffix(".splitinfo.json")


def concatenate_ap_binaries(
    bin_files: Sequence[str],
    meta_files: Sequence[str],
    target_bin: str | Path,
    *,
    svd_clean: bool = True,
    n_svd_components: int = 5,
    batch_seconds: float = 0.5,
    progress_cb: Callable[[int], None] | None = None,
    log_cb: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> List[int]:
    """Stream-concatenate AP binaries into one file, returning per-file samples.

    When ``svd_clean`` is on, the leading ``n_svd_components`` shared spatial
    components of the AP channel block are removed batch by batch (the same
    denoising the legacy MATLAB ``concatenateAndCleanBinaries`` performed).
    Non-AP channels (for example the sync channel) are passed through unchanged
    so the output keeps the original channel layout and stays a drop-in AP file.
    """
    import numpy as np

    meta0 = _read_meta_keyvals(Path(meta_files[0]))
    info = spikeglx_meta_channel_info(meta0)
    n_chan = int(info["n_saved"])
    if n_chan <= 0:
        raise ValueError("Could not determine channel count from the first meta file.")
    ap_count = int(info["ap"]) or n_chan
    ap_count = min(ap_count, n_chan)
    sample_rate = float(info["sample_rate"]) or 30000.0
    batch_samples = max(1, int(round(batch_seconds * sample_rate)))
    k = max(0, int(n_svd_components))

    samplelist: List[int] = []
    for bin_file in bin_files:
        size_bytes = Path(bin_file).stat().st_size
        samplelist.append(int(size_bytes // (2 * n_chan)))

    total_batches = sum(int(math.ceil(s / batch_samples)) for s in samplelist if s > 0) or 1
    done_batches = 0

    target = Path(target_bin)
    target.parent.mkdir(parents=True, exist_ok=True)

    with open(target, "wb") as fout:
        for file_idx, (bin_file, n_samp) in enumerate(zip(bin_files, samplelist)):
            if log_cb:
                log_cb(
                    f"File {file_idx + 1}/{len(bin_files)}: {Path(bin_file).name} "
                    f"({n_samp} samples, {n_samp / sample_rate:.1f} s)"
                )
            if n_samp <= 0:
                continue
            mm = np.memmap(bin_file, dtype=np.int16, mode="r", shape=(n_samp, n_chan))
            try:
                n_batch = int(math.ceil(n_samp / batch_samples))
                for batch_idx in range(n_batch):
                    if should_cancel is not None and should_cancel():
                        raise RuntimeError("Concatenation cancelled by user.")
                    s0 = batch_idx * batch_samples
                    s1 = min(s0 + batch_samples, n_samp)
                    batch = mm[s0:s1, :]
                    if svd_clean and k > 0 and ap_count > 0 and (s1 - s0) > 1:
                        ap_block = batch[:, :ap_count].astype(np.float32)
                        cov = ap_block.T @ ap_block
                        _evals, evecs = np.linalg.eigh(cov)
                        kk = min(k, evecs.shape[1])
                        top = evecs[:, evecs.shape[1] - kk :]
                        cleaned = ap_block - (ap_block @ top) @ top.T
                        out = np.array(batch, dtype=np.int16)
                        out[:, :ap_count] = np.clip(np.rint(cleaned), -32768, 32767).astype(np.int16)
                        fout.write(out.tobytes())
                    else:
                        fout.write(np.ascontiguousarray(batch).tobytes())
                    done_batches += 1
                    if progress_cb is not None:
                        progress_cb(int(done_batches * 100 / total_batches))
            finally:
                del mm

    return samplelist


def write_concat_meta(
    meta_files: Sequence[str],
    target_bin: str | Path,
    samplelist: Sequence[int],
    *,
    n_out_chans: int | None = None,
) -> Path:
    """Write a combined .meta beside the concatenated bin using the first meta as template."""
    target = Path(target_bin)
    template_path = Path(meta_files[0])
    text = template_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    info = spikeglx_meta_channel_info(_read_meta_keyvals(template_path))
    source_chans = int(info["n_saved"])
    sample_rate = float(info["sample_rate"]) or 30000.0
    n_chan = int(n_out_chans) if n_out_chans else source_chans
    total_samples = int(sum(int(s) for s in samplelist))
    file_size_bytes = 2 * n_chan * total_samples
    file_time_secs = total_samples / sample_rate if sample_rate else 0.0

    def set_key(key: str, value: str) -> None:
        prefix = f"{key}="
        for idx, line in enumerate(lines):
            if line.startswith(prefix):
                lines[idx] = f"{key}={value}"
                return
        lines.append(f"{key}={value}")

    set_key("fileName", str(target))
    set_key("fileSizeBytes", str(file_size_bytes))
    set_key("fileTimeSecs", f"{file_time_secs:.10f}")
    set_key("fileCreateTime", datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
    set_key("fileSHA1", "0")
    if n_out_chans is not None and int(n_out_chans) != source_chans:
        set_key("nSavedChans", str(n_chan))
        set_key("snsApLfSy", f"{n_chan},0,0")
        set_key("snsSaveChanSubset", f"0:{n_chan - 1}")
    lines.append("concatenatedFrom=" + ";".join(str(Path(m)) for m in meta_files))
    lines.append("concatSampleList=" + ",".join(str(int(s)) for s in samplelist))

    meta_path = _concat_meta_path_for_bin(target)
    meta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return meta_path


def write_concat_splitinfo(
    meta_files: Sequence[str],
    target_bin: str | Path,
    bin_files: Sequence[str],
    samplelist: Sequence[int],
) -> Path:
    """Write the per-session sample boundaries JSON next to the concatenated bin."""
    target = Path(target_bin)
    sample_rate = 30000.0
    if meta_files:
        info = spikeglx_meta_channel_info(_read_meta_keyvals(Path(meta_files[0])))
        if info["sample_rate"]:
            sample_rate = float(info["sample_rate"])

    segments: List[Dict[str, object]] = []
    cumulative = 0
    for idx, (bin_file, n_samp) in enumerate(zip(bin_files, samplelist)):
        start = cumulative
        end = cumulative + int(n_samp)
        cumulative = end
        segments.append(
            {
                "index": idx,
                "source_bin": str(Path(bin_file)),
                "source_meta": str(Path(meta_files[idx])) if idx < len(meta_files) else "",
                "n_samples": int(n_samp),
                "start_sample": start,
                "end_sample": end,
                "start_time": start / sample_rate if sample_rate else 0.0,
                "end_time": end / sample_rate if sample_rate else 0.0,
            }
        )

    payload = {
        "target_bin": str(target),
        "sampling_rate": sample_rate,
        "n_segments": len(segments),
        "segments": segments,
    }
    json_path = _concat_splitinfo_path_for_bin(target)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return json_path


def write_concat_manifest(
    target_bin: str | Path,
    bin_files: Sequence[str],
    meta_files: Sequence[str],
    samplelist: Sequence[int],
    *,
    svd_clean: bool,
    n_svd_components: int,
    batch_seconds: float,
    meta_path: str | Path,
    splitinfo_path: str | Path,
) -> Path:
    """Write a human-readable summary of a concatenation next to the fused bin."""
    target = Path(target_bin)
    payload = {
        "target_bin": str(target),
        "meta_path": str(meta_path),
        "splitinfo_path": str(splitinfo_path),
        "n_sources": len(list(bin_files)),
        "total_samples": int(sum(int(s) for s in samplelist)),
        "svd_clean": bool(svd_clean),
        "n_svd_components": int(n_svd_components),
        "batch_seconds": float(batch_seconds),
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "sources": [
            {
                "index": idx,
                "bin": str(Path(b)),
                "meta": str(Path(meta_files[idx])) if idx < len(meta_files) else "",
                "n_samples": int(s),
            }
            for idx, (b, s) in enumerate(zip(bin_files, samplelist))
        ],
    }
    manifest_path = target.parent / "concat_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def concatenate_ap_session(
    bin_files: Sequence[str],
    meta_files: Sequence[str],
    target_bin: str | Path,
    *,
    svd_clean: bool = True,
    n_svd_components: int = 5,
    batch_seconds: float = 0.5,
    progress_cb: Callable[[int], None] | None = None,
    log_cb: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Dict[str, object]:
    """Concatenate AP binaries and write the combined .bin, .meta, split-info and manifest."""
    samplelist = concatenate_ap_binaries(
        bin_files,
        meta_files,
        target_bin,
        svd_clean=svd_clean,
        n_svd_components=n_svd_components,
        batch_seconds=batch_seconds,
        progress_cb=progress_cb,
        log_cb=log_cb,
        should_cancel=should_cancel,
    )
    meta_path = write_concat_meta(meta_files, target_bin, samplelist)
    splitinfo_path = write_concat_splitinfo(meta_files, target_bin, bin_files, samplelist)
    manifest_path = write_concat_manifest(
        target_bin,
        bin_files,
        meta_files,
        samplelist,
        svd_clean=svd_clean,
        n_svd_components=n_svd_components,
        batch_seconds=batch_seconds,
        meta_path=meta_path,
        splitinfo_path=splitinfo_path,
    )
    return {
        "samplelist": [int(s) for s in samplelist],
        "target_bin": str(Path(target_bin)),
        "meta_path": str(meta_path),
        "splitinfo_path": str(splitinfo_path),
        "manifest_path": str(manifest_path),
    }


# ---------------------------------------------------------------------------
# Splitting a joint sort back into per-session results
# ---------------------------------------------------------------------------
#
# IMPORTANT: this splits the *spike trains*, never the binary.  The joint sort
# defines unit identity once across all sessions; physically re-cutting the
# .bin per session would force a re-extraction of templates and throw that
# shared identity away.  Instead we take the sorted result as a read-only view:
# for each session we mask the spikes whose sample index falls inside that
# session's window from the split-info map, shift the times to be session-local,
# and write a phy-loadable folder that reuses the same global templates/cluster
# IDs and points back at the original per-session recording (where that
# session's own TPrime-aligned NI events live).

# Per-spike arrays carry one row per spike and must be masked + (for times) shifted.
_PER_SPIKE_NPY = {
    "spike_times",
    "spike_times_corrected",
    "spike_clusters",
    "spike_templates",
    "amplitudes",
    "pc_features",
    "template_features",
    "spike_positions",
    "spike_depths",
    "spike_amplitudes",
}
# Cluster/template/channel arrays are shared across sessions and copied verbatim.
_SHARED_NPY = {
    "templates",
    "templates_ind",
    "templates_unw",
    "channel_map",
    "channel_positions",
    "channel_shanks",
    "whitening_mat",
    "whitening_mat_inv",
    "similar_templates",
    "pc_feature_ind",
    "template_feature_ind",
    "ops",
}


def read_concat_splitinfo(path: str | Path) -> Dict[str, object]:
    """Load a concatenation split-info JSON written by :func:`write_concat_splitinfo`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "segments" not in data:
        raise ValueError(f"Not a valid concatenation split-info file: {path}")
    return data


def find_concat_splitinfo_for_ks_folder(ks_folder: str | Path) -> Path | None:
    """Locate the split-info JSON for a (joint) Kilosort output folder.

    Resolution order: the bin recorded in ``params.py`` (sibling
    ``*.ap.splitinfo.json``), then a shallow search around the bin and the KS
    folder. Returns ``None`` when the folder is not a concatenated sort.
    """
    ks = Path(ks_folder)
    candidates: List[Path] = []
    params_py = ks / "params.py"
    if params_py.exists():
        dat_path = parse_kilosort_params_dat_path(params_py)
        if dat_path:
            bin_path = Path(dat_path)
            candidates.append(_concat_splitinfo_path_for_bin(bin_path))
            candidates.extend(sorted(bin_path.parent.glob("*.splitinfo.json")))
    for root in (ks, ks.parent, ks.parent.parent):
        if root.exists():
            candidates.extend(sorted(root.glob("*.splitinfo.json")))
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None


def _session_event_files(
    source_bin: str | Path,
    *,
    run_name: str = "",
    extra_roots: Sequence[str | Path] | None = None,
    max_files: int = 128,
) -> List[Path]:
    """Best-effort search for a session's CatGT/TPrime NI event text files.

    Searches the raw recording's own folder tree (covers in-place CatGT output)
    and any ``extra_roots`` such as the processed ``output_root`` (covers the
    standard pipeline layout ``<output_root>/<run_name>/catgt_.../*.nidq.*.txt``).
    When ``run_name`` is given, matches under ``extra_roots`` are restricted to
    files whose name carries that run name, so sessions don't pick up each
    other's events.
    """
    bin_path = Path(source_bin)
    near_roots: List[Path] = []
    cur = bin_path.parent
    # Stay within the session tree (probe -> run -> session); going higher risks
    # rglob-ing a large shared raw root. The processed output_root is searched
    # separately via extra_roots with a run-name filter.
    for _ in range(3):
        if cur.exists() and cur not in near_roots:
            near_roots.append(cur)
        if cur.parent == cur:
            break
        cur = cur.parent

    patterns = ["*.nidq.*.txt", "*.xa_*.txt", "*.xd_*.txt", "*.xia_*.txt", "*.xid_*.txt", "*.adj.txt"]
    found: List[Path] = []
    seen: set[str] = set()

    def _collect(root: Path, *, require_run_name: bool) -> None:
        if not root.exists():
            return
        for pattern in patterns:
            for hit in root.rglob(pattern):
                if require_run_name and run_name and run_name not in hit.name:
                    continue
                key = str(hit.resolve()).lower()
                if key in seen:
                    continue
                seen.add(key)
                found.append(hit)
                if len(found) >= max_files:
                    return

    for root in near_roots:
        _collect(root, require_run_name=False)
        if len(found) >= max_files:
            return found
    for root in extra_roots or []:
        _collect(Path(root), require_run_name=True)
        if len(found) >= max_files:
            return found
    return found


def _rewrite_params_dat_path(original_params: Path, dest_params: Path, new_dat_path: str | Path) -> None:
    """Copy params.py with dat_path repointed to the per-session recording."""
    new_line = f"dat_path = r'{Path(new_dat_path)}'"
    if original_params.exists():
        text = original_params.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"(?m)^\s*dat_path\s*=", text):
            text = re.sub(r"(?m)^\s*dat_path\s*=.*$", lambda _m: new_line, text, count=1)
        else:
            text = new_line + "\n" + text
    else:
        text = new_line + "\n"
    dest_params.write_text(text, encoding="utf-8")


def split_concatenated_sort(
    ks_folder: str | Path,
    *,
    output_dir: str | Path | None = None,
    splitinfo_path: str | Path | None = None,
    copy_events: bool = True,
    event_search_roots: Sequence[str | Path] | None = None,
    log_cb: Callable[[str], None] | None = None,
) -> Dict[str, object]:
    """Split a joint Kilosort sort into per-session phy folders (spike trains only).

    Each session folder reuses the global templates and cluster IDs, holds only
    that session's spikes with session-local sample times, points ``params.py``
    back at the original per-session recording, and (optionally) gathers that
    session's TPrime-aligned NI event text files. The concatenated binary is
    never re-cut.
    """
    import numpy as np

    def log(message: str) -> None:
        if log_cb is not None:
            log_cb(message)

    ks = Path(ks_folder)
    if not ks.is_dir():
        raise ValueError(f"Kilosort folder not found: {ks}")

    if splitinfo_path is None:
        splitinfo_path = find_concat_splitinfo_for_ks_folder(ks)
    if not splitinfo_path:
        raise FileNotFoundError(
            "No concatenation split-info file found for this folder. Splitting only applies "
            "to a joint sort produced from a concatenated recording."
        )
    info = read_concat_splitinfo(str(splitinfo_path))
    segments = list(info.get("segments") or [])
    if not segments:
        raise ValueError(f"Split-info file has no segments: {splitinfo_path}")

    spike_times_path = ks / "spike_times.npy"
    if not spike_times_path.exists():
        raise FileNotFoundError(f"Missing spike_times.npy in {ks}")
    spike_times = np.load(spike_times_path).squeeze()
    n_spikes_total = int(spike_times.shape[0])

    # Classify the top-level files once.
    per_spike_files: List[Path] = []
    shared_files: List[Path] = []
    other_files: List[Path] = []
    for entry in sorted(ks.iterdir()):
        if entry.is_dir():
            continue
        if entry.suffix == ".npy":
            stem = entry.stem
            if stem in _PER_SPIKE_NPY:
                per_spike_files.append(entry)
            elif stem in _SHARED_NPY:
                shared_files.append(entry)
            else:
                try:
                    arr = np.load(entry, mmap_mode="r")
                    if arr.ndim >= 1 and int(arr.shape[0]) == n_spikes_total:
                        per_spike_files.append(entry)
                    else:
                        shared_files.append(entry)
                except Exception:
                    other_files.append(entry)
        elif entry.name == "params.py":
            continue
        else:
            other_files.append(entry)

    out_root = Path(output_dir) if output_dir is not None else ks / "sessions"
    out_root.mkdir(parents=True, exist_ok=True)
    original_params = ks / "params.py"

    session_results: List[Dict[str, object]] = []
    for seg in segments:
        idx = int(seg.get("index", len(session_results)))
        start = int(seg.get("start_sample", 0))
        end = int(seg.get("end_sample", 0))
        source_bin = str(seg.get("source_bin") or "")
        run_name = ""
        if source_bin:
            run_name = str(parse_spikeglx_bin_name(source_bin).get("run_name") or "")
        if not run_name:
            run_name = f"session{idx}"
        session_dir = out_root / f"{idx:02d}_{run_name}"
        session_dir.mkdir(parents=True, exist_ok=True)

        mask = (spike_times >= start) & (spike_times < end)
        n_session_spikes = int(np.count_nonzero(mask))
        log(f"Session {idx} ({run_name}): {n_session_spikes} spikes in samples [{start}, {end})")

        # Per-spike arrays: subset by the session mask; shift spike-time arrays to local.
        n_clusters = 0
        for src in per_spike_files:
            arr = np.load(src)
            subset = arr[mask]
            if "spike_times" in src.stem:
                subset = (subset - np.asarray(start, dtype=subset.dtype)).astype(arr.dtype, copy=False)
            np.save(session_dir / src.name, subset)
            if src.stem == "spike_clusters":
                n_clusters = int(np.unique(subset).size)

        # Shared arrays + auxiliary files: copied verbatim so identity is preserved.
        for src in (*shared_files, *other_files):
            shutil.copy2(src, session_dir / src.name)

        # params.py repointed at the original per-session recording.
        _rewrite_params_dat_path(original_params, session_dir / "params.py", source_bin or original_params)

        events_copied: List[str] = []
        if copy_events and source_bin:
            event_files = _session_event_files(
                source_bin,
                run_name=run_name,
                extra_roots=event_search_roots,
            )
            if event_files:
                events_dir = session_dir / "events"
                events_dir.mkdir(exist_ok=True)
                for ev in event_files:
                    try:
                        shutil.copy2(ev, events_dir / ev.name)
                        events_copied.append(ev.name)
                    except Exception as exc:
                        log(f"  Could not copy event file {ev.name}: {exc}")
            else:
                log(
                    f"  No NI/TPrime event files found near {source_bin}. "
                    "Run CatGT extract-only + TPrime on the raw session to generate them."
                )

        session_results.append(
            {
                "index": idx,
                "run_name": run_name,
                "output_dir": str(session_dir),
                "source_bin": source_bin,
                "n_spikes": n_session_spikes,
                "n_clusters": n_clusters,
                "start_sample": start,
                "end_sample": end,
                "events_copied": events_copied,
            }
        )

    manifest = {
        "source_ks_folder": str(ks),
        "splitinfo_path": str(splitinfo_path),
        "sampling_rate": info.get("sampling_rate"),
        "n_sessions": len(session_results),
        "n_spikes_total": n_spikes_total,
        "output_root": str(out_root),
        "sessions": session_results,
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_root / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"Wrote {len(session_results)} per-session folder(s) under {out_root}")
    return manifest

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


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

"""Build an :class:`~neuropyguin.workers.EcephysPipelineConfig` for the CLI.

The Preprocessing tab keeps every pipeline knob in widgets and mirrors them into
``QSettings``. The CLI needs the same configuration without a window, so this
module resolves it from four layers, each overriding the previous one:

1. :func:`default_pipeline_settings` - the values the GUI widgets are built with,
2. :func:`settings_pipeline_overrides` - whatever the GUI last saved (skippable),
3. ``--config file.json`` - a saved config snapshot, same field names,
4. explicit command line flags.

That ordering means a user who has already set up the GUI can run
``neuropyguin preprocess run <bin>`` with no flags at all and get exactly the
run the window would have produced, while a scripted/headless install can pin
everything explicitly and ignore the stored state with ``--no-saved-settings``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

from ._common import (
    CommandError,
    PROJECT_ROOT,
    load_json_file,
    open_settings,
    parse_csv_list,
    setting_bool,
    setting_float,
    setting_str,
)


#: Pipeline stages in execution order. Each maps to a ``run_<name>`` config field
#: and is selectable on the command line by this short name.
PIPELINE_STEPS: tuple[str, ...] = (
    "catgt",
    "catgt_extract_only",
    "tprime",
    "kilosort",
    "kilosort_postproc",
    "noise_templates",
    "mean_waveforms",
    "quality_metrics",
    "pybombcell",
)

#: Human labels for ``preprocess steps`` and help text.
STEP_LABELS: Dict[str, str] = {
    "catgt": "CatGT filtering, CAR and event extraction",
    "catgt_extract_only": "CatGT event extraction only (no AP output)",
    "tprime": "TPrime cross-stream event alignment",
    "kilosort": "Kilosort spike sorting",
    "kilosort_postproc": "Kilosort postprocessing (duplicate removal)",
    "noise_templates": "Noise template identification",
    "mean_waveforms": "Mean waveform extraction (C_Waves)",
    "quality_metrics": "Quality metrics computation",
    "pybombcell": "py_bombcell unit quality classification",
}

#: Config fields holding a filesystem path, expanded to absolute form on build.
_PATH_FIELDS = (
    "output_root",
    "json_root",
    "catgt_path",
    "tprime_path",
    "cwaves_path",
    "ks4_repo_path",
    "kilosort_output_tmp",
)


def default_pipeline_settings() -> Dict[str, Any]:
    """Return the built-in defaults, matching the Preprocessing tab's widgets.

    These are the values a fresh install starts from: CatGT + Kilosort +
    postprocessing + mean waveforms + quality metrics + py_bombcell enabled, the
    stock CatGT flag string, and the bundled tool folders under ``tools/``.
    """
    from ..tool_installer import default_tool_paths

    tools = default_tool_paths(PROJECT_ROOT)
    return {
        # Output placement
        "output_root": str((Path.cwd() / "NeuroPyGuiN_output").resolve()),
        "json_root": str((Path.cwd() / "NeuroPyGuiN_json").resolve()),
        "output_layout": "mirror",
        "mirror_raw_hierarchy_output": True,
        "save_catgt_ap_bin": False,
        # Stage toggles
        "run_catgt": True,
        "run_catgt_extract_only": False,
        "run_tprime": False,
        "run_kilosort": True,
        "run_kilosort_postproc": True,
        "run_noise_templates": False,
        "run_mean_waveforms": True,
        "run_quality_metrics": True,
        "run_pybombcell": True,
        # Run identity (per-file values are re-parsed from each bin name)
        "ks_ver": "4",
        "gate_string": "0",
        "trigger_string": "0,0",
        "probe_string": "0",
        "region_name": "default",
        # CatGT / TPrime strings
        "ni_extract_string": "-xd=0,0,8,7,0 -xd=0,0,8,5,0 -xd=0,0,8,6,0 -xd=0,0,8,3,0",
        "catgt_cmd_string": "-prb_fld -out_prb_fld -apfilter=butter,12,300,10000 -gfix=0.4,0.10,0.02",
        "catgt_car_mode": "gbldmx",
        "catgt_loccar_min_um": 40.0,
        "catgt_loccar_max_um": 160.0,
        "sync_period": 1.0,
        "tostream_sync_params": "imec0",
        # Sorting / metrics parameters
        "ks_th": "[8,9]",
        "qm_isi_thresh": 0.002,
        "ks4_duplicate_spike_ms": 0.25,
        "ks4_min_template_size_um": 10.0,
        "c_waves_snr_um": 160.0,
        "ks4_advanced_params": {},
        # External tools
        "catgt_path": tools.get("catgt", ""),
        "tprime_path": tools.get("tprime", ""),
        "cwaves_path": tools.get("cwaves", ""),
        "ks4_repo_path": tools.get("kilosort", ""),
        "kilosort_output_tmp": str((Path.cwd() / "kilosort_datatemp").resolve()),
    }


def settings_pipeline_overrides(settings=None) -> Dict[str, Any]:
    """Read the pipeline configuration the GUI last persisted.

    Only keys actually present in the store produce an override, so an install
    that has never opened a given section keeps the built-in default for it.
    Mirrors :meth:`PreprocessingTab._restore_settings` key for key.
    """
    import json

    from ..preprocessing import normalize_output_layout

    store = settings if settings is not None else open_settings()
    defaults = default_pipeline_settings()
    out: Dict[str, Any] = {}

    def put_str(field: str, key: str) -> None:
        value = setting_str(store, key, "")
        if value:
            out[field] = value

    def put_float(field: str, key: str) -> None:
        if store.contains(key):
            out[field] = setting_float(store, key, float(defaults[field]))

    def put_bool(field: str, key: str) -> None:
        if store.contains(key):
            out[field] = setting_bool(store, key, bool(defaults[field]))

    put_str("output_root", "preproc/output_root")
    put_str("json_root", "preproc/json_root")

    # Layout: the explicit key wins, otherwise fall back to the legacy boolean so
    # an older install keeps writing where it always has.
    legacy_mirror = setting_bool(store, "preproc/mirror_raw_hierarchy_output", True)
    if store.contains("preproc/output_layout") or store.contains("preproc/mirror_raw_hierarchy_output"):
        layout = normalize_output_layout(
            setting_str(store, "preproc/output_layout", ""),
            mirror_raw_hierarchy=legacy_mirror,
        )
        out["output_layout"] = layout
        out["mirror_raw_hierarchy_output"] = layout == "mirror"

    put_bool("save_catgt_ap_bin", "preproc/save_catgt_ap_bin")
    for step in PIPELINE_STEPS:
        put_bool(f"run_{step}", f"preproc/run_{step}")

    put_str("ks_ver", "preproc/ks_ver")
    put_str("gate_string", "preproc/gate_string")
    put_str("trigger_string", "preproc/trigger_string")
    put_str("probe_string", "preproc/probe_string")
    put_str("region_name", "preproc/region_name")
    put_str("ks_th", "preproc/ks_th")
    put_str("ni_extract_string", "preproc/ni_extract_string")
    put_str("catgt_cmd_string", "preproc/catgt_cmd_string")
    put_str("tostream_sync_params", "preproc/tostream_sync_params")
    put_str("catgt_car_mode", "preproc/catgt_car_mode")

    put_float("qm_isi_thresh", "preproc/qm_isi_thresh")
    put_float("sync_period", "preproc/sync_period")
    put_float("catgt_loccar_min_um", "preproc/catgt_loccar_min_um")
    put_float("catgt_loccar_max_um", "preproc/catgt_loccar_max_um")
    put_float("ks4_duplicate_spike_ms", "preproc/ks4_duplicate_spike_ms")
    put_float("ks4_min_template_size_um", "preproc/ks4_min_template_size_um")
    put_float("c_waves_snr_um", "preproc/c_waves_snr_um")

    put_str("catgt_path", "preproc/catgt_path")
    put_str("tprime_path", "preproc/tprime_path")
    put_str("cwaves_path", "preproc/cwaves_path")
    put_str("ks4_repo_path", "preproc/ks4_repo_path")
    put_str("kilosort_output_tmp", "preproc/kilosort_output_tmp")

    raw_adv = setting_str(store, "preproc/ks4_advanced_params_json", "")
    if raw_adv:
        try:
            parsed = json.loads(raw_adv)
            if isinstance(parsed, dict):
                out["ks4_advanced_params"] = parsed
        except ValueError:
            pass  # a corrupt stored blob must not block a CLI run

    return out


def resolve_steps(
    current: Dict[str, bool],
    *,
    only: str | None = None,
    enable: Sequence[str] = (),
    disable: Sequence[str] = (),
) -> Dict[str, bool]:
    """Apply the ``--steps`` / ``--with`` / ``--without`` selection to the toggles.

    ``only`` replaces the whole selection (``all``, ``none``, or a comma list).
    ``enable`` and ``disable`` then adjust it, so ``--steps kilosort --with tprime``
    is a valid and readable way to say "just these two".
    """
    resolved = dict(current)
    if only is not None:
        text = str(only).strip().lower()
        if text == "all":
            chosen = set(PIPELINE_STEPS)
        elif text in {"none", ""}:
            chosen = set()
        else:
            chosen = set()
            for name in parse_csv_list(text):
                if name not in PIPELINE_STEPS:
                    raise CommandError(
                        f"Unknown pipeline step {name!r}. Valid steps: {', '.join(PIPELINE_STEPS)}"
                    )
                chosen.add(name)
        resolved = {step: (step in chosen) for step in PIPELINE_STEPS}

    for name in enable:
        if name not in PIPELINE_STEPS:
            raise CommandError(f"Unknown pipeline step {name!r}")
        resolved[name] = True
    for name in disable:
        if name not in PIPELINE_STEPS:
            raise CommandError(f"Unknown pipeline step {name!r}")
        resolved[name] = False
    return resolved


def _apply_cli_overrides(values: Dict[str, Any], args) -> None:
    """Overlay the explicit command line flags onto the resolved values dict.

    Only flags the user actually passed (i.e. not ``None``) override anything,
    which is what lets the stored GUI configuration show through.
    """
    simple = {
        "output_root": "output_root",
        "json_root": "json_root",
        "output_layout": "output_layout",
        "ks_ver": "ks_ver",
        "gate_string": "gate",
        "trigger_string": "trigger",
        "probe_string": "probe",
        "region_name": "region",
        "ni_extract_string": "ni_extract",
        "catgt_cmd_string": "catgt_cmd",
        "catgt_car_mode": "car_mode",
        "tostream_sync_params": "tostream",
        "ks_th": "ks_th",
        "catgt_path": "catgt_path",
        "tprime_path": "tprime_path",
        "cwaves_path": "cwaves_path",
        "ks4_repo_path": "ks4_repo",
        "kilosort_output_tmp": "ks_tmp",
        "sync_period": "sync_period",
        "qm_isi_thresh": "qm_isi",
        "catgt_loccar_min_um": "loccar_min",
        "catgt_loccar_max_um": "loccar_max",
        "ks4_duplicate_spike_ms": "ks4_duplicate_ms",
        "ks4_min_template_size_um": "ks4_min_template_um",
        "c_waves_snr_um": "cwaves_snr_um",
    }
    for field, attr in simple.items():
        value = getattr(args, attr, None)
        if value is not None:
            values[field] = value

    if getattr(args, "save_catgt_ap_bin", None) is not None:
        values["save_catgt_ap_bin"] = bool(args.save_catgt_ap_bin)

    if getattr(args, "ks4_param", None):
        # Repeatable ``--ks4-param name=value``; values are parsed as JSON when
        # possible so numbers and booleans reach Kilosort with the right type.
        import json

        advanced = dict(values.get("ks4_advanced_params") or {})
        for item in args.ks4_param:
            if "=" not in item:
                raise CommandError(f"--ks4-param expects name=value, got {item!r}")
            name, _, raw = item.partition("=")
            try:
                advanced[name.strip()] = json.loads(raw)
            except ValueError:
                advanced[name.strip()] = raw
        values["ks4_advanced_params"] = advanced

    steps = {step: bool(values[f"run_{step}"]) for step in PIPELINE_STEPS}
    steps = resolve_steps(
        steps,
        only=getattr(args, "steps", None),
        enable=parse_csv_list(getattr(args, "with_steps", None)),
        disable=parse_csv_list(getattr(args, "without_steps", None)),
    )
    for step, enabled in steps.items():
        values[f"run_{step}"] = enabled

    # Keep the legacy mirror flag consistent with whatever layout won.
    values["mirror_raw_hierarchy_output"] = str(values.get("output_layout")) == "mirror"


def build_pipeline_config(args, reporter=None):
    """Resolve the layered configuration into an ``EcephysPipelineConfig``.

    Returns the dataclass the worker consumes. Raises :class:`CommandError` when
    the resulting configuration cannot possibly run (for example a required
    external tool folder that does not exist for an enabled stage).
    """
    from ..preprocessing import OUTPUT_LAYOUTS, normalize_output_layout
    from ..workers import EcephysPipelineConfig

    values = default_pipeline_settings()

    if not getattr(args, "no_saved_settings", False):
        values.update(settings_pipeline_overrides())

    config_path = getattr(args, "config", None)
    if config_path:
        snapshot = load_json_file(config_path)
        unknown = sorted(set(snapshot) - set(values))
        if unknown and reporter is not None:
            reporter.warn(f"Ignoring unknown config keys: {', '.join(unknown)}")
        values.update({k: v for k, v in snapshot.items() if k in values})

    _apply_cli_overrides(values, args)

    layout = normalize_output_layout(str(values.get("output_layout") or ""))
    if str(values.get("output_layout") or "").strip().lower() not in OUTPUT_LAYOUTS:
        if reporter is not None and values.get("output_layout"):
            reporter.warn(f"Unknown output layout {values['output_layout']!r}; using {layout}.")
    values["output_layout"] = layout
    values["mirror_raw_hierarchy_output"] = layout == "mirror"

    for field in _PATH_FIELDS:
        raw = str(values.get(field) or "").strip()
        values[field] = str(Path(raw).expanduser()) if raw else ""

    return EcephysPipelineConfig(**values)


def validate_config_for_run(cfg) -> List[str]:
    """Return human readable problems that would make the run fail immediately.

    Mirrors the guard clauses inside :class:`EcephysPipelineWorker` so the CLI can
    refuse a doomed batch up front instead of failing on the first recording.
    """
    problems: List[str] = []
    needs_catgt = cfg.run_catgt or cfg.run_catgt_extract_only
    if needs_catgt and not Path(cfg.catgt_path or "").is_dir():
        problems.append(f"CatGT folder is not a directory: {cfg.catgt_path or '(unset)'}")
    if cfg.run_tprime and not Path(cfg.tprime_path or "").is_dir():
        problems.append(f"TPrime folder is not a directory: {cfg.tprime_path or '(unset)'}")
    if (cfg.run_kilosort_postproc or cfg.run_mean_waveforms) and not Path(cfg.cwaves_path or "").is_dir():
        problems.append(f"C_Waves folder is not a directory: {cfg.cwaves_path or '(unset)'}")
    if not any(getattr(cfg, f"run_{step}") for step in PIPELINE_STEPS):
        problems.append("No pipeline steps are enabled. Use --steps to pick at least one.")
    if not str(cfg.output_root or "").strip():
        problems.append("Output root is empty. Pass --output-root.")
    return problems


def config_as_dict(cfg) -> Dict[str, Any]:
    """Return the config dataclass as a plain JSON-serialisable dict."""
    from dataclasses import asdict

    return asdict(cfg)


def enabled_steps(cfg) -> List[str]:
    """List the enabled stage names in execution order."""
    return [step for step in PIPELINE_STEPS if bool(getattr(cfg, f"run_{step}"))]


def add_pipeline_arguments(parser) -> None:
    """Attach every pipeline configuration flag to ``parser``.

    Shared by ``preprocess run`` and ``preprocess plan`` so the dry run accepts
    exactly the options the real run does. Defaults are deliberately ``None``:
    that is the signal that the user did not ask for an override, which keeps
    the stored GUI configuration visible through the layering.
    """
    where = parser.add_argument_group("output placement")
    where.add_argument("--output-root", help="Root folder for processed output.")
    where.add_argument("--json-path", dest="json_root", help="Fallback folder for pipeline JSON files.")
    where.add_argument(
        "--output-layout",
        choices=["mirror", "run_folder", "exact"],
        help=(
            "Where a run lands under the output root: mirror the rawData tree, "
            "one folder per run, or the output root itself."
        ),
    )
    where.add_argument(
        "--save-catgt-ap-bin",
        dest="save_catgt_ap_bin",
        action="store_true",
        default=None,
        help="Keep the filtered CatGT AP binary instead of discarding it.",
    )
    where.add_argument(
        "--no-save-catgt-ap-bin",
        dest="save_catgt_ap_bin",
        action="store_false",
        help="Discard the filtered CatGT AP binary.",
    )

    stages = parser.add_argument_group("pipeline stages")
    stages.add_argument(
        "--steps",
        help=(
            "Run exactly these comma-separated steps (or 'all' / 'none'). "
            f"Available: {', '.join(PIPELINE_STEPS)}."
        ),
    )
    stages.add_argument("--with", dest="with_steps", help="Additionally enable these comma-separated steps.")
    stages.add_argument("--without", dest="without_steps", help="Disable these comma-separated steps.")

    run = parser.add_argument_group("run identity")
    run.add_argument("--ks-ver", choices=["4", "3.0", "2.5", "2.0"], help="Kilosort version to drive.")
    run.add_argument("--gate", help="SpikeGLX gate string (default: parsed from each file name).")
    run.add_argument("--trigger", help="SpikeGLX trigger string (default: parsed from each file name).")
    run.add_argument("--probe", help="Probe index string (default: parsed from each file name).")
    run.add_argument("--region", help="Region label recorded with the run.")

    strings = parser.add_argument_group("CatGT and TPrime")
    strings.add_argument("--catgt-cmd", help="Raw CatGT flag string.")
    strings.add_argument("--ni-extract", help="CatGT/TPrime NI event extractor string.")
    strings.add_argument("--car-mode", choices=["gbldmx", "loccar", "none"], help="CatGT CAR mode.")
    strings.add_argument("--loccar-min", type=float, help="CatGT loccar inner radius (um).")
    strings.add_argument("--loccar-max", type=float, help="CatGT loccar outer radius (um).")
    strings.add_argument("--tostream", help="TPrime reference stream, e.g. imec0 or ni.")
    strings.add_argument("--sync-period", type=float, help="TPrime sync pulse period (s).")

    params = parser.add_argument_group("sorting and metrics")
    params.add_argument("--ks-th", help="Kilosort detection thresholds, e.g. '[8,9]'.")
    params.add_argument("--qm-isi", type=float, help="Quality metrics refractory threshold (s).")
    params.add_argument("--ks4-duplicate-ms", type=float, help="KS4 duplicate spike window (ms).")
    params.add_argument("--ks4-min-template-um", type=float, help="KS4 minimum template size (um).")
    params.add_argument("--cwaves-snr-um", type=float, help="C_Waves SNR radius (um).")
    params.add_argument(
        "--ks4-param",
        action="append",
        metavar="NAME=VALUE",
        help="Advanced Kilosort4 parameter override. Repeatable. Values are parsed as JSON.",
    )

    tools = parser.add_argument_group("external tools")
    tools.add_argument("--catgt-path", help="Folder containing the CatGT executable.")
    tools.add_argument("--tprime-path", help="Folder containing the TPrime executable.")
    tools.add_argument("--cwaves-path", help="Folder containing the C_Waves executable.")
    tools.add_argument("--ks4-repo", help="Kilosort repository/installation path.")
    tools.add_argument("--ks-tmp", help="Kilosort scratch folder for temporary data.")

    source = parser.add_argument_group("configuration source")
    source.add_argument("--config", help="JSON file with pipeline settings to overlay.")
    source.add_argument(
        "--no-saved-settings",
        action="store_true",
        help="Ignore the settings saved by the GUI and start from built-in defaults.",
    )

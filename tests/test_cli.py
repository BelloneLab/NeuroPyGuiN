"""Tests for the NeuroPyGuiN command line interface.

Covers the argument tree, the layered pipeline-configuration resolution, the
shared helpers, and the read-only commands that can run without external tools
or real recordings. Tests never touch the user's real settings store: every
pipeline configuration is built with ``--no-saved-settings``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from neuropyguin.cli._common import (
    CommandError,
    Reporter,
    format_table,
    parse_csv_list,
    parse_int_list,
    write_json_file,
)
from neuropyguin.cli.parser import build_parser, main
from neuropyguin.cli.pipeline_config import (
    PIPELINE_STEPS,
    STEP_LABELS,
    default_pipeline_settings,
    resolve_steps,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


N_CHAN = 5
N_SAMPLES = 600


def _write_spikeglx_run(root: Path, run: str = "testrun") -> Path:
    """Create a minimal but valid SpikeGLX AP recording and return its bin path."""
    probe_dir = root / f"{run}_g0" / f"{run}_g0_imec0"
    probe_dir.mkdir(parents=True, exist_ok=True)
    bin_path = probe_dir / f"{run}_g0_t0.imec0.ap.bin"
    bin_path.write_bytes(b"\x00" * (2 * N_CHAN * N_SAMPLES))
    meta_path = probe_dir / f"{run}_g0_t0.imec0.ap.meta"
    meta_path.write_text(
        "\n".join(
            [
                f"fileName={bin_path}",
                f"nSavedChans={N_CHAN}",
                f"snsApLfSy={N_CHAN - 1},0,1",
                "imSampRate=30000",
                "~snsChanMap=(5,0,1)(AP0;0:0)(AP1;1:1)(AP2;2:2)(AP3;3:3)(SY0;4:4)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return bin_path


@pytest.fixture
def raw_session(tmp_path: Path) -> Path:
    """A folder holding one valid SpikeGLX recording."""
    _write_spikeglx_run(tmp_path / "rawData")
    return tmp_path / "rawData"


def _namespace(**overrides) -> argparse.Namespace:
    """Build the namespace ``build_pipeline_config`` expects, all flags unset."""
    base = {
        "no_saved_settings": True,
        "config": None,
        "steps": None,
        "with_steps": None,
        "without_steps": None,
        "save_catgt_ap_bin": None,
        "ks4_param": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def _iter_parsers(parser, path=()):
    """Yield every (path, parser) pair in the subcommand tree."""
    yield path, parser
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                yield from _iter_parsers(sub, path + (name,))


def test_parser_builds_every_group() -> None:
    parser = build_parser()
    groups = {path[0] for path, _ in _iter_parsers(parser) if len(path) == 1}
    assert {"preprocess", "curate", "postproc", "histology", "doctor", "tools", "config"} <= groups


def test_every_subcommand_renders_help() -> None:
    """A malformed argument definition raises here rather than at run time."""
    for path, sub in _iter_parsers(build_parser()):
        assert sub.format_help(), f"empty help for {' '.join(path)}"


def test_every_leaf_command_has_a_handler() -> None:
    """Each terminal subcommand must set ``func`` so the dispatcher can call it."""
    for path, sub in _iter_parsers(build_parser()):
        has_children = any(isinstance(a, argparse._SubParsersAction) for a in sub._actions)
        if path and not has_children:
            assert "func" in sub._defaults, f"{' '.join(path)} has no handler"


def test_unknown_command_exits_with_usage_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["preprocess", "not-a-command"])
    assert excinfo.value.code == 2


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def test_parse_int_list_handles_ranges_and_dedupes() -> None:
    assert parse_int_list("3,7,10-12,7") == [3, 7, 10, 11, 12]
    assert parse_int_list("8-5") == [5, 6, 7, 8]
    assert parse_int_list("") == []
    assert parse_int_list(None) == []


def test_parse_int_list_rejects_garbage() -> None:
    with pytest.raises(CommandError):
        parse_int_list("1,abc")


def test_parse_csv_list_strips_and_drops_blanks() -> None:
    assert parse_csv_list(" a , ,b ") == ["a", "b"]


def test_reporter_json_mode_writes_parsable_stdout(capsys) -> None:
    reporter = Reporter(as_json=True)
    reporter.log("progress line")
    reporter.emit({"value": 1, "path": Path("x")})
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"value": 1, "path": "x"}
    assert "progress line" in captured.err  # progress must not pollute stdout


def test_reporter_text_mode_renders_nested_payload(capsys) -> None:
    Reporter().emit({"a": 1, "b": {"c": 2}, "d": [1, 2]})
    out = capsys.readouterr().out
    assert "a: 1" in out and "c: 2" in out and "- 1" in out


def test_format_table_pads_columns() -> None:
    lines = format_table([{"x": "a", "y": "long value"}], ["x", "y"])
    assert lines[0].startswith("x")
    assert "long value" in lines[2]


def test_write_json_file_round_trips(tmp_path: Path) -> None:
    """Paths are serialised as strings and parent folders are created."""
    source = Path("a/b")
    target = write_json_file(tmp_path / "nested" / "out.json", {"p": source})
    assert json.loads(target.read_text(encoding="utf-8")) == {"p": str(source)}


# --------------------------------------------------------------------------- #
# Pipeline configuration
# --------------------------------------------------------------------------- #


def test_every_step_has_a_label() -> None:
    assert set(STEP_LABELS) == set(PIPELINE_STEPS)


def test_resolve_steps_only_replaces_the_whole_selection() -> None:
    current = {step: True for step in PIPELINE_STEPS}
    resolved = resolve_steps(current, only="kilosort,tprime")
    assert resolved["kilosort"] is True and resolved["tprime"] is True
    assert resolved["catgt"] is False


def test_resolve_steps_with_and_without_adjust_the_selection() -> None:
    current = {step: False for step in PIPELINE_STEPS}
    resolved = resolve_steps(current, only="kilosort", enable=["tprime"], disable=["kilosort"])
    assert resolved["tprime"] is True and resolved["kilosort"] is False


def test_resolve_steps_all_and_none() -> None:
    current = {step: False for step in PIPELINE_STEPS}
    assert all(resolve_steps(current, only="all").values())
    assert not any(resolve_steps(current, only="none").values())


def test_resolve_steps_rejects_unknown_names() -> None:
    with pytest.raises(CommandError):
        resolve_steps({step: False for step in PIPELINE_STEPS}, only="nope")


def test_default_settings_cover_every_config_field() -> None:
    """The defaults must fully populate ``EcephysPipelineConfig``."""
    pytest.importorskip("PySide6")
    from dataclasses import fields

    from neuropyguin.workers import EcephysPipelineConfig

    expected = {f.name for f in fields(EcephysPipelineConfig)}
    assert set(default_pipeline_settings()) == expected


def test_build_pipeline_config_applies_flags_over_defaults(tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    from neuropyguin.cli.pipeline_config import build_pipeline_config, enabled_steps

    cfg = build_pipeline_config(
        _namespace(
            output_root=str(tmp_path / "out"),
            output_layout="run_folder",
            steps="kilosort,quality_metrics",
            ks_ver="4",
        )
    )
    assert cfg.output_root == str(tmp_path / "out")
    assert cfg.output_layout == "run_folder"
    assert cfg.mirror_raw_hierarchy_output is False
    assert enabled_steps(cfg) == ["kilosort", "quality_metrics"]


def test_build_pipeline_config_reads_a_config_file(tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    from neuropyguin.cli.pipeline_config import build_pipeline_config

    snapshot = tmp_path / "cfg.json"
    snapshot.write_text(json.dumps({"ks_th": "[6,7]", "run_tprime": True}), encoding="utf-8")

    cfg = build_pipeline_config(_namespace(config=str(snapshot)))
    assert cfg.ks_th == "[6,7]"
    assert cfg.run_tprime is True

    # An explicit flag still wins over the file.
    cfg = build_pipeline_config(_namespace(config=str(snapshot), ks_th="[9,9]"))
    assert cfg.ks_th == "[9,9]"


def test_build_pipeline_config_parses_ks4_params() -> None:
    pytest.importorskip("PySide6")
    from neuropyguin.cli.pipeline_config import build_pipeline_config

    cfg = build_pipeline_config(_namespace(ks4_param=["nblocks=5", "tag=fast"]))
    assert cfg.ks4_advanced_params["nblocks"] == 5
    assert cfg.ks4_advanced_params["tag"] == "fast"


def test_validate_config_reports_missing_tools_and_empty_selection(tmp_path: Path) -> None:
    pytest.importorskip("PySide6")
    from neuropyguin.cli.pipeline_config import build_pipeline_config, validate_config_for_run

    cfg = build_pipeline_config(
        _namespace(steps="catgt", catgt_path=str(tmp_path / "missing"), output_root=str(tmp_path))
    )
    problems = validate_config_for_run(cfg)
    assert any("CatGT" in p for p in problems)

    cfg = build_pipeline_config(_namespace(steps="none", output_root=str(tmp_path)))
    assert any("No pipeline steps" in p for p in validate_config_for_run(cfg))


# --------------------------------------------------------------------------- #
# Read-only commands
# --------------------------------------------------------------------------- #


def test_discover_lists_valid_recordings(raw_session: Path, capsys) -> None:
    assert main(["--json", "preprocess", "discover", str(raw_session)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 1
    assert payload["recordings"][0]["run_name"] == "testrun"
    assert payload["recordings"][0]["valid"] is True


def test_discover_errors_when_nothing_matches(tmp_path: Path, capsys) -> None:
    assert main(["preprocess", "discover", str(tmp_path)]) == 2
    assert "No SpikeGLX AP binaries" in capsys.readouterr().err


def test_validate_flags_a_sync_only_recording(tmp_path: Path, capsys) -> None:
    """A calibration/sync-only file must be reported as unusable (exit code 1)."""
    probe_dir = tmp_path / "bad_g0" / "bad_g0_imec0"
    probe_dir.mkdir(parents=True)
    bin_path = probe_dir / "bad_g0_t0.imec0.ap.bin"
    bin_path.write_bytes(b"\x00" * 100)
    (probe_dir / "bad_g0_t0.imec0.ap.meta").write_text(
        "nSavedChans=1\nsnsApLfSy=0,0,1\n~snsChanMap=(1,0,1)(SY0;0:0)\n", encoding="utf-8"
    )
    assert main(["--json", "preprocess", "validate", str(tmp_path)]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["invalid"] == 1


def test_plan_predicts_the_mirrored_output_layout(raw_session: Path, tmp_path: Path, capsys) -> None:
    pytest.importorskip("PySide6")
    code = main(
        [
            "--json",
            "preprocess",
            "plan",
            str(raw_session),
            "--no-saved-settings",
            "--output-root",
            str(tmp_path / "out"),
            "--output-layout",
            "run_folder",
            "--steps",
            "kilosort",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["steps"] == ["kilosort"]
    job = payload["jobs"][0]
    assert job["ks_folder"].endswith(str(Path("out") / "testrun" / "imec0_ks4"))
    # Kilosort alone needs no external tool folder, so the plan is runnable.
    assert code == 0


def test_steps_command_lists_all_stages(capsys) -> None:
    assert main(["--json", "preprocess", "steps"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [row["step"] for row in payload["steps"]] == list(PIPELINE_STEPS)


def test_build_catgt_renders_a_flag_string(capsys) -> None:
    assert main(
        [
            "--json",
            "preprocess",
            "build-catgt",
            "--probe-folders",
            "--ap-filter",
            "--highpass",
            "300",
            "--lowpass",
            "9000",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "-prb_fld" in payload["catgt_cmd_string"]
    assert "-apfilter=butter,12,300,9000" in payload["catgt_cmd_string"]


def test_build_tprime_ni_analog_preset(capsys) -> None:
    assert main(
        [
            "--json",
            "preprocess",
            "build-tprime",
            "--ni-analog-preset",
            "0-1",
            "--tostream-kind",
            "imec",
            "--tostream-index",
            "0",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tostream_sync_params"] == "imec0"
    # Two channels, each with a rising and a falling extractor.
    assert payload["ni_extract_string"].count("-xa=") == 2
    assert payload["ni_extract_string"].count("-xia=") == 2


def test_curate_thresholds_returns_the_defaults(capsys) -> None:
    assert main(["--json", "curate", "thresholds"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {"noise", "mua", "non-somatic"} <= set(payload)


def test_version_reports_the_interpreter(capsys) -> None:
    assert main(["--json", "version"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["application"] == "NeuroPyGuiN"
    assert payload["python"]


def test_good_label_normalisation() -> None:
    from neuropyguin.cli.postproc_cmds import _is_good_label

    assert _is_good_label("good")
    assert _is_good_label("NON-SOMA")
    assert not _is_good_label("mua")
    assert not _is_good_label("nan")
    assert not _is_good_label("")


def test_command_error_is_reported_without_a_traceback(capsys) -> None:
    assert main(["postproc", "info", "definitely-not-a-folder"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "Traceback" not in err

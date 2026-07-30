from __future__ import annotations

from pathlib import Path

from neuropyguin.ecephys_runtime import ensure_ecephys_on_sys_path

ensure_ecephys_on_sys_path()

from ecephys_spike_sorting.modules.tPrime_helper.__main__ import call_TPrime  # noqa: E402


def test_tprime_skips_when_catgt_has_only_reference_stream(tmp_path: Path, monkeypatch, capsys) -> None:
    run_name = "recording_g0"
    run_directory = tmp_path / f"catgt_{run_name}"
    run_directory.mkdir()
    sync_file = run_directory / f"{run_name}.imec0.ap.xd_384_6_500.txt"
    sync_file.write_text("0.0\n1.0\n", encoding="utf-8")
    (run_directory / f"{run_name}_all_fyi.txt").write_text(
        f"sync_imec0={sync_file}\n",
        encoding="utf-8",
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("TPrime executable must not run without event streams")

    monkeypatch.setattr(
        "ecephys_spike_sorting.modules.tPrime_helper.__main__.subprocess.run",
        fail_if_called,
    )
    args = {
        "directories": {"extracted_data_directory": str(tmp_path)},
        "catGT_helper_params": {"run_name": "recording", "gate_string": "0"},
        "tPrime_helper_params": {
            "sync_period": 1.0,
            "sort_out_tag": "ks4",
            "toStream_sync_params": "imec0",
            "tPrime_path": str(tmp_path / "TPrime-linux"),
            "psth_ex_str": "",
        },
    }

    result = call_TPrime(args)

    assert result["skipped"] is True
    assert "no event streams" in result["skip_reason"]
    assert "TPrime skipped" in capsys.readouterr().out

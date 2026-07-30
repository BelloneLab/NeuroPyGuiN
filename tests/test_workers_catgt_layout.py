from __future__ import annotations

from pathlib import Path

from neuropyguin.preprocessing import catgt_command_for_input_layout, catgt_input_layout


RUN = "recording"
META = f"{RUN}_g0_t0.imec0.ap.meta"


def test_catgt_detects_flat_probe_layout_and_removes_only_input_flag(tmp_path: Path) -> None:
    gate_dir = tmp_path / f"{RUN}_g0"
    gate_dir.mkdir()
    (gate_dir / META).touch()

    layout = catgt_input_layout(
        str(tmp_path), RUN, "0", "0,0", "0"
    )
    command = catgt_command_for_input_layout(
        "-prb_fld -out_prb_fld -apfilter=butter,12,300,10000", layout
    )

    assert layout == "flat"
    assert "-prb_fld" not in command.split()
    assert "-out_prb_fld" in command.split()


def test_catgt_keeps_input_flag_for_probe_folder_layout(tmp_path: Path) -> None:
    probe_dir = tmp_path / f"{RUN}_g0" / f"{RUN}_g0_imec0"
    probe_dir.mkdir(parents=True)
    (probe_dir / META).touch()

    layout = catgt_input_layout(
        str(tmp_path), RUN, "0", "0", "0"
    )
    command = catgt_command_for_input_layout(
        "-prb_fld -out_prb_fld", layout
    )

    assert layout == "probe_folders"
    assert "-prb_fld" in command.split()

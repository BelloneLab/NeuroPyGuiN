from __future__ import annotations

from pathlib import Path

from neuropyguin.preprocessing import (
    catgt_extract_only_flags,
    catgt_extract_only_stream_string,
    catgt_stream_string,
    default_local_ks_output_dir,
    default_pipeline_ks_output_dir,
    expected_ni_catgt_output_patterns,
    has_ni_catgt_extractors,
    is_catgt_processed_bin,
    merge_extractors_into_catgt_command,
    parse_catgt_processed_bin_context,
    parse_spikeglx_bin_name,
)


def test_is_catgt_processed_bin_matches_tcat_and_catgt_names() -> None:
    assert is_catgt_processed_bin(r"D:\data\run_tcat.imec0.ap.bin")
    assert is_catgt_processed_bin(r"D:\data\catgt_run.imec1.ap.bin")
    assert not is_catgt_processed_bin(r"D:\data\plain_run.imec0.ap.bin")


def test_default_local_ks_output_dir_uses_bin_parent_and_probe() -> None:
    bin_file = Path(r"B:\NPX\processedData\VTA_NPX\29538\2\spike_sorting\catgt_29538_2_trial1_g0\29538_2_trial1_g0_imec0\29538_2_trial1_g0_tcat.imec0.ap.bin")
    expected = Path(r"B:\NPX\processedData\VTA_NPX\29538\2\spike_sorting\catgt_29538_2_trial1_g0\29538_2_trial1_g0_imec0\imec0_ks4")
    assert default_local_ks_output_dir(str(bin_file), "ks4", "0") == expected


def test_default_pipeline_ks_output_dir_uses_root_for_raw_inputs() -> None:
    bin_file = Path(r"B:\NPX\rawData\pups_NAc_NPX\vocal01\vocal01_g0_t0.imec0.ap.bin")
    expected = Path(r"D:\sorting\vocal01\ks4")
    assert (
        default_pipeline_ks_output_dir(
            str(bin_file),
            "ks4",
            "0",
            output_root=r"D:\sorting",
            run_name="vocal01",
        )
        == expected
    )


def test_default_pipeline_ks_output_dir_uses_processed_bin_parent_after_catgt() -> None:
    bin_file = Path(r"B:\NPX\processedData\pups_NAc_NPX\vocal01\catgt_vocal01_g0\vocal01_g0_imec0\vocal01_g0_tcat.imec0.ap.bin")
    expected = Path(r"B:\NPX\processedData\pups_NAc_NPX\vocal01\catgt_vocal01_g0\vocal01_g0_imec0\imec0_ks4")
    assert (
        default_pipeline_ks_output_dir(
            str(bin_file),
            "ks4",
            "0",
            output_root=r"D:\sorting",
            run_name="vocal01",
            store_next_to_bin=True,
        )
        == expected
    )


def test_parse_spikeglx_bin_name_supports_tcat_processed_files() -> None:
    parsed = parse_spikeglx_bin_name(
        r"B:\NPX\processedData\pups_NAc_NPX\vocal01\catgt_vocal01_g0\vocal01_g0_imec0\vocal01_g0_tcat.imec0.ap.bin"
    )
    assert parsed == {
        "run_name": "vocal01",
        "gate_string": "0",
        "trigger_string": "cat",
        "probe_string": "0",
    }


def test_catgt_stream_string_adds_ni_when_ni_extractors_are_present() -> None:
    cmd = "-prb_fld -out_prb_fld -apfilter=butter,12,300,10000 -xa=0,0,0,2,0,0"
    assert catgt_stream_string(cmd) == "-ap -ni"


def test_catgt_extract_only_stream_string_uses_ni_only_for_ni_extractors() -> None:
    cmd = "-prb_fld -out_prb_fld -xa=0,0,0,2,0,0 -xd=0,0,8,3,0"
    assert catgt_extract_only_stream_string(cmd) == "-ni"


def test_catgt_extract_only_stream_string_keeps_ap_when_imec_extractors_are_present() -> None:
    cmd = "-prb_fld -xd=2,0,6,3,0 -xa=0,0,0,2,0,0"
    assert catgt_extract_only_stream_string(cmd) == "-ap -ni"


def test_has_ni_catgt_extractors_detects_ni_only_rows() -> None:
    assert has_ni_catgt_extractors("-xa=0,0,0,2,0,0 -xd=2,0,-1,6,500")
    assert not has_ni_catgt_extractors("-xd=2,0,-1,6,500")


def test_expected_ni_catgt_output_patterns_cover_rise_fall_and_bitfield() -> None:
    cmd = "-xa=0,0,0,2,0,0 -xia=0,0,0,2,0,0 -xd=0,0,8,3,0 -xid=0,0,8,3,0 -bf=0,0,8,3,4,3"
    assert expected_ni_catgt_output_patterns(cmd, "vocal01", "0") == [
        "vocal01_g0_tcat.nidq.xa_0_0.txt",
        "vocal01_g0_tcat.nidq.xia_0_0.txt",
        "vocal01_g0_tcat.nidq.xd_8_3_0.txt",
        "vocal01_g0_tcat.nidq.xid_8_3_0.txt",
        "vocal01_g0_tcat.nidq.bfv_8_3_4.txt",
        "vocal01_g0_tcat.nidq.bft_8_3_4.txt",
    ]


def test_catgt_extract_only_flags_keep_extractors_but_strip_filtering() -> None:
    cmd = "-prb_fld -out_prb_fld -apfilter=butter,12,300,10000 -gfix=0.4,0.1,0.02 -xa=0,0,0,2,0,0 -xd=0,0,8,3,0"
    assert catgt_extract_only_flags(cmd) == "-prb_fld -out_prb_fld -xa=0,0,0,2,0,0 -xd=0,0,8,3,0 -no_tshift"


def test_merge_extractors_into_catgt_command_replaces_existing_extractor_tokens() -> None:
    cmd = "-prb_fld -out_prb_fld -gfix=0.4,0.1,0.02 -xa=0,0,0,2,0,0"
    extractors = "-xa=0,0,0,1.1,0,0 -xa=0,0,1,1.1,0,0"
    assert (
        merge_extractors_into_catgt_command(cmd, extractors)
        == "-prb_fld -out_prb_fld -gfix=0.4,0.1,0.02 -xa=0,0,0,1.1,0,0 -xa=0,0,1,1.1,0,0"
    )


def test_parse_catgt_processed_bin_context_returns_run_root_details() -> None:
    parsed = parse_catgt_processed_bin_context(
        r"B:\NPX\processedData\pups_NAc_NPX\vocal01\catgt_vocal01_g0\vocal01_g0_imec0\vocal01_g0_tcat.imec0.ap.bin"
    )
    assert parsed == {
        "catgt_dest": r"B:\NPX\processedData\pups_NAc_NPX\vocal01",
        "catgt_run_dir": r"B:\NPX\processedData\pups_NAc_NPX\vocal01\catgt_vocal01_g0",
        "catgt_run_name": "catgt_vocal01",
        "source_run_name": "vocal01",
        "gate_string": "0",
        "trigger_string": "cat",
        "probe_string": "0",
    }

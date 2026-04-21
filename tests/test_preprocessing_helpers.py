from __future__ import annotations

from pathlib import Path

from neuropyguin.preprocessing import (
    catgt_extract_command_string,
    catgt_extract_only_flags,
    catgt_extract_only_stream_string,
    catgt_extract_stream_selection,
    catgt_stream_string,
    completed_run_target_folders,
    default_kilosort_output_name,
    default_local_ks_output_dir,
    default_pipeline_output_dir,
    default_pipeline_ks_output_dir,
    default_pipeline_raw_output_layout,
    discover_completed_runs,
    expected_ni_catgt_output_patterns,
    extractor_label_rename_map,
    has_ni_catgt_extractors,
    infer_completed_run_name,
    is_catgt_processed_bin,
    merge_extractors_into_catgt_command,
    parse_kilosort_params_dat_path,
    parse_catgt_processed_bin_context,
    resolve_labelled_output_context,
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


def test_default_kilosort_output_name_uses_probe_when_available() -> None:
    assert default_kilosort_output_name("ks4", "0") == "imec0_ks4"
    assert default_kilosort_output_name("ks4", "") == "ks4"


def test_default_pipeline_output_dir_can_mirror_raw_hierarchy_into_spike_sorting() -> None:
    bin_file = Path(
        r"B:\NPX\rawData\VTA_NPX\31096\test\31096_test_sync_test_1_g0\31096_test_sync_test_1_g0_imec0\31096_test_sync_test_1_g0_t0.imec0.ap.bin"
    )
    expected = Path(r"B:\NPX\processedData\VTA_NPX\31096\test\spike_sorting")
    assert (
        default_pipeline_output_dir(
            str(bin_file),
            r"B:\NPX\processedData",
            run_name="31096_test_sync_test_1",
            mirror_raw_hierarchy=True,
        )
        == expected
    )


def test_default_pipeline_output_dir_can_mirror_numeric_session_hierarchy_into_spike_sorting() -> None:
    bin_file = Path(
        r"B:\NPX\rawData\VTA_NPX\31098\1\31098_1_NPX_basal_g0\31098_1_NPX_basal_g0_imec0\31098_1_NPX_basal_g0_t0.imec0.ap.bin"
    )
    expected = Path(r"B:\NPX\processedData\VTA_NPX\31098\1\spike_sorting")
    assert (
        default_pipeline_output_dir(
            str(bin_file),
            r"B:\NPX\processedData",
            run_name="31098_1_NPX_basal",
            mirror_raw_hierarchy=True,
        )
        == expected
    )


def test_default_pipeline_ks_output_dir_uses_root_for_raw_inputs() -> None:
    bin_file = Path(r"B:\NPX\rawData\pups_NAc_NPX\vocal01\vocal01_g0_t0.imec0.ap.bin")
    expected = Path(r"D:\sorting\vocal01\imec0_ks4")
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


def test_default_pipeline_ks_output_dir_can_mirror_raw_inputs_into_spike_sorting() -> None:
    bin_file = Path(
        r"B:\NPX\rawData\VTA_NPX\31101\1\31101_1_NPX_basal_g0\31101_1_NPX_basal_g0_imec0\31101_1_NPX_basal_g0_t0.imec0.ap.bin"
    )
    expected = Path(
        r"B:\NPX\processedData\VTA_NPX\31101\1\spike_sorting\imec0_ks4"
    )
    assert (
        default_pipeline_ks_output_dir(
            str(bin_file),
            "ks4",
            "0",
            output_root=r"B:\NPX\processedData",
            run_name="31101_1_NPX_basal",
            mirror_raw_hierarchy=True,
        )
        == expected
    )


def test_default_pipeline_raw_output_layout_keeps_root_and_ks_folder_in_same_mirrored_tree() -> None:
    bin_file = Path(
        r"B:\NPX\rawData\VTA_NPX\31102\7\31102_7_NPX_omaze_g0\31102_7_NPX_omaze_g0_imec0\31102_7_NPX_omaze_g0_t0.imec0.ap.bin"
    )
    extracted_root, ks_folder = default_pipeline_raw_output_layout(
        str(bin_file),
        r"B:\NPX\processedData",
        "ks4",
        "0",
        run_name="31102_7_NPX_omaze",
        mirror_raw_hierarchy=True,
    )
    assert extracted_root == Path(r"B:\NPX\processedData\VTA_NPX\31102\7\spike_sorting")
    assert ks_folder == Path(r"B:\NPX\processedData\VTA_NPX\31102\7\spike_sorting\imec0_ks4")


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


def test_extractor_label_rename_map_preserves_labelled_outputs() -> None:
    mapping = extractor_label_rename_map(
        "-xd=0,0,8,3,0[laser_on] -xa=0,0,1,1.1,0,0[reward]",
        "vocal01",
        "0",
    )
    assert mapping == {
        "vocal01_g0_tcat.nidq.xd_8_3_0.txt": "laser_on_vocal01_g0_tcat.nidq.xd_8_3_0.txt",
        "vocal01_g0_tcat.nidq.xd_8_3_0.adj.txt": "laser_on_vocal01_g0_tcat.nidq.xd_8_3_0.adj.txt",
        "vocal01_g0_tcat.nidq.xa_1_0.txt": "reward_vocal01_g0_tcat.nidq.xa_1_0.txt",
        "vocal01_g0_tcat.nidq.xa_1_0.adj.txt": "reward_vocal01_g0_tcat.nidq.xa_1_0.adj.txt",
    }


def test_catgt_extract_only_flags_keep_extractors_but_strip_filtering() -> None:
    cmd = "-prb_fld -out_prb_fld -apfilter=butter,12,300,10000 -gfix=0.4,0.1,0.02 -xa=0,0,0,2,0,0 -xd=0,0,8,3,0"
    assert catgt_extract_only_flags(cmd) == "-prb_fld -out_prb_fld -xa=0,0,0,2,0,0 -xd=0,0,8,3,0 -no_tshift"


def test_catgt_extract_command_string_keeps_full_command_when_ap_save_enabled() -> None:
    cmd = "-prb_fld -out_prb_fld -apfilter=butter,12,300,10000 -gfix=0.4,0.1,0.02 -xd=0,0,8,3,0"
    assert catgt_extract_command_string(cmd, save_ap_bin=True) == cmd
    assert catgt_extract_command_string(cmd, save_ap_bin=False) == "-prb_fld -out_prb_fld -xd=0,0,8,3,0 -no_tshift"


def test_catgt_extract_stream_selection_forces_ap_when_ap_save_enabled() -> None:
    cmd = "-prb_fld -out_prb_fld"
    assert catgt_extract_stream_selection(cmd, save_ap_bin=False) == ""
    assert catgt_extract_stream_selection(cmd, save_ap_bin=True) == "-ap"


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


def test_parse_kilosort_params_dat_path_reads_dat_path_line(tmp_path: Path) -> None:
    params = tmp_path / "params.py"
    params.write_text("sample_rate = 30000\ndat_path = 'B:/NPX/rawData/VTA_NPX/31098/2/run_g0/run_g0_imec0/run_g0_t0.imec0.ap.bin'\n", encoding="utf-8")
    assert parse_kilosort_params_dat_path(params) == r"B:\NPX\rawData\VTA_NPX\31098\2\run_g0\run_g0_imec0\run_g0_t0.imec0.ap.bin"


def test_infer_completed_run_name_prefers_catgt_ancestor() -> None:
    ks_folder = Path(
        r"B:\NPX\processedData\VTA_NPX\31098\2\spike_sorting\catgt_31098_2_NPX_object_social_food_g0\31098_2_NPX_object_social_food_g0_imec0\imec0_ks4"
    )
    assert infer_completed_run_name(ks_folder) == "31098_2_NPX_object_social_food"


def test_discover_completed_runs_finds_kilosort_outputs_under_processed_root(tmp_path: Path) -> None:
    processed_root = tmp_path / "processedData"
    ks_folder = (
        processed_root
        / "VTA_NPX"
        / "31098"
        / "2"
        / "spike_sorting"
        / "catgt_31098_2_NPX_object_social_food_g0"
        / "31098_2_NPX_object_social_food_g0_imec0"
        / "imec0_ks4"
    )
    ks_folder.mkdir(parents=True)
    params_file = ks_folder / "params.py"
    params_file.write_text(
        "sample_rate = 30000\n"
        "dat_path = 'B:/NPX/rawData/VTA_NPX/31098/2/31098_2_NPX_object_social_food_g0/"
        "31098_2_NPX_object_social_food_g0_imec0/31098_2_NPX_object_social_food_g0_t0.imec0.ap.bin'\n",
        encoding="utf-8",
    )

    entries = discover_completed_runs(processed_root)

    assert len(entries) == 1
    assert entries[0]["run_name"] == "31098_2_NPX_object_social_food"
    assert entries[0]["ks_folder"] == str(ks_folder.resolve())
    assert entries[0]["bin_file"] == (
        r"B:\NPX\rawData\VTA_NPX\31098\2\31098_2_NPX_object_social_food_g0"
        r"\31098_2_NPX_object_social_food_g0_imec0\31098_2_NPX_object_social_food_g0_t0.imec0.ap.bin"
    )
    assert entries[0]["params_file"] == str(params_file.resolve())
    assert entries[0]["source_root"] == str(processed_root.resolve())


def test_completed_run_target_folders_dedupes_and_skips_empty_entries() -> None:
    folders = completed_run_target_folders(
        [
            {"ks_folder": r"D:\runs\imec0_ks4"},
            {"ks_folder": ""},
            {"ks_folder": r"d:\runs\imec0_ks4"},
            {"ks_folder": r"D:\runs\imec1_ks4"},
        ]
    )
    assert folders == [r"D:\runs\imec0_ks4", r"D:\runs\imec1_ks4"]


def test_resolve_labelled_output_context_falls_back_to_existing_catgt_context_for_raw_input() -> None:
    fallback = {
        "catgt_dest": r"B:\NPX\processedData\VTA_NPX\31096\1\spike_sorting",
        "catgt_run_dir": r"B:\NPX\processedData\VTA_NPX\31096\1\spike_sorting\catgt_31096_1_NPX_basal_g0",
        "catgt_run_name": "catgt_31096_1_NPX_basal",
        "source_run_name": "31096_1_NPX_basal",
        "gate_string": "0",
        "trigger_string": "cat",
        "probe_string": "0",
    }
    raw_bin = (
        r"B:\NPX\rawData\VTA_NPX\31096\1\31096_1_NPX_basal_g0"
        r"\31096_1_NPX_basal_g0_imec0\31096_1_NPX_basal_g0_t0.imec0.ap.bin"
    )

    assert resolve_labelled_output_context(raw_bin, fallback) == fallback


def test_resolve_labelled_output_context_prefers_parsed_catgt_bin_context() -> None:
    fallback = {
        "catgt_dest": "fallback_dest",
        "catgt_run_dir": "fallback_run_dir",
        "catgt_run_name": "fallback_run_name",
        "source_run_name": "fallback_run",
        "gate_string": "9",
        "trigger_string": "cat",
        "probe_string": "9",
    }
    catgt_bin = r"B:\NPX\processedData\VTA_NPX\31096\1\spike_sorting\catgt_31096_1_NPX_basal_g0\31096_1_NPX_basal_g0_imec0\31096_1_NPX_basal_g0_tcat.imec0.ap.bin"

    assert resolve_labelled_output_context(catgt_bin, fallback) == {
        "catgt_dest": r"B:\NPX\processedData\VTA_NPX\31096\1\spike_sorting",
        "catgt_run_dir": r"B:\NPX\processedData\VTA_NPX\31096\1\spike_sorting\catgt_31096_1_NPX_basal_g0",
        "catgt_run_name": "catgt_31096_1_NPX_basal",
        "source_run_name": "31096_1_NPX_basal",
        "gate_string": "0",
        "trigger_string": "cat",
        "probe_string": "0",
    }

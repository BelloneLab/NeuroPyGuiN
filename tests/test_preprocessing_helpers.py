from __future__ import annotations

import json
from pathlib import Path

from neuropyguin.preprocessing import (
    build_concat_run_name,
    catgt_extract_command_string,
    catgt_extract_only_flags,
    catgt_extract_only_stream_string,
    catgt_extract_stream_selection,
    catgt_stream_string,
    completed_run_target_folders,
    concatenate_ap_session,
    default_concat_run_layout,
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
    is_concatenated_run_bin,
    merge_extractors_into_catgt_command,
    mirrored_concat_base_dir,
    parse_kilosort_params_dat_path,
    parse_catgt_processed_bin_context,
    resolve_labelled_output_context,
    parse_spikeglx_bin_name,
    strip_ni_catgt_extractor_flags,
    validate_concat_inputs,
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


def test_mirror_output_dir_mirrors_raw_input() -> None:
    out = "B:/NPX/processedData"
    run = "51543_object_mPFC_NAc_week1"
    raw = f"B:/NPX/rawData/mPFC-NAc/51543/mPFC_NAc_week1/object/{run}_g0/{run}_g0_imec0/{run}_g0_t0.imec0.ap.bin"
    got = default_pipeline_output_dir(raw, out, run_name=run, mirror_raw_hierarchy=True)
    assert got == Path(r"B:\NPX\processedData\mPFC-NAc\51543\mPFC_NAc_week1\object\spike_sorting")


def test_mirror_output_dir_without_experiment_level() -> None:
    # The experiment folder is optional for some projects; the mirror keeps whatever
    # levels sit between rawData and the session, then appends spike_sorting.
    out = "B:/NPX/processedData"
    run = "subj_sessA"
    raw = f"B:/NPX/rawData/ProjX/subj/sessA/{run}_g0/{run}_g0_imec0/{run}_g0_t0.imec0.ap.bin"
    got = default_pipeline_output_dir(raw, out, run_name=run, mirror_raw_hierarchy=True)
    assert got == Path(r"B:\NPX\processedData\ProjX\subj\sessA\spike_sorting")


def test_mirror_output_dir_handles_bin_already_under_output_root() -> None:
    # A concatenated bin we wrote into processedData has no 'rawData' token; it must
    # still mirror relative to the output root, not collapse to <output_root>/<run>.
    out = "B:/NPX/processedData"
    concat = (
        "B:/NPX/processedData/mPFC-NAc/51543/mPFC_NAc_week1/object/"
        "object_healthy_sick_g0/object_healthy_sick_g0_imec0/object_healthy_sick_g0_t0.imec0.ap.bin"
    )
    got = default_pipeline_output_dir(concat, out, run_name="object_healthy_sick", mirror_raw_hierarchy=True)
    assert got == Path(r"B:\NPX\processedData\mPFC-NAc\51543\mPFC_NAc_week1\object\spike_sorting")


def test_mirror_output_dir_flat_fallback_when_unmappable() -> None:
    out = "B:/NPX/processedData"
    stray = "C:/somewhere/else/run_g0/run_g0_imec0/run_g0_t0.imec0.ap.bin"
    got = default_pipeline_output_dir(stray, out, run_name="run", mirror_raw_hierarchy=True)
    assert got == Path(r"B:\NPX\processedData\run")


def test_mirrored_concat_base_dir_places_new_session_under_output_root() -> None:
    out = "B:/NPX/processedData"
    combined = "51543_object_healthy_sick_mPFC_NAc_week1"
    first = "B:/NPX/rawData/mPFC-NAc/51543/mPFC_NAc_week1/object/a_g0/a_g0_imec0/a_g0_t0.imec0.ap.bin"
    got = mirrored_concat_base_dir(first, out, combined, mirror_raw_hierarchy=True)
    assert got == Path(rf"B:\NPX\processedData\mPFC-NAc\51543\mPFC_NAc_week1\{combined}")


def test_concat_in_spike_sorting_round_trips_without_double_nesting() -> None:
    # The fused run lives inside <session>/spike_sorting/; sorting that bin later
    # must mirror back to the SAME spike_sorting folder, not spike_sorting/spike_sorting.
    out = "B:/NPX/processedData"
    combined = "object_healthy_sick"
    first = "B:/NPX/rawData/mPFC-NAc/51556/mPFC_NAc_week1/object/a_g0/a_g0_imec0/a_g0_t0.imec0.ap.bin"
    dest = mirrored_concat_base_dir(first, out, combined, mirror_raw_hierarchy=True) / "spike_sorting"
    fused = default_concat_run_layout(str(dest), combined, "0")["bin"]
    sort_dir = default_pipeline_output_dir(str(fused), out, run_name=combined, mirror_raw_hierarchy=True)
    assert dest == Path(r"B:\NPX\processedData\mPFC-NAc\51556\mPFC_NAc_week1\object_healthy_sick\spike_sorting")
    assert sort_dir == dest


def test_mirrored_concat_base_dir_legacy_when_mirror_off() -> None:
    out = "B:/NPX/processedData"
    first = Path("B:/NPX/rawData/mPFC-NAc/51543/mPFC_NAc_week1/object/a_g0/a_g0_imec0/a_g0_t0.imec0.ap.bin")
    got = mirrored_concat_base_dir(first, out, "combined", mirror_raw_hierarchy=False)
    assert got == first.parents[2]


def test_discover_completed_runs_finds_mirrored_run_and_ignores_pipeline_json(tmp_path: Path) -> None:
    run = "51542_object_healthy_sick_mPFC_NAc_week1"
    ss = tmp_path / "mPFC-NAc" / "51542" / "mPFC_NAc_week1" / "object_healthy_sick" / "spike_sorting"
    ks = ss / "imec0_ks4"
    ks.mkdir(parents=True)
    bin_path = ss / "catgt_x_g0" / f"{run}_g0_imec0" / f"{run}_g0_tcat.imec0.ap.bin"
    (ks / "params.py").write_text(f'dat_path = r"{bin_path}"\nn_channels_dat = 385\n', encoding="utf-8")
    # A per-run JSON folder beside the KS output must NOT be mistaken for a run.
    pj = ss / "pipeline_json"
    pj.mkdir()
    (pj / f"{run}_modules-input.json").write_text("{}", encoding="utf-8")

    entries = discover_completed_runs(tmp_path)
    assert len(entries) == 1
    assert Path(entries[0]["ks_folder"]) == ks.resolve()


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


def test_strip_ni_catgt_extractor_flags_drops_ni_keeps_ap() -> None:
    # Probe-only / concatenated run: NI (js=0) extractors must go, but the
    # AP-stream sync extractor (js=2) and filtering flags must survive so CatGT
    # is never asked to read a non-existent nidq stream.
    cmd = (
        "-prb_fld -out_prb_fld -apfilter=butter,12,300,10000 -gfix=0.4,0.1,0.02 "
        "-ni -xd=0,0,8,3,0 -xd=0,0,8,4,0 -xd=0,0,8,0,0 -xd=2,0,384,6,500"
    )
    stripped = strip_ni_catgt_extractor_flags(cmd)
    assert stripped == (
        "-prb_fld -out_prb_fld -apfilter=butter,12,300,10000 -gfix=0.4,0.1,0.02 "
        "-xd=2,0,384,6,500"
    )
    assert not has_ni_catgt_extractors(stripped)
    # The stream selector must no longer request -ni.
    assert "-ni" not in catgt_stream_string(stripped).split()
    assert catgt_stream_string(stripped) == "-ap"


def test_strip_ni_catgt_extractor_flags_drops_labelled_ni_extractors() -> None:
    cmd = "-prb_fld -xd=0,0,8,3,0[laser] -xd=2,0,384,6,500[sync]"
    assert strip_ni_catgt_extractor_flags(cmd) == "-prb_fld -xd=2,0,384,6,500[sync]"


def test_is_concatenated_run_bin_detects_splitinfo_and_manifest(tmp_path: Path) -> None:
    imec = tmp_path / "run_g0" / "run_g0_imec0"
    imec.mkdir(parents=True)
    bin_file = imec / "run_g0_t0.imec0.ap.bin"
    bin_file.write_bytes(b"")
    # Plain probe-only run (no concat markers) is not concatenated.
    assert not is_concatenated_run_bin(bin_file)
    # A *.ap.splitinfo.json beside the bin marks a concatenated run.
    (imec / "run_g0_t0.imec0.ap.splitinfo.json").write_text("{}", encoding="utf-8")
    assert is_concatenated_run_bin(bin_file)
    # The concat_manifest.json alone is also sufficient.
    (imec / "run_g0_t0.imec0.ap.splitinfo.json").unlink()
    assert not is_concatenated_run_bin(bin_file)
    (imec / "concat_manifest.json").write_text("{}", encoding="utf-8")
    assert is_concatenated_run_bin(bin_file)


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


# ---------------------------------------------------------------------------
# Multi-session concatenation
# ---------------------------------------------------------------------------


def test_build_concat_run_name_joins_runs_and_collapses_large_sets() -> None:
    assert build_concat_run_name(["sessA", "sessB"]) == "concat_sessA__sessB"
    assert build_concat_run_name(["a b", "c/d"]) == "concat_a_b__c_d"
    assert build_concat_run_name(["r1", "r2", "r3", "r4", "r5"]) == "concat_r1__and4more"


def test_default_concat_run_layout_builds_spikeglx_standard_paths() -> None:
    layout = default_concat_run_layout(r"D:\out", "concat_a__b", "0")
    assert layout["bin"] == Path(r"D:\out\concat_a__b_g0\concat_a__b_g0_imec0\concat_a__b_g0_t0.imec0.ap.bin")
    assert layout["meta"] == Path(r"D:\out\concat_a__b_g0\concat_a__b_g0_imec0\concat_a__b_g0_t0.imec0.ap.meta")
    # The fused file parses back to the combined run name like a normal recording.
    assert parse_spikeglx_bin_name(str(layout["bin"]))["run_name"] == "concat_a__b"


def _write_fake_ap(path: Path, data) -> Path:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(data, dtype=np.int16).tofile(path)
    meta = Path(str(path).replace(".ap.bin", ".ap.meta"))
    n_chan = np.asarray(data).shape[1]
    meta.write_text(
        "\n".join(
            [
                f"nSavedChans={n_chan}",
                f"snsApLfSy={n_chan - 1},0,1",
                "imSampRate=30000",
                f"fileName={path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return meta


def test_concatenate_ap_session_glues_raw_samples_and_writes_split_map(tmp_path: Path) -> None:
    import numpy as np

    c = 4
    data1 = np.arange(5 * c, dtype=np.int16).reshape(5, c)
    data2 = (np.arange(3 * c, dtype=np.int16) + 100).reshape(3, c)
    bin1 = tmp_path / "runA_g0_t0.imec0.ap.bin"
    bin2 = tmp_path / "runB_g0_t0.imec0.ap.bin"
    meta1 = _write_fake_ap(bin1, data1)
    meta2 = _write_fake_ap(bin2, data2)

    target = tmp_path / "concat_runA__runB_g0_t0.imec0.ap.bin"
    # batch shorter than each file to exercise the streaming path; cleaning off
    # so the output is an exact raw concatenation we can compare byte for byte.
    result = concatenate_ap_session(
        [str(bin1), str(bin2)],
        [str(meta1), str(meta2)],
        target,
        svd_clean=False,
        batch_seconds=2 / 30000,
    )

    assert result["samplelist"] == [5, 3]
    assert Path(result["manifest_path"]).exists()
    fused = np.fromfile(target, dtype=np.int16).reshape(-1, c)
    assert np.array_equal(fused, np.vstack([data1, data2]))

    split = json.loads(Path(result["splitinfo_path"]).read_text(encoding="utf-8"))
    assert split["sampling_rate"] == 30000.0
    assert [(s["start_sample"], s["end_sample"]) for s in split["segments"]] == [(0, 5), (5, 8)]

    combined_meta = Path(result["meta_path"]).read_text(encoding="utf-8")
    assert "fileSizeBytes=64" in combined_meta  # 2 bytes * 4 chan * 8 samples
    assert "concatSampleList=5,3" in combined_meta


def test_concatenate_svd_clean_removes_shared_component(tmp_path: Path) -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    n_samp, ap = 64, 8
    # A strong shared component on all AP channels plus small per-channel noise.
    shared = rng.normal(0, 50, size=(n_samp, 1)) * np.ones((1, ap))
    noise = rng.normal(0, 5, size=(n_samp, ap))
    ap_data = np.clip(np.rint(shared + noise), -32768, 32767).astype(np.int16)
    sync = np.full((n_samp, 1), 7, dtype=np.int16)
    data = np.hstack([ap_data, sync]).astype(np.int16)

    b = tmp_path / "runA_g0_t0.imec0.ap.bin"
    m = _write_fake_ap(b, data)
    # Two copies so the helper has the >=1 file it expects; identical content.
    b2 = tmp_path / "runB_g0_t0.imec0.ap.bin"
    m2 = _write_fake_ap(b2, data)

    target = tmp_path / "concat_g0_t0.imec0.ap.bin"
    result = concatenate_ap_session(
        [str(b), str(b2)],
        [str(m), str(m2)],
        target,
        svd_clean=True,
        n_svd_components=1,
        batch_seconds=10.0,
    )

    fused = np.fromfile(target, dtype=np.int16).reshape(-1, data.shape[1])
    first_half = fused[:n_samp]
    # The dominant shared component should be strongly attenuated on AP channels.
    assert np.var(first_half[:, :ap]) < np.var(data[:, :ap]) * 0.2
    # Sync (non-AP) channel is passed through untouched.
    assert np.array_equal(first_half[:, ap], data[:, ap])
    assert result["samplelist"] == [n_samp, n_samp]


def test_validate_concat_inputs_flags_channel_mismatch(tmp_path: Path) -> None:
    import numpy as np

    m1 = _write_fake_ap(tmp_path / "a_g0_t0.imec0.ap.bin", np.zeros((4, 4), dtype=np.int16))
    m2 = _write_fake_ap(tmp_path / "b_g0_t0.imec0.ap.bin", np.zeros((4, 5), dtype=np.int16))

    ok, reason, _info = validate_concat_inputs([str(m1), str(m2)])
    assert not ok
    assert "mismatch" in reason.lower()

    ok2, reason2, info = validate_concat_inputs([str(m1), str(m1)])
    assert ok2 and reason2 == ""
    assert int(info["n_saved"]) == 4


def test_split_concatenated_sort_masks_shifts_and_preserves_identity(tmp_path: Path) -> None:
    import numpy as np

    from neuropyguin.preprocessing import (
        split_concatenated_sort,
        write_concat_splitinfo,
    )

    # Two source sessions with a combined length of 100 samples (60 + 40).
    sess_a = tmp_path / "runA_g0" / "runA_g0_imec0" / "runA_g0_t0.imec0.ap.bin"
    sess_b = tmp_path / "runB_g0" / "runB_g0_imec0" / "runB_g0_t0.imec0.ap.bin"
    for b in (sess_a, sess_b):
        b.parent.mkdir(parents=True, exist_ok=True)
        b.write_bytes(b"\x00" * 16)
        Path(str(b).replace(".ap.bin", ".ap.meta")).write_text("imSampRate=30000\n", encoding="utf-8")

    ks = tmp_path / "concat_runA__runB_g0" / "concat_runA__runB_g0_imec0" / "imec0_ks4"
    ks.mkdir(parents=True)
    target_bin = ks.parent / "concat_runA__runB_g0_t0.imec0.ap.bin"
    splitinfo = write_concat_splitinfo(
        [str(sess_a).replace(".ap.bin", ".ap.meta"), str(sess_b).replace(".ap.bin", ".ap.meta")],
        target_bin,
        [str(sess_a), str(sess_b)],
        [60, 40],
    )

    # Minimal phy-like outputs: per-spike arrays + one shared array.
    spike_times = np.array([5, 30, 65, 95], dtype=np.uint64)
    spike_clusters = np.array([0, 1, 0, 2], dtype=np.int32)
    amplitudes = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    np.save(ks / "spike_times.npy", spike_times)
    np.save(ks / "spike_clusters.npy", spike_clusters)
    np.save(ks / "amplitudes.npy", amplitudes)
    templates = np.zeros((3, 61, 4), dtype=np.float32)
    np.save(ks / "templates.npy", templates)  # shared, must be copied verbatim
    (ks / "params.py").write_text(
        f"dat_path = r'{target_bin}'\nn_channels_dat = 385\nsample_rate = 30000\n", encoding="utf-8"
    )

    manifest = split_concatenated_sort(ks, splitinfo_path=str(splitinfo), copy_events=False)

    assert manifest["n_sessions"] == 2
    sessions = manifest["sessions"]
    dir_a = Path(sessions[0]["output_dir"])
    dir_b = Path(sessions[1]["output_dir"])

    # Session A owns samples [0, 60): spikes at 5 and 30, unshifted because start=0.
    assert np.array_equal(np.load(dir_a / "spike_times.npy"), np.array([5, 30], dtype=np.uint64))
    assert np.array_equal(np.load(dir_a / "spike_clusters.npy"), np.array([0, 1], dtype=np.int32))
    # Session B owns samples [60, 100): spikes at 65 and 95, shifted by -60 -> 5 and 35.
    assert np.array_equal(np.load(dir_b / "spike_times.npy"), np.array([5, 35], dtype=np.uint64))
    assert np.array_equal(np.load(dir_b / "spike_clusters.npy"), np.array([0, 2], dtype=np.int32))
    # Shared templates copied verbatim into each session (identity preserved).
    assert np.load(dir_a / "templates.npy").shape == (3, 61, 4)
    assert np.load(dir_b / "templates.npy").shape == (3, 61, 4)
    # params.py repointed at the per-session recording, not the concatenated bin.
    assert str(sess_b) in (dir_b / "params.py").read_text(encoding="utf-8")


def test_find_concat_splitinfo_for_ks_folder_via_params(tmp_path: Path) -> None:
    from neuropyguin.preprocessing import find_concat_splitinfo_for_ks_folder

    bin_dir = tmp_path / "concat_x_g0" / "concat_x_g0_imec0"
    bin_dir.mkdir(parents=True)
    target_bin = bin_dir / "concat_x_g0_t0.imec0.ap.bin"
    split = bin_dir / "concat_x_g0_t0.imec0.ap.splitinfo.json"
    split.write_text(json.dumps({"segments": [], "sampling_rate": 30000}), encoding="utf-8")

    ks = tmp_path / "elsewhere" / "imec0_ks4"
    ks.mkdir(parents=True)
    (ks / "params.py").write_text(f"dat_path = r'{target_bin}'\n", encoding="utf-8")

    found = find_concat_splitinfo_for_ks_folder(ks)
    assert found is not None and Path(found) == split


def test_session_event_files_filters_extra_roots_by_run_name(tmp_path: Path) -> None:
    from neuropyguin.preprocessing import _session_event_files

    raw = tmp_path / "raw" / "runA_g0" / "runA_g0_imec0" / "runA_g0_t0.imec0.ap.bin"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"\x00" * 8)

    out = tmp_path / "processed"
    cat_a = out / "runA" / "catgt_runA_g0" / "runA_g0_imec0"
    cat_a.mkdir(parents=True)
    (cat_a / "runA_g0_tcat.nidq.xd_8_3_0.txt").write_text("0.1\n", encoding="utf-8")
    cat_b = out / "runB" / "catgt_runB_g0" / "runB_g0_imec0"
    cat_b.mkdir(parents=True)
    (cat_b / "runB_g0_tcat.nidq.xd_8_3_0.txt").write_text("0.2\n", encoding="utf-8")

    names = [f.name for f in _session_event_files(raw, run_name="runA", extra_roots=[str(out)])]
    assert "runA_g0_tcat.nidq.xd_8_3_0.txt" in names
    assert "runB_g0_tcat.nidq.xd_8_3_0.txt" not in names

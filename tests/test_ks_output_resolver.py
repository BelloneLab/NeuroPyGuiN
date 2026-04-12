from __future__ import annotations

from pathlib import Path

from neuropyguin.ks_output_resolver import (
    archive_output_dir,
    find_kilosort_output_dir,
    find_metrics_file,
    has_kilosort_output,
    next_archived_output_dir,
)


REQUIRED = (
    "spike_times.npy",
    "spike_clusters.npy",
    "spike_templates.npy",
    "amplitudes.npy",
    "templates.npy",
    "channel_map.npy",
    "channel_positions.npy",
    "whitening_mat_inv.npy",
)


def _make_ks_folder(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED:
        (path / name).write_bytes(b"0")
    (path / "params.py").write_text("sample_rate = 30000\n", encoding="utf-8")
    (path / "metrics.csv").write_text("cluster_id,firing_rate\n1,5.0\n", encoding="utf-8")
    return path


def test_resolver_prefers_nested_matching_imec_ks_folder(tmp_path: Path) -> None:
    root = tmp_path / "spike_sorting"
    requested = root / "29538_2_trial1_g0_tcat.imec0.ap"
    requested.mkdir(parents=True)
    actual = _make_ks_folder(root / "catgt_29538_2_trial1_g0" / "29538_2_trial1_g0_imec0" / "imec0_ks4")

    resolved = find_kilosort_output_dir(requested / "ks4", ks_tag="ks4", probe_string="0", extra_roots=[root], max_depth=4)

    assert resolved == actual
    assert has_kilosort_output(resolved)


def test_find_metrics_file_uses_resolved_ks_folder(tmp_path: Path) -> None:
    root = tmp_path / "spike_sorting"
    requested = root / "29538_2_trial1_g0_tcat.imec0.ap"
    requested.mkdir(parents=True)
    actual = _make_ks_folder(root / "catgt_29538_2_trial1_g0" / "29538_2_trial1_g0_imec0" / "imec0_ks4")

    metrics_path = find_metrics_file(requested, ks_tag="ks4", probe_string="0", extra_roots=[root], max_depth=4)

    assert metrics_path == actual / "metrics.csv"


def test_resolver_prefers_mirrored_catgt_probe_folder_over_legacy_flat_run_folder(tmp_path: Path) -> None:
    processed_root = tmp_path / "processedData"
    requested = processed_root / "VTA_NPX" / "31098" / "1" / "spike_sorting" / "imec0_ks4"
    requested.parent.mkdir(parents=True, exist_ok=True)

    legacy = _make_ks_folder(processed_root / "31098_1_NPX_basal" / "imec0_ks4")
    mirrored = _make_ks_folder(
        processed_root
        / "VTA_NPX"
        / "31098"
        / "1"
        / "spike_sorting"
        / "catgt_31098_1_NPX_basal_g0"
        / "31098_1_NPX_basal_g0_imec0"
        / "imec0_ks4"
    )

    resolved = find_kilosort_output_dir(
        requested,
        ks_tag="ks4",
        probe_string="0",
        extra_roots=[processed_root],
        max_depth=4,
    )

    assert resolved == mirrored
    assert resolved != legacy


def test_archive_output_dir_versions_existing_kilosort_folders(tmp_path: Path) -> None:
    root = tmp_path / "archive_case"
    current = _make_ks_folder(root / "ks4")

    assert next_archived_output_dir(current) == root / "ks4_0"

    archived0 = archive_output_dir(current)
    assert archived0 == root / "ks4_0"
    assert archived0.exists()

    _make_ks_folder(root / "ks4")
    archived1 = archive_output_dir(root / "ks4")
    assert archived1 == root / "ks4_1"
    assert archived1.exists()

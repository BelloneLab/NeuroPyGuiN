from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neuropyguin.pybombcell_integration import (
    _ensure_saved_metrics_maxchannels,
    _write_pybombcell_manifest,
    pybombcell_metadata_path,
    pybombcell_default_settings,
    pybombcell_settings_signature,
    run_pybombcell_on_folder,
    run_pybombcell_on_folders,
    summarize_saved_pybombcell_results,
)


def test_summarize_saved_pybombcell_results_counts_labels_and_reuses_matching_manifest(tmp_path: Path) -> None:
    ks = tmp_path / "imec0_ks4"
    save_dir = ks / "bombcell"
    plots_dir = ks / "bombcell_plots"
    save_dir.mkdir(parents=True)
    plots_dir.mkdir()
    metrics_csv = save_dir / "templates._bc_qMetrics.csv"
    metrics_csv.write_text("cluster_id\n0\n1\n2\n", encoding="utf-8")
    labels_csv = ks / "bombcell_labels.csv"
    labels_csv.write_text(
        "cluster_id,bombcell_label\n0,good\n1,mua\n2,noise\n",
        encoding="utf-8",
    )

    settings = pybombcell_default_settings()
    _write_pybombcell_manifest(
        ks,
        settings,
        metrics_csv=metrics_csv,
        labels_csv=labels_csv,
        plots_dir=plots_dir,
        counts={"good": 1, "mua": 1, "noise": 1},
        n_units=3,
    )

    summary = summarize_saved_pybombcell_results(ks, settings=settings)

    assert summary["counts"] == {"good": 1, "mua": 1, "noise": 1}
    assert summary["n_units"] == 3
    assert summary["can_reuse"] is True
    assert summary["cache_reason"] == "matching_signature"


def test_summarize_saved_pybombcell_results_normalizes_uppercase_labels_and_writes_metadata(tmp_path: Path) -> None:
    ks = tmp_path / "imec0_ks4"
    save_dir = ks / "bombcell"
    save_dir.mkdir(parents=True)
    metrics_csv = save_dir / "templates._bc_qMetrics.csv"
    metrics_csv.write_text("cluster_id\n0\n1\n2\n", encoding="utf-8")
    labels_csv = ks / "bombcell_labels.csv"
    labels_csv.write_text(
        "cluster_id,bombcell_label\n0,GOOD\n1,MUA\n2,NON-SOMA GOOD\n",
        encoding="utf-8",
    )

    settings = pybombcell_default_settings()
    manifest_path = _write_pybombcell_manifest(
        ks,
        settings,
        metrics_csv=metrics_csv,
        labels_csv=labels_csv,
        plots_dir=ks / "bombcell_plots",
        counts={"GOOD": 1, "MUA": 1, "NON-SOMA GOOD": 1},
        n_units=3,
    )

    summary = summarize_saved_pybombcell_results(ks, settings=settings)

    assert manifest_path == pybombcell_metadata_path(ks)
    assert pybombcell_metadata_path(ks).exists()
    assert summary["counts"] == {"good": 1, "mua": 1, "non_soma": 1}
    assert summary["n_units"] == 3


def test_summarize_saved_pybombcell_results_requires_rerun_when_settings_change(tmp_path: Path) -> None:
    ks = tmp_path / "imec0_ks4"
    save_dir = ks / "bombcell"
    save_dir.mkdir(parents=True)
    metrics_csv = save_dir / "templates._bc_qMetrics.csv"
    metrics_csv.write_text("cluster_id\n0\n", encoding="utf-8")
    labels_csv = ks / "bombcell_labels.csv"
    labels_csv.write_text("cluster_id,bombcell_label\n0,good\n", encoding="utf-8")

    settings = pybombcell_default_settings()
    _write_pybombcell_manifest(
        ks,
        settings,
        metrics_csv=metrics_csv,
        labels_csv=labels_csv,
        plots_dir=ks / "bombcell_plots",
        counts={"good": 1},
        n_units=1,
    )

    changed = dict(settings)
    changed["minAmplitude"] = 55
    summary = summarize_saved_pybombcell_results(ks, settings=changed)

    assert summary["can_reuse"] is False
    assert summary["cache_reason"] == "settings_changed"
    assert summary["manifest_signature"] == pybombcell_settings_signature(settings)
    assert summary["settings_signature"] == pybombcell_settings_signature(changed)


def test_summarize_saved_pybombcell_results_reuses_legacy_default_outputs(tmp_path: Path) -> None:
    ks = tmp_path / "imec0_ks4"
    save_dir = ks / "bombcell"
    save_dir.mkdir(parents=True)
    (save_dir / "templates._bc_qMetrics.csv").write_text("cluster_id\n0\n1\n", encoding="utf-8")
    (ks / "bombcell_labels.csv").write_text(
        "cluster_id,bombcell_label\n0,good\n1,non_soma\n",
        encoding="utf-8",
    )

    summary = summarize_saved_pybombcell_results(ks, settings=pybombcell_default_settings())

    assert summary["can_reuse"] is True
    assert summary["cache_reason"] == "legacy_default_assumed"
    assert summary["counts"] == {"good": 1, "non_soma": 1}


def test_summarize_saved_pybombcell_results_uses_manifest_counts_when_labels_missing(tmp_path: Path) -> None:
    ks = tmp_path / "imec0_ks4"
    save_dir = ks / "bombcell"
    save_dir.mkdir(parents=True)
    metrics_csv = save_dir / "templates._bc_qMetrics.csv"
    metrics_csv.write_text("cluster_id\n0\n1\n", encoding="utf-8")

    settings = pybombcell_default_settings()
    _write_pybombcell_manifest(
        ks,
        settings,
        metrics_csv=metrics_csv,
        labels_csv=ks / "bombcell_labels.csv",
        plots_dir=ks / "bombcell_plots",
        counts={"GOOD": 1, "NON-SOMA MUA": 1},
        n_units=2,
        metrics_reused=True,
        mode="reclassified_from_saved_metrics",
    )

    summary = summarize_saved_pybombcell_results(ks, settings=settings)

    assert summary["counts"] == {"good": 1, "non_soma": 1}
    assert summary["n_units"] == 2
    assert summary["metrics_reused"] is True
    assert summary["mode"] == "reclassified_from_saved_metrics"


def test_run_pybombcell_on_folder_reuses_saved_metrics_when_settings_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ks = tmp_path / "imec0_ks4"
    save_dir = ks / "bombcell"
    save_dir.mkdir(parents=True)
    metrics_csv = save_dir / "templates._bc_qMetrics.csv"
    metrics_csv.write_text("cluster_id\n0\n", encoding="utf-8")
    (ks / "bombcell_labels.csv").write_text("cluster_id,bombcell_label\n0,GOOD\n", encoding="utf-8")

    settings = pybombcell_default_settings()
    _write_pybombcell_manifest(
        ks,
        settings,
        metrics_csv=metrics_csv,
        labels_csv=ks / "bombcell_labels.csv",
        plots_dir=ks / "bombcell_plots",
        counts={"GOOD": 1},
        n_units=1,
    )

    changed = dict(settings)
    changed["minAmplitude"] = 55
    called = {"value": False}

    def fake_refresh(folder, normalized_settings, *, save_plots):
        called["value"] = True
        assert Path(folder) == ks
        assert normalized_settings["minAmplitude"] == 55
        assert save_plots is True
        return {
            "metrics_csv": str(metrics_csv),
            "labels_csv": str(ks / "bombcell_labels.csv"),
            "plots_dir": str(ks / "bombcell_plots"),
            "counts": {"good": 1},
            "n_units": 1,
            "cached": False,
            "metrics_reused": True,
            "cache_reason": "reused_saved_metrics",
        }

    monkeypatch.setattr("neuropyguin.pybombcell_integration._refresh_pybombcell_outputs_from_saved_metrics", fake_refresh)

    result = run_pybombcell_on_folder(str(ks), settings=changed)

    assert called["value"] is True
    assert result["metrics_reused"] is True
    assert result["cache_reason"] == "reused_saved_metrics"


def test_run_pybombcell_on_folder_refreshes_saved_metrics_even_when_settings_match(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ks = tmp_path / "imec0_ks4"
    save_dir = ks / "bombcell"
    plots_dir = ks / "bombcell_plots"
    save_dir.mkdir(parents=True)
    plots_dir.mkdir()
    metrics_csv = save_dir / "templates._bc_qMetrics.csv"
    metrics_csv.write_text("cluster_id\n0\n", encoding="utf-8")
    labels_csv = ks / "bombcell_labels.csv"
    labels_csv.write_text("cluster_id,bombcell_label\n0,GOOD\n", encoding="utf-8")

    settings = pybombcell_default_settings()
    _write_pybombcell_manifest(
        ks,
        settings,
        metrics_csv=metrics_csv,
        labels_csv=labels_csv,
        plots_dir=plots_dir,
        counts={"GOOD": 1},
        n_units=1,
    )

    called = {"value": False}

    def fake_refresh(folder, normalized_settings, *, save_plots):
        called["value"] = True
        assert Path(folder) == ks
        assert normalized_settings == settings
        assert save_plots is True
        return {
            "metrics_csv": str(metrics_csv),
            "labels_csv": str(labels_csv),
            "plots_dir": str(plots_dir),
            "counts": {"good": 1},
            "n_units": 1,
            "cached": False,
            "metrics_reused": True,
            "cache_reason": "reused_saved_metrics",
        }

    monkeypatch.setattr("neuropyguin.pybombcell_integration._refresh_pybombcell_outputs_from_saved_metrics", fake_refresh)

    result = run_pybombcell_on_folder(str(ks), settings=settings)

    assert called["value"] is True
    assert result["metrics_reused"] is True
    assert result["cached"] is False


def test_ensure_saved_metrics_maxchannels_rebuilds_from_template_waveforms() -> None:
    metrics_df = pd.DataFrame({"cluster_id": [0, 2], "phy_clusterID": [0, 2], "nSpikes": [10, 12]})
    template_waveforms = np.zeros((3, 2, 4), dtype=float)
    template_waveforms[0, 0, 1] = -5.0
    template_waveforms[1, 0, 3] = -4.0
    template_waveforms[2, 0, 2] = -6.0

    rebuilt, full_max_channels = _ensure_saved_metrics_maxchannels(metrics_df, template_waveforms)

    assert rebuilt["maxChannels"].tolist() == [1, 2]
    assert full_max_channels.tolist() == [1, 3, 2]


def test_run_pybombcell_on_folders_aggregates_success_cache_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(folder: str, save_plots: bool = True, force_recompute: bool = False, settings=None):
        if folder.endswith("bad"):
            raise RuntimeError("boom")
        return {
            "cached": folder.endswith("cached"),
            "counts": {"good": 2, "noise": 1, "mua": 0, "non_soma": 0},
            "n_units": 3,
        }

    monkeypatch.setattr("neuropyguin.pybombcell_integration.run_pybombcell_on_folder", fake_run)

    payload = run_pybombcell_on_folders(["run_a", "run_cached", "run_bad"])

    assert payload["summary"] == {
        "total": 3,
        "reran": 1,
        "reused_metrics": 0,
        "cached": 1,
        "failed": 1,
        "good": 4,
        "noise": 2,
        "mua": 0,
        "non_soma": 0,
    }
    assert [item["ok"] for item in payload["results"]] == [True, True, False]

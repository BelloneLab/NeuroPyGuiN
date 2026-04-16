from __future__ import annotations

from pathlib import Path

import pandas as pd

from neuropyguin.bombcell_core import run_bombcell_on_folder_with_thresholds, sync_phy_cluster_group


def test_sync_phy_cluster_group_uses_kslabel_when_group_is_all_noise(tmp_path: Path) -> None:
    ks = tmp_path / "imec0_ks4"
    ks.mkdir(parents=True, exist_ok=True)
    (ks / "cluster_KSLabel.tsv").write_text(
        "cluster_id\tKSLabel\n0\tgood\n1\tmua\n2\tnoise\n",
        encoding="utf-8",
    )
    (ks / "cluster_group.tsv").write_text(
        "cluster_id\tgroup\n0\tnoise\n1\tnoise\n2\tnoise\n",
        encoding="utf-8",
    )

    result = sync_phy_cluster_group(ks, force=False)
    assert result["updated"] is True
    out = pd.read_csv(ks / "cluster_group.tsv", sep="\t")
    assert out["group"].tolist() == ["good", "mua", "noise"]


def test_sync_phy_cluster_group_keeps_existing_non_noise_groups(tmp_path: Path) -> None:
    ks = tmp_path / "imec0_ks4"
    ks.mkdir(parents=True, exist_ok=True)
    (ks / "cluster_KSLabel.tsv").write_text(
        "cluster_id\tKSLabel\n0\tnoise\n1\tnoise\n",
        encoding="utf-8",
    )
    (ks / "cluster_group.tsv").write_text(
        "cluster_id\tgroup\n0\tgood\n1\tmua\n",
        encoding="utf-8",
    )

    result = sync_phy_cluster_group(ks, force=False)
    assert result["updated"] is False
    out = pd.read_csv(ks / "cluster_group.tsv", sep="\t")
    assert out["group"].tolist() == ["good", "mua"]


def test_run_bombcell_saves_phy_cluster_group(tmp_path: Path) -> None:
    ks = tmp_path / "imec0_ks4"
    ks.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"cluster_id": [0, 1, 2]}).to_csv(ks / "metrics.csv", index=False)

    result = run_bombcell_on_folder_with_thresholds(str(ks), thresholds=None)
    sync_result = result.get("phy_group_sync", {})
    assert sync_result.get("updated") is True
    assert (ks / "bombcell_labels.csv").exists()
    group = pd.read_csv(ks / "cluster_group.tsv", sep="\t")
    assert sorted(group["group"].unique().tolist()) == ["good"]

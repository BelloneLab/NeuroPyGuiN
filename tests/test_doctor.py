from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest

from neuropyguin import doctor


def test_version_tuple_parses_local_and_partial_versions() -> None:
    assert doctor._version_tuple("2.8.0+cu126") == (2, 8, 0)
    assert doctor._version_tuple("6.7.3") == (6, 7, 3)
    assert doctor._version_tuple("4.1") == (4, 1)
    assert doctor._version_tuple("2.0.0rc1") == (2, 0, 0)
    assert doctor._version_tuple("unknown") == ()


def test_module_group_lists_missing_imports_and_builds_a_pip_fix(monkeypatch) -> None:
    present = {"argschema"}
    monkeypatch.setattr(doctor, "_is_importable", lambda name: name in present)

    result = doctor._module_group(
        key="ecephys_imports",
        category="Spike sorting",
        label="ecephys pipeline imports",
        specs=doctor._ECEPHYS_IMPORTS,
        severity=doctor.REQUIRED,
        ok_detail="all importable",
    )

    assert result.status == doctor.FAIL
    assert result.is_blocking
    assert "git" in result.detail and "xarray" in result.detail
    assert "argschema" not in result.detail
    # The distribution name, not the import name, must appear in the fix.
    assert '"GitPython"' in result.fix
    assert "-m pip install" in result.fix


def test_module_group_is_ok_when_everything_imports(monkeypatch) -> None:
    monkeypatch.setattr(doctor, "_is_importable", lambda name: True)
    result = doctor._module_group(
        key="scientific",
        category="Application core",
        label="Scientific stack",
        specs=doctor._CORE_SCIENTIFIC,
        severity=doctor.REQUIRED,
        ok_detail="all present",
    )
    assert result.status == doctor.OK
    assert not result.is_blocking
    assert result.fix == ""


def test_ecephys_schema_stack_requires_both_packages(monkeypatch) -> None:
    monkeypatch.setattr(doctor, "_is_importable", lambda name: name == "argschema")

    result = doctor._check_ecephys_schema_stack()

    assert result.status == doctor.FAIL
    assert result.is_blocking
    assert "marshmallow" in result.detail
    assert "argschema==1.17.5" in result.fix
    assert "marshmallow>=2.15,<3" in result.fix


def test_ecephys_schema_stack_rejects_modern_unknown_field_behavior(monkeypatch) -> None:
    versions = {"argschema": "3.0.4", "marshmallow": "3.26.2"}
    monkeypatch.setattr(doctor, "_is_importable", lambda name: True)
    monkeypatch.setattr(doctor, "_module_version", lambda name: versions[name])

    result = doctor._check_ecephys_schema_stack()

    assert result.status == doctor.FAIL
    assert result.is_blocking
    assert "unknown fields" in result.detail
    assert "--force-reinstall" in result.fix


def test_ecephys_schema_stack_accepts_legacy_pipeline_versions(monkeypatch) -> None:
    versions = {"argschema": "1.17.5", "marshmallow": "2.21.0"}
    monkeypatch.setattr(doctor, "_is_importable", lambda name: True)
    monkeypatch.setattr(doctor, "_module_version", lambda name: versions[name])

    result = doctor._check_ecephys_schema_stack()

    assert result.status == doctor.OK
    assert not result.is_blocking
    assert "1.17.5" in result.detail


def test_missing_kilosort_fails_without_proposing_a_torch_change(monkeypatch) -> None:
    """The fix must not reinstall torch: that is how a working CUDA build gets downgraded."""
    monkeypatch.setattr(doctor, "_is_importable", lambda name: name != "kilosort")

    result = doctor._check_kilosort()

    assert result.status == doctor.FAIL
    assert result.severity == doctor.REQUIRED
    assert "kilosort>=4.1,<4.2" in result.fix
    assert "torch" not in result.fix


def test_kilosort_outside_tested_series_only_warns(monkeypatch) -> None:
    monkeypatch.setattr(doctor, "_is_importable", lambda name: True)
    monkeypatch.setattr(doctor, "_module_version", lambda name: "4.0.12")
    result = doctor._check_kilosort()
    assert result.status == doctor.WARN
    assert not result.is_blocking


def test_cpu_only_torch_warns_and_points_at_the_cuda_index(monkeypatch) -> None:
    monkeypatch.setattr(doctor, "_is_importable", lambda name: True)
    monkeypatch.setattr(doctor, "_module_version", lambda name: "2.8.0")
    result = doctor._check_torch()
    assert result.status == doctor.WARN
    assert "download.pytorch.org/whl/cu126" in result.fix


def test_cuda_torch_build_is_ok(monkeypatch) -> None:
    monkeypatch.setattr(doctor, "_is_importable", lambda name: True)
    monkeypatch.setattr(doctor, "_module_version", lambda name: "2.8.0+cu126")
    assert doctor._check_torch().status == doctor.OK


def test_qt_above_the_tested_ceiling_warns(monkeypatch) -> None:
    monkeypatch.setattr(doctor, "_is_importable", lambda name: True)
    monkeypatch.setattr(doctor, "_module_version", lambda name: "6.10.0")
    result = doctor._check_qt()
    assert result.status == doctor.WARN
    assert result.fix == doctor.ENV_UPDATE_CMD
    assert "pip install" not in result.fix


@pytest.mark.parametrize("filename", ["environment-linux.yml", "environment-windows.yml"])
def test_environment_keeps_pyside_and_shiboken_out_of_pip(filename: str) -> None:
    import yaml

    document = yaml.safe_load((doctor.REPO_ROOT / filename).read_text(encoding="utf-8"))
    dependencies = document["dependencies"]
    conda_packages = [item.lower() for item in dependencies if isinstance(item, str)]
    pip_packages = next(item["pip"] for item in dependencies if isinstance(item, dict) and "pip" in item)
    pip_names = [str(item).lower() for item in pip_packages]

    assert any(item.startswith("pyside6>=6.5,<6.8") for item in conda_packages)
    assert not any(item.startswith(("pyside6", "shiboken6")) for item in pip_names)


def test_numpy_above_numba_ceiling_warns(monkeypatch) -> None:
    monkeypatch.setattr(doctor, "_is_importable", lambda name: True)
    monkeypatch.setattr(doctor, "_module_version", lambda name: "2.6.0")
    assert doctor._check_numpy().status == doctor.WARN


def test_atlas_check_reads_the_configured_folder(tmp_path: Path) -> None:
    for name in (
        "template_volume_10um.npy",
        "annotation_volume_10um_by_index.npy",
        "structure_tree_safe_2017.csv",
    ):
        (tmp_path / name).write_bytes(b"0")

    settings: Dict[str, str] = {"histology/atlas_path": str(tmp_path)}
    read = doctor._settings_reader(lambda key, default: settings.get(key, default))

    result = doctor._check_atlas(read)
    assert result.status == doctor.OK
    assert str(tmp_path) in result.detail


def test_atlas_check_warns_when_volumes_are_absent(tmp_path: Path) -> None:
    settings = {"histology/atlas_path": str(tmp_path / "nope")}
    read = doctor._settings_reader(lambda key, default: settings.get(key, default))
    result = doctor._check_atlas(read)
    assert result.status == doctor.WARN
    assert result.severity == doctor.RECOMMENDED
    assert "osf.io" in result.fix


def test_settings_reader_survives_a_raising_getter() -> None:
    def broken(_key, _default):
        raise RuntimeError("settings backend is gone")

    read = doctor._settings_reader(broken)
    assert read("any/key", "fallback") == "fallback"


def test_external_tool_rows_cover_every_native_tool() -> None:
    from neuropyguin.tool_installer import NATIVE_TOOLS

    read = doctor._settings_reader(None)
    rows = doctor._check_external_tools(read)

    assert {row.key for row in rows} == {f"tool_{tool.key}" for tool in NATIVE_TOOLS}
    assert all(row.category == "External tools" for row in rows)


def test_bundled_toolboxes_are_found_in_this_checkout() -> None:
    rows = {row.key: row for row in doctor._check_bundled_repos()}
    assert set(rows) == {"ecephys_repo", "bombcell_repo", "npyx_repo"}
    for row in rows.values():
        assert row.status == doctor.OK, f"{row.label}: {row.detail}"


def test_run_diagnostics_returns_grouped_rows_and_never_raises() -> None:
    results = doctor.run_diagnostics()

    assert results
    keys = {row.key for row in results}
    for expected in ("python", "pyside6", "numpy", "kilosort", "torch", "ecephys_imports"):
        assert expected in keys
    # Categories arrive in display order, and every row carries a known status.
    positions = [doctor.CATEGORY_ORDER.index(row.category) for row in results]
    assert positions == sorted(positions)
    assert all(row.status in (doctor.OK, doctor.WARN, doctor.FAIL) for row in results)


def test_run_diagnostics_reports_progress_for_every_step() -> None:
    seen: list[tuple[int, int]] = []
    doctor.run_diagnostics(progress=lambda done, total, label: seen.append((done, total)))
    assert seen
    totals = {total for _done, total in seen}
    assert len(totals) == 1
    assert [done for done, _total in seen] == list(range(1, seen[-1][1] + 1))


def test_summary_helpers_agree_with_the_rows() -> None:
    rows = [
        doctor.CheckResult("a", "Application core", "A", doctor.OK, "fine"),
        doctor.CheckResult("b", "Spike sorting", "B", doctor.WARN, "meh", doctor.RECOMMENDED),
        doctor.CheckResult("c", "Spike sorting", "C", doctor.FAIL, "gone", doctor.REQUIRED, fix="pip install c"),
    ]

    assert doctor.summarize(rows) == {doctor.OK: 1, doctor.WARN: 1, doctor.FAIL: 1}
    assert doctor.has_blocking_failures(rows)
    assert "1 required component" in doctor.headline(rows)

    report = doctor.report_text(rows)
    assert "[FAIL] C: gone" in report
    assert "fix: pip install c" in report
    # Healthy rows never advertise a fix.
    assert "[ ok ] A: fine" in report


def test_headline_distinguishes_clean_from_warnings_only() -> None:
    clean = [doctor.CheckResult("a", "Application core", "A", doctor.OK, "fine")]
    warned = [doctor.CheckResult("b", "Histology", "B", doctor.WARN, "meh", doctor.OPTIONAL)]
    assert doctor.headline(clean).startswith("Everything")
    assert "optional component" in doctor.headline(warned)


def test_env_file_matches_this_platform() -> None:
    assert doctor.ENV_FILE in ("environment-linux.yml", "environment-windows.yml")
    assert (doctor.REPO_ROOT / doctor.ENV_FILE).exists()
    assert doctor.ENV_FILE in doctor.ENV_UPDATE_CMD


def test_module_entry_point_exits_nonzero_when_something_required_is_missing(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        doctor,
        "run_diagnostics",
        lambda *_a, **_k: [
            doctor.CheckResult("x", "Spike sorting", "X", doctor.FAIL, "gone", doctor.REQUIRED)
        ],
    )
    code = doctor.main([])
    assert code == 1
    assert "X: gone" in capsys.readouterr().out


@pytest.mark.parametrize("status,blocking", [(doctor.OK, False), (doctor.WARN, False), (doctor.FAIL, True)])
def test_is_blocking_only_for_failures(status: str, blocking: bool) -> None:
    row = doctor.CheckResult("k", "Application core", "L", status, "d")
    assert row.is_blocking is blocking

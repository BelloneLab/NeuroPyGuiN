from __future__ import annotations

from pathlib import Path
import zipfile

import pytest

from neuropyguin import tool_installer


def test_detected_os_normalizes_supported_platforms() -> None:
    assert tool_installer.detected_os("Windows") == "windows"
    assert tool_installer.detected_os("Linux") == "linux"
    assert tool_installer.detected_os("Darwin") == "macos"


def test_default_tool_paths_match_detected_os(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tool_installer, "installed_kilosort_path", lambda: None)

    linux = tool_installer.default_tool_paths(tmp_path, "Linux")
    windows = tool_installer.default_tool_paths(tmp_path, "Windows")

    assert Path(linux["catgt"]).name == "CatGT-linux"
    assert Path(linux["tprime"]).name == "TPrime-linux"
    assert Path(linux["cwaves"]).name == "C_Waves-linux"
    assert Path(windows["catgt"]).name == "CatGT-win"
    assert Path(windows["tprime"]).name == "TPrime-win"
    assert Path(windows["cwaves"]).name == "C_Waves-win"


def test_linux_tool_requires_configured_launcher(tmp_path: Path) -> None:
    tool = tool_installer.NATIVE_TOOLS[0]
    folder = tmp_path / "CatGT-linux"
    folder.mkdir()
    (folder / "CatGT").touch()
    (folder / "runit.sh").touch()

    assert not tool_installer.native_tool_is_installed(tool, folder, "Linux")

    links = folder / "links"
    links.mkdir()
    (links / "ld-linux-x86-64.so.2").touch()
    assert tool_installer.native_tool_is_installed(tool, folder, "Linux")


def test_safe_extract_rejects_parent_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../outside.txt", "unsafe")

    with pytest.raises(RuntimeError, match="Unsafe path"):
        tool_installer._safe_extract_zip(archive, tmp_path / "output")


def test_missing_tools_uses_platform_specific_executables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tool_installer, "installed_kilosort_path", lambda: tmp_path / "kilosort")
    (tmp_path / "kilosort").mkdir()
    paths: dict[str, str] = {}
    for tool in tool_installer.NATIVE_TOOLS:
        folder = tmp_path / tool.directory_name("windows")
        folder.mkdir()
        (folder / f"{tool.executable}.exe").touch()
        paths[tool.key] = str(folder)
    paths["kilosort"] = str(tmp_path / "kilosort")

    assert tool_installer.missing_tools(paths, "Windows") == []
    assert set(tool_installer.missing_tools(paths, "Linux")) == {"catgt", "tprime", "cwaves"}

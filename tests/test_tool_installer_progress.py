"""Progress-reporting tests for the external-tool installer.

These cover the ``(label, fraction)`` contract the GUI progress window relies on:
byte-level progress during a download, an indeterminate marker while pip runs, and
correct rescaling of each step into overall completion.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import List, Tuple

import pytest

from neuropyguin import tool_installer


class _FakeResponse:
    """Minimal urlopen stand-in that streams ``payload`` in fixed-size chunks."""

    def __init__(self, payload: bytes, *, send_length: bool = True) -> None:
        self._buffer = io.BytesIO(payload)
        self.headers = {"Content-Length": str(len(payload))} if send_length else {}

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_exc) -> None:
        return None


def test_download_reports_bytes_against_content_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"x" * (tool_installer._DOWNLOAD_CHUNK * 3 + 17)
    monkeypatch.setattr(tool_installer, "urlopen", lambda *_a, **_k: _FakeResponse(payload))
    seen: List[Tuple[int, int]] = []

    target = tmp_path / "archive.zip"
    tool_installer._download(
        "https://example.invalid/x.zip", target, lambda read, total: seen.append((read, total))
    )

    assert target.read_bytes() == payload
    assert seen[-1] == (len(payload), len(payload))
    # Monotonic, chunked, and never overshooting the total.
    assert [read for read, _total in seen] == sorted(read for read, _total in seen)
    assert all(read <= total for read, total in seen)
    assert len(seen) == 4


def test_download_without_content_length_reports_zero_total(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"y" * 1024
    monkeypatch.setattr(
        tool_installer, "urlopen", lambda *_a, **_k: _FakeResponse(payload, send_length=False)
    )
    seen: List[Tuple[int, int]] = []

    tool_installer._download(
        "https://example.invalid/x.zip", tmp_path / "a.zip", lambda read, total: seen.append((read, total))
    )

    assert seen == [(len(payload), 0)]


def _fake_linux_package(tool: tool_installer.NativeTool) -> bytes:
    """Build a ZIP shaped like an official Linux tool package."""
    folder = tool.directory_name("linux")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr(f"{folder}/{tool.executable}", "binary")
        bundle.writestr(f"{folder}/runit.sh", "#!/bin/sh\n")
        bundle.writestr(f"{folder}/install.sh", "#!/bin/sh\nexit 0\n")
        bundle.writestr(f"{folder}/links/ld-linux-x86-64.so.2", "linker")
    return buffer.getvalue()


def test_install_native_tool_emits_a_monotonic_labelled_progression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = tool_installer.NATIVE_TOOLS[0]
    payload = _fake_linux_package(tool)
    monkeypatch.setattr(tool_installer, "urlopen", lambda *_a, **_k: _FakeResponse(payload))
    # The package's install.sh is a no-op here; skip running it.
    monkeypatch.setattr(tool_installer, "_configure_linux_tool", lambda *_a, **_k: None)
    events: List[Tuple[str, float]] = []

    target = tool_installer.install_native_tool(
        tool, tmp_path, "linux", lambda _m: None, lambda label, fraction: events.append((label, fraction))
    )

    assert target == (tmp_path / tool.directory_name("linux")).resolve()
    fractions = [fraction for _label, fraction in events]
    assert fractions[0] == 0.0
    assert fractions[-1] == 1.0
    assert fractions == sorted(fractions)
    labels = " | ".join(label for label, _f in events)
    for phase in ("Downloading", "Extracting", "Copying", "Configuring", "Verifying", "Installed"):
        assert phase in labels


def test_install_kilosort_marks_pip_as_indeterminate(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[Tuple[str, float]] = []

    class _Process:
        stdout = io.StringIO("Collecting kilosort\nSuccessfully installed kilosort-4.1.7\n")

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(tool_installer.subprocess, "Popen", lambda *_a, **_k: _Process())
    # First lookup (before pip) finds nothing; the one after pip "ran" must succeed.
    lookups = iter([None, Path("/env/site-packages/kilosort")])
    monkeypatch.setattr(tool_installer, "installed_kilosort_path", lambda: next(lookups))

    tool_installer.install_kilosort(lambda _m: None, lambda label, fraction: calls.append((label, fraction)))

    assert calls[0][1] == tool_installer.BUSY
    assert "pip" in calls[0][0]
    assert calls[-1] == ("Installed Kilosort4", 1.0)


def test_install_kilosort_short_circuits_when_already_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tool_installer, "installed_kilosort_path", lambda: Path("/env/kilosort"))
    calls: List[Tuple[str, float]] = []

    tool_installer.install_kilosort(lambda _m: None, lambda label, fraction: calls.append((label, fraction)))

    assert calls == [("Kilosort4 is already installed", 1.0)]


def test_forced_kilosort_reinstall_never_resolves_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A refresh of an installed Kilosort must not let pip touch the CUDA PyTorch."""
    commands: List[List[str]] = []

    class _Process:
        stdout = io.StringIO("Successfully installed kilosort-4.1.7\n")

        def wait(self) -> int:
            return 0

    def fake_popen(cmd, *_a, **_k):
        commands.append(list(cmd))
        return _Process()

    monkeypatch.setattr(tool_installer, "installed_kilosort_path", lambda: Path("/env/kilosort"))
    monkeypatch.setattr(tool_installer.subprocess, "Popen", fake_popen)
    logs: List[str] = []

    tool_installer.install_kilosort(logs.append, None, force=True)

    assert len(commands) == 1
    command = commands[0]
    assert "--force-reinstall" in command and "--no-deps" in command
    assert tool_installer.KILOSORT_REQUIREMENT in command
    assert tool_installer.PYTORCH_REQUIREMENT not in command
    assert any("PyTorch build is left alone" in line for line in logs)


def test_force_is_ignored_when_kilosort_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A first install still needs its dependency resolution, so --no-deps must not appear."""
    commands: List[List[str]] = []

    class _Process:
        stdout = io.StringIO("Successfully installed kilosort-4.1.7\n")

        def wait(self) -> int:
            return 0

    lookups = iter([None, Path("/env/kilosort")])
    monkeypatch.setattr(tool_installer, "installed_kilosort_path", lambda: next(lookups))
    monkeypatch.setattr(
        tool_installer.subprocess, "Popen", lambda cmd, *_a, **_k: (commands.append(list(cmd)), _Process())[1]
    )

    tool_installer.install_kilosort(lambda _m: None, None, force=True)

    assert "--no-deps" not in commands[0]
    assert tool_installer.PYTORCH_REQUIREMENT in commands[0]


def test_install_missing_tools_passes_force_through_to_kilosort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: List[bool] = []
    monkeypatch.setattr(tool_installer, "installed_kilosort_path", lambda: tmp_path / "kilosort")
    monkeypatch.setattr(
        tool_installer,
        "install_kilosort",
        lambda _report, _progress=None, force=False: (seen.append(force), tmp_path / "kilosort")[1],
    )

    tool_installer.install_missing_tools(
        tmp_path, {}, requested=["kilosort"], system_name="Linux", force=True
    )

    assert seen == [True]


def test_installed_kilosort_version_is_a_string(monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(tool_installer.installed_kilosort_version(), str)


def test_install_missing_tools_rescales_each_step_into_overall_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two steps: step 0 spans 0.0-0.5 of the bar, step 1 spans 0.5-1.0."""
    monkeypatch.setattr(tool_installer, "installed_kilosort_path", lambda: tmp_path / "kilosort")

    def fake_native(tool, _root, _os_name, _report, on_progress=None):
        if on_progress is not None:
            on_progress(f"start {tool.name}", 0.0)
            on_progress(f"half {tool.name}", 0.5)
            on_progress(f"done {tool.name}", 1.0)
        return tmp_path / tool.directory_name("linux")

    def fake_kilosort(_report, on_progress=None, force=False):
        if on_progress is not None:
            on_progress("pip", tool_installer.BUSY)
            on_progress("done kilosort", 1.0)
        return tmp_path / "kilosort"

    monkeypatch.setattr(tool_installer, "install_native_tool", fake_native)
    monkeypatch.setattr(tool_installer, "install_kilosort", fake_kilosort)
    events: List[Tuple[str, float]] = []

    installed = tool_installer.install_missing_tools(
        tmp_path,
        {},
        requested=["catgt", "kilosort"],
        progress=lambda label, fraction: events.append((label, fraction)),
        system_name="Linux",
    )

    assert set(installed) == {"catgt", "kilosort"}
    assert events[0] == ("start CatGT", 0.0)
    assert events[1] == ("half CatGT", 0.25)
    assert events[2] == ("done CatGT", 0.5)
    # An indeterminate step keeps its sentinel instead of being rescaled.
    assert events[3] == ("pip", tool_installer.BUSY)
    assert events[4] == ("done kilosort", 1.0)
    assert events[-1][1] == 1.0


def test_install_missing_tools_without_progress_callback_still_installs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tool_installer, "installed_kilosort_path", lambda: tmp_path / "kilosort")
    monkeypatch.setattr(
        tool_installer,
        "install_native_tool",
        lambda tool, _root, _os, _report, on_progress=None: tmp_path / tool.directory_name("linux"),
    )

    installed = tool_installer.install_missing_tools(
        tmp_path, {}, requested=["tprime"], system_name="Linux"
    )

    assert set(installed) == {"tprime"}
    assert (tmp_path / "tools" / "neuropyguin-tools.json").exists()

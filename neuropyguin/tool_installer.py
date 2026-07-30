"""Cross-platform installer for NeuroPyGuiN's external preprocessing tools.

CatGT, TPrime, and C_Waves are downloaded from the official SpikeGLX download
page. Kilosort is installed into the currently running Python environment.
The module contains no GUI code so detection and installation can be tested
independently from Qt.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Callable, Dict, Iterable
from urllib.request import Request, urlopen
import zipfile


SPIKEGLX_DOWNLOAD_PAGE = "https://billkarsh.github.io/SpikeGLX/"
KILOSORT_REPOSITORY = "https://github.com/MouseLand/Kilosort"
PYTORCH_REQUIREMENT = "torch==2.5.1"
KILOSORT_REQUIREMENT = "kilosort>=4.1,<4.2"

#: Download block size. Small enough that the progress bar moves smoothly on a
#: few-megabyte archive, large enough not to flood the GUI with signals.
_DOWNLOAD_CHUNK = 256 * 1024

#: Progress callback signature: ``(label, fraction)`` where ``fraction`` is overall
#: completion in [0, 1], or :data:`BUSY` for a step with no measurable progress.
ProgressCallback = Callable[[str, float], None]

#: Sentinel fraction meaning "this step is running but cannot be measured".
BUSY = -1.0


@dataclass(frozen=True)
class NativeTool:
    key: str
    name: str
    executable: str
    windows_url: str
    linux_url: str

    def archive_url(self, os_name: str) -> str:
        if os_name == "windows":
            return self.windows_url
        if os_name == "linux":
            return self.linux_url
        raise RuntimeError(
            f"{self.name} has no official prebuilt package for {platform.system() or os_name}. "
            "Only Windows and Linux are supported."
        )

    def directory_name(self, os_name: str) -> str:
        suffix = "win" if os_name == "windows" else "linux"
        return f"{self.name}-{suffix}"


NATIVE_TOOLS = (
    NativeTool(
        key="catgt",
        name="CatGT",
        executable="CatGT",
        windows_url=f"{SPIKEGLX_DOWNLOAD_PAGE}Support/CatGTWinApp.zip",
        linux_url=f"{SPIKEGLX_DOWNLOAD_PAGE}Support/CatGTLnxApp.zip",
    ),
    NativeTool(
        key="tprime",
        name="TPrime",
        executable="TPrime",
        windows_url=f"{SPIKEGLX_DOWNLOAD_PAGE}Support/TPrimeWinApp.zip",
        linux_url=f"{SPIKEGLX_DOWNLOAD_PAGE}Support/TPrimeLnxApp.zip",
    ),
    NativeTool(
        key="cwaves",
        name="C_Waves",
        executable="C_Waves",
        windows_url=f"{SPIKEGLX_DOWNLOAD_PAGE}Support/C_WavesWinApp.zip",
        linux_url=f"{SPIKEGLX_DOWNLOAD_PAGE}Support/C_WavesLnxApp.zip",
    ),
)


def detected_os(system_name: str | None = None) -> str:
    """Return the supported installer platform name."""
    value = (system_name or platform.system()).strip().lower()
    if value.startswith("win"):
        return "windows"
    if value.startswith("linux"):
        return "linux"
    if value in {"darwin", "mac", "macos"}:
        return "macos"
    return value or "unknown"


def installed_kilosort_path() -> Path | None:
    """Return the importable Kilosort package directory, if installed."""
    importlib.invalidate_caches()
    spec = importlib.util.find_spec("kilosort")
    if spec is None:
        return None
    if spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            path = Path(location).resolve()
            if path.is_dir():
                return path
    if spec.origin:
        path = Path(spec.origin).resolve().parent
        if path.is_dir():
            return path
    return None


def default_tool_paths(project_root: Path, system_name: str | None = None) -> Dict[str, str]:
    """Build platform-correct defaults rooted in the application's tools folder."""
    os_name = detected_os(system_name)
    tools_root = Path(project_root).resolve() / "tools"
    paths = {
        tool.key: str((tools_root / tool.directory_name(os_name)).resolve())
        for tool in NATIVE_TOOLS
        if os_name in {"windows", "linux"}
    }
    ks_path = installed_kilosort_path()
    paths["kilosort"] = str(ks_path or (tools_root / "Kilosort" / "kilosort").resolve())
    return paths


def native_tool_is_installed(tool: NativeTool, folder: str | Path, os_name: str | None = None) -> bool:
    """Check for the executable and platform launcher required by the pipeline."""
    platform_name = detected_os(os_name)
    root = Path(folder).expanduser()
    if platform_name == "windows":
        return root.is_dir() and (root / f"{tool.executable}.exe").is_file()
    if platform_name == "linux":
        return (
            root.is_dir()
            and (root / tool.executable).is_file()
            and (root / "runit.sh").is_file()
            and (root / "links" / "ld-linux-x86-64.so.2").exists()
        )
    return False


def missing_tools(paths: Dict[str, str], system_name: str | None = None) -> list[str]:
    """Return missing or incompatible tool keys for the supplied configured paths."""
    os_name = detected_os(system_name)
    missing = [
        tool.key
        for tool in NATIVE_TOOLS
        if not native_tool_is_installed(tool, paths.get(tool.key, ""), os_name)
    ]
    if installed_kilosort_path() is None:
        missing.append("kilosort")
    return missing


def tool_display_name(key: str) -> str:
    if key == "kilosort":
        return "Kilosort4"
    for tool in NATIVE_TOOLS:
        if tool.key == key:
            return tool.name
    return key


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    """Extract a ZIP while rejecting traversal paths and archived symlinks."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            member_path = (destination / member.filename).resolve()
            try:
                member_path.relative_to(destination)
            except ValueError as exc:
                raise RuntimeError(f"Unsafe path in downloaded archive: {member.filename}") from exc
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise RuntimeError(f"Downloaded archive contains an unsupported symlink: {member.filename}")
        bundle.extractall(destination)


def _download(
    url: str,
    destination: Path,
    on_bytes: Callable[[int, int], None] | None = None,
) -> None:
    """Stream ``url`` to ``destination``, reporting ``(read, total)`` bytes as it goes.

    ``total`` is 0 when the server sends no ``Content-Length``, which is the caller's
    cue to show an indeterminate progress bar instead of a percentage.
    """
    request = Request(url, headers={"User-Agent": "NeuroPyGuiN-tool-installer/1"})
    with urlopen(request, timeout=120) as response, destination.open("wb") as output:
        try:
            total = int(response.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            total = 0
        read = 0
        while True:
            chunk = response.read(_DOWNLOAD_CHUNK)
            if not chunk:
                break
            output.write(chunk)
            read += len(chunk)
            if on_bytes is not None:
                on_bytes(read, total)


def _find_extracted_tool(root: Path, tool: NativeTool, os_name: str) -> Path:
    expected = tool.directory_name(os_name)
    matches = [path for path in root.rglob(expected) if path.is_dir()]
    if len(matches) != 1:
        raise RuntimeError(
            f"The {tool.name} archive did not contain exactly one {expected} directory."
        )
    return matches[0]


def _configure_linux_tool(folder: Path, tool: NativeTool) -> None:
    installer = folder / "install.sh"
    if not installer.is_file():
        raise RuntimeError(f"{tool.name} Linux package is missing install.sh")
    completed = subprocess.run(
        ["/bin/sh", str(installer)],
        cwd=str(folder),
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"{tool.name} Linux setup failed: {details or completed.returncode}")


def install_native_tool(
    tool: NativeTool,
    tools_root: Path,
    os_name: str,
    report: Callable[[str], None],
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Download, extract, configure, and verify one official native tool.

    ``on_progress`` receives ``(label, fraction)`` for THIS tool only, with
    ``fraction`` in [0, 1]; :func:`install_missing_tools` rescales it into overall
    progress across all requested tools.
    """

    def step(label: str, fraction: float) -> None:
        if on_progress is not None:
            on_progress(label, fraction)

    url = tool.archive_url(os_name)
    target = tools_root / tool.directory_name(os_name)
    report(f"Downloading {tool.name} for {os_name.title()} from the official SpikeGLX site...")
    step(f"Downloading {tool.name}...", 0.0)
    tools_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"neuropyguin-{tool.key}-") as temp_name:
        temp_root = Path(temp_name)
        archive = temp_root / f"{tool.key}.zip"
        extracted = temp_root / "extracted"
        extracted.mkdir()

        def on_bytes(read: int, total: int) -> None:
            got = read / 1e6
            if total > 0:
                # The download dominates the wall clock, so it owns most of the bar.
                step(f"Downloading {tool.name}: {got:.1f} / {total / 1e6:.1f} MB", 0.75 * read / total)
            else:
                step(f"Downloading {tool.name}: {got:.1f} MB", BUSY)

        _download(url, archive, on_bytes)
        step(f"Extracting {tool.name}...", 0.78)
        _safe_extract_zip(archive, extracted)
        source = _find_extracted_tool(extracted, tool, os_name)
        step(f"Copying {tool.name} into {target.name}...", 0.86)
        shutil.copytree(source, target, dirs_exist_ok=True)

    if os_name == "linux":
        report(f"Configuring the {tool.name} Linux runtime...")
        step(f"Configuring the {tool.name} Linux runtime...", 0.93)
        _configure_linux_tool(target, tool)
    executable = target / (f"{tool.executable}.exe" if os_name == "windows" else tool.executable)
    if executable.exists() and os_name == "linux":
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        launcher = target / "runit.sh"
        launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    step(f"Verifying {tool.name}...", 0.98)
    if not native_tool_is_installed(tool, target, os_name):
        raise RuntimeError(f"{tool.name} installation could not be verified at {target}")
    report(f"Installed {tool.name}: {target}")
    step(f"Installed {tool.name}", 1.0)
    return target.resolve()


def install_kilosort(
    report: Callable[[str], None],
    on_progress: ProgressCallback | None = None,
    force: bool = False,
) -> Path:
    """Install Kilosort4 into the interpreter running NeuroPyGuiN.

    ``force`` reinstalls an already-present Kilosort. That path adds ``--no-deps``
    on purpose: the package is only being refreshed, so pip must not be allowed to
    resolve (and possibly downgrade) the CUDA PyTorch this environment depends on.
    """

    def step(label: str, fraction: float) -> None:
        if on_progress is not None:
            on_progress(label, fraction)

    existing = installed_kilosort_path()
    if existing is not None and not force:
        report(f"Kilosort4 is already installed: {existing}")
        step("Kilosort4 is already installed", 1.0)
        return existing
    reinstall = existing is not None
    # pip reports no machine-readable progress, so this step is marked indeterminate
    # and the live log is what the user watches.
    step(
        ("Reinstalling" if reinstall else "Installing")
        + " Kilosort4 with pip (this can take several minutes)...",
        BUSY,
    )
    if reinstall:
        report(
            f"Reinstalling {KILOSORT_REQUIREMENT} into {sys.executable} "
            "(--no-deps, so the installed PyTorch build is left alone)..."
        )
        requirements = ["--upgrade", "--force-reinstall", "--no-deps", KILOSORT_REQUIREMENT]
    else:
        report(
            f"Installing {KILOSORT_REQUIREMENT} with the tested {PYTORCH_REQUIREMENT} baseline "
            f"into {sys.executable}..."
        )
        requirements = [PYTORCH_REQUIREMENT, KILOSORT_REQUIREMENT]
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            *requirements,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.strip()
        if line:
            report(line)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"pip failed to install Kilosort4 (exit code {return_code})")
    importlib.invalidate_caches()
    path = installed_kilosort_path()
    if path is None:
        raise RuntimeError("pip completed, but Kilosort4 is still not importable")
    report(f"Installed Kilosort4: {path}")
    step("Installed Kilosort4", 1.0)
    return path


def installed_kilosort_version() -> str:
    """Return the installed Kilosort distribution version, or an empty string."""
    import importlib.metadata as md

    try:
        return str(md.version("kilosort"))
    except Exception:
        return ""


def install_missing_tools(
    project_root: Path,
    configured_paths: Dict[str, str],
    requested: Iterable[str] | None = None,
    report: Callable[[str], None] | None = None,
    system_name: str | None = None,
    progress: ProgressCallback | None = None,
    force: bool = False,
) -> Dict[str, str]:
    """Install requested missing tools and return their verified paths.

    ``report`` receives log lines; ``progress`` receives ``(label, fraction)`` with
    ``fraction`` being OVERALL completion in [0, 1] across every requested tool, or
    :data:`BUSY` while a step (a pip install) cannot be measured. ``force`` refreshes
    tools that are already present (native archives are always re-downloaded; see
    :func:`install_kilosort` for how the pip case protects PyTorch).
    """
    reporter = report or (lambda _message: None)
    os_name = detected_os(system_name)
    wanted = list(requested) if requested is not None else missing_tools(configured_paths, os_name)
    native_keys = {tool.key for tool in NATIVE_TOOLS}
    if os_name not in {"windows", "linux"} and any(key in native_keys for key in wanted):
        names = ", ".join(tool_display_name(key) for key in wanted if key in native_keys)
        raise RuntimeError(
            f"No official prebuilt {names} packages are available for {platform.system()}. "
            "Use Windows or Linux, or build the tools from source."
        )

    tools_root = Path(project_root).resolve() / "tools"
    installed: Dict[str, str] = {}

    # Steps run in this order, and each one owns an equal slice of the bar.
    step_keys = [tool.key for tool in NATIVE_TOOLS if tool.key in wanted]
    if "kilosort" in wanted:
        step_keys.append("kilosort")
    total_steps = max(len(step_keys), 1)

    def scaled(step_index: int) -> ProgressCallback:
        """Map one step's 0..1 progress into the overall 0..1 range."""

        def emit(label: str, fraction: float) -> None:
            if progress is None:
                return
            if fraction < 0:
                progress(label, BUSY)
                return
            within = max(0.0, min(1.0, fraction))
            progress(label, (step_index + within) / total_steps)

        return emit

    for tool in NATIVE_TOOLS:
        if tool.key not in wanted:
            continue
        index = step_keys.index(tool.key)
        installed[tool.key] = str(
            install_native_tool(tool, tools_root, os_name, reporter, scaled(index))
        )
    if "kilosort" in wanted:
        installed["kilosort"] = str(
            install_kilosort(reporter, scaled(step_keys.index("kilosort")), force=force)
        )

    manifest = {
        "platform": os_name,
        "installed_at": datetime.now(timezone.utc).isoformat(),
        "paths": installed,
        "sources": {
            "spikeglx": SPIKEGLX_DOWNLOAD_PAGE,
            "kilosort": KILOSORT_REPOSITORY,
        },
    }
    if installed:
        tools_root.mkdir(parents=True, exist_ok=True)
        (tools_root / "neuropyguin-tools.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    if progress is not None:
        progress("All requested tools are installed", 1.0)
    return installed

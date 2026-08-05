"""Environment self-check ("doctor") for NeuroPyGuiN.

The app is a front-end for a stack of heavyweight scientific tools that live in
several places: Python packages in the conda env (PySide6, Kilosort 4, PyTorch,
argschema, ibllib, ...), repositories bundled inside the app folder
(``ecephys_spike_sorting``, ``py_bombcell``, ``npyx``), native SpikeGLX binaries
(CatGT, TPrime, C_Waves), and large downloaded data (the Allen CCF atlas). When
one of them is missing the failure usually surfaces much later, deep inside a
subprocess log - for example every ecephys module dies on ``import argschema``
long after a queue has started running.

This module answers the question "is everything the app needs actually here?"
before the user presses any button. It is deliberately Qt-free so the checks can
run in a worker thread, in tests, or from a terminal:

    python -m neuropyguin.doctor

:func:`run_diagnostics` returns a flat list of :class:`CheckResult` rows grouped
by :attr:`CheckResult.category`; :mod:`neuropyguin.doctor_dialog` renders them and
is what the GUI shows on the very first launch.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]

# Status values, ordered from healthy to broken.
OK = "ok"
WARN = "warn"
FAIL = "fail"

# Severity decides what a missing item means: a required item that is absent is a
# hard failure (the app cannot do that job at all), recommended degrades one
# feature, optional is purely informational.
REQUIRED = "required"
RECOMMENDED = "recommended"
OPTIONAL = "optional"

CATEGORY_ORDER = (
    "Application core",
    "Spike sorting",
    "GPU acceleration",
    "Bundled toolboxes",
    "Curation",
    "Post processing",
    "Histology",
    "External tools",
)

# The environment file that fixes almost everything, per platform.
ENV_FILE = "environment-windows.yml" if os.name == "nt" else "environment-linux.yml"
ENV_UPDATE_CMD = f"conda env update -n neuropygui -f {ENV_FILE} --prune"


@dataclass(frozen=True)
class CheckResult:
    """One diagnostic row.

    ``status`` is :data:`OK`, :data:`WARN` or :data:`FAIL`; ``severity`` records
    how much of the app depends on the item, and ``fix`` holds a copy-pasteable
    command (or a short instruction) that repairs it.
    """

    key: str
    category: str
    label: str
    status: str
    detail: str
    severity: str = REQUIRED
    fix: str = ""
    install_keys: Tuple[str, ...] = ()

    @property
    def is_blocking(self) -> bool:
        """True when this row means a whole workflow cannot run."""
        return self.status == FAIL


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _module_version(name: str) -> str:
    """Return a printable version for an installed module, without importing it.

    Uses installed distribution metadata (cheap) and falls back to an empty
    string, so the doctor never pays for importing torch just to name it.
    """
    import importlib.metadata as md

    for dist in (name, name.replace("_", "-")):
        try:
            return str(md.version(dist))
        except Exception:
            continue
    return ""


def _is_importable(name: str) -> bool:
    """True if ``name`` can be located by the import system."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _version_tuple(text: str) -> Tuple[int, ...]:
    """Parse a leading dotted numeric version into a tuple (``"2.8.0+cu126"`` -> (2, 8, 0))."""
    parts: List[int] = []
    for chunk in str(text).split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


@dataclass(frozen=True)
class ModuleSpec:
    """An import name plus the distribution name used to install it."""

    module: str
    package: str = ""
    pip_installable: bool = True

    @property
    def install_name(self) -> str:
        return self.package or self.module


def _missing_modules(specs: Sequence[ModuleSpec]) -> List[ModuleSpec]:
    return [spec for spec in specs if not _is_importable(spec.module)]


def _module_group(
    *,
    key: str,
    category: str,
    label: str,
    specs: Sequence[ModuleSpec],
    severity: str,
    ok_detail: str,
    fix_prefix: str = "",
) -> CheckResult:
    """Check a group of imports and summarize them as a single row."""
    missing = _missing_modules(specs)
    if not missing:
        return CheckResult(key, category, label, OK, ok_detail, severity)
    names = ", ".join(spec.module for spec in missing)
    installable = [spec for spec in missing if spec.pip_installable]
    installs = " ".join(f'"{spec.install_name}"' for spec in installable)
    status = FAIL if severity == REQUIRED else WARN
    fix = f"{fix_prefix}{sys.executable} -m pip install {installs}" if installs else fix_prefix.strip()
    return CheckResult(
        key,
        category,
        label,
        status,
        f"Not importable: {names}",
        severity,
        fix=fix,
        install_keys=tuple(f"pip:{spec.install_name}" for spec in installable),
    )


def _settings_reader(
    settings_get: Optional[Callable[[str, object], object]],
) -> Callable[[str, str], str]:
    """Wrap a QSettings-style getter into a plain ``(key, default) -> str`` reader."""

    def read(key: str, default: str = "") -> str:
        if settings_get is None:
            return default
        try:
            value = settings_get(key, default)
        except Exception:
            return default
        return "" if value is None else str(value)

    return read


# --------------------------------------------------------------------------- #
# Individual check families
# --------------------------------------------------------------------------- #


def _check_python() -> CheckResult:
    version = ".".join(str(v) for v in sys.version_info[:3])
    detail = f"Python {version} at {sys.executable}"
    if sys.version_info < (3, 10):
        return CheckResult(
            "python", "Application core", "Python interpreter", FAIL,
            f"{detail} - phy 2.1 and the tested stack need Python >= 3.10",
            REQUIRED, fix=ENV_UPDATE_CMD,
        )
    if sys.version_info >= (3, 12):
        return CheckResult(
            "python", "Application core", "Python interpreter", WARN,
            f"{detail} - the app is tested on Python 3.10", RECOMMENDED, fix=ENV_UPDATE_CMD,
        )
    return CheckResult("python", "Application core", "Python interpreter", OK, detail, REQUIRED)


def _check_qt() -> CheckResult:
    """PySide6 must be present and stay below 6.8 (Qt 6.10 crashes pyqtgraph)."""
    if not _is_importable("PySide6"):
        return CheckResult(
            "pyside6", "Application core", "PySide6 (Qt runtime)", FAIL,
            "PySide6 is not importable - the GUI cannot start", REQUIRED, fix=ENV_UPDATE_CMD,
        )
    version = _module_version("PySide6")
    if version and _version_tuple(version) >= (6, 8):
        return CheckResult(
            "pyside6", "Application core", "PySide6 (Qt runtime)", WARN,
            f"PySide6 {version} ships Qt 6.10, where pyqtgraph can crash the paint thread. "
            "6.7.x is the tested baseline.",
            RECOMMENDED,
            fix=ENV_UPDATE_CMD,
        )
    return CheckResult(
        "pyside6", "Application core", "PySide6 (Qt runtime)", OK,
        f"PySide6 {version or 'installed'}", REQUIRED,
    )


def _check_numpy() -> CheckResult:
    """numpy must stay in the window numba/cupy and the bundled code agree on."""
    if not _is_importable("numpy"):
        return CheckResult(
            "numpy", "Application core", "numpy", FAIL, "numpy is not importable",
            REQUIRED, fix=ENV_UPDATE_CMD,
        )
    version = _module_version("numpy")
    parsed = _version_tuple(version)
    if parsed and parsed >= (2, 5):
        return CheckResult(
            "numpy", "Application core", "numpy", WARN,
            f"numpy {version} is above the tested ceiling (numba requires < 2.5, and bundled "
            "code still calls the deprecated np.trapz / np.in1d aliases)",
            RECOMMENDED,
            fix=f'{sys.executable} -m pip install "numpy>=1.26,<2.5"',
            install_keys=("pip:numpy>=1.26,<2.5",),
        )
    if parsed and parsed < (1, 26):
        return CheckResult(
            "numpy", "Application core", "numpy", WARN,
            f"numpy {version} is below the tested floor (pyqtgraph 0.14 requires >= 1.25)",
            RECOMMENDED,
            fix=f'{sys.executable} -m pip install "numpy>=1.26,<2.5"',
            install_keys=("pip:numpy>=1.26,<2.5",),
        )
    return CheckResult("numpy", "Application core", "numpy", OK, f"numpy {version or 'installed'}", REQUIRED)


_CORE_SCIENTIFIC = (
    ModuleSpec("scipy"),
    ModuleSpec("pandas"),
    ModuleSpec("matplotlib"),
    ModuleSpec("numba"),
    ModuleSpec("sklearn", "scikit-learn"),
    ModuleSpec("imblearn", "imbalanced-learn"),
    ModuleSpec("statsmodels"),
    ModuleSpec("networkx"),
    ModuleSpec("h5py"),
    ModuleSpec("joblib"),
    ModuleSpec("psutil"),
    ModuleSpec("tqdm"),
    ModuleSpec("seaborn"),
    ModuleSpec("pyarrow"),
    ModuleSpec("PIL", "pillow"),
    ModuleSpec("cmcrameri"),
)

# Third-party imports every bundled ecephys_spike_sorting module performs before
# it does any work: argschema (its CLI layer), git (common/utils.py), xarray
# (mean_waveforms output), phylib (IBL quality metrics), tkinter (SGLXMetaToCoords).
_ECEPHYS_IMPORTS = (
    ModuleSpec("argschema"),
    ModuleSpec("git", "GitPython"),
    ModuleSpec("xarray"),
    ModuleSpec("phylib"),
    ModuleSpec("tkinter", pip_installable=False),
)

_ECEPHYS_SCHEMA_FIX = (
    f'{sys.executable} -m pip install --upgrade --force-reinstall '
    '"argschema==1.17.5" "marshmallow>=2.15,<3"'
)


def _check_ecephys_schema_stack() -> CheckResult:
    """Reject schema-library versions that cannot read the combined pipeline JSON.

    The bundled ecephys code passes one JSON document containing every module's
    settings to each individual module.  Its schemas rely on the Marshmallow 2
    behavior used by argschema 1.17.5, which ignores fields owned by other
    modules.  argschema 3 with Marshmallow 3 instead raises ``Unknown field``
    before CatGT, Kilosort, or any downstream module can start.
    """
    missing = [name for name in ("argschema", "marshmallow") if not _is_importable(name)]
    if missing:
        return CheckResult(
            "ecephys_schema_stack",
            "Spike sorting",
            "ecephys schema compatibility",
            FAIL,
            f"Not importable: {', '.join(missing)}",
            REQUIRED,
            fix=_ECEPHYS_SCHEMA_FIX,
            install_keys=("pip-force:argschema==1.17.5", "pip-force:marshmallow>=2.15,<3"),
        )

    argschema_version = _module_version("argschema")
    marshmallow_version = _module_version("marshmallow")
    compatible = (
        _version_tuple(argschema_version) == (1, 17, 5)
        and (2,) <= _version_tuple(marshmallow_version) < (3,)
    )
    detail = (
        f"argschema {argschema_version or 'unknown'}, "
        f"marshmallow {marshmallow_version or 'unknown'}"
    )
    if not compatible:
        return CheckResult(
            "ecephys_schema_stack",
            "Spike sorting",
            "ecephys schema compatibility",
            FAIL,
            f"{detail}. This combination rejects the pipeline JSON as unknown fields.",
            REQUIRED,
            fix=_ECEPHYS_SCHEMA_FIX,
            install_keys=("pip-force:argschema==1.17.5", "pip-force:marshmallow>=2.15,<3"),
        )
    return CheckResult(
        "ecephys_schema_stack",
        "Spike sorting",
        "ecephys schema compatibility",
        OK,
        detail,
        REQUIRED,
    )


def _check_kilosort() -> CheckResult:
    if not _is_importable("kilosort"):
        return CheckResult(
            "kilosort", "Spike sorting", "Kilosort 4", FAIL,
            "kilosort is not importable, so the Kilosort step cannot run. Install it WITHOUT "
            "touching the existing torch build.",
            REQUIRED,
            fix=f'{sys.executable} -m pip install "kilosort>=4.1,<4.2"',
            install_keys=("kilosort",),
        )
    version = _module_version("kilosort")
    parsed = _version_tuple(version)
    if parsed and not ((4, 1) <= parsed < (4, 2)):
        return CheckResult(
            "kilosort", "Spike sorting", "Kilosort 4", WARN,
            f"kilosort {version} is outside the tested 4.1.x series; the bundled ks4_helper "
            "calls run_kilosort(clear_cache=, save_preprocessed_copy=, verbose_console=)",
            RECOMMENDED,
            fix=f'{sys.executable} -m pip install "kilosort>=4.1,<4.2"',
            install_keys=("kilosort",),
        )
    return CheckResult(
        "kilosort", "Spike sorting", "Kilosort 4", OK, f"kilosort {version or 'installed'}", REQUIRED,
    )


def _check_torch() -> CheckResult:
    """Report the torch build. Importing torch is slow, hence the metadata-first path."""
    if not _is_importable("torch"):
        return CheckResult(
            "torch", "Spike sorting", "PyTorch", FAIL,
            "torch is not importable - Kilosort 4 cannot run", REQUIRED, fix=ENV_UPDATE_CMD,
        )
    version = _module_version("torch")
    parsed = _version_tuple(version)
    if parsed and parsed < (2, 1):
        return CheckResult(
            "torch", "Spike sorting", "PyTorch", WARN,
            f"torch {version} is below Kilosort 4's minimum (2.1)", RECOMMENDED, fix=ENV_UPDATE_CMD,
        )
    cuda_tag = "+cu" in version
    detail = f"torch {version or 'installed'}"
    if not cuda_tag and version:
        return CheckResult(
            "torch", "Spike sorting", "PyTorch", WARN,
            f"{detail} - this looks like a CPU-only build. Kilosort 4 on CPU is impractically "
            "slow for Neuropixels data.",
            RECOMMENDED,
            fix=(
                "python -m pip install --extra-index-url "
                "https://download.pytorch.org/whl/cu126 torch==2.8.0+cu126"
            ),
            install_keys=("pip-extra:https://download.pytorch.org/whl/cu126:torch==2.8.0+cu126",),
        )
    return CheckResult("torch", "Spike sorting", "PyTorch", OK, detail, REQUIRED)


def _check_cuda() -> CheckResult:
    """Actually import torch and ask the driver, so a broken CUDA install shows up here."""
    if not _is_importable("torch"):
        return CheckResult(
            "cuda", "GPU acceleration", "CUDA device (PyTorch)", WARN,
            "torch is missing, so no GPU could be probed", RECOMMENDED, fix=ENV_UPDATE_CMD,
        )
    try:
        import torch  # noqa: PLC0415 - imported lazily: this is the slow part of the doctor

        if torch.cuda.is_available():
            names = ", ".join(
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            )
            return CheckResult(
                "cuda", "GPU acceleration", "CUDA device (PyTorch)", OK,
                f"{torch.cuda.device_count()} device(s): {names} (CUDA {torch.version.cuda})",
                RECOMMENDED,
            )
        return CheckResult(
            "cuda", "GPU acceleration", "CUDA device (PyTorch)", WARN,
            f"torch {torch.__version__} reports no usable CUDA device; sorting would fall back "
            "to CPU. Check the NVIDIA driver, or reinstall the CUDA wheel.",
            RECOMMENDED,
            fix=(
                "python -m pip install --extra-index-url "
                "https://download.pytorch.org/whl/cu126 torch==2.8.0+cu126"
            ),
        )
    except Exception as exc:  # noqa: BLE001 - a broken install must not abort the doctor
        return CheckResult(
            "cuda", "GPU acceleration", "CUDA device (PyTorch)", WARN,
            f"torch failed to initialize: {exc}", RECOMMENDED, fix=ENV_UPDATE_CMD,
        )


def _check_cupy() -> CheckResult:
    if not _is_importable("cupy"):
        return CheckResult(
            "cupy", "GPU acceleration", "CuPy (GPU filtering)", WARN,
            "cupy is not importable; GPU filtering/whitening falls back to CPU", OPTIONAL,
            fix="conda install -n neuropygui -c conda-forge cupy cuda-version=12",
        )
    return CheckResult(
        "cupy", "GPU acceleration", "CuPy (GPU filtering)", OK,
        f"cupy {_module_version('cupy') or 'installed'}", OPTIONAL,
    )


def _check_bundled_repos() -> List[CheckResult]:
    """Verify the vendored toolboxes are where the runtime resolvers expect them."""
    out: List[CheckResult] = []

    try:
        from .ecephys_runtime import _is_ecephys_repo, resolve_ecephys_repo

        repo = resolve_ecephys_repo()
        ok = repo.exists() and _is_ecephys_repo(repo)
        out.append(
            CheckResult(
                "ecephys_repo", "Bundled toolboxes", "ecephys_spike_sorting", OK if ok else FAIL,
                str(repo) if ok else f"Not a valid checkout: {repo}", REQUIRED,
                fix="" if ok else "Restore the bundled ecephys_spike_sorting folder from the repository.",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(
            CheckResult(
                "ecephys_repo", "Bundled toolboxes", "ecephys_spike_sorting", FAIL,
                f"Resolver failed: {exc}", REQUIRED,
            )
        )

    try:
        from .pybombcell_runtime import _is_pybombcell_repo, resolve_pybombcell_repo

        repo = resolve_pybombcell_repo()
        ok = repo.exists() and _is_pybombcell_repo(repo)
        out.append(
            CheckResult(
                "bombcell_repo", "Bundled toolboxes", "py_bombcell", OK if ok else FAIL,
                str(repo) if ok else f"Not a valid checkout: {repo}", REQUIRED,
                fix="" if ok else "Restore the bundled py_bombcell folder from the repository.",
            )
        )
    except Exception as exc:  # noqa: BLE001
        out.append(
            CheckResult(
                "bombcell_repo", "Bundled toolboxes", "py_bombcell", FAIL,
                f"Resolver failed: {exc}", REQUIRED,
            )
        )

    npyx_init = REPO_ROOT / "npyx" / "__init__.py"
    out.append(
        CheckResult(
            "npyx_repo", "Bundled toolboxes", "npyx (NeuroPyxels)",
            OK if npyx_init.exists() else FAIL,
            str(npyx_init.parent) if npyx_init.exists() else f"Missing: {npyx_init}",
            REQUIRED,
            fix="" if npyx_init.exists() else "Restore the bundled npyx folder from the repository.",
        )
    )
    return out


def _check_phy() -> CheckResult:
    """phy may live in this env or in a sibling conda env; both are supported."""
    try:
        from .phy_launch import resolve_phy_executable

        exe = resolve_phy_executable()
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "phy", "Curation", "phy (manual curation)", WARN,
            f"Could not resolve a phy executable: {exc}", RECOMMENDED,
            fix=f'{sys.executable} -m pip install "phy>=2.1,<3"',
            install_keys=("pip:phy>=2.1,<3",),
        )
    resolved = exe if Path(exe).exists() else (shutil.which(exe) or "")
    if resolved:
        return CheckResult("phy", "Curation", "phy (manual curation)", OK, str(resolved), RECOMMENDED)
    return CheckResult(
        "phy", "Curation", "phy (manual curation)", WARN,
        "No phy executable found in this env, a sibling conda env, or on PATH", RECOMMENDED,
        fix=f'{sys.executable} -m pip install "phy>=2.1,<3"',
        install_keys=("pip:phy>=2.1,<3",),
    )


def _check_c4() -> CheckResult:
    """C4 deliberately runs in its own env, so this is informational only."""
    try:
        from .c4_runner import find_c4_python

        interpreter = find_c4_python()
    except Exception:  # noqa: BLE001
        interpreter = None
    if interpreter:
        return CheckResult(
            "c4", "Post processing", "C4 cell-type classifier (separate env)", OK,
            str(interpreter), OPTIONAL,
        )
    hint = (
        "conda create -n npyx_c4 python=3.10 -y && "
        'conda run -n npyx_c4 python -m pip install "npyx[c4]"'
    )
    if os.name != "nt":
        hint += "  # then set NPYX_C4_PYTHON to that env's python"
    return CheckResult(
        "c4", "Post processing", "C4 cell-type classifier (separate env)", WARN,
        "No npyx_c4 interpreter found. Cell Types > C4 stays unavailable; Bombcell "
        "classification still works.",
        OPTIONAL, fix=hint,
    )


def _check_atlas(read: Callable[[str, str], str]) -> CheckResult:
    """The Allen CCF volumes are a large one-off download, not a pip package."""
    configured = read("histology/atlas_path", "")
    try:
        from .histology import atlas as hatlas

        path = hatlas.resolve_atlas_path(configured or None)
        present = hatlas.atlas_files_present(configured or None)
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "atlas", "Histology", "Allen CCF atlas volumes", WARN,
            f"Could not probe the atlas folder: {exc}", RECOMMENDED,
        )
    if present:
        return CheckResult("atlas", "Histology", "Allen CCF atlas volumes", OK, str(path), RECOMMENDED)
    return CheckResult(
        "atlas", "Histology", "Allen CCF atlas volumes", WARN,
        f"Atlas volumes not found in {path}. Atlas matching and probe tracing stay disabled.",
        RECOMMENDED,
        fix="Download the 10um CCF volumes from https://osf.io/fv7ed/overview, then set "
            "Histology > Setup > Atlas folder (or NPG_ATLAS_PATH).",
    )


def _check_iblapps(read: Callable[[str, str], str]) -> CheckResult:
    """The optional IBL ephys-alignment GUI needs the iblapps repo on the path."""
    configured = read("histology/iblapps_path", "")
    candidates = [configured] if configured else []
    for candidate in candidates:
        if (Path(candidate).expanduser() / "atlaselectrophysiology").is_dir():
            return CheckResult(
                "iblapps", "Histology", "IBL ephys-alignment GUI (iblapps)", OK,
                str(Path(candidate).expanduser()), OPTIONAL,
            )
    if _is_importable("atlaselectrophysiology"):
        return CheckResult(
            "iblapps", "Histology", "IBL ephys-alignment GUI (iblapps)", OK,
            "atlaselectrophysiology is importable", OPTIONAL,
        )
    return CheckResult(
        "iblapps", "Histology", "IBL ephys-alignment GUI (iblapps)", WARN,
        "iblapps not configured. The native AP_histology path (including the per-channel map) "
        "works without it; only the optional IBL refinement GUI is unavailable.",
        OPTIONAL,
        fix="git clone https://github.com/int-brain-lab/iblapps, then set "
            "Histology > Setup > iblapps folder.",
        install_keys=("iblapps",),
    )


def _check_external_tools(read: Callable[[str, str], str]) -> List[CheckResult]:
    """CatGT / TPrime / C_Waves: native SpikeGLX binaries, installed by tool_installer."""
    try:
        from .tool_installer import NATIVE_TOOLS, default_tool_paths, detected_os, native_tool_is_installed
    except Exception as exc:  # noqa: BLE001
        return [
            CheckResult(
                "external_tools", "External tools", "CatGT / TPrime / C_Waves", WARN,
                f"Tool detection unavailable: {exc}", RECOMMENDED,
            )
        ]

    os_name = detected_os()
    defaults = default_tool_paths(REPO_ROOT)
    settings_keys = {"catgt": "preproc/catgt_path", "tprime": "preproc/tprime_path", "cwaves": "preproc/cwaves_path"}
    out: List[CheckResult] = []
    for tool in NATIVE_TOOLS:
        configured = read(settings_keys[tool.key], "") or defaults.get(tool.key, "")
        installed = bool(configured) and native_tool_is_installed(tool, configured, os_name)
        if installed:
            out.append(
                CheckResult(
                    f"tool_{tool.key}", "External tools", tool.name, OK, str(configured), RECOMMENDED,
                )
            )
            continue
        if os_name not in {"windows", "linux"}:
            detail = f"No official prebuilt {tool.name} package exists for {platform.system()}"
            fix = ""
        else:
            detail = f"Not found at {configured or '(not configured)'}"
            fix = "Preprocessing > Settings > Tool and outputs > Install missing tools"
        out.append(
            CheckResult(
                f"tool_{tool.key}",
                "External tools",
                tool.name,
                WARN,
                detail,
                RECOMMENDED,
                fix=fix,
                install_keys=(tool.key,) if fix else (),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def run_diagnostics(
    settings_get: Optional[Callable[[str, object], object]] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> List[CheckResult]:
    """Run every check and return the results in display order.

    ``settings_get`` is a QSettings-style ``(key, default) -> value`` callable used
    to read configured folders (tool paths, atlas, iblapps); pass ``None`` to use
    defaults only. ``progress`` is called as ``(done, total, label)`` after each
    step so a GUI can drive a progress bar.
    """
    read = _settings_reader(settings_get)

    steps: List[Tuple[str, Callable[[], Iterable[CheckResult] | CheckResult]]] = [
        ("Python interpreter", _check_python),
        ("Qt runtime", _check_qt),
        ("numpy", _check_numpy),
        (
            "Plotting stack",
            lambda: _module_group(
                key="pyqtgraph_group", category="Application core", label="pyqtgraph",
                specs=(ModuleSpec("pyqtgraph"),), severity=REQUIRED,
                ok_detail=f"pyqtgraph {_module_version('pyqtgraph') or 'installed'}",
            ),
        ),
        (
            "Scientific stack",
            lambda: _module_group(
                key="scientific", category="Application core", label="Scientific stack",
                specs=_CORE_SCIENTIFIC, severity=REQUIRED,
                ok_detail=f"{len(_CORE_SCIENTIFIC)} packages present "
                          "(scipy, pandas, matplotlib, numba, scikit-learn, ...)",
            ),
        ),
        ("Kilosort 4", _check_kilosort),
        ("PyTorch", _check_torch),
        ("ecephys schema compatibility", _check_ecephys_schema_stack),
        (
            "ecephys pipeline imports",
            lambda: _module_group(
                key="ecephys_imports", category="Spike sorting", label="ecephys pipeline imports",
                specs=_ECEPHYS_IMPORTS, severity=REQUIRED,
                ok_detail="argschema, git, xarray, phylib, tkinter all importable",
            ),
        ),
        ("CUDA device", _check_cuda),
        ("CuPy", _check_cupy),
        ("Bundled toolboxes", _check_bundled_repos),
        (
            "Bombcell GUI stack",
            lambda: _module_group(
                key="bombcell_gui", category="Curation", label="Bombcell notebook GUI",
                specs=(ModuleSpec("ipywidgets"), ModuleSpec("jupyterlab"), ModuleSpec("IPython", "ipython")),
                severity=RECOMMENDED,
                ok_detail="ipywidgets, jupyterlab, ipython present",
            ),
        ),
        ("phy", _check_phy),
        (
            "phy plugin extras",
            lambda: _module_group(
                key="phy_plugins", category="Curation", label="phy plugin extras",
                specs=(ModuleSpec("seaborn"), ModuleSpec("umap", "umap-learn")),
                severity=OPTIONAL,
                ok_detail="seaborn (mahalanobis) and umap-learn (UMAP/recluster) present",
            ),
        ),
        (
            "Post-processing stack",
            lambda: _module_group(
                key="postproc", category="Post processing", label="Post-processing stack",
                specs=(ModuleSpec("cachecache"), ModuleSpec("upsetplot"), ModuleSpec("mtscomp")),
                severity=RECOMMENDED,
                ok_detail="cachecache, upsetplot, mtscomp present",
            ),
        ),
        ("C4 classifier", _check_c4),
        (
            "Histology imaging stack",
            lambda: _module_group(
                key="hist_imaging", category="Histology", label="Imaging stack",
                specs=(ModuleSpec("tifffile"), ModuleSpec("imagecodecs"), ModuleSpec("cv2", "opencv-python-headless")),
                severity=RECOMMENDED,
                ok_detail="tifffile, imagecodecs, opencv present",
            ),
        ),
        (
            "IBL stack",
            lambda: _module_group(
                key="hist_ibl", category="Histology", label="IBL stack (channel map)",
                specs=(ModuleSpec("ibllib"), ModuleSpec("iblatlas"), ModuleSpec("SimpleITK")),
                severity=RECOMMENDED,
                ok_detail="ibllib, iblatlas, SimpleITK present",
            ),
        ),
        ("Atlas volumes", lambda: _check_atlas(read)),
        ("iblapps", lambda: _check_iblapps(read)),
        ("External tools", lambda: _check_external_tools(read)),
    ]

    results: List[CheckResult] = []
    total = len(steps)
    for index, (label, step) in enumerate(steps, start=1):
        try:
            outcome = step()
        except Exception as exc:  # noqa: BLE001 - one broken probe must not hide the rest
            outcome = CheckResult(
                f"error_{index}", "Application core", label, WARN,
                f"Check failed to run: {exc}", OPTIONAL,
            )
        if isinstance(outcome, CheckResult):
            results.append(outcome)
        else:
            results.extend(outcome)
        if progress is not None:
            try:
                progress(index, total, label)
            except Exception:
                pass

    order = {name: i for i, name in enumerate(CATEGORY_ORDER)}
    results.sort(key=lambda r: (order.get(r.category, len(order)), r.label.lower()))
    return results


def summarize(results: Sequence[CheckResult]) -> Dict[str, int]:
    """Return ``{"ok": n, "warn": n, "fail": n}`` counts for ``results``."""
    counts = {OK: 0, WARN: 0, FAIL: 0}
    for result in results:
        if result.status in counts:
            counts[result.status] += 1
    return counts


def has_blocking_failures(results: Sequence[CheckResult]) -> bool:
    """True when at least one required component is missing."""
    return any(result.is_blocking for result in results)


def headline(results: Sequence[CheckResult]) -> str:
    """One-sentence verdict suitable for a dialog header or a status bar."""
    counts = summarize(results)
    if counts[FAIL]:
        return (
            f"{counts[FAIL]} required component(s) missing - some workflows cannot run yet."
        )
    if counts[WARN]:
        return f"Core install is complete. {counts[WARN]} optional component(s) need attention."
    return "Everything NeuroPyGuiN needs is installed."


def report_text(results: Sequence[CheckResult]) -> str:
    """Render the results as a plain-text report (used by "Copy report")."""
    counts = summarize(results)
    lines = [
        "NeuroPyGuiN diagnostics",
        f"  platform : {platform.platform()}",
        f"  python   : {sys.version.split()[0]} ({sys.executable})",
        f"  app root : {REPO_ROOT}",
        f"  summary  : {counts[OK]} ok, {counts[WARN]} warnings, {counts[FAIL]} failures",
        "",
    ]
    symbols = {OK: "[ ok ]", WARN: "[warn]", FAIL: "[FAIL]"}
    current = ""
    for result in results:
        if result.category != current:
            current = result.category
            lines.append(f"{current}:")
        lines.append(f"  {symbols.get(result.status, '[ ?? ]')} {result.label}: {result.detail}")
        if result.fix and result.status != OK:
            lines.append(f"         fix: {result.fix}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Terminal entry point: print the report and exit non-zero on hard failures."""
    results = run_diagnostics()
    print(report_text(results))
    print()
    print(headline(results))
    return 1 if has_blocking_failures(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())

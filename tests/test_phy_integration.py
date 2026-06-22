from __future__ import annotations

import os
from pathlib import Path

from neuropyguin.phy_gamepad_plugin import _gamepad_plugin_source
from neuropyguin.phy_integration import PHY_PLUGIN_CLASS, PHY_PLUGIN_FILE, ensure_phy_short_isi_plugin
from neuropyguin.phy_launch import (
    PHY_EXECUTABLE_ENV,
    _phy_exe_name,
    _scripts_dir_name,
    phy_child_environment,
    resolve_phy_executable,
)


def test_ensure_phy_short_isi_plugin_creates_plugin_and_config(tmp_path: Path) -> None:
    result = ensure_phy_short_isi_plugin(tmp_path)
    assert result["plugin_updated"] is True
    assert result["config_updated"] is True

    plugin_path = tmp_path / "plugins" / PHY_PLUGIN_FILE
    config_path = tmp_path / "phy_config.py"
    assert plugin_path.exists()
    assert config_path.exists()
    assert "Split short ISI" in plugin_path.read_text(encoding="utf-8")
    assert PHY_PLUGIN_CLASS in config_path.read_text(encoding="utf-8")


def test_ensure_phy_short_isi_plugin_is_idempotent(tmp_path: Path) -> None:
    first = ensure_phy_short_isi_plugin(tmp_path)
    second = ensure_phy_short_isi_plugin(tmp_path)
    assert first["plugin_updated"] is True
    assert first["config_updated"] is True
    assert second["plugin_updated"] is False
    assert second["config_updated"] is False


def test_ensure_phy_short_isi_plugin_replaces_previous_registration_block(tmp_path: Path) -> None:
    cfg = tmp_path / "phy_config.py"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        "\n".join(
            [
                "from phy import IPlugin",
                "c = get_config()",
                "# >>> NeuroPyGuiN auto plugin registration >>>",
                "c.TemplateGUI.plugins = ['OldPlugin']",
                "# <<< NeuroPyGuiN auto plugin registration <<<",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ensure_phy_short_isi_plugin(tmp_path)
    text = cfg.read_text(encoding="utf-8")
    assert text.count(">>> NeuroPyGuiN auto plugin registration >>>") == 1
    assert PHY_PLUGIN_CLASS in text


def test_phy_plugin_source_contains_scroll_optimizer(tmp_path: Path) -> None:
    result = ensure_phy_short_isi_plugin(tmp_path)
    plugin_path = Path(result["plugin_path"])
    text = plugin_path.read_text(encoding="utf-8")

    assert "class _ClusterScrollOptimizer(QObject):" in text
    assert "_SCROLL_SELECTION_DEBOUNCE_MS = 120" in text
    assert "_SCROLL_SETTLE_MS = 250" in text
    assert "scroll_optimizer.install()" in text
    assert "scroll_optimizer.note_selected(cluster_ids)" in text


def test_gamepad_plugin_source_contains_layout_aware_mapping_hooks() -> None:
    source = _gamepad_plugin_source()

    assert '_LAYOUT_SETTINGS_KEY = "gamepad/layout_v1"' in source
    assert '_BUTTON_MAP_SETTINGS_KEY_TEMPLATE = "gamepad/button_map_{layout}_v1"' in source
    assert 'LAYOUT_SWITCH = "switch"' in source
    assert 'def _processed_controller_asset(image_path):' in source
    assert 'class _AssignmentRow(QFrame):' in source
    assert 'class _ControllerMapWidget(QWidget):' in source
    assert 'QScrollArea' in source
    assert 'QPushButton#layoutButton' in source
    assert 'Nintendo Switch' in source
    assert 'xbox.png' in source
    assert 'switch.png' in source
    assert '__XBOX_LAYOUT_IMAGE__' not in source
    assert '__SWITCH_LAYOUT_IMAGE__' not in source
    assert 'self._layout_key = self._load_layout_key()' in source
    assert 'self._button_maps = {' in source
    assert 'self._plugin._save_layout_key(self._plugin._layout_key)' in source
    assert 'self._plugin._save_button_map(layout_key, button_map)' in source
    assert '_run_edit_action(' in source
    assert 'split_short_isi' in source
    assert '_run_file_action(' in source
    assert 'save' in source


def test_gamepad_plugin_source_contains_xinput_fallback() -> None:
    source = _gamepad_plugin_source()

    assert "_HAS_XINPUT = _XINPUT_DLL is not None" in source
    assert "class _XInputController:" in source
    assert "def _detect_xinput_controller(self):" in source
    assert 'self._lbl_backend = QLabel()' in source
    assert 'if _HAS_PYGAME or _HAS_XINPUT:' in source
    assert 'logger.info("Gamepad connected via %s: %s", backend, name)' in source
    assert 'backend == _XINPUT_BACKEND' in source


def test_gamepad_plugin_source_forces_sdl_dummy_before_pygame_import() -> None:
    source = _gamepad_plugin_source()

    video_idx = source.index('os.environ.setdefault("SDL_VIDEODRIVER", "dummy")')
    pygame_idx = source.index("    import pygame")
    assert video_idx < pygame_idx
    assert 'os.environ.setdefault("SDL_AUDIODRIVER", "dummy")' in source


def test_resolve_phy_executable_honors_explicit_env_var(tmp_path: Path) -> None:
    fake_phy = tmp_path / _phy_exe_name()

    assert resolve_phy_executable({PHY_EXECUTABLE_ENV: str(fake_phy), "PATH": ""}) == str(fake_phy)


def test_phy_child_environment_forces_pyqt5_and_sdl_dummy() -> None:
    env = {
        "PATH": "original",
        "PYQTGRAPH_QT_LIB": "PySide6",
        "QT_API": "pyside6",
    }

    child = phy_child_environment(env)

    assert child["PYQTGRAPH_QT_LIB"] == "PyQt5"
    assert child["QT_API"] == "pyqt5"
    assert child["SDL_VIDEODRIVER"] == "dummy"
    assert child["SDL_AUDIODRIVER"] == "dummy"
    assert child["PYGAME_HIDE_SUPPORT_PROMPT"] == "1"


def test_phy_child_environment_prepends_conda_runtime_paths(tmp_path: Path) -> None:
    conda_root = tmp_path / "phy2"
    (conda_root / "conda-meta").mkdir(parents=True)
    scripts = conda_root / _scripts_dir_name()
    scripts.mkdir()
    phy_exe = scripts / _phy_exe_name()
    phy_exe.write_text("", encoding="utf-8")
    if _scripts_dir_name() == "Scripts":
        (conda_root / "Library" / "bin").mkdir(parents=True)
        (conda_root / "Library" / "usr" / "bin").mkdir(parents=True)
        site_packages = conda_root / "Lib" / "site-packages"
    else:
        site_packages = conda_root / "lib" / "python3.10" / "site-packages"
    site_packages.mkdir(parents=True)

    child = phy_child_environment({"PATH": "original"}, str(phy_exe))

    expected_first = str(conda_root) if _scripts_dir_name() == "Scripts" else str(conda_root / "bin")
    assert child["PATH"].split(os.pathsep)[0] == expected_first
    # phy is no longer hard-isolated from the user site (some user phy plugins need
    # user-site packages like seaborn/umap-learn). Instead the env's own
    # site-packages lead PYTHONPATH so its tested numpy/scipy/scikit-learn win.
    assert "PYTHONNOUSERSITE" not in child
    assert child["PYTHONPATH"].split(os.pathsep)[0] == str(site_packages)

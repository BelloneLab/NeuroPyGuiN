from __future__ import annotations

from pathlib import Path

from neuropyguin.phy_gamepad_plugin import _gamepad_plugin_source
from neuropyguin.phy_integration import PHY_PLUGIN_CLASS, PHY_PLUGIN_FILE, ensure_phy_short_isi_plugin


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

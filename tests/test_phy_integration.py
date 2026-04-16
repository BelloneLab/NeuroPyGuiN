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


def test_gamepad_plugin_source_contains_persistent_xbox_mapping_hooks() -> None:
    source = _gamepad_plugin_source()

    assert '_BUTTON_MAP_SETTINGS_KEY = "gamepad/button_map_v2"' in source
    assert 'class _ControllerMapWidget(QWidget):' in source
    assert 'Xbox control' in source
    assert 'self._button_map = self._load_button_map()' in source
    assert '_save_button_map(self._plugin._button_map)' in source
    assert '_run_edit_action(' in source
    assert 'split_short_isi' in source
    assert '_run_file_action(' in source
    assert 'save' in source

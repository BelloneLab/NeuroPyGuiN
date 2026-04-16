from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Dict

from .phy_gamepad_plugin import _gamepad_plugin_source


PHY_PLUGIN_CLASS = "NeuroPyGuiNSplitShortISIContext"
PHY_PLUGIN_FILE = "neuropygui_split_short_isi_context.py"

PHY_GAMEPAD_PLUGIN_CLASS = "NeuroPyGuiNGamepadController"
PHY_GAMEPAD_PLUGIN_FILE = "neuropyguin_gamepad_controller.py"

_REGISTRATION_BEGIN = "# >>> NeuroPyGuiN auto plugin registration >>>"
_REGISTRATION_END = "# <<< NeuroPyGuiN auto plugin registration <<<"


def _plugin_source() -> str:
    return dedent(
        """
        from phy import IPlugin, connect
        from phylib.utils import unconnect
        from phy.gui.qt import QMenu, QPoint
        import logging
        import numpy as np

        logger = logging.getLogger("phy")


        class NeuroPyGuiNSplitShortISIContext(IPlugin):
            def attach_to_controller(self, controller):
                @connect
                def on_gui_ready(sender, gui):
                    supervisor = controller.supervisor

                    @supervisor.actions.add(
                        name="Split short ISI",
                        alias="split_short_isi",
                        shortcut="alt+shift+i",
                    )
                    def split_short_isi():
                        selected = list(supervisor.selected or [])
                        if len(selected) != 1:
                            logger.info("Split short ISI: select exactly one cluster.")
                            return

                        cluster_id = int(selected[0])
                        spike_ids = np.asarray(
                            supervisor.clustering.spikes_in_clusters([cluster_id]),
                            dtype=np.int64,
                        )
                        if spike_ids.size < 3:
                            logger.info("Split short ISI: cluster %d has too few spikes.", cluster_id)
                            return

                        spike_times = np.asarray(controller.model.spike_times[spike_ids], dtype=float)
                        order = np.argsort(spike_times)
                        sorted_ids = spike_ids[order]
                        sorted_times = spike_times[order]
                        dt = np.diff(sorted_times)

                        short_isi = dt < 0.0015
                        if not np.any(short_isi):
                            logger.info("Split short ISI: no short-ISI spikes in cluster %d.", cluster_id)
                            return

                        labels = np.ones(sorted_ids.shape[0], dtype=np.int64)
                        labels[1:][short_isi] = 2
                        labels[:-1][short_isi] = 2

                        if np.count_nonzero(labels == 2) < 2:
                            logger.info("Split short ISI: no robust short-ISI subset in cluster %d.", cluster_id)
                            return

                        supervisor.actions.split(sorted_ids, labels)
                        logger.info(
                            "Split short ISI: cluster %d, moved %d/%d spikes.",
                            cluster_id,
                            int(np.count_nonzero(labels == 2)),
                            int(labels.size),
                        )

                    def _find_short_isi_action():
                        actions = supervisor.actions
                        for name in ("split_short_isi", "Split short ISI", "split_short_isi_1"):
                            action = actions.get(name)
                            if action is not None:
                                return action
                        for name in sorted(getattr(actions, "_actions_dict", {})):
                            lowered = str(name).lower().replace("-", "_")
                            if "short" in lowered and "isi" in lowered:
                                action = actions.get(name)
                                if action is not None:
                                    return action
                        return None

                    try:
                        unconnect(supervisor._show_cluster_context_menu)
                    except Exception:
                        pass

                    @connect(sender=supervisor.cluster_view, event="context_menu")
                    def on_cluster_context_menu(sender, obj):
                        if sender != supervisor.cluster_view or not obj:
                            return

                        selected = [int(cluster_id) for cluster_id in obj.get("selected", []) or []]
                        if not selected:
                            return
                        supervisor._sync_context_menu_selection(selected)

                        actions = getattr(supervisor, "actions", None)
                        merge_action = actions.get("merge") if actions is not None else None
                        noise_action = actions.get("move_best_to_noise") if actions is not None else None
                        mua_action = actions.get("move_best_to_mua") if actions is not None else None
                        good_action = actions.get("move_best_to_good") if actions is not None else None
                        kmeans_action = supervisor._find_cluster_context_action("K_means_clustering")
                        short_isi_action = _find_short_isi_action()

                        menu = QMenu(supervisor.cluster_view)
                        supervisor._add_context_menu_action(menu, "Noise", noise_action, enabled=len(selected) >= 1)
                        supervisor._add_context_menu_action(menu, "Good", good_action, enabled=len(selected) >= 1)
                        supervisor._add_context_menu_action(menu, "MUA", mua_action, enabled=len(selected) >= 1)
                        menu.addSeparator()
                        supervisor._add_context_menu_action(
                            menu,
                            "Split short ISI",
                            short_isi_action,
                            enabled=len(selected) == 1,
                        )
                        supervisor._add_context_menu_action(
                            menu,
                            "Split With K-Means...",
                            kmeans_action,
                            enabled=len(selected) >= 1,
                        )
                        supervisor._add_context_menu_action(
                            menu,
                            "Merge",
                            merge_action,
                            enabled=len(selected) >= 2,
                        )

                        point = QPoint(int(obj.get("x", 0) or 0), int(obj.get("y", 0) or 0))
                        menu.exec(supervisor.cluster_view.mapToGlobal(point))
        """
    ).strip() + "\n"


def _registration_block(plugin_dir: Path) -> str:
    plugin_dir = plugin_dir.resolve()
    return dedent(
        f"""
        {_REGISTRATION_BEGIN}
        try:
            _neuropygui_plugin_dirs = list(getattr(c.Plugins, "dirs", []))
        except Exception:
            _neuropygui_plugin_dirs = []
        _neuropygui_plugin_dir = r"{plugin_dir}"
        if _neuropygui_plugin_dir not in _neuropygui_plugin_dirs:
            _neuropygui_plugin_dirs.append(_neuropygui_plugin_dir)
        c.Plugins.dirs = _neuropygui_plugin_dirs

        try:
            _neuropygui_template_plugins = list(getattr(c.TemplateGUI, "plugins", []))
        except Exception:
            _neuropygui_template_plugins = []
        for _cls in ["{PHY_PLUGIN_CLASS}", "{PHY_GAMEPAD_PLUGIN_CLASS}"]:
            if _cls not in _neuropygui_template_plugins:
                _neuropygui_template_plugins.append(_cls)
        c.TemplateGUI.plugins = _neuropygui_template_plugins
        {_REGISTRATION_END}
        """
    ).strip() + "\n"


def _ensure_default_phy_config(config_path: Path) -> None:
    if config_path.exists():
        return
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        dedent(
            """
            from phy import IPlugin

            c = get_config()
            c.Plugins.dirs = [r'~/.phy/plugins/']
            c.TemplateGUI.plugins = []
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def _upsert_registration_block(config_text: str, block: str) -> str:
    if block in config_text:
        return config_text

    start = config_text.find(_REGISTRATION_BEGIN)
    end = config_text.find(_REGISTRATION_END)
    if start != -1 and end != -1 and end >= start:
        end_idx = end + len(_REGISTRATION_END)
        while end_idx < len(config_text) and config_text[end_idx] in "\r\n":
            end_idx += 1
        before = config_text[:start].rstrip()
        after = config_text[end_idx:].lstrip()
        pieces = [before, block]
        if after:
            pieces.append(after)
        return "\n\n".join(piece for piece in pieces if piece) + "\n"

    base = config_text.rstrip()
    if base:
        return f"{base}\n\n{block}"
    return block


def ensure_phy_short_isi_plugin(phy_home: Path | None = None) -> Dict[str, object]:
    home = Path(phy_home) if phy_home is not None else (Path.home() / ".phy")
    plugins_dir = home / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)

    # ── Split-short-ISI plugin ────────────────────────────────────────
    plugin_path = plugins_dir / PHY_PLUGIN_FILE
    plugin_text = _plugin_source()
    old_plugin = plugin_path.read_text(encoding="utf-8", errors="ignore") if plugin_path.exists() else ""
    plugin_updated = old_plugin != plugin_text
    if plugin_updated:
        plugin_path.write_text(plugin_text, encoding="utf-8")

    # ── Gamepad controller plugin ─────────────────────────────────────
    gamepad_path = plugins_dir / PHY_GAMEPAD_PLUGIN_FILE
    gamepad_text = _gamepad_plugin_source()
    old_gamepad = gamepad_path.read_text(encoding="utf-8", errors="ignore") if gamepad_path.exists() else ""
    gamepad_updated = old_gamepad != gamepad_text
    if gamepad_updated:
        gamepad_path.write_text(gamepad_text, encoding="utf-8")

    # ── phy_config.py registration ────────────────────────────────────
    config_path = home / "phy_config.py"
    _ensure_default_phy_config(config_path)
    config_old = config_path.read_text(encoding="utf-8", errors="ignore")
    config_new = _upsert_registration_block(config_old, _registration_block(plugins_dir))
    config_updated = config_old != config_new
    if config_updated:
        config_path.write_text(config_new, encoding="utf-8")

    return {
        "plugin_path": str(plugin_path),
        "gamepad_plugin_path": str(gamepad_path),
        "config_path": str(config_path),
        "plugin_updated": plugin_updated,
        "gamepad_plugin_updated": gamepad_updated,
        "config_updated": config_updated,
    }

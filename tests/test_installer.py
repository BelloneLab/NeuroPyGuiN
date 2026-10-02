"""Installer regression checks with isolated paths and no scientific downloads."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(name):
    """Load standalone installer modules without importing the GUI package."""
    spec = importlib.util.spec_from_file_location(name, ROOT / 'install' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup = load_module('setup')
launch = load_module('launch')


def test_desktop_exec_escapes_shell_and_field_codes():
    """Desktop entries have two escaping passes and percent placeholders."""
    rendered = setup.desktop_exec(['/a folder/100%/$HOME/`quote`/"app"', 'a\\b'])
    assert '100%%' in rendered
    assert '\\\\$HOME' in rendered
    assert '\\\\`quote\\\\`' in rendered
    assert '\\\\"app\\\\"' in rendered
    assert 'a\\\\\\\\b' in rendered


def test_linux_shortcuts_are_valid_and_repeatable(tmp_path, monkeypatch):
    """Create native desktop entries in paths with spaces, never the real home."""
    home = tmp_path / 'test home'
    desktop = home / 'Desktop'
    desktop.mkdir(parents=True)
    prefix = home / 'app install'
    source = home / 'code % $ with spaces'
    state = prefix / 'installation.json'
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: home))
    monkeypatch.setenv('XDG_DATA_HOME', str(home / 'share'))
    monkeypatch.setattr(setup.shutil, 'which', lambda name: None)
    setup.linux_shortcuts(prefix, source, state)
    first = (desktop / 'neuropyguin.desktop').read_text()
    setup.linux_shortcuts(prefix, source, state)
    assert (desktop / 'neuropyguin.desktop').read_text() == first
    assert 'Terminal=false' in first
    assert '%%' in first
    update = (desktop / 'neuropyguin-update.desktop').read_text()
    assert 'Terminal=true' in update and '--update' in update
    assert '--prefix' in update and '--source' in update
    assert (home / 'share/applications/neuropyguin.desktop').exists()
    validator = '/usr/bin/desktop-file-validate'
    if Path(validator).exists():
        subprocess.run([validator, str(desktop / 'neuropyguin.desktop')], check=True)
        subprocess.run([validator, str(desktop / 'neuropyguin-update.desktop')], check=True)


def test_headless_shortcuts_do_not_create_desktop(tmp_path, monkeypatch):
    """Do not clutter home with an English Desktop folder on headless machines."""
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: tmp_path))
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'share'))
    monkeypatch.setattr(setup.shutil, 'which', lambda name: None)
    setup.linux_shortcuts(tmp_path, tmp_path / 'source', tmp_path / 'state.json')
    assert not (tmp_path / 'Desktop').exists()


@pytest.mark.parametrize('branch,dirty,message', [
    ('feature', '', 'requires branch main'),
    ('main', ' M main.py', 'local changes'),
])
def test_update_preserves_local_work(tmp_path, monkeypatch, branch, dirty, message):
    """Unsafe updates must stop before any pull or environment mutation."""
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        output = branch if 'rev-parse' in command else dirty
        return subprocess.CompletedProcess(command, 0, output)
    monkeypatch.setattr(setup, 'run', fake_run)
    with pytest.raises(RuntimeError, match=message):
        setup.update_checkout(tmp_path)
    assert not any('pull' in command for command in calls)


def test_update_only_fast_forwards(tmp_path, monkeypatch):
    """No reset, force, stash, or merge commit is introduced by the updater."""
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, 'main' if 'rev-parse' in command else '')
    monkeypatch.setattr(setup, 'run', fake_run)
    setup.update_checkout(tmp_path)
    assert calls[-1][-4:] == ['pull', '--ff-only', 'origin', 'main']


@pytest.mark.parametrize('existing', [False, True])
def test_setup_installs_platform_yaml_and_writes_state(tmp_path, monkeypatch, existing):
    """Fresh install and repair share the manual environment specification."""
    prefix = tmp_path / 'install space'
    source = tmp_path / 'source space'
    prefix.mkdir()
    source.mkdir()
    (source / 'environment-linux.yml').write_text('name: example\n')
    if existing:
        history = prefix / 'runtime/conda-meta/history'
        history.parent.mkdir(parents=True)
        history.touch()
    calls = []
    monkeypatch.setattr(setup.sys, 'platform', 'linux')
    monkeypatch.setattr(setup, 'run', lambda command, **kwargs: calls.append(command))
    shortcuts = []
    monkeypatch.setattr(setup, 'linux_shortcuts', lambda *args: shortcuts.append(args))
    setup.configure(argparse.Namespace(prefix=prefix, source=source, update=False, no_launch=True))
    assert calls[0][1:3] == ['env', 'update' if existing else 'create']
    assert calls[0][-1] == str(source / 'environment-linux.yml')
    assert '--no-rc' in calls[0]
    assert 'prepare_icon.py' in calls[1][5]
    state = json.loads((prefix / 'installation.json').read_text())
    assert state['source'] == str(source)
    assert state['runtime'] == str(prefix / 'runtime')
    assert len(shortcuts) == 1


def test_failed_dependency_install_does_not_publish_shortcuts(tmp_path, monkeypatch):
    """A failed runtime install must not claim that the app is ready."""
    (tmp_path / 'environment-linux.yml').touch()
    monkeypatch.setattr(setup.sys, 'platform', 'linux')
    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(setup, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        setup.configure(argparse.Namespace(prefix=tmp_path, source=tmp_path, update=False, no_launch=True))
    assert not (tmp_path / 'installation.json').exists()


def test_windows_shortcuts_preserve_unicode_and_quotes(tmp_path, monkeypatch):
    """No user path is interpolated into a PowerShell command as raw code."""
    import base64
    calls = []
    monkeypatch.setattr(setup, 'run', lambda command, **kwargs: calls.append(command))
    prefix = tmp_path / "Zoë's app $data"
    setup.windows_shortcuts(prefix, tmp_path / 'source space', prefix / 'installation.json')
    script = base64.b64decode(calls[0][-1]).decode('utf-16-le')
    assert 'WScript.Shell' in script
    assert 'Zoë' not in script
    assert "GetFolderPath('DesktopDirectory')" in script
    assert "GetFolderPath('Programs')" in script
    payload = script.split("FromBase64String('")[1].split("')")[0]
    data = json.loads(base64.b64decode(payload))
    assert data['python'] == str(prefix / 'bootstrap/pythonw.exe')
    update = base64.b64decode(data['update_arguments'].split()[-1]).decode('utf-16-le')
    assert "Zoë''s app $data" in update


def test_shortcut_launcher_uses_environment_and_logs(tmp_path, monkeypatch):
    """Launch through micromamba so Windows DLL and Linux runtime paths work."""
    state = dict(prefix=str(tmp_path), source=str(tmp_path / 'source'),
                 runtime=str(tmp_path / 'runtime'), mamba=str(tmp_path / 'micromamba'))
    state_path = tmp_path / 'installation.json'
    state_path.write_text(json.dumps(state))
    monkeypatch.setattr(sys, 'argv', ['launch.py', '--state', str(state_path)])
    monkeypatch.setenv('PYTHONPATH', '/unrelated/python')
    calls = []
    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(launch.subprocess, 'run', fake_run)
    assert launch.main() == 0
    command, kwargs = calls[0]
    assert command[1:4] == ['run', '-p', state['runtime']]
    assert kwargs['cwd'] == state['source']
    assert 'PYTHONPATH' not in kwargs['env']
    assert kwargs['env']['PYTHONNOUSERSITE'] == '1'
    assert (tmp_path / 'logs/launcher.log').exists()


def test_launcher_failure_is_visible(tmp_path, monkeypatch):
    """Broken installation metadata produces an error, not a silent shortcut."""
    monkeypatch.setattr(sys, 'argv', ['launch.py', '--state', str(tmp_path / 'missing.json')])
    errors = []
    monkeypatch.setattr(launch, 'report_error', errors.append)
    assert launch.main() == 1
    assert 'launcher.log' in errors[0]


def test_update_restarts_new_bootstrap_without_pulling_twice(tmp_path, monkeypatch):
    """Updates must also apply changes to the pinned environment manager."""
    (tmp_path / 'environment-linux.yml').touch()
    monkeypatch.setattr(setup.sys, 'platform', 'linux')
    updates, calls = [], []
    monkeypatch.setattr(setup, 'update_checkout', updates.append)
    monkeypatch.setattr(setup, 'run', lambda command, **kwargs: calls.append(command))
    setup.configure(argparse.Namespace(prefix=tmp_path, source=tmp_path, update=True, no_launch=True))
    assert updates == [tmp_path]
    assert calls[0][0] == 'bash'
    assert calls[0][1].endswith('Install-NeuroPyGuiN.sh')
    assert '--update' not in calls[0]
    assert '--no-launch' in calls[0]

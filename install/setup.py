"""Finish a per-user installation using the same environment files as manual setup.

The bootstrap scripts supply Python and Git. This module owns dependency setup,
conservative Git updates, and OS shortcuts. It never edits the user's Conda
configuration, shell profile, app settings, or recording directories.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run one process without a shell, propagating failures to the installer."""
    return subprocess.run([str(part) for part in command], check=True, **kwargs)


def update_checkout(source: Path) -> None:
    """Fast-forward main, retaining local work and refusing other branches."""
    def git(*args: str) -> str:
        return run(['git', '-C', str(source), *args], capture_output=True, text=True).stdout.strip()

    if git('rev-parse', '--abbrev-ref', 'HEAD') != 'main':
        raise RuntimeError('Update requires branch main. Switch branches yourself, then retry.')
    if git('status', '--porcelain', '--untracked-files=no'):
        raise RuntimeError('This checkout has local changes. Commit or move them before updating.')
    # Git also refuses overwriting untracked files if an incoming path conflicts.
    git('pull', '--ff-only', 'origin', 'main')


def desktop_value(value: str) -> str:
    """Escape Desktop Entry string values before writing UTF-8 text."""
    return value.replace('\\', '\\\\').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')


def desktop_exec(parts: list[str]) -> str:
    """Quote argv using Desktop Entry rules, which differ from shell quoting."""
    quoted = []
    for part in parts:
        # Exec performs its own escape pass after the desktop string escape pass.
        arg = str(part).replace('%', '%%')
        for char in ('\\', '"', '`', '$'):
            arg = arg.replace(char, '\\' + char)
        quoted.append('"' + arg + '"')
    return desktop_value(' '.join(quoted))


def linux_shortcuts(prefix: Path, source: Path, state_path: Path) -> None:
    """Create discoverable menu entries and desktop copies when a desktop exists."""
    python = prefix / 'bootstrap/bin/python'
    launcher = source / 'install/launch.py'
    menu = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local/share')) / 'applications'
    menu.mkdir(parents=True, exist_ok=True)
    desktop = Path.home() / 'Desktop'
    if shutil.which('xdg-user-dir'):
        result = run(['xdg-user-dir', 'DESKTOP'], capture_output=True, text=True).stdout.strip()
        if result:
            desktop = Path(result)
    entries = [
        ('NeuroPyGuiN', 'Neural recording analysis',
         [str(python), str(launcher), '--state', str(state_path)], False),
        ('Update NeuroPyGuiN', 'Update the GitHub checkout and application dependencies',
         ['bash', str(source / 'Install-NeuroPyGuiN.sh'), '--prefix', str(prefix),
          '--source', str(source), '--update', '--no-launch', '--terminal'], True),
    ]
    for name, description, command, terminal in entries:
        filename = 'neuropyguin-update.desktop' if terminal else 'neuropyguin.desktop'
        content = '\n'.join([
            '[Desktop Entry]', 'Type=Application', f'Name={name}', f'Comment={description}',
            f'Exec={desktop_exec(command)}',
            f'Path={desktop_value(str(source))}',
            f'Icon={desktop_value(str(source / "neuropyguin/assets/neuropyguin-icon.svg"))}',
            f'Terminal={str(terminal).lower()}', 'Categories=Science;',
            'StartupNotify=true', 'StartupWMClass=NeuroPyGuiN', '',
        ])
        (menu / filename).write_text(content, encoding='utf-8')
        # Do not create an English Desktop folder on a localized/headless system.
        if desktop.is_dir() and desktop != Path.home():
            target = desktop / filename
            target.write_text(content, encoding='utf-8')
            target.chmod(0o755)
            if shutil.which('gio'):
                subprocess.run(['gio', 'set', str(target), 'metadata::trusted', 'true'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    if shutil.which('update-desktop-database'):
        subprocess.run(['update-desktop-database', str(menu)], check=False)


def windows_shortcuts(prefix: Path, source: Path, state_path: Path) -> None:
    """Use known Windows folders and COM to create app and updater shortcuts."""
    # JSON and EncodedCommand preserve Unicode, quotes, and spaces in paths.
    data = {
        'python': str(prefix / 'bootstrap/pythonw.exe'),
        'arguments': subprocess.list2cmdline([str(source / 'install/launch.py'), '--state', str(state_path)]),
        'source': str(source), 'icon': str(prefix / 'neuropyguin.ico'),
    }
    updater = source / 'install/install.ps1'
    ps_quote = lambda value: "'" + str(value).replace("'", "''") + "'"
    update_script = (
        f"& {ps_quote(updater)} -Prefix {ps_quote(prefix)} -Source {ps_quote(source)} -Update -NoLaunch; "
        "if ($LASTEXITCODE -ne 0) { Write-Host 'Update failed. Review the error above.' }; "
        "Read-Host 'Press Enter to close'"
    )
    data['update_arguments'] = '-NoProfile -ExecutionPolicy Bypass -EncodedCommand ' + base64.b64encode(
        update_script.encode('utf-16-le')).decode('ascii')
    encoded_data = base64.b64encode(json.dumps(data).encode('utf-8')).decode('ascii')
    script = """
$ErrorActionPreference = 'Stop'
$d = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('DATA')) | ConvertFrom-Json
$desktop = [Environment]::GetFolderPath('DesktopDirectory')
$menu = Join-Path ([Environment]::GetFolderPath('Programs')) 'NeuroPyGuiN'
New-Item -ItemType Directory -Force -Path $menu | Out-Null
$w = New-Object -ComObject WScript.Shell
foreach ($folder in @($desktop, $menu)) {
    if (-not $folder) { continue }
    $link = $w.CreateShortcut((Join-Path $folder 'NeuroPyGuiN.lnk'))
    $link.TargetPath = $d.python
    $link.Arguments = $d.arguments
    $link.WorkingDirectory = $d.source
    $link.IconLocation = $d.icon
    $link.Description = 'Neural recording analysis'
    $link.Save()
    $link = $w.CreateShortcut((Join-Path $folder 'Update NeuroPyGuiN.lnk'))
    $link.TargetPath = Join-Path $env:SystemRoot 'System32\\WindowsPowerShell\\v1.0\\powershell.exe'
    $link.Arguments = $d.update_arguments
    $link.WorkingDirectory = $d.source
    $link.IconLocation = $d.icon
    $link.Save()
}
""".replace('DATA', encoded_data)
    run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-EncodedCommand',
         base64.b64encode(script.encode('utf-16-le')).decode('ascii')])


def configure(args: argparse.Namespace) -> None:
    """Install dependencies, check the GUI runtime, and publish usable shortcuts."""
    prefix, source = args.prefix.resolve(), args.source.resolve()
    windows = sys.platform == 'win32'
    platform_name = 'windows' if windows else 'linux'
    mamba = prefix / 'bin' / ('micromamba.exe' if windows else 'micromamba')
    runtime = prefix / 'runtime'
    environment = source / f'environment-{platform_name}.yml'
    if not environment.is_file():
        raise RuntimeError(f'Missing environment file: {environment}')
    os.environ['MAMBA_ROOT_PREFIX'] = str(prefix / 'mamba')
    os.environ['PYTHONNOUSERSITE'] = '1'
    os.environ['NEUROPYGUIN_NO_ENV_RELAUNCH'] = '1'
    if args.update:
        update_checkout(source)
        # Restart the updated bootstrap too, so manager pins and setup changes apply.
        if windows:
            command = ['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                       '-File', str(source / 'install/install.ps1'), '-Prefix', str(prefix),
                       '-Source', str(source)]
            if args.no_launch:
                command.append('-NoLaunch')
        else:
            command = ['bash', str(source / 'Install-NeuroPyGuiN.sh'), '--prefix', str(prefix),
                       '--source', str(source)]
            if args.no_launch:
                command.append('--no-launch')
        run(command)
        return

    print(f'Installing NeuroPyGuiN in {source}\nRuntime: {runtime}', flush=True)
    print('The scientific and GPU packages are large. The first download can take a while.', flush=True)
    operation = 'update' if (runtime / 'conda-meta/history').exists() else 'create'
    run([str(mamba), 'env', operation, '-y', '--no-rc', '-p', str(runtime), '-f', str(environment)])
    # Render with the newly installed Qt, not the bootstrap Python. This also
    # produces a multi-resolution Windows icon directly from the SVG master.
    run([str(mamba), 'run', '-p', str(runtime), 'python', str(source / 'install/prepare_icon.py'),
         str(source / 'neuropyguin/assets/neuropyguin-icon.svg'), str(prefix / 'neuropyguin.ico')])
    state_path = prefix / 'installation.json'
    state_path.write_text(json.dumps({'source': str(source), 'prefix': str(prefix),
                                     'runtime': str(runtime), 'mamba': str(mamba)}, indent=2), encoding='utf-8')
    if windows:
        windows_shortcuts(prefix, source, state_path)
    else:
        linux_shortcuts(prefix, source, state_path)
    print('\nReady. Open NeuroPyGuiN from your desktop or applications menu.\n'
          'Close the app before using the Update NeuroPyGuiN shortcut.\n'
          f'Installation details: {state_path}', flush=True)
    if not args.no_launch:
        command = [sys.executable, str(source / 'install/launch.py'), '--state', str(state_path)]
        if windows:
            command[0] = str(prefix / 'bootstrap/pythonw.exe')
        subprocess.Popen(command, start_new_session=not windows)


def main() -> int:
    """Parse shared installer arguments and keep failure messages actionable."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--update', action='store_true')
    parser.add_argument('--no-launch', action='store_true')
    args = parser.parse_args()
    try:
        configure(args)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f'Installation stopped: {error}', file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            print(error.stderr, file=sys.stderr)
        print('Fix the reported error and re-run the installer. Existing recordings and settings are retained.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

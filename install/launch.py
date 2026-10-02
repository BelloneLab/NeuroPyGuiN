"""Launch from a shortcut with the managed environment and persistent error logs."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime


def report_error(message: str) -> None:
    """Surface startup failures even when launched without a console window."""
    if sys.platform == 'win32':
        ctypes.windll.user32.MessageBoxW(0, message, 'NeuroPyGuiN could not start', 0x10)
    elif shutil.which('zenity'):
        subprocess.run(['zenity', '--error', '--title=NeuroPyGuiN could not start', '--text=' + message], check=False)
    elif shutil.which('xmessage'):
        subprocess.run(['xmessage', '-center', message], check=False)
    if sys.stderr:
        print(message, file=sys.stderr)


def main() -> int:
    """Run with activated DLL/library paths, independently of the user's shell."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    args = parser.parse_args()
    log_path = args.state.parent / 'logs/launcher.log'
    try:
        state = json.loads(args.state.read_text(encoding='utf-8'))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        for key in ('PYTHONHOME', 'PYTHONPATH', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH'):
            env.pop(key, None)
        env.update(PYTHONNOUSERSITE='1', NEUROPYGUIN_NO_ENV_RELAUNCH='1',
                   MAMBA_ROOT_PREFIX=str(Path(state['prefix']) / 'mamba'))
        command = [state['mamba'], 'run', '-p', state['runtime'], 'python',
                   str(Path(state['source']) / 'main.py')]
        with log_path.open('a', encoding='utf-8') as log:
            log.write(f'\nStartup: {datetime.now().isoformat()}\n')
            log.flush()
            result = subprocess.run(command, cwd=state['source'], env=env, stdout=log,
                                    stderr=subprocess.STDOUT, check=False,
                                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
        if result.returncode:
            raise RuntimeError(f'The application exited with code {result.returncode}.')
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        report_error(f'{error}\n\nDetails: {log_path}\nRe-run the installer to repair dependencies. '
                     'On Linux, also check the Qt system libraries listed in the installation guide.')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

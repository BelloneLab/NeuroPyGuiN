from __future__ import annotations

import subprocess
import sys

from neuropyguin.processes import terminate_child_processes, tracked_popen, tracked_run, unregister_process


def test_terminate_child_processes_stops_tracked_popen() -> None:
    proc = tracked_popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )
    try:
        assert proc.poll() is None
        result = terminate_child_processes(timeout=0.5, kill_timeout=0.5)
        proc.wait(timeout=5)
        touched = set(result["terminated"]) | set(result["killed"])
        if touched:
            assert proc.pid in touched
        assert proc.poll() is not None
    finally:
        unregister_process(proc)
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_tracked_run_returns_completed_process() -> None:
    proc = tracked_run(
        [sys.executable, "-c", "print('ok')"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == "ok"

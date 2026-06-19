from __future__ import annotations

import os
import subprocess
import threading
from typing import Dict, Iterable, List, Tuple


_TRACKED_LOCK = threading.RLock()
_TRACKED_PROCESSES: Dict[int, Tuple[subprocess.Popen, float | None]] = {}


def register_process(proc: subprocess.Popen) -> subprocess.Popen:
    """Track a child process so app shutdown can terminate it explicitly."""
    pid = getattr(proc, "pid", None)
    if pid is None:
        return proc
    create_time: float | None = None
    try:
        import psutil

        create_time = float(psutil.Process(int(pid)).create_time())
    except Exception:
        create_time = None
    with _TRACKED_LOCK:
        _TRACKED_PROCESSES[int(pid)] = (proc, create_time)
    return proc


def tracked_popen(*args, **kwargs) -> subprocess.Popen:
    """subprocess.Popen wrapper that registers the process for shutdown."""
    return register_process(subprocess.Popen(*args, **kwargs))


def tracked_run(*popenargs, input=None, capture_output: bool = False, timeout=None, check: bool = False, **kwargs):
    """subprocess.run variant that keeps the child registered while it runs."""
    if input is not None:
        if kwargs.get("stdin") is not None:
            raise ValueError("stdin and input arguments may not both be used")
        kwargs["stdin"] = subprocess.PIPE
    if capture_output:
        if kwargs.get("stdout") is not None or kwargs.get("stderr") is not None:
            raise ValueError("stdout and stderr arguments may not be used with capture_output")
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE

    proc = tracked_popen(*popenargs, **kwargs)
    try:
        try:
            stdout, stderr = proc.communicate(input, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            stdout, stderr = proc.communicate()
            exc.stdout = stdout
            exc.stderr = stderr
            raise
        args = popenargs[0] if popenargs else kwargs.get("args")
        completed = subprocess.CompletedProcess(args, proc.returncode, stdout, stderr)
        if check and completed.returncode:
            raise subprocess.CalledProcessError(
                completed.returncode,
                completed.args,
                output=completed.stdout,
                stderr=completed.stderr,
            )
        return completed
    finally:
        unregister_process(proc)


def unregister_process(proc_or_pid: subprocess.Popen | int | None) -> None:
    if proc_or_pid is None:
        return
    try:
        pid = int(proc_or_pid if isinstance(proc_or_pid, int) else proc_or_pid.pid)
    except Exception:
        return
    with _TRACKED_LOCK:
        _TRACKED_PROCESSES.pop(pid, None)


def _matching_psutil_process(pid: int, create_time: float | None):
    import psutil

    try:
        proc = psutil.Process(pid)
        if create_time is not None and abs(float(proc.create_time()) - create_time) > 1.0:
            return None
        return proc
    except psutil.Error:
        return None


def _unique_processes(processes: Iterable) -> List:
    seen: set[int] = set()
    out = []
    current_pid = os.getpid()
    for proc in processes:
        try:
            pid = int(proc.pid)
        except Exception:
            continue
        if pid == current_pid or pid in seen:
            continue
        seen.add(pid)
        out.append(proc)
    return out


def _tracked_psutil_processes() -> List:
    with _TRACKED_LOCK:
        tracked = list(_TRACKED_PROCESSES.items())
    out = []
    for pid, (popen_proc, create_time) in tracked:
        try:
            if popen_proc.poll() is not None:
                unregister_process(pid)
                continue
        except Exception:
            pass
        proc = _matching_psutil_process(pid, create_time)
        if proc is not None:
            out.append(proc)
    return out


def terminate_child_processes(timeout: float = 1.5, kill_timeout: float = 0.75) -> dict:
    """Terminate tracked subprocesses and every live child of this Python process."""
    try:
        import psutil
    except Exception:
        with _TRACKED_LOCK:
            tracked = list(_TRACKED_PROCESSES.values())
        popens = [popen_proc for popen_proc, _create_time in tracked]
        terminated: List[int] = []
        for popen_proc, _create_time in tracked:
            try:
                if popen_proc.poll() is None:
                    popen_proc.terminate()
                    terminated.append(int(popen_proc.pid))
            except Exception:
                pass
        killed: List[int] = []
        alive: List[int] = []
        for popen_proc in popens:
            try:
                if popen_proc.poll() is None:
                    popen_proc.wait(timeout=max(0.0, float(timeout)))
            except subprocess.TimeoutExpired:
                try:
                    popen_proc.kill()
                    killed.append(int(popen_proc.pid))
                    popen_proc.wait(timeout=max(0.0, float(kill_timeout)))
                except Exception:
                    pass
            except Exception:
                pass
            try:
                if popen_proc.poll() is None:
                    alive.append(int(popen_proc.pid))
                else:
                    unregister_process(popen_proc)
            except Exception:
                pass
        return {"terminated": terminated, "killed": killed, "alive": alive}

    candidates = []
    candidates.extend(_tracked_psutil_processes())
    try:
        candidates.extend(psutil.Process(os.getpid()).children(recursive=True))
    except psutil.Error:
        pass

    processes = _unique_processes(candidates)
    terminated: List[int] = []
    for proc in processes:
        try:
            proc.terminate()
            terminated.append(int(proc.pid))
        except psutil.NoSuchProcess:
            pass
        except psutil.Error:
            pass

    gone, alive = psutil.wait_procs(processes, timeout=max(0.0, float(timeout)))
    killed: List[int] = []
    for proc in alive:
        try:
            proc.kill()
            killed.append(int(proc.pid))
        except psutil.NoSuchProcess:
            pass
        except psutil.Error:
            pass

    if alive:
        _gone_after_kill, alive = psutil.wait_procs(alive, timeout=max(0.0, float(kill_timeout)))

    for proc in list(gone) + [p for p in processes if not p.is_running()]:
        try:
            unregister_process(int(proc.pid))
        except Exception:
            pass

    still_alive: List[int] = []
    for proc in alive:
        try:
            if proc.is_running():
                still_alive.append(int(proc.pid))
        except psutil.Error:
            pass
    return {"terminated": terminated, "killed": killed, "alive": still_alive}

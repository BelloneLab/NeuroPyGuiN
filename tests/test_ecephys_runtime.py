from __future__ import annotations

import os

from neuropyguin.ecephys_runtime import ecephys_subprocess_env, resolve_ecephys_repo


def test_ecephys_subprocess_env_forces_noninteractive_matplotlib(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "QtAgg")

    env = ecephys_subprocess_env()

    assert env["MPLBACKEND"] == "Agg"


def test_ecephys_subprocess_env_includes_bundled_repo(monkeypatch):
    inherited = os.pathsep.join(("/existing/one", "/existing/two"))
    monkeypatch.setenv("PYTHONPATH", inherited)

    env = ecephys_subprocess_env()

    entries = env["PYTHONPATH"].split(os.pathsep)
    assert entries[0] == str(resolve_ecephys_repo())
    assert entries[1:] == inherited.split(os.pathsep)

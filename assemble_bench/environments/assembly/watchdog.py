"""Fail-closed watchdog for the sim process.

A hung PhysX or RTX call holds the GIL, so an in-process timer never fires. A side
process owns the deadline instead and SIGKILLs the sim when a guarded section overruns;
the env server then reports the dead sim rather than waiting out the run timeout.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import gymnasium as gym

# Reads "<deadline_unix_s> <label>" from the deadline file; 0 means nothing is guarded.
_KILLER = """
import os, signal, sys, time
pid, path = int(sys.argv[1]), sys.argv[2]
while True:
    time.sleep(2)
    try:
        os.kill(pid, 0)
    except OSError:
        sys.exit(0)
    deadline, _, label = open(path).read().partition(" ")
    if float(deadline) and time.time() > float(deadline):
        print(f"[watchdog] {label} overran; killing {pid}", flush=True)
        os.kill(pid, signal.SIGKILL)
        sys.exit(0)
"""


class Watchdog:
    """Kills this process when a ``guard`` section outlives its budget."""

    def __init__(self) -> None:
        self._path = Path(f"/tmp/assemble-bench-deadline-{os.getpid()}")
        self._path.write_text("0")
        subprocess.Popen(
            [sys.executable, "-c", _KILLER, str(os.getpid()), str(self._path)],
            start_new_session=True,
        )

    @contextmanager
    def guard(self, label: str, budget_s: float):
        self._set(f"{time.time() + budget_s} {label}")
        try:
            yield
        finally:
            self._set("0")

    def _set(self, text: str) -> None:
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(text)
        tmp.replace(self._path)


class GuardedEnv(gym.Wrapper):
    """Runs every reset and step under the watchdog."""

    def __init__(self, env: gym.Env, watchdog: Watchdog, reset_s: float, step_s: float) -> None:
        super().__init__(env)
        self._watchdog = watchdog
        self._reset_s = reset_s
        self._step_s = step_s

    def reset(self, **kwargs):
        with self._watchdog.guard("reset", self._reset_s):
            return super().reset(**kwargs)

    def step(self, action):
        with self._watchdog.guard("step", self._step_s):
            return super().step(action)

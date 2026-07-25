"""Unit tests for tilt-fail gate (no Isaac)."""

from __future__ import annotations

import math

import torch

from assembly_bench.environments.assembly.tilt_fail import (
    STALL_STEPS,
    TILT_COS_INSERT,
    TILT_DEG_INSERT,
    nearly_seated,
    peg_tilt_deg,
    peg_upright_cos,
    tilt_fail_fire,
    update_tilt_stall,
)


def _xyzw_tilt_about_x(deg: float) -> torch.Tensor:
    """IsaacLab xyzw quat: rotate ``deg`` about world x."""
    a = math.radians(deg) * 0.5
    return torch.tensor([[math.sin(a), 0.0, 0.0, math.cos(a)]])


def test_insert_tilt_threshold():
    assert float(peg_upright_cos(_xyzw_tilt_about_x(TILT_DEG_INSERT))[0]) <= TILT_COS_INSERT + 1e-3
    assert float(peg_tilt_deg(_xyzw_tilt_about_x(TILT_DEG_INSERT))[0]) >= TILT_DEG_INSERT - 0.5


def test_stall_fires():
    stall = torch.zeros(2, dtype=torch.long)
    for _ in range(STALL_STEPS):
        stall = update_tilt_stall(
            stall,
            tipped=torch.tensor([True, False]),
            seated=torch.tensor([False, False]),
        )
    fire = tilt_fail_fire(stall)
    assert bool(fire[0]) and not bool(fire[1])


def test_seated_clears():
    stall = torch.full((1,), STALL_STEPS - 1, dtype=torch.long)
    stall = update_tilt_stall(
        stall,
        tipped=torch.tensor([True]),
        seated=torch.tensor([True]),
    )
    assert int(stall[0]) == 0


def test_nearly_seated():
    peg = torch.tensor([[0.005, 0.0, 0.01], [0.05, 0.0, 0.05]])
    hole = torch.zeros(2, 3)
    s = nearly_seated(peg, hole)
    assert bool(s[0]) and not bool(s[1])

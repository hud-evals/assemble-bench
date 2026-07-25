"""Unit tests for grasp-fail gate (no Isaac)."""

from __future__ import annotations

import torch

from assembly_bench.environments.assembly.grasp_fail import (
    STALL_STEPS,
    ee_peg_xy_dist,
    grasp_fail_fire,
    grip_closed,
    peg_on_stand,
    update_stall,
)


def test_retreat_and_fire():
    n = 2
    stall = torch.zeros(n, dtype=torch.long)
    # env0: closed + retreating + on stand → accumulates; env1: open → stays 0
    for _ in range(STALL_STEPS):
        closed = torch.tensor([True, False])
        retreating = torch.tensor([True, True])
        on_stand = torch.tensor([True, True])
        stall = update_stall(stall, closed=closed, retreating=retreating, on_stand=on_stand)
    fire = grasp_fail_fire(stall)
    assert bool(fire[0]) and not bool(fire[1])


def test_breaks_on_lift():
    stall = torch.full((1,), STALL_STEPS - 1, dtype=torch.long)
    stall = update_stall(
        stall,
        closed=torch.tensor([True]),
        retreating=torch.tensor([True]),
        on_stand=torch.tensor([False]),  # peg lifted
    )
    assert int(stall[0]) == 0
    assert not bool(grasp_fail_fire(stall)[0])


def test_grip_and_dist_helpers():
    action = torch.tensor([[0.0] * 7 + [1.0], [0.0] * 7 + [0.0]])
    assert bool(grip_closed(action)[0]) and not bool(grip_closed(action)[1])
    ee = torch.tensor([[0.0, 0.0, 0.1], [0.1, 0.0, 0.1]])
    peg = torch.zeros(2, 3)
    d = ee_peg_xy_dist(ee, peg)
    assert abs(float(d[0]) - 0.0) < 1e-6
    assert abs(float(d[1]) - 0.1) < 1e-6
    assert bool(peg_on_stand(torch.tensor([0.01]), 0.0, 0.03)[0])
    assert not bool(peg_on_stand(torch.tensor([0.05]), 0.0, 0.03)[0])

"""Unit tests for insert-fail gate (no Isaac)."""

from __future__ import annotations

import torch

from assembly_bench.environments.assembly.insert_fail import (
    ALIGN_HANDOFF_XY,
    STALL_STEPS,
    at_seat,
    half_radius_m,
    insert_fail_fire,
    peg_diameter_mm,
    peg_hole_xy,
    seat_cost,
    update_insert_stall,
)


def test_off_center_at_seat_fires():
    """Ram beside the bore (xy > half-radius, at seat) → latch."""
    stall = torch.zeros(1, dtype=torch.long)
    for _ in range(STALL_STEPS):
        stall = update_insert_stall(
            stall,
            closed=torch.tensor([True]),
            at_seat_now=torch.tensor([True]),
            off_center=torch.tensor([True]),
            improving=torch.tensor([True]),  # progress must not excuse off-center
        )
    assert bool(insert_fail_fire(stall)[0])


def test_on_center_progress_clears():
    stall = torch.full((1,), STALL_STEPS - 1, dtype=torch.long)
    stall = update_insert_stall(
        stall,
        closed=torch.tensor([True]),
        at_seat_now=torch.tensor([True]),
        off_center=torch.tensor([False]),
        improving=torch.tensor([True]),
    )
    assert int(stall[0]) == 0


def test_approach_does_not_count():
    """Still descending (above seat band) → stall resets even if off-center."""
    stall = torch.full((1,), STALL_STEPS - 1, dtype=torch.long)
    stall = update_insert_stall(
        stall,
        closed=torch.tensor([True]),
        at_seat_now=torch.tensor([False]),
        off_center=torch.tensor([True]),
        improving=torch.tensor([False]),
    )
    assert int(stall[0]) == 0


def test_at_seat_and_half_radius():
    hole = torch.zeros(1, 3)
    xy_ok = half_radius_m(peg_diameter_mm("peg_round_8mm"))
    assert abs(xy_ok - 0.002) < 1e-9  # half of 4 mm radius
    # Seat height, 6 mm off → at seat + clearly off center.
    peg = torch.tensor([[0.006, 0.0, 0.026]])
    assert bool(at_seat(peg, hole)[0])
    assert float(peg_hole_xy(peg, hole)[0]) > xy_ok
    # High carry over hole → not at seat yet.
    high = torch.tensor([[0.002, 0.0, 0.10]])
    assert not bool(at_seat(high, hole)[0])


def test_seat_cost_and_handoff():
    peg = torch.tensor([[0.01, 0.0, 0.05], [0.05, 0.0, 0.05]])
    hole = torch.zeros(2, 3)
    xy = peg_hole_xy(peg, hole)
    assert float(xy[0]) < ALIGN_HANDOFF_XY
    assert float(xy[1]) > ALIGN_HANDOFF_XY
    assert float(seat_cost(peg, hole)[0]) < float(seat_cost(peg, hole)[1])

"""Insert-fail detector for CG-DAgger expert takeover (privileged poses).

Only judges the peg once it is **at the seat** (mouth / boss contact), not while
still descending. Latch when, with a closed grip:

  1. **off-center** — peg–hole xy > half the peg radius (clearly missing the bore), or
  2. **seat hit / jam** — at seat height and seating cost not improving (ramming).

Tilt is a separate gate. Do **not** use stand lift-clear (mouth height sits
below that threshold).
"""

from __future__ import annotations

import re

import torch

# ~0.67 s at 15 Hz once at the seat (not during approach).
STALL_STEPS = 10
# At-seat height band (peg base above hole root). Bore mouth ~25 mm.
AT_SEAT_GAP_LO = 0.002
AT_SEAT_GAP_HI = 0.030
# Must be near the hole laterally before we call it an insert attempt.
NEAR_HOLE = 0.04  # 4 cm
# Cost drop that counts as progress (jam detection).
PROGRESS_EPS = 1e-3  # 1 mm
# Jump expert to align when already this close; else transport.
ALIGN_HANDOFF_XY = 0.02


def peg_diameter_mm(task: str) -> float:
    """Parse ``peg_round_8mm`` → 8.0; default 8 if missing."""
    m = re.search(r"_(\d+)mm", task)
    return float(m.group(1)) if m else 8.0


def half_radius_m(diameter_mm: float) -> float:
    """Half a peg radius (m): offset larger than this is clearly off-bore."""
    return (diameter_mm / 1000.0) / 4.0  # 0.5 * (d/2)


def peg_hole_xy(peg_pos: torch.Tensor, hole_pos: torch.Tensor) -> torch.Tensor:
    """Horizontal peg → hole distance (m)."""
    return torch.norm(peg_pos[:, :2] - hole_pos[:, :2], dim=-1)


def seat_cost(peg_pos: torch.Tensor, hole_pos: torch.Tensor) -> torch.Tensor:
    """Scalar seating cost: xy error + half the remaining positive height gap."""
    xy = peg_hole_xy(peg_pos, hole_pos)
    gap = peg_pos[:, 2] - hole_pos[:, 2]
    return xy + 0.5 * torch.clamp(gap, min=0.0)


def at_seat(
    peg_pos: torch.Tensor,
    hole_pos: torch.Tensor,
    *,
    gap_lo: float = AT_SEAT_GAP_LO,
    gap_hi: float = AT_SEAT_GAP_HI,
    near_xy: float = NEAR_HOLE,
) -> torch.Tensor:
    """Peg near the hole and down at mouth / seat-contact height (not approach)."""
    xy = peg_hole_xy(peg_pos, hole_pos)
    gap = peg_pos[:, 2] - hole_pos[:, 2]
    return (xy < near_xy) & (gap > gap_lo) & (gap < gap_hi)


# Back-compat alias used by tilt gate (at-seat = insert-relevant height).
in_mouth_band = at_seat


def update_insert_stall(
    stall: torch.Tensor,
    *,
    closed: torch.Tensor,
    at_seat_now: torch.Tensor,
    off_center: torch.Tensor,
    improving: torch.Tensor,
) -> torch.Tensor:
    """Increment only at seat: off-center miss or jammed press; else reset."""
    active = closed & at_seat_now & (off_center | ~improving)
    return torch.where(active, stall + 1, torch.zeros_like(stall))


def insert_fail_fire(stall: torch.Tensor, *, window: int = STALL_STEPS) -> torch.Tensor:
    return stall >= window

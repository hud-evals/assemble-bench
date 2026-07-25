"""Grasp-fail detector for CG-DAgger expert takeover (privileged poses).

Fires when the policy has closed, is retreating from the peg, and the peg is
still on the stand — i.e. a committed missed grasp, not a mid-approach close.
"""

from __future__ import annotations

import torch

# Sustained window (~0.8 s at 15 Hz) so a single noisy frame never latches.
STALL_STEPS = 12
# EE–peg distance must grow by this much vs the previous tick (retreat).
RETREAT_EPS = 5e-4  # 0.5 mm
# Gripper command (DROID: 0 open, 1 close) or finger joint (rad) closed thresh.
GRIP_CMD_CLOSED = 0.5
# Finger joint ~0 at open, ~0.785 (π/4) fully closed; mid is enough for "shut".
FINGER_CLOSED = 0.25


def ee_peg_xy_dist(ee_pos: torch.Tensor, peg_pos: torch.Tensor) -> torch.Tensor:
    """Horizontal distance EE → peg (m). Retreat is clearest in XY after a miss."""
    return torch.norm(ee_pos[:, :2] - peg_pos[:, :2], dim=-1)


def peg_on_stand(peg_z: torch.Tensor, stand_z: float, lift_clear: float) -> torch.Tensor:
    """True while the peg has not cleared the presentation stand."""
    return (peg_z - stand_z) < lift_clear


def grip_closed(action: torch.Tensor, finger: torch.Tensor | None = None) -> torch.Tensor:
    """Closed from commanded grip and/or measured finger joint."""
    closed = action[:, 7] >= GRIP_CMD_CLOSED
    if finger is not None:
        closed = closed | (finger >= FINGER_CLOSED)
    return closed


def update_stall(
    stall: torch.Tensor,
    *,
    closed: torch.Tensor,
    retreating: torch.Tensor,
    on_stand: torch.Tensor,
) -> torch.Tensor:
    """Increment consecutive-fail counter; reset to 0 when any predicate drops."""
    active = closed & retreating & on_stand
    return torch.where(active, stall + 1, torch.zeros_like(stall))


def grasp_fail_fire(stall: torch.Tensor, *, window: int = STALL_STEPS) -> torch.Tensor:
    """Latch candidate: stall counter has reached the sustained window."""
    return stall >= window

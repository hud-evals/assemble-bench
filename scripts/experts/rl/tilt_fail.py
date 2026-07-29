"""Tilt-fail detector for CG-DAgger expert takeover (privileged poses).

Two thresholds:

  - **Insert band** (closed, near mouth): latch at ``TILT_DEG_INSERT`` (~12°) —
    catches sideways jam / bending in the bore before the peg flips over.
  - **Elsewhere** off the table plane: latch at ``TILT_DEG`` (30°) — tip-pinch
    or dropped-on-side.

Never uses stand ``lift_clear`` (mouth height sits below that threshold).
Expert handoff regrasps from hover. Complements grasp/insert gates.

Upright geometry lives in ``tasks.peg_upright_cos`` (shared with success).
"""

from __future__ import annotations

import math

import torch

from assemble_bench.environments.assembly.tasks import peg_upright_cos

TILT_DEG = 30.0
TILT_DEG_INSERT = 12.0
TILT_COS = math.cos(math.radians(TILT_DEG))
TILT_COS_INSERT = math.cos(math.radians(TILT_DEG_INSERT))
# Short debounce (~0.2 s @ 15 Hz) — intervene before the jam hardens.
STALL_STEPS = 3
# Don't steal a nearly-seated upright peg.
SEATED_XY = 0.01
SEATED_GAP = 0.015


def peg_tilt_deg(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Peg lean from vertical in degrees."""
    c = peg_upright_cos(quat_xyzw).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(c))


def nearly_seated(
    peg_pos: torch.Tensor,
    hole_pos: torch.Tensor,
    *,
    xy_tol: float = SEATED_XY,
    gap_tol: float = SEATED_GAP,
) -> torch.Tensor:
    """True when the peg is already deep in the bore (skip tilt latch)."""
    xy = torch.norm(peg_pos[:, :2] - hole_pos[:, :2], dim=-1)
    gap = peg_pos[:, 2] - hole_pos[:, 2]
    return (xy < xy_tol) & (gap.abs() < gap_tol)


def update_tilt_stall(
    stall: torch.Tensor,
    *,
    tipped: torch.Tensor,
    seated: torch.Tensor,
) -> torch.Tensor:
    """Increment while tipped and not seated; else reset."""
    active = tipped & ~seated
    return torch.where(active, stall + 1, torch.zeros_like(stall))


def tilt_fail_fire(stall: torch.Tensor, *, window: int = STALL_STEPS) -> torch.Tensor:
    return stall >= window

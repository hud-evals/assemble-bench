"""Scripted privileged peg-insert expert (round pegs first).

Phase machine over true sim poses: hover above the peg, descend gated on xy
centering, close, lift gently, carry over the hole, free-space align to the
true bore center, then a straight-down chamfer-guided press. Targets follow
the two hard-won rules: after grasping, anchors are FIXED (captured at grasp)
or fixed-asset-relative servos `ee + (hole - peg)` -- never `peg + offset`.

Geometry (authored parts): pegs are 50 mm long, bores 25 mm deep with a ~1 mm
chamfer mouth; the presentation stand is a second bore, so the exposed shaft
is ~25 mm. Success = peg base at the hole root within 2.5 mm xy / 3 mm seat.
"""

import torch

import warp as wp
from isaaclab.utils.math import quat_apply, quat_error_magnitude

from .base import Machine, Phase, home_quat, pos_of

PEG_LEN = 0.050
BORE_H = 0.025
# Pad-TIP depth below the peg top at grasp: tips mid-exposed-shaft, matching
# the source benchmark's proven Robotiq recipe (pads 35 mm above the root of
# a 50 mm peg). All earlier depth "findings" were artifacts of aiming the
# crank frames instead of the real fingertip plane.
GRASP_BELOW_TOP = 0.015
SAFE_BASE_Z = 0.075         # peg base clears the 25 mm bosses with margin
ALIGN_MARGIN = 0.006        # peg base above the bore mouth while aligning
SEAT_OVERSHOOT = 0.002


# Binary gripper, the CANONICAL DROID/openpi/RoboLab convention (finger_joint
# 0 rad = open, pi/4 = closed): action 0 opens, 1 closes. The "inverted" probe
# reading was a misread of the parallelogram linkage (knuckle arms fan OUT as
# the pads close).
OPEN, CLOSE = 0.0, 1.0


def make_machine(base, servo, grasp_below_top=None, aim_off=None):
    """Build the peg phase machine for a live env (call after reset+settle).

    grasp_below_top: optional per-env (N,) grasp depths for sweeps.
    aim_off: optional per-env (N,3) offset added to the grasp aim point --
    used to calibrate the true pinch position against physics.
    """
    N, dev = base.num_envs, base.device
    # The DROID home already points the tool straight down, so we HOLD the home
    # orientation for the whole trajectory and grasp by pure translation.
    hold_quat = home_quat(servo)
    grasp_anchor = torch.zeros((N, 3), device=dev)
    lift_anchor = torch.zeros((N, 3), device=dev)
    grasp_depth = (grasp_below_top if grasp_below_top is not None
                   else torch.full((N,), GRASP_BELOW_TOP, device=dev))

    peg = lambda: pos_of(base, "held_part")
    hole = lambda: pos_of(base, "fixed_part")

    def peg_upright():
        """Cos of the peg's tilt from vertical (1 = perfectly upright)."""
        q = wp.to_torch(base.scene["held_part"].data.root_quat_w)
        z = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(N, 3)
        return quat_apply(q, z)[:, 2]

    def peg_speed():
        return torch.norm(wp.to_torch(base.scene["held_part"].data.root_lin_vel_w), dim=-1)

    # The approach tracks the peg's RESTING pose: updated whenever the peg is
    # at rest (so a bumped peg is re-approached at its new spot), frozen while
    # it moves (so a peg wedged in the gripper is never chased -- run 7 dragged
    # one 60 cm across the table).
    peg_rest = peg().clone()

    def peg_top_tcp():
        """Grasp aim point: on the resting peg axis, grasp_depth under the tip."""
        settled = peg_speed() < 0.01
        peg_rest[settled] = peg()[settled]
        p = peg_rest.clone()
        p[:, 2] += PEG_LEN - grasp_depth
        return p if aim_off is None else p + aim_off

    def toward(target, m):
        """EE target that drives the tool-axis aim point (open tool-frame
        midpoint) onto `target` 1:1."""
        return m.servo.ee() + (target - m.servo.tcp())

    def xy_err(a, b):
        return torch.norm(a[:, :2] - b[:, :2], dim=-1)

    # --- phase targets (full batch) ----------------------------------------
    def t_hover(m):
        t = peg_top_tcp()
        t[:, 2] += 0.05
        return toward(t, m), hold_quat, OPEN

    def t_descend(m):
        return toward(peg_top_tcp(), m), hold_quat, OPEN

    def enter_grasp(m, ids):
        grasp_anchor[ids] = m.servo.ee()[ids]    # freeze: peg-relative would drift

    def t_grasp(m):
        return grasp_anchor, hold_quat, CLOSE

    def t_microlift(m):
        # 12 mm straight up, very slowly: verifies the grasp with the only
        # honest signal (the peg rising) before committing to the transport,
        # and lets a marginal pinch settle instead of being yanked (the
        # 2F-85's asymmetric-contact torque is worst under acceleration).
        t = grasp_anchor.clone()
        t[:, 2] += 0.012
        return t, hold_quat, CLOSE

    def enter_lift(m, ids):
        # Fixed target: a diverging peg-relative climb was the failed-grasp
        # symptom (arm rode +7.5 cm/step to the ceiling and flipped).
        lift_anchor[ids] = grasp_anchor[ids]
        lift_anchor[ids, 2] += SAFE_BASE_Z - peg()[ids, 2]

    def t_lift(m):
        return lift_anchor, hold_quat, CLOSE

    def t_transport(m):
        # Stable 1:1 servo toward the static hole; keep the safe height.
        t = m.servo.ee() + (hole() - peg())
        t[:, 2] = m.servo.ee()[:, 2] + (SAFE_BASE_Z - peg()[:, 2])
        return t, hold_quat, CLOSE

    def t_align(m):
        t = m.servo.ee() + (hole() - peg())
        t[:, 2] = m.servo.ee()[:, 2] + (hole()[:, 2] + BORE_H + ALIGN_MARGIN - peg()[:, 2])
        return t, hold_quat, CLOSE

    def t_insert(m):
        t = m.servo.ee() + (hole() - peg())
        t[:, 2] = m.servo.ee()[:, 2] + (hole()[:, 2] - SEAT_OVERSHOOT - peg()[:, 2])
        return t, hold_quat, CLOSE

    def t_hold(m):
        return m.servo.ee(), hold_quat, CLOSE

    # --- phase completion (full batch bool) ---------------------------------
    # Via-point tolerances stay LOOSE (the IK+PD chain has a few-mm steady-state
    # error and via points don't need precision); only contact phases are tight.
    def upright(m):
        return quat_error_magnitude(m.servo.ee_quat(), hold_quat) < 0.15

    def near(target_fn, tol=0.012, need_upright=False, z_tol=None):
        """Position convergence; z_tol adds a TIGHT z check on top of a loose
        3D one (descend must actually reach grasp depth before closing)."""
        def done(m):
            t, _, _ = target_fn(m)
            ok = torch.norm(m.servo.ee() - t, dim=-1) < tol
            if z_tol is not None:
                ok &= (m.servo.ee()[:, 2] - t[:, 2]).abs() < z_tol
            return ok & upright(m) if need_upright else ok
        return done

    def grasp_ok(m):
        # Close, then wait for the CONTACT to settle (peg at rest again): the
        # underactuated close knocks the peg around before the pads seat, and
        # advancing early yanks a live contact. Finger angle is useless here
        # (reads ~fully closed even around an 8 mm peg).
        return (m.timer >= 12) & (peg_speed() < 0.005)

    def microlifted(m):
        # The peg rose off its stand base: the grasp is real.
        return peg()[:, 2] > peg_rest[:, 2] + 0.006

    def lifted(m):
        # Full-lift verification: still holding AND still near-vertical. A
        # tip-pinched peg pivots and dangles (run 24) -- it can never insert,
        # so fail here, not 300 steps later at the hole.
        return near(t_lift)(m) & (peg()[:, 2] > 0.03) & (peg_upright() > 0.95)

    def seated(m):
        gap = peg()[:, 2] - hole()[:, 2]
        return (xy_err(peg(), hole()) < 0.002) & (gap < 0.003)

    never = lambda m: torch.zeros(N, dtype=torch.bool, device=dev)

    return Machine(base, servo, [
        # Generous timeouts: the PD arm tracks the IK targets slower than the
        # rate limit; phases advance on CONVERGENCE, timeout means failure.
        Phase("hover", t_hover, near(t_hover, need_upright=True), timeout=150),
        # Descend: loose in xy (the close centers the peg) but TIGHT in z --
        # exiting 8 mm high left only the pad tips' edge on the peg (run 30).
        Phase("descend", t_descend, near(t_descend, 0.008, z_tol=0.002), timeout=90,
              gate=0.004, zcap=0.5),
        Phase("grasp", t_grasp, grasp_ok, timeout=45, fail_on_timeout=False,
              on_enter=enter_grasp),
        Phase("microlift", t_microlift, microlifted, timeout=45, zcap=0.12),
        Phase("lift", t_lift, lifted, timeout=90, zcap=0.25, on_enter=enter_lift),
        Phase("transport", t_transport, lambda m: xy_err(peg(), hole()) < 0.002, timeout=150),
        Phase("align", t_align, lambda m: xy_err(peg(), hole()) < 0.0008, timeout=90, zcap=0.4),
        Phase("insert", t_insert, seated, timeout=200, gate=0.002, zcap=0.15),
        Phase("hold", t_hold, never, timeout=10**9),
    ])

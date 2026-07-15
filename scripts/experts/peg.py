"""Scripted privileged peg-insert expert (round + rectangular "square" pegs).

Phase machine over true sim poses: hover above the peg, descend gated on xy
centering, close, lift gently, carry over the hole, free-space align to the
true bore center, then a straight-down chamfer-guided press. Targets follow
the two hard-won rules: after grasping, anchors are FIXED (captured at grasp)
or fixed-asset-relative servos `ee + (hole - peg)` -- never `peg + offset`.

Round pegs are rotationally symmetric (no clocking). Rectangular ("square")
pegs are NOT: the hole is yaw-jittered per episode, so `clock=True` adds a
CLOSED-LOOP yaw servo -- roll the wrist about world z during transport/align to
null the live `wrap(hole_yaw - peg_yaw, pi)` (2-fold), gated tight before the
press. An open-loop clock measured once at grasp is not enough (in-grip slip
leaves a few-degree residual that jams the ~0.1 mm rect clearance).

Geometry (authored parts): pegs are 50 mm long, bores 25 mm deep with a ~1 mm
chamfer mouth; the presentation stand is a second bore, so the exposed shaft
is ~25 mm. Success = peg base at the hole root within 2.5 mm xy / 3 mm seat.
"""

import torch

import warp as wp
from isaaclab.utils.math import quat_apply, quat_error_magnitude

from .base import Machine, Phase, home_quat, pos_of, quat_of, rolled

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

# Yaw clocking (rectangular pegs). CLOCK_RATE caps the per-step wrist roll so it
# never runs ahead of what the servo's rot_cap can actually track (no windup);
# reading the LIVE peg yaw each step then converges monotonically. YAW_TOL gates
# align->insert: a rect peg with ~0.1 mm clearance and ~4 mm half-width jams
# past ~1.5deg, so null to well under that before pressing.
CLOCK_RATE = 0.08
# Gate align->insert on yaw: a rect peg jams past ~1.5deg, and 0.02 rad (1.15deg)
# left no margin for in-grip drift during the press -> ~half the square inserts
# jammed. 0.01 rad (0.57deg) gives 2x headroom below the jam threshold.
YAW_TOL = 0.01


# Binary gripper, the CANONICAL DROID/openpi/RoboLab convention (finger_joint
# 0 rad = open, pi/4 = closed): action 0 opens, 1 closes. The "inverted" probe
# reading was a misread of the parallelogram linkage (knuckle arms fan OUT as
# the pads close).
OPEN, CLOSE = 0.0, 1.0


def make_machine(base, servo, grasp_below_top=None, aim_off=None, seed=None, clock=False):
    """Build the peg phase machine for a live env (call after reset+settle).

    grasp_below_top: optional per-env (N,) grasp depths for sweeps.
    aim_off: optional per-env (N,3) offset added to the grasp aim point --
    used to calibrate the true pinch position against physics.
    seed: if set (and no calibration aim_off given), draw per-env, seeded
    task-meaningful randomization -- wrist approach yaw, grasp depth, in-grip
    xy, hover/lift heights, grasp dwell -- so episodes differ in start state,
    wrist angle, and timing rather than being one template translated in xy.
    clock: rectangular pegs -- clock the grasped peg's yaw to the (jittered)
    hole yaw with a closed-loop wrist-roll servo during transport/align. Round
    pegs leave it False (rotationally symmetric); clock_yaw then stays 0 so the
    orientation is byte-identical to the round path.
    """
    N, dev = base.num_envs, base.device
    # The DROID home already points the tool straight down, so we HOLD the home
    # orientation for the whole trajectory and grasp by pure translation.
    hold_quat = home_quat(servo)
    grasp_anchor = torch.zeros((N, 3), device=dev)
    lift_anchor = torch.zeros((N, 3), device=dev)
    inhand = torch.zeros((N, 3), device=dev)   # peg base offset in EE frame, set at insert entry
    grasp_depth = (grasp_below_top if grasp_below_top is not None
                   else torch.full((N,), GRASP_BELOW_TOP, device=dev))

    # --- per-env, seeded trajectory diversity ------------------------------
    # Round peg is rotationally symmetric, so wrist yaw is pure approach-angle
    # diversity (no clocking needed); the rest vary start state and timing.
    hover_extra = torch.zeros(N, device=dev)
    safe_base_z = torch.full((N,), SAFE_BASE_Z, device=dev)
    grasp_dwell = torch.full((N,), 12, device=dev)
    if seed is not None and aim_off is None:
        g = torch.Generator(device=dev).manual_seed(seed)
        u = lambda lo, hi: lo + (hi - lo) * torch.rand(N, generator=g, device=dev)
        # Rect pegs must grasp near-square so the pads land on the flats (the
        # clocking servo sets the final yaw anyway); round pegs are symmetric so
        # a wide roll is free approach-angle diversity.
        roll_amp = 0.10 if clock else 0.35
        hold_quat = rolled(hold_quat, u(-roll_amp, roll_amp))
        # Grasp-affecting jitter stays SMALL to keep the grasp rock-solid (an
        # 8 mm peg tips/slips if pinched too high or too off-center).
        grasp_depth = grasp_depth + u(-0.0025, 0.0025)         # grasp depth
        aim_off = torch.zeros((N, 3), device=dev)
        aim_off[:, 0], aim_off[:, 1] = u(-0.001, 0.001), u(-0.001, 0.001)  # in-grip pose
        hover_extra = u(0.0, 0.03)                             # approach height / timing
        safe_base_z = safe_base_z + u(-0.01, 0.02)             # lift height / timing
        grasp_dwell = u(12.0, 28.0).long()                     # phase-duration jitter

    peg = lambda: pos_of(base, "held_part")
    hole = lambda: pos_of(base, "fixed_part")

    # --- yaw clocking (rect pegs) ------------------------------------------
    # clock_yaw is the wrist roll about world z we add on top of the grasp
    # orientation; 0 until transport, integrated closed-loop through align, then
    # frozen for the press. `name2idx` (filled just before return) lets the
    # transport/align targets integrate ONLY for their own envs.
    clock_yaw = torch.zeros(N, device=dev)
    name2idx = {}

    def part_yaw(name):
        """World yaw of a part's local +x axis (parts sit ~flat, so this is the
        clocking DOF)."""
        x = quat_apply(quat_of(base, name), torch.tensor([1.0, 0.0, 0.0], device=dev).expand(N, 3))
        return torch.atan2(x[:, 1], x[:, 0])

    def yaw_err():
        """hole - peg yaw, wrapped to (-pi/2, pi/2] (rect is 2-fold symmetric)."""
        e = part_yaw("fixed_part") - part_yaw("held_part")
        return e - torch.pi * torch.round(e / torch.pi)

    def clock_step(phase_name, m):
        """Integrate the wrist roll toward nulling the live yaw error, but only
        for envs currently in `phase_name`, and rate-limited so the wrist keeps
        up (no integrator windup)."""
        if not clock:
            return
        mask = m.phase == name2idx[phase_name]
        if bool(mask.any()):
            clock_yaw[mask] += torch.clamp(yaw_err()[mask], -CLOCK_RATE, CLOCK_RATE)

    qhold = (lambda: rolled(hold_quat, clock_yaw)) if clock else (lambda: hold_quat)

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
        t[:, 2] += 0.05 + hover_extra
        return toward(t, m), qhold(), OPEN

    def t_descend(m):
        return toward(peg_top_tcp(), m), qhold(), OPEN

    def enter_grasp(m, ids):
        grasp_anchor[ids] = m.servo.ee()[ids]    # freeze: peg-relative would drift

    def t_grasp(m):
        return grasp_anchor, qhold(), CLOSE

    def t_microlift(m):
        # 12 mm straight up, very slowly: verifies the grasp with the only
        # honest signal (the peg rising) before committing to the transport,
        # and lets a marginal pinch settle instead of being yanked (the
        # 2F-85's asymmetric-contact torque is worst under acceleration).
        t = grasp_anchor.clone()
        t[:, 2] += 0.012
        return t, qhold(), CLOSE

    def enter_lift(m, ids):
        # Fixed target: a diverging peg-relative climb was the failed-grasp
        # symptom (arm rode +7.5 cm/step to the ceiling and flipped).
        lift_anchor[ids] = grasp_anchor[ids]
        lift_anchor[ids, 2] += safe_base_z[ids] - peg()[ids, 2]

    def t_lift(m):
        return lift_anchor, qhold(), CLOSE

    def t_transport(m):
        # Free space: LIVE servo toward the static hole (self-corrects any
        # in-grip slip); keep the safe height. Start clocking the yaw here so it
        # has the whole carry to converge before the tight align gate.
        clock_step("transport", m)
        t = m.servo.ee() + (hole() - peg())
        t[:, 2] = m.servo.ee()[:, 2] + (safe_base_z - peg()[:, 2])
        return t, qhold(), CLOSE

    def t_align(m):
        clock_step("align", m)
        t = m.servo.ee() + (hole() - peg())
        t[:, 2] = m.servo.ee()[:, 2] + (hole()[:, 2] + BORE_H + ALIGN_MARGIN - peg()[:, 2])
        return t, qhold(), CLOSE

    def enter_insert(m, ids):
        # Capture the in-hand transform (peg base in the EE frame) right before
        # contact -- align has just put the live peg on the bore xy, so this is
        # accurate AND includes any in-grip tilt. Orientation is held constant,
        # so the world-axis offset stays valid through the press.
        inhand[ids] = peg()[ids] - m.servo.ee()[ids]

    def t_insert(m):
        # Keep clocking through the press: the rect peg slips in the grip once
        # the align-phase clocking stops, so yaw drifts back up (measured 0.5 ->
        # 5-8 deg during insert) and jams. Nulling the LIVE yaw error every step
        # holds it under the jam threshold all the way down. No-op for round.
        clock_step("insert", m)
        # Fixed anchor onto the bore: xy is a continuous CORRECTION (not a hard
        # gate) so a contact bump no longer stalls the descent, and -- unlike
        # the live `ee + (hole - peg)` servo -- the target never becomes
        # `ee + const` and marches the arm off once a jammed peg stops tracking.
        t = hole() - inhand
        # Clamp the press if the peg stops tracking the EE (jam/slip): hold z
        # instead of grinding the arm down into a wedged peg.
        slip = torch.norm(peg() - (m.servo.ee() + inhand), dim=-1) > 0.015
        dz = hole()[:, 2] - SEAT_OVERSHOOT - peg()[:, 2]
        t[:, 2] = m.servo.ee()[:, 2] + torch.where(slip, torch.zeros_like(dz), dz)
        return t, qhold(), CLOSE

    def t_release(m):
        # Press done: open the gripper in place so the peg drops/settles into the
        # bore under gravity. A peg pressed to partial depth but held by the pads
        # never satisfies the seat check (it hangs proud); releasing lets it seat.
        return m.servo.ee(), qhold(), OPEN

    def t_hold(m):
        # Stay clear and open while the freed peg settles and the success check
        # debounces -- never re-grip.
        return m.servo.ee(), qhold(), OPEN

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
        return (m.timer >= grasp_dwell) & (peg_speed() < 0.005)

    def microlifted(m):
        # The peg rose off its stand base: the grasp is real.
        return peg()[:, 2] > peg_rest[:, 2] + 0.006

    def lifted(m):
        # Full-lift verification: still holding AND still near-vertical. A
        # tip-pinched peg pivots and dangles (run 24) -- it can never insert,
        # so fail here, not 300 steps later at the hole.
        return near(t_lift)(m) & (peg()[:, 2] > 0.03) & (peg_upright() > 0.95)

    def aligned(m):
        # Tight xy AND (for rect pegs) tight yaw before the press -- inserting a
        # mis-clocked rect peg jams the corners.
        ok = xy_err(peg(), hole()) < 0.0008
        if clock:
            ok = ok & (yaw_err().abs() < YAW_TOL)
        return ok

    def seated(m):
        gap = peg()[:, 2] - hole()[:, 2]
        return (xy_err(peg(), hole()) < 0.002) & (gap < 0.003)

    def insert_done(m):
        # Advance to release once the peg is aligned and engaged in the bore --
        # either fully seated already, or the press has run long enough to have
        # bottomed out. A peg that never centered (xy off) keeps pressing and
        # times out (a real miss), so we don't release into thin air.
        engaged = (xy_err(peg(), hole()) < 0.002) & ((peg()[:, 2] - hole()[:, 2]) < 0.02)
        return engaged & (seated(m) | (m.timer > 80))

    def released(m):
        # The freed peg has settled (dropped into the bore and come to rest).
        return peg_speed() < 0.005

    never = lambda m: torch.zeros(N, dtype=torch.bool, device=dev)

    machine = Machine(base, servo, [
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
        Phase("align", t_align, aligned, timeout=120, zcap=0.4),
        # No xy gate: xy is corrected continuously by the fixed anchor while the
        # gentle zcap presses down, so contact can't stall the descent.
        Phase("insert", t_insert, insert_done, timeout=200, zcap=0.15, on_enter=enter_insert),
        # Release the peg and let it settle into the bore, then hold clear/open.
        # Timeout just advances to hold (a slow-settling peg isn't a failure).
        Phase("release", t_release, released, timeout=90, fail_on_timeout=False),
        Phase("hold", t_hold, never, timeout=10**9),
    ])
    name2idx.update({ph.name: i for i, ph in enumerate(machine.phases)})
    return machine

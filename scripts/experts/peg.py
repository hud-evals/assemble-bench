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

Near the hole: keep grip CLOSED, freeze wrist. XY is dual and height must NOT
override it: if the peg **base** is off the hole, servo base→hole; once the
base is on (or already in the bore), servo **tip→hole** (tip = base + 50 mm
along quat). A centered base with a leaning tip still jams — tip-led XY stands
the shaft up (and holds Z while tip is far). Never open early.

Post-release stuck: if open still leaves the peg proud on the seat, one
top-regrasp nudge toward the bore then re-press/release.
"""

import torch

import warp as wp
from isaaclab.utils.math import quat_apply, quat_error_magnitude, quat_mul

from .base import Machine, Phase, home_quat, pos_of, quat_of, rolled


def quat_align_vectors(v_from: torch.Tensor, v_to: torch.Tensor) -> torch.Tensor:
    """Batch xyzw quat rotating each ``v_from`` onto ``v_to`` (world frame)."""
    a = v_from / v_from.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    b = v_to / v_to.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    c = (a * b).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    axis = torch.linalg.cross(a, b, dim=-1)
    opp = (c.squeeze(-1) < -0.999)
    if bool(opp.any()):
        helper = torch.zeros_like(a)
        helper[:, 0] = 1.0
        alt = torch.linalg.cross(a, helper, dim=-1)
        thin = alt.norm(dim=-1) < 1e-6
        helper[thin] = a.new_tensor([0.0, 1.0, 0.0])
        alt = torch.linalg.cross(a, helper, dim=-1)
        alt = alt / alt.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        axis = torch.where(opp.unsqueeze(-1), alt, axis)
        c = torch.where(opp.unsqueeze(-1), c.new_full((1, 1), -1.0).expand_as(c), c)
    q = torch.cat([axis, 1.0 + c], dim=-1)
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)


PEG_LEN = 0.050
BORE_H = 0.025
# Pad-TIP depth below the peg top at grasp. Deeper pinch (was 15 mm mid-shaft)
# so demos teach a more stable grip. Exposed shaft ~25 mm → tips near stand mouth.
GRASP_BELOW_TOP = 0.025
SAFE_BASE_Z = 0.075         # peg base clears the 25 mm bosses with margin
ALIGN_MARGIN = 0.006        # peg base above the bore mouth while aligning
SEAT_OVERSHOOT = 0.004  # press a few mm past seat root so pads don't leave it proud

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
# Soft xy servo — high enough to finish centering under light contact, low
# enough to avoid the old shimmy/jitter. 0.35 asymptoted at ~1.5 mm (run fail).
SERVO_XY_GAIN = 0.55
# Tip recovery: base-first centering; tip fold only after base is on-hole.
RECOVER_UPRIGHT = 0.985  # cos(10°): enter correct when clearly tipped
PRESS_UPRIGHT = 0.978    # cos(12°): abort press only past this
BASE_READY_XY = 0.0015   # 1.5 mm — base on bore before tip/wrist fold
# Release only inside this xy — opening off-center dumps the peg (bad demo).
RELEASE_XY = 0.0012
INSERT_ON_BORE_XY = 0.0015  # base on bore; also tip-led threshold
STUCK_STEPS = 8
STUCK_EPS = 0.00025
XY_CLAMP = 0.006            # base-led live target band
TIP_CLAMP = 0.018           # tip-led needs room for ~13 mm tip lean
TIP_HARD_XY = 0.003         # tip this far → hard tip→hole every step
MOUTH_GAP = 0.012           # still on chamfer / not in bore
SERVO_TIP_GAIN = 0.85       # tip-led needs a firmer shove than base soft


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
    # Insert: base-led on mouth; tip-led once bottom is in / on-bore.
    best_gap = torch.full((N,), float("inf"), device=dev)
    stuck_ticks = torch.zeros(N, dtype=torch.long, device=dev)
    # The DROID home already points the tool straight down, so we HOLD the home
    # orientation for the whole trajectory and grasp by pure translation.
    hold_quat = home_quat(servo)
    grasp_anchor = torch.zeros((N, 3), device=dev)
    lift_anchor = torch.zeros((N, 3), device=dev)
    # One-shot post-release nudge (proud peg that didn't drop into the bore).
    nudge_used = torch.zeros(N, dtype=torch.bool, device=dev)
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

    # Snapshot tool-down grasp orientation (recovery must not freeze a tilted wrist).
    home_hold = hold_quat.clone()

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

    def peg_axis():
        """Unit shaft axis in world (held_part local +z, flipped up if inverted)."""
        q = wp.to_torch(base.scene["held_part"].data.root_quat_w)
        z = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(N, 3)
        axis = quat_apply(q, z)
        return torch.where(axis[:, 2:3] < 0, -axis, axis)

    def peg_tip():
        """World position of the peg tip (base + length along shaft)."""
        return peg() + PEG_LEN * peg_axis()

    def peg_needs_correct():
        return peg_upright() < RECOVER_UPRIGHT

    def peg_too_tipped_to_press():
        return peg_upright() < PRESS_UPRIGHT

    def peg_speed():
        return torch.norm(wp.to_torch(base.scene["held_part"].data.root_lin_vel_w), dim=-1)

    def _goto_straighten(m, mask):
        """Jump tipped align slots to translation-only tip→hole recenter."""
        if not bool(mask.any()):
            return
        ids = mask.nonzero(as_tuple=False).squeeze(-1)
        if ids.ndim == 0:
            ids = ids.unsqueeze(0)
        m.goto(ids, name2idx["straighten"])

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
        # Free space: soft LIVE xy toward the hole (self-corrects slip); keep
        # safe height. Start yaw clocking here so it converges before align.
        clock_step("transport", m)
        ee = m.servo.ee()
        err = hole() - peg()
        t = ee.clone()
        t[:, :2] = ee[:, :2] + SERVO_XY_GAIN * err[:, :2]
        t[:, 2] = ee[:, 2] + (safe_base_z - peg()[:, 2])
        return t, qhold(), CLOSE

    def enter_align(m, ids):
        # Freeze wrist as-is — never rotate near the hole.
        hold_quat[ids] = m.servo.ee_quat()[ids]

    def tip_led_mask(peg_p, hole_p, gap):
        # Tip owns XY once the base is on OR already below the mouth — never
        # gated on "at mouth" height (that was the bug: ignored 13 mm tip lean).
        base_ok = xy_err(peg_p, hole_p) < INSERT_ON_BORE_XY
        in_bore = gap < MOUTH_GAP
        return base_ok | in_bore

    def xy_dual_target(ee_xy, peg_p, hole_p, tip_p, tip_led, *, use_hard):
        """EE XY: base→hole or tip→hole; hard maps the active point onto hole."""
        base_err = hole_p[:, :2] - peg_p[:, :2]
        tip_err = hole_p[:, :2] - tip_p[:, :2]
        err = torch.where(tip_led.unsqueeze(-1), tip_err, base_err)
        gain = torch.where(tip_led, SERVO_TIP_GAIN, SERVO_XY_GAIN)
        soft = ee_xy + gain.unsqueeze(-1) * err
        hard_pt = torch.where(tip_led.unsqueeze(-1), tip_p[:, :2], peg_p[:, :2])
        hard = hole_p[:, :2] + (ee_xy - hard_pt)
        xy = torch.where(use_hard.unsqueeze(-1), hard, soft)
        clamp = torch.where(tip_led, TIP_CLAMP, XY_CLAMP)
        return (torch.max(torch.min(xy, hole_p[:, :2] + clamp.unsqueeze(-1)),
                          hole_p[:, :2] - clamp.unsqueeze(-1)),
                tip_err)

    def t_align(m):
        # Dual XY + ease to mouth. Wrist frozen.
        clock_step("align", m)
        ee, peg_p, hole_p = m.servo.ee(), peg(), hole()
        tip_p = peg_tip()
        gap = peg_p[:, 2] - hole_p[:, 2]
        tip_led = tip_led_mask(peg_p, hole_p, gap)
        tip_far = tip_led & (xy_err(tip_p, hole_p) > TIP_HARD_XY)
        xy, _ = xy_dual_target(
            ee[:, :2], peg_p, hole_p, tip_p, tip_led, use_hard=tip_far)
        t = ee.clone()
        t[:, :2] = xy
        # Hold Z while standing the tip up; else ease to mouth.
        ease_z = ee[:, 2] + (hole_p[:, 2] + BORE_H + ALIGN_MARGIN - peg_p[:, 2]).clamp(max=0.0)
        t[:, 2] = torch.where(tip_far, ee[:, 2], ease_z)
        return t, hold_quat, CLOSE

    def enter_insert(m, ids):
        hold_quat[ids] = m.servo.ee_quat()[ids]
        best_gap[ids] = float("inf")
        stuck_ticks[ids] = 0
        peg_p, hole_p, tip_p = peg()[ids], hole()[ids], peg_tip()[ids]
        xy = torch.norm(peg_p[:, :2] - hole_p[:, :2], dim=-1)
        tip_xy = torch.norm(tip_p[:, :2] - hole_p[:, :2], dim=-1)
        gap = peg_p[:, 2] - hole_p[:, 2]
        print(f"[insert] enter ids={ids.tolist()} base_xy_mm={(xy*1e3).tolist()} "
              f"tip_xy_mm={(tip_xy*1e3).tolist()} tip_dx_mm={((tip_p[:,0]-hole_p[:,0])*1e3).tolist()} "
              f"gap_mm={(gap*1e3).tolist()} tip_led={tip_led_mask(peg_p, hole_p, gap).tolist()} "
              f"up={peg_upright()[ids].tolist()}", flush=True)

    def t_insert(m):
        # Dual XY (tip once base on — height never overrides). Hold Z while tip
        # is far; else press. Stuck / tip-far → hard map of the active point.
        ee, peg_p, hole_p = m.servo.ee(), peg(), hole()
        tip_p = peg_tip()
        gap = peg_p[:, 2] - hole_p[:, 2]
        tip_led = tip_led_mask(peg_p, hole_p, gap)
        tip_xy = xy_err(tip_p, hole_p)
        tip_far = tip_led & (tip_xy > TIP_HARD_XY)

        improved = gap < (best_gap - STUCK_EPS)
        reject = gap > (best_gap + 0.001)
        best_gap[:] = torch.minimum(best_gap, gap)
        stuck_ticks[:] = torch.where(
            improved & ~tip_far, torch.zeros_like(stuck_ticks), stuck_ticks + 1)
        stuck = (stuck_ticks >= STUCK_STEPS) | reject
        stuck_ticks[:] = torch.where(stuck, torch.zeros_like(stuck_ticks), stuck_ticks)

        if int(m.timer.max().item()) % 60 == 0 and int(m.timer.max().item()) > 0:
            base_xy = xy_err(peg_p, hole_p)
            print(f"[insert] t={int(m.timer.max().item())} "
                  f"base_xy_mm={(base_xy*1e3).tolist()} "
                  f"tip_xy_mm={(tip_xy*1e3).tolist()} "
                  f"tip_dx_mm={((tip_p[:, 0] - hole_p[:, 0]) * 1e3).tolist()} "
                  f"gap_mm={(gap*1e3).tolist()} tip_led={tip_led.tolist()} "
                  f"up={peg_upright().tolist()}", flush=True)

        # Base still off → hard base when stuck; tip-led → hard tip whenever tip_far.
        use_hard = stuck | tip_far | (~tip_led & (xy_err(peg_p, hole_p) > 0.0008))
        xy, _ = xy_dual_target(
            ee[:, :2], peg_p, hole_p, tip_p, tip_led, use_hard=use_hard)
        t = ee.clone()
        t[:, :2] = xy
        press_z = ee[:, 2] + (hole_p[:, 2] - SEAT_OVERSHOOT - peg_p[:, 2]).clamp(max=0.0)
        # Don't ram the rim while the tip is still leaning.
        t[:, 2] = torch.where(tip_far, ee[:, 2], press_z)
        return t, hold_quat, CLOSE

    def enter_straighten(m, ids):
        hold_quat[ids] = m.servo.ee_quat()[ids]

    def t_straighten(m):
        # Same dual XY as align.
        ee, peg_p, hole_p = m.servo.ee(), peg(), hole()
        tip_p = peg_tip()
        gap = peg_p[:, 2] - hole_p[:, 2]
        tip_led = tip_led_mask(peg_p, hole_p, gap)
        tip_far = tip_led & (xy_err(tip_p, hole_p) > TIP_HARD_XY)
        xy, _ = xy_dual_target(
            ee[:, :2], peg_p, hole_p, tip_p, tip_led, use_hard=tip_far)
        t = ee.clone()
        t[:, :2] = xy
        ease_z = ee[:, 2] + (hole_p[:, 2] + BORE_H + ALIGN_MARGIN - peg_p[:, 2]).clamp(max=0.0)
        t[:, 2] = torch.where(tip_far, ee[:, 2], ease_z)
        return t, hold_quat, CLOSE

    def enter_straighten_done(m, ids):
        hold_quat[ids] = m.servo.ee_quat()[ids]
        m.goto(ids, name2idx["insert"])

    def t_release(m):
        # Peg already at/near seat under a closed grip — open in place so the
        # success check sees a freed, seated peg (not a mid-press drop).
        return m.servo.ee(), qhold(), OPEN

    def t_hold(m):
        # Stay clear and open while the freed peg settles and the success check
        # debounces -- never re-grip.
        return m.servo.ee(), qhold(), OPEN

    def t_fail_closed(m):
        # Insert/align bail sink: keep pinching. Opening off-bore dumps the peg
        # and poisons DAgger with a drop-beside-hole demo.
        return m.servo.ee(), qhold(), CLOSE

    def nudge_top_tcp():
        # Pinch near the tip of a proud-on-seat peg (live pose, not stand rest).
        p = peg().clone()
        p[:, 2] += PEG_LEN - 0.008
        return p

    def t_nudge_approach(m):
        t = nudge_top_tcp()
        t[:, 2] += 0.025
        return toward(t, m), qhold(), OPEN

    def t_nudge_descend(m):
        return toward(nudge_top_tcp(), m), qhold(), OPEN

    def enter_nudge_grasp(m, ids):
        grasp_anchor[ids] = m.servo.ee()[ids]

    def t_nudge_grasp(m):
        return grasp_anchor, qhold(), CLOSE

    def t_nudge_shift(m):
        # Free the jam a few mm, soft-servo XY onto the bore, then re-press.
        ee, peg_p, hole_p = m.servo.ee(), peg(), hole()
        t = ee.clone()
        t[:, :2] = ee[:, :2] + SERVO_XY_GAIN * (hole_p[:, :2] - peg_p[:, :2])
        t[:, 2] = ee[:, 2] + 0.008
        return t, qhold(), CLOSE

    def enter_nudge_to_insert(m, ids):
        m.goto(ids, name2idx["insert"])

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
        # Peg base roughly on the bore — then keep holding and press.
        ok = xy_err(peg(), hole()) < INSERT_ON_BORE_XY
        if clock:
            ok = ok & (yaw_err().abs() < YAW_TOL)
        return ok

    def seated(m):
        gap = peg()[:, 2] - hole()[:, 2]
        return (xy_err(peg(), hole()) < 0.002) & (gap < 0.003)

    def insert_done(m):
        # Keep grip closed until truly near the seat. Opening while still ~6 mm
        # proud looked "in the hole" but never finished the press.
        gap = peg()[:, 2] - hole()[:, 2]
        well = xy_err(peg(), hole()) < RELEASE_XY
        return well & (gap < 0.003) & (m.timer > 20)

    def released(m):
        # Brief open dwell so drop-in (or stuck-proud) can be observed.
        return m.timer >= 12

    def peg_stuck_proud():
        # Open left the peg near the seat but still proud — didn't fall in.
        gap = peg()[:, 2] - hole()[:, 2]
        xy = xy_err(peg(), hole())
        return (xy < 0.015) & (gap > 0.008) & (gap < 0.040) & (peg_speed() < 0.01)

    def straighten_ok(m):
        return (xy_err(peg(), hole()) < INSERT_ON_BORE_XY) & (m.timer >= 4)

    def nudge_grasped(m):
        return (m.timer >= 10) & (peg_speed() < 0.008)

    def nudge_shift_ok(m):
        return (xy_err(peg(), hole()) < 0.002) | (m.timer > 30)

    never = lambda m: torch.zeros(N, dtype=torch.bool, device=dev)

    def enter_to_hold(m, ids):
        # After release: if peg is still proud on the seat, one top-regrasp
        # nudge toward the bore then re-insert. Else park open. Skip straighten
        # (sequential neighbor) — never re-close on a freed seated peg.
        stuck = peg_stuck_proud()
        can = stuck & ~nudge_used
        to_nudge = ids[can[ids]]
        to_park = ids[~can[ids]]
        if to_nudge.numel():
            nudge_used[to_nudge] = True
            hold_quat[to_nudge] = home_hold[to_nudge]
            m.goto(to_nudge, name2idx["nudge_approach"])
        if to_park.numel():
            m.goto(to_park, name2idx["hold"])

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
        # Slow carry / align / press: lower caps + soft xy gain → smooth demos.
        Phase("transport", t_transport, lambda m: xy_err(peg(), hole()) < 0.002,
              timeout=220, pos_cap=0.006, zcap=0.15, rot_cap=0.06),
        # Align until on-bore — timeout handoff at ~2 mm pins the lip.
        Phase("align", t_align, aligned, timeout=10**9,
              pos_cap=0.005, zcap=0.4, rot_cap=0.05, on_enter=enter_align,
              fail_on_timeout=False),
        # Tip-led XY needs a bit more lateral rate to unwind ~13 mm tip lean.
        Phase("insert", t_insert, insert_done, timeout=450,
              pos_cap=0.008, zcap=1.0, rot_cap=0.04, on_enter=enter_insert,
              fail_on_timeout=False),
        Phase("release", t_release, released, timeout=20, fail_on_timeout=False),
        Phase("to_hold", t_hold, never, timeout=10**9, on_enter=enter_to_hold),
        Phase("straighten", t_straighten, straighten_ok, timeout=90,
              pos_cap=0.005, zcap=0.4, rot_cap=0.04, on_enter=enter_straighten,
              fail_on_timeout=False),
        Phase("straighten_done", t_straighten, never, timeout=10**9,
              on_enter=enter_straighten_done),
        # Post-release stuck: top-regrasp → XY nudge to bore → insert again.
        Phase("nudge_approach", t_nudge_approach,
              near(t_nudge_approach, 0.012, need_upright=True), timeout=60,
              pos_cap=0.005, zcap=0.3, rot_cap=0.06),
        Phase("nudge_descend", t_nudge_descend,
              near(t_nudge_descend, 0.008, z_tol=0.003), timeout=60,
              gate=0.005, zcap=0.4, pos_cap=0.004),
        Phase("nudge_grasp", t_nudge_grasp, nudge_grasped, timeout=30,
              fail_on_timeout=False, on_enter=enter_nudge_grasp),
        Phase("nudge_shift", t_nudge_shift, nudge_shift_ok, timeout=40,
              pos_cap=0.003, zcap=0.2, rot_cap=0.04, fail_on_timeout=False),
        Phase("nudge_to_insert", t_nudge_shift, never, timeout=10**9,
              on_enter=enter_nudge_to_insert),
        # Success park (open). Machine bail jumps to LAST — keep that closed.
        Phase("hold", t_hold, never, timeout=10**9),
        Phase("fail_closed", t_fail_closed, never, timeout=10**9),
    ])
    name2idx.update({ph.name: i for i, ph in enumerate(machine.phases)})
    return machine

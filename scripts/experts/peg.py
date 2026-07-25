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

Tip recovery (near the hole): keep the grip. A tilted peg's tip is NOT the
centering signal — always put the peg **base** on the hole first (tiny lift if
jammed on the lip), then fold the tip in (small wrist + tip nudge), then press.
No park swings or drop/regrasp for tip.
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
# Soft xy servo — high enough to finish centering under light contact, low
# enough to avoid the old shimmy/jitter. 0.35 asymptoted at ~1.5 mm (run fail).
SERVO_XY_GAIN = 0.55
# Tip recovery: base-first centering; tip fold only after base is on-hole.
RECOVER_UPRIGHT = 0.985  # cos(10°): enter correct when clearly tipped
PRESS_UPRIGHT = 0.978    # cos(12°): abort press only past this
BASE_READY_XY = 0.0015   # 1.5 mm — base on bore before tip/wrist fold
# Release only inside this xy — opening off-center dumps the peg (bad demo).
RELEASE_XY = 0.0012


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
        """Jump tipped align/insert slots to tip→hole nudge (keep grip)."""
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

    def t_align(m):
        # ONE continuous target: soft xy servo onto the bore at just-above-mouth
        # height. No engage/park mode switches — boundary flips read as jitter.
        # The press itself belongs to insert.
        _goto_straighten(m, peg_needs_correct() & (m.phase == name2idx.get("align", -1)))
        clock_step("align", m)
        ee = m.servo.ee()
        err = hole() - peg()
        t = ee.clone()
        t[:, :2] = ee[:, :2] + SERVO_XY_GAIN * err[:, :2]
        t[:, 2] = ee[:, 2] + (hole()[:, 2] + BORE_H + ALIGN_MARGIN - peg()[:, 2])
        return t, qhold(), CLOSE

    def enter_insert(m, ids):
        # Capture the in-hand transform (peg base in the EE frame) right before
        # contact -- align has just put the live peg on the bore xy, so this is
        # accurate AND includes any in-grip tilt. Orientation is held constant,
        # so the world-axis offset stays valid through the press.
        inhand[ids] = peg()[ids] - m.servo.ee()[ids]

    def t_insert(m):
        # Steady press: blend xy onto the bore anchor and drive z to the seat.
        # No probes. When still off-center, favor recentering over ramming the lip.
        _goto_straighten(
            m, peg_too_tipped_to_press() & (m.phase == name2idx.get("insert", -1)))
        clock_step("insert", m)
        ee = m.servo.ee()
        peg_p, hole_p = peg(), hole()
        t = hole_p - inhand
        t[:, :2] = ee[:, :2] + SERVO_XY_GAIN * (t[:, :2] - ee[:, :2])
        xy = xy_err(peg_p, hole_p)
        on_bore = xy < 0.002
        # Pause z only on a real off-bore slip, or while still clearly off-center
        # (pressing the lip teaches jam-then-drop, not insertion).
        slip = torch.norm(peg_p - (ee + inhand), dim=-1) > 0.015
        dz = hole_p[:, 2] - SEAT_OVERSHOOT - peg_p[:, 2]
        freeze_z = (slip & ~on_bore) | (xy > 0.0025)
        t[:, 2] = ee[:, 2] + torch.where(freeze_z, torch.zeros_like(dz), dz)
        return t, qhold(), CLOSE

    def enter_straighten(m, ids):
        # Correct from the grasp tool-down pose (never a previously snapped wrist).
        hold_quat[ids] = home_hold[ids]

    def t_straighten(m):
        # Tilt recovery, smooth and base-first: soft-servo the BASE onto the bore
        # at mouth height; once it's on, blend in the tip and stand the wrist up
        # (rate-limited, no-op when already upright). Insert does the press.
        ee, peg_p, hole_p, tip_p = m.servo.ee(), peg(), hole(), peg_tip()
        base_off = hole_p[:, :2] - peg_p[:, :2]
        tip_off = hole_p[:, :2] - tip_p[:, :2]
        base_ready = xy_err(peg_p, hole_p) < BASE_READY_XY
        xy_cmd = SERVO_XY_GAIN * torch.where(
            base_ready.unsqueeze(-1),
            0.5 * base_off + 0.5 * tip_off,
            base_off,
        )
        t = ee.clone()
        t[:, :2] = ee[:, :2] + xy_cmd
        t[:, 2] = ee[:, 2] + (hole_p[:, 2] + BORE_H + ALIGN_MARGIN - peg_p[:, 2])
        # Wrist upright once base is on the bore — folds the tip over the hole.
        q_tgt = qhold()
        if bool(base_ready.any()):
            world_z = peg_p.new_tensor([0.0, 0.0, 1.0]).expand(N, 3)
            q_up = quat_mul(quat_align_vectors(peg_axis(), world_z), m.servo.ee_quat())
            q_tgt = torch.where(base_ready.unsqueeze(-1), q_up, q_tgt)
        return t, q_tgt, CLOSE

    def enter_straighten_done(m, ids):
        # Freeze whatever upright wrist we reached, then press.
        hold_quat[ids] = m.servo.ee_quat()[ids]
        m.goto(ids, name2idx["insert"])

    def t_release(m):
        # Press done: open the gripper in place so the peg drops/settles into the
        # bore under gravity. A peg pressed to partial depth but held by the pads
        # never satisfies the seat check (it hangs proud); releasing lets it seat.
        return m.servo.ee(), qhold(), OPEN

    def t_hold(m):
        # Stay clear and open while the freed peg settles and the success check
        # debounces -- never re-grip.
        return m.servo.ee(), qhold(), OPEN

    def t_fail_closed(m):
        # Insert/align bail sink: keep pinching. Opening off-bore dumps the peg
        # and poisons DAgger with a drop-beside-hole demo.
        return m.servo.ee(), qhold(), CLOSE

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
        # Tight enough that insert presses into the bore, not the lip.
        ok = (xy_err(peg(), hole()) < RELEASE_XY) & (~peg_too_tipped_to_press())
        if clock:
            ok = ok & (yaw_err().abs() < YAW_TOL)
        return ok

    def seated(m):
        gap = peg()[:, 2] - hole()[:, 2]
        return (xy_err(peg(), hole()) < 0.002) & (gap < 0.003)

    def insert_done(m):
        # Release only when ON-BORE. Prefer a real seat; if pads hit the face
        # (~20 mm proud) after a short firm press, open so the peg drops in —
        # never idle-crawl for 90 ticks, never open while off-center.
        gap = peg()[:, 2] - hole()[:, 2]
        well = xy_err(peg(), hole()) < RELEASE_XY
        deep = gap < 0.010
        # Fingers often bottom on the socket before the peg seats; drop-in is OK
        # once centered and we've actually tried to press for a bit.
        lip_drop = (gap < 0.022) & (m.timer > 28)
        return well & (seated(m) | deep | lip_drop)

    def released(m):
        # Peg has dropped/settled into the bore (brief settle, don't idle).
        return (m.timer >= 6) | (peg_speed() < 0.005)

    def straighten_ok(m):
        # Base on bore → hand to insert promptly; don't wait out tip perfection.
        base_ok = xy_err(peg(), hole()) < BASE_READY_XY
        tip_ok = xy_err(peg_tip(), hole()) < 0.004
        good = base_ok & tip_ok & (m.timer >= 4)
        mild = base_ok & (~peg_too_tipped_to_press()) & (m.timer >= 12)
        return good | mild | (base_ok & (m.timer >= 30))

    never = lambda m: torch.zeros(N, dtype=torch.bool, device=dev)

    def enter_to_hold(m, ids):
        # Sequential advance from release must NOT fall into the straighten
        # branch below (it would re-close on the freed peg) — jump to hold.
        m.goto(ids, name2idx["hold"])

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
        # Align = smooth centering just above the mouth; timeout still advances
        # (insert refuses to press/release while off-center).
        Phase("align", t_align, aligned, timeout=70,
              pos_cap=0.004, zcap=0.3, rot_cap=0.05, fail_on_timeout=False),
        # Firm press once on-bore: ~1.6 mm/tick (zcap*pos_cap) — decisive, smooth.
        Phase("insert", t_insert, insert_done, timeout=120,
              pos_cap=0.004, zcap=0.4, rot_cap=0.04, on_enter=enter_insert),
        Phase("release", t_release, released, timeout=20, fail_on_timeout=False),
        # Park: reroute release's sequential advance past the recovery branch.
        Phase("to_hold", t_hold, never, timeout=10**9, on_enter=enter_to_hold),
        # Tip recovery (goto entry): base-first center → tip fold → insert.
        Phase("straighten", t_straighten, straighten_ok, timeout=60,
              pos_cap=0.004, zcap=0.15, rot_cap=0.04, on_enter=enter_straighten,
              fail_on_timeout=False),
        Phase("straighten_done", t_straighten, never, timeout=10**9,
              on_enter=enter_straighten_done),
        # Success park (open). Machine bail jumps to LAST — keep that closed.
        Phase("hold", t_hold, never, timeout=10**9),
        Phase("fail_closed", t_fail_closed, never, timeout=10**9),
    ])
    name2idx.update({ph.name: i for i, ph in enumerate(machine.phases)})
    return machine

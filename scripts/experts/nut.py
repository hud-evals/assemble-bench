"""Privileged scripted expert for the M8--M20 nut-thread tasks.

The expert grasps the nut flat from the board, places its bore over the bolt,
and follows the thread helix with coupled wrist yaw and axial motion.
The required 1.5+ turns exceed one comfortable Franka wrist sweep, so each
short turn is followed by a physical release, yaw reset, and regrasp while the
nut remains engaged on the bolt.
"""

from math import pi

import torch

import warp as wp
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_error_magnitude

from assembly_bench.environments.assembly.variants import NUTBOLT, NUT_BASE_TOP

from .base import Machine, Phase, home_quat, pos_of, quat_of, rolled


OPEN, CLOSE = 0.0, 1.0

# Factory asset body heights (m), from authored mesh bounds.
NUT_HEIGHT = {8: 0.0065, 12: 0.010, 16: 0.013, 20: 0.016}
# Fallback only; make_machine sets transport height above the bolt tip.
SAFE_BASE_Z = 0.070
# Hover just above the tip before engage. M20's large hex + tall tip perch
# nearer ~2.5 mm (analytical tip vs SDF contact); 4 mm fights the perch.
ALIGN_CLEARANCE = {8: 0.004, 12: 0.004, 16: 0.004, 20: 0.0025}
# Capture the thread mouth, not merely its neighbourhood. With an 0.8 mm gate,
# the first helix stroke was consumed approaching the tip and left no thread
# depth to support the nut during release.
ENGAGE_GAP = 0.00005
# Lead-search drives the nut FIRMLY down (past the chamfer, ~1 pitch) while
# sweeping the wrist, so it hunts the thread start and drops in instead of
# perching on the tip. A shallow press never reached the first full thread.
LEAD_DEPTH = 0.0015
LEAD_CAPTURE_DEPTH = 0.0001
# Pad face sits ~3–4 mm above the TCP plane. Sink TCP below the nut base so the
# pads bite the lower hex; scale with nut height.
REGRASP_SINK = {8: 0.0035, 12: 0.0035, 16: 0.004, 20: 0.004}
# Positive = aim TCP below nut base (board-relative pick).
GRASP_BOARD_PRELOAD = {8: 0.001, 12: 0.001, 16: 0.001, 20: 0.001}
# Require this much tip capture before opening for a ratchet (else the nut
# pops off the lead and the next close grabs air).
# One 120° stroke drops ~pitch/3; gate just below a partial flank catch.
RELEASE_MIN_DEPTH = {8: 0.0004, 12: 0.0008, 16: 0.0010, 20: 0.0012}

# One stroke is 120 degrees so the hex presents equivalent flats after each
# unwind/regrasp. (90° looks gentler but lands pads on corners.)
THREAD_CYCLES = 7
TURN_ANGLE = {8: 2.0 * pi / 3.0, 12: 2.0 * pi / 3.0, 16: 2.0 * pi / 3.0, 20: 2.0 * pi / 3.0}
TURN_SIGN = -1.0
TURN_RATE = 0.08


def make_machine(base, servo, *, size, aim_off=None, lead_phase_off=None, seed=None):
    """Build a pick, engage, and release-regrip threading phase machine."""
    if size not in NUT_HEIGHT:
        raise ValueError(f"unsupported nut size: M{size}")

    N, dev = base.num_envs, base.device
    dims = NUTBOLT[size]
    head_h, shank, pitch = dims["head_h"], dims["shank"], dims["pitch"]
    nut_h = NUT_HEIGHT[size]
    grasp_preload = GRASP_BOARD_PRELOAD[size]
    regrasp_sink = REGRASP_SINK[size]
    release_min_depth = RELEASE_MIN_DEPTH[size]
    turn_angle = TURN_ANGLE[size]
    align_clearance = ALIGN_CLEARANCE[size]
    # Small nuts need a firmer lead press to catch the first flank.
    lead_depth = LEAD_DEPTH if size >= 12 else LEAD_DEPTH + 0.0005

    hold_quat = home_quat(servo)
    grasp_anchor = torch.zeros((N, 3), device=dev)
    lift_anchor = torch.zeros((N, 3), device=dev)
    release_anchor = torch.zeros((N, 3), device=dev)
    release_quat = hold_quat.clone()
    inhand = torch.zeros((N, 3), device=dev)
    turn_base = torch.zeros((N, 3), device=dev)
    turn_yaw_start = torch.zeros(N, device=dev)
    turn_command = torch.zeros(N, device=dev)
    force_baseline = torch.zeros((N, 3), device=dev)
    force_delta = torch.zeros(N, device=dev)
    peak_force_delta = torch.zeros(N, device=dev)
    lead_direction = torch.ones(N, device=dev)
    thread_pending = torch.zeros(N, dtype=torch.bool, device=dev)

    calibrating = aim_off is not None
    aim_off = (
        torch.zeros((N, 3), device=dev)
        if not calibrating
        else aim_off.clone().to(device=dev)
    )
    hover_extra = torch.zeros(N, device=dev)
    # Clear the bolt tip in transport (M20 tip is ~74 mm; the old 70 mm floor
    # jammed the nut into the shank during the carry).
    tip_clear_z = NUT_BASE_TOP + head_h + shank + 0.012
    safe_base_z = torch.full((N,), max(SAFE_BASE_Z, tip_clear_z), device=dev)
    grasp_dwell = torch.full((N,), 18, dtype=torch.long, device=dev)
    release_dwell = torch.full((N,), 10, dtype=torch.long, device=dev)
    turn_rate = torch.full((N,), TURN_RATE, device=dev)

    # A supplied aim offset is a calibration sweep: keep it exact. Production
    # runs randomize the same small, task-meaningful axes as the other experts.
    if seed is not None and not calibrating:
        g = torch.Generator(device=dev).manual_seed(seed)
        u = lambda lo, hi: lo + (hi - lo) * torch.rand(N, generator=g, device=dev)
        # Keep the pads near hex flats, but vary the captured thread phase.
        # Small nuts: tighter randomization — large aim/roll misses the hex.
        roll = 0.06 if size <= 8 else 0.12
        xy_j = 0.00025 if size <= 8 else 0.0005
        hold_quat = rolled(hold_quat, u(-roll, roll))
        aim_off[:, 0], aim_off[:, 1] = u(-xy_j, xy_j), u(-xy_j, xy_j)
        aim_off[:, 2] = u(-0.0002, 0.0002) if size <= 8 else u(-0.0004, 0.0004)
        hover_extra = u(0.0, 0.010)
        # Only randomize upward so we never dip below the bolt tip again.
        safe_base_z += u(0.0, 0.012)
        grasp_dwell = u(18.0, 24.0).long()
        release_dwell = u(9.0, 12.0).long()
        if size <= 8:
            turn_rate = u(0.055, 0.075)
        else:
            turn_rate = u(0.070, 0.090)

    ratchet_quat = hold_quat.clone()
    # The generated loose M8 groove has only ~0.06 mm radial crest clearance.
    # Pause the turn whenever contact pushes farther off-axis, recenter, then
    # continue; advancing an open-loop wrist target guarantees a flank clash.
    # M20 needs a wider align/thread gate — tip contact leaves ~0.9 mm residual.
    thread_xy_tol = min(0.0012 if size >= 20 else 0.0005, 0.06 * size / 1000.0)
    engage_xy_tol = max(0.0005, 0.06 * size / 1000.0)
    engage_tol = max(0.0006, 0.075 * size / 1000.0)
    force_limit = 5.0
    nut = lambda: pos_of(base, "held_part")

    def bolt_tip():
        off = torch.tensor([0.0, 0.0, head_h + shank], device=dev).expand(N, 3)
        return pos_of(base, "fixed_part") + quat_apply(quat_of(base, "fixed_part"), off)

    def nut_base():
        off = torch.tensor([0.0, 0.0, head_h], device=dev).expand(N, 3)
        return nut() + quat_apply(quat_of(base, "held_part"), off)

    def nut_speed():
        return torch.norm(wp.to_torch(base.scene["held_part"].data.root_lin_vel_w), dim=-1)

    def nut_ang_speed():
        return torch.norm(wp.to_torch(base.scene["held_part"].data.root_ang_vel_w), dim=-1)

    def finger_pos():
        return wp.to_torch(base.scene["robot"].data.joint_pos)[:, 7]

    def wrist_force():
        wrench = wp.to_torch(
            base.scene["robot"].root_view.get_link_incoming_joint_force()
        )[:, servo.ee_idx]
        return wrench[:, :3]

    def nut_upright():
        z = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(N, 3)
        return quat_apply(quat_of(base, "held_part"), z)[:, 2]

    def nut_yaw():
        x = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(N, 3)
        axis = quat_apply(quat_of(base, "held_part"), x)
        return torch.atan2(axis[:, 1], axis[:, 0])

    def bolt_yaw():
        x = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(N, 3)
        axis = quat_apply(quat_of(base, "fixed_part"), x)
        return torch.atan2(axis[:, 1], axis[:, 0])

    def xy_err(a, b):
        return torch.norm(a[:, :2] - b[:, :2], dim=-1)

    def toward(target, m):
        return m.servo.ee() + (target - m.servo.tcp())

    # Track only a resting table nut. Freeze the aim once contact moves it.
    nut_rest = nut().clone()
    approaching = torch.ones(N, dtype=torch.bool, device=dev)

    def grasp_tcp():
        settled = approaching & (nut_speed() < 0.01)
        nut_rest[settled] = nut()[settled]
        p = nut_rest + aim_off
        # Negative preload (M4) aims into the hex so pads meet the thin body.
        p[:, 2] = nut_rest[:, 2] + head_h - grasp_preload + aim_off[:, 2]
        return p

    def t_hover(m):
        p = grasp_tcp()
        p[:, 2] += 0.025 + hover_extra
        return toward(p, m), hold_quat, OPEN

    def t_descend(m):
        return toward(grasp_tcp(), m), hold_quat, OPEN

    def enter_grasp(m, ids):
        # The thin M4 nut may stop the open fingertips slightly above the ideal
        # mid-plane. Close at the achieved pose instead of ramming the board.
        approaching[ids] = False
        nut_rest[ids] = nut()[ids]
        grasp_anchor[ids] = m.servo.ee()[ids]

    def t_grasp(m):
        return grasp_anchor, hold_quat, CLOSE

    def t_microlift(m):
        p = grasp_anchor.clone()
        p[:, 2] += 0.010
        return p, hold_quat, CLOSE

    def enter_lift(m, ids):
        lift_anchor[ids] = grasp_anchor[ids]
        lift_anchor[ids, 2] += safe_base_z[ids] - nut_base()[ids, 2]

    def t_lift(m):
        return lift_anchor, hold_quat, CLOSE

    def t_transport(m):
        p = m.servo.ee() + (bolt_tip() - nut_base())
        p[:, 2] = m.servo.ee()[:, 2] + (safe_base_z - nut_base()[:, 2])
        return p, hold_quat, CLOSE

    def t_align(m):
        # Center XY; for M20 keep a soft Z hold once we are near the tip so we
        # do not grind against the lead-in (that chatters and never settles).
        p = m.servo.ee() + (bolt_tip() - nut_base())
        desired_z = bolt_tip()[:, 2] + align_clearance
        if size >= 20:
            gap = nut_base()[:, 2] - bolt_tip()[:, 2]
            near_tip = (gap > 0.0005) & (gap < 0.006)
            # Hold current height when already perched; otherwise approach slowly.
            dz = torch.where(
                near_tip,
                torch.zeros_like(gap),
                torch.clamp(desired_z - nut_base()[:, 2], -0.002, 0.002),
            )
            p[:, 2] = m.servo.ee()[:, 2] + dz
        else:
            p[:, 2] = m.servo.ee()[:, 2] + (desired_z - nut_base()[:, 2])
        return p, hold_quat, CLOSE

    def enter_engage(m, ids):
        # Freeze the nut-base offset before contact. Live nut-relative targets
        # become an unreachable ee+constant target if the thread mouth catches.
        inhand[ids] = quat_apply_inverse(
            m.servo.ee_quat()[ids], nut_base()[ids] - m.servo.ee()[ids]
        )

    def t_engage(m):
        desired = bolt_tip().clone()
        desired[:, 2] += ENGAGE_GAP
        target = desired - quat_apply(hold_quat, inhand)
        actual_off = quat_apply(m.servo.ee_quat(), inhand)
        tracking = torch.norm(nut_base() - (m.servo.ee() + actual_off), dim=-1) < 0.006
        xy_step = torch.clamp(bolt_tip()[:, :2] - nut_base()[:, :2], -0.002, 0.002)
        target[:, :2] = m.servo.ee()[:, :2] + torch.where(
            tracking.unsqueeze(-1), xy_step, torch.zeros_like(xy_step)
        )
        dz = target[:, 2] - m.servo.ee()[:, 2]
        target[:, 2] = m.servo.ee()[:, 2] + torch.where(
            tracking, dz, torch.zeros_like(dz)
        )
        return target, hold_quat, CLOSE

    def enter_lead(m, ids):
        lead_direction[ids] = 1.0
        inhand[ids] = quat_apply_inverse(
            m.servo.ee_quat()[ids], nut_base()[ids] - m.servo.ee()[ids]
        )

    def t_lead(m):
        # Sweep most of a full turn within the safe wrist range. Reverse at
        # +/-2.4 rad and stop only when depth proves that the first flank caught.
        relative_yaw = nut_yaw() - bolt_yaw()
        relative_yaw -= 2.0 * torch.pi * torch.round(relative_yaw / (2.0 * torch.pi))
        lead_direction[relative_yaw > 2.4] = -1.0
        lead_direction[relative_yaw < -2.4] = 1.0
        step = 0.15 * lead_direction
        target_quat = rolled(m.servo.ee_quat(), step)

        desired = bolt_tip().clone()
        # Search the lead under a bounded downward preload. At the matching
        # angular phase the nut can drop onto the first flank; otherwise the
        # SDF contact holds it at the mouth.
        desired[:, 2] -= lead_depth
        target = desired - quat_apply(target_quat, inhand)
        actual_off = quat_apply(m.servo.ee_quat(), inhand)
        tracking = torch.norm(nut_base() - (m.servo.ee() + actual_off), dim=-1) < 0.006
        xy_step = torch.clamp(bolt_tip()[:, :2] - nut_base()[:, :2], -0.002, 0.002)
        target[:, :2] = m.servo.ee()[:, :2] + torch.where(
            tracking.unsqueeze(-1), xy_step, torch.zeros_like(xy_step)
        )
        dz = torch.clamp(target[:, 2] - m.servo.ee()[:, 2], -0.0005, 0.00015)
        target[:, 2] = m.servo.ee()[:, 2] + dz
        return target, target_quat, CLOSE

    def enter_thread(m, ids):
        # Machine enters a phase before the preceding phase's final action is
        # simulated. Defer the capture to the first thread target evaluation.
        thread_pending[ids] = True

    def turn_progress():
        delta = nut_yaw() - turn_yaw_start
        delta -= 2.0 * torch.pi * torch.round(delta / (2.0 * torch.pi))
        return torch.clamp(TURN_SIGN * delta, 0.0, turn_angle)

    def t_thread(m):
        ids = thread_pending.nonzero().squeeze(-1)
        if ids.numel():
            # Re-capture after every regrasp. Each stroke starts from the
            # achieved depth, so a blocked stroke cannot build a ram target.
            ratchet_quat[ids] = m.servo.ee_quat()[ids]
            turn_base[ids] = nut_base()[ids]
            turn_yaw_start[ids] = nut_yaw()[ids]
            turn_command[ids] = 0.0
            force_baseline[ids] = wrist_force()[ids]
            force_delta[ids] = 0.0
            peak_force_delta[ids] = 0.0
            inhand[ids] = quat_apply_inverse(
                m.servo.ee_quat()[ids], nut_base()[ids] - m.servo.ee()[ids]
            )
            thread_pending[ids] = False
        actual_off = quat_apply(m.servo.ee_quat(), inhand)
        # Descent/turn gate is the loose "still roughly on the bolt" tolerance,
        # not the tight success tolerance: the wrist-roll makes the gripped nut
        # orbit by up to ~1 mm, and gating the screw feed on the tight tolerance
        # let the descent fire almost never (nut spun without advancing).
        aligned = xy_err(nut_base(), bolt_tip()) < engage_xy_tol
        tracking = torch.norm(nut_base() - (m.servo.ee() + actual_off), dim=-1) < 0.003
        progress = turn_progress()
        expected_drop = pitch * progress / (2.0 * torch.pi)
        actual_drop = turn_base[:, 2] - nut_base()[:, 2]
        depth_error = expected_drop - actual_drop
        force_delta.copy_(torch.norm(wrist_force() - force_baseline, dim=-1))
        peak_force_delta.copy_(torch.maximum(peak_force_delta, force_delta))
        force_safe = force_delta < force_limit
        can_turn = aligned & tracking & (nut_upright() > 0.95)
        turn_command.add_(torch.where(can_turn, turn_rate, torch.zeros_like(turn_rate)))
        turn_command.clamp_(max=turn_angle)

        target_quat = rolled(ratchet_quat, TURN_SIGN * turn_command)
        desired = turn_base.clone()
        desired[:, :2] = bolt_tip()[:, :2]
        target = desired - quat_apply(target_quat, inhand)
        xy_step = torch.clamp(bolt_tip()[:, :2] - nut_base()[:, :2], -0.002, 0.002)
        target[:, :2] = m.servo.ee()[:, :2] + torch.where(
            tracking.unsqueeze(-1), xy_step, torch.zeros_like(xy_step)
        )

        # Couple Z to yaw as a real screw: drive the nut base to its ideal
        # helical depth (turn_base - pitch*yaw/2pi). A bounded step per frame
        # forces the flanks to mate as the wrist turns; high wrist force unloads
        # Z so a mis-phased press backs off instead of jamming.
        # Small nuts get a light extra press so one stroke captures a flank
        # before the ratchet opens.
        tip_gap = nut_base()[:, 2] - bolt_tip()[:, 2]
        # Until the nut is ~1.5 mm onto the tip, bias below the pure helix so
        # M8 does not free-spin on the chamfer. Prefer seating within the first
        # two strokes — later strokes often lose the thin hex.
        shallow = tip_gap > -0.0015
        extra = torch.where(
            shallow,
            torch.full_like(tip_gap, 0.0006 if size <= 8 else 0.0002),
            torch.zeros_like(tip_gap),
        )
        helix_z = turn_base[:, 2] - expected_drop - extra
        z_err = helix_z - nut_base()[:, 2]
        dz = torch.clamp(z_err, -0.0005, 0.0002)
        dz = torch.where(force_safe, dz, torch.full_like(dz, 0.0001))
        target[:, 2] = m.servo.ee()[:, 2] + torch.where(
            aligned & tracking, dz, torch.zeros_like(dz)
        )
        return target, target_quat, CLOSE

    def enter_release(m, ids):
        # Hold XY/Z — sinking while opening dragged small nuts back off the tip.
        release_anchor[ids] = m.servo.ee()[ids]
        release_quat[ids] = m.servo.ee_quat()[ids]

    def enter_lead_release(m, ids):
        # The caught lead sits near the positive wrist limit. Reset one
        # hex-symmetric sector while open, then tighten back through it.
        release_anchor[ids] = m.servo.ee()[ids]
        release_quat[ids] = m.servo.ee_quat()[ids]
        ratchet_quat[ids] = rolled(
            m.servo.ee_quat()[ids],
            torch.full_like(turn_yaw_start[ids], -turn_angle),
        )

    def t_release(m):
        # Opening laterally leaves the nut supported by the real threads.
        return release_anchor, release_quat, OPEN

    def t_reset(m):
        # Cage-track the nut while unwinding. A frozen anchor lets M8 tip off
        # the lead and leave the open jaws empty for regrasp.
        target = release_anchor.clone()
        target[:, :2] = m.servo.ee()[:, :2] + (
            nut_base()[:, :2] - m.servo.tcp()[:, :2]
        )
        return target, ratchet_quat, OPEN

    def t_realign(m):
        target_tcp = nut_base().clone()
        target_tcp[:, 2] -= regrasp_sink
        return toward(target_tcp, m), ratchet_quat, OPEN

    def enter_regrasp(m, ids):
        target_tcp = nut_base()[ids].clone()
        target_tcp[:, 2] -= regrasp_sink
        grasp_anchor[ids] = (
            m.servo.ee()[ids] + target_tcp - m.servo.tcp()[ids]
        )

    def t_regrasp(m):
        # Keep tracking the nut while closing — a frozen anchor misses when the
        # lead nut drifts during the open-jaw unwind.
        target_tcp = nut_base().clone()
        target_tcp[:, 2] -= regrasp_sink
        return toward(target_tcp, m), ratchet_quat, CLOSE

    def t_center(m):
        # Recenter XY after a ratchet; Z press is a separate phase so we do not
        # twist and press at once (that pops M8 out of the pads).
        target = m.servo.ee().clone()
        target[:, :2] += bolt_tip()[:, :2] - nut_base()[:, :2]
        return target, ratchet_quat, CLOSE

    def t_press(m):
        target = m.servo.ee().clone()
        target[:, :2] += bolt_tip()[:, :2] - nut_base()[:, :2]
        tip_gap = nut_base()[:, 2] - bolt_tip()[:, 2]
        target[:, 2] += torch.clamp(-0.0012 - tip_gap, -0.0007, 0.0002)
        return target, ratchet_quat, CLOSE

    def t_hold(m):
        return m.servo.ee(), m.servo.ee_quat(), OPEN

    def near(target_fn, tol=0.012, z_tol=None):
        def done(m):
            target, _, _ = target_fn(m)
            err = m.servo.ee() - target
            ok = torch.norm(err, dim=-1) < tol
            if z_tol is not None:
                ok &= err[:, 2].abs() < z_tol
            return ok
        return done

    def grasped(m):
        return (
            (m.timer >= grasp_dwell)
            & (finger_pos() < 0.77)
            & (nut_speed() < 0.05)
            & (nut_ang_speed() < 5.0)
            & (nut_upright() > 0.9)
        )

    def microlifted(m):
        return nut_base()[:, 2] > nut_rest[:, 2] + head_h + 0.004

    def lifted(m):
        return (
            near(t_lift)(m)
            & (nut_base()[:, 2] > 0.045)
            & (nut_upright() > 0.92)
        )

    def aligned(m):
        gap = nut_base()[:, 2] - bolt_tip()[:, 2]
        # Tall bolts (M20) leave less IK headroom; allow a wider z band.
        z_tol = 0.0025 if size >= 20 else 0.0015
        xy_tol = max(0.0015 if size >= 20 else 0.0008, thread_xy_tol)
        # M20 tip contact chatters at ~55 mm/s forever — don't gate on speed.
        return (
            (xy_err(nut_base(), bolt_tip()) < xy_tol)
            & ((gap - align_clearance).abs() < z_tol)
            & (nut_upright() > 0.95)
        )

    def engaged(m):
        gap = nut_base()[:, 2] - bolt_tip()[:, 2]
        return (
            (xy_err(nut_base(), bolt_tip()) < engage_xy_tol)
            & ((gap - ENGAGE_GAP).abs() < engage_tol)
            & (nut_upright() > 0.95)
        )

    def lead_aligned(m):
        gap = nut_base()[:, 2] - bolt_tip()[:, 2]
        return (
            (xy_err(nut_base(), bolt_tip()) < max(0.0005, 2.0 * thread_xy_tol))
            # SDF contact only exposes a shallow first-flank click; the helix
            # phase supplies the real depth while the gripper stays closed.
            & (gap < -LEAD_CAPTURE_DEPTH)
            & (gap > -2.0 * lead_depth)
            & (nut_upright() > 0.95)
        )

    def thread_done(m):
        final_quat = rolled(
            ratchet_quat, torch.full((N,), TURN_SIGN * turn_angle, device=dev)
        )
        progress = turn_progress()
        tip_gap = nut_base()[:, 2] - bolt_tip()[:, 2]
        return (
            (turn_command > 0.95 * turn_angle)
            & (progress > 0.95 * turn_angle)
            & (quat_error_magnitude(m.servo.ee_quat(), final_quat) < 0.12)
            & (xy_err(nut_base(), bolt_tip()) < 2.0 * thread_xy_tol)
            & (nut_upright() > 0.9)
            # Keep the nut captured on the helix before opening for a ratchet.
            & (tip_gap < -release_min_depth)
        )

    def released(m):
        return (
            (m.timer >= release_dwell)
            & (nut_speed() < 0.02)
            & (nut_ang_speed() < 0.5)
        )

    def reset(m):
        return (
            (m.timer >= 4)
            & (torch.norm(m.servo.ee() - release_anchor, dim=-1) < 0.010)
            & (quat_error_magnitude(m.servo.ee_quat(), ratchet_quat) < 0.08)
        )

    def realigned(m):
        target_tcp = nut_base().clone()
        target_tcp[:, 2] -= regrasp_sink
        return (
            (m.timer >= 4)
            & (xy_err(m.servo.tcp(), target_tcp) < 0.0005)
            & ((m.servo.tcp()[:, 2] - target_tcp[:, 2]).abs() < 0.001)
            & (nut_upright() > 0.9)
        )

    def regrasped(m):
        return (
            (m.timer >= grasp_dwell)
            & (finger_pos() < 0.77)
            & (nut_speed() < 0.08)
            # Closing around an engaged nut leaves small contact oscillations;
            # the finger stop + upright checks are the honest grasp signals.
            & (nut_ang_speed() < 8.0)
            & (nut_upright() > 0.95)
        )

    def centered(m):
        return (
            (xy_err(nut_base(), bolt_tip()) < max(0.0004, thread_xy_tol))
            & (finger_pos() < 0.77)
            & (nut_upright() > 0.95)
        )

    def pressed(m):
        tip_gap = nut_base()[:, 2] - bolt_tip()[:, 2]
        return (
            (tip_gap < -0.0008)
            & (finger_pos() < 0.77)
            & (nut_upright() > 0.95)
        )

    never = lambda m: torch.zeros(N, dtype=torch.bool, device=dev)
    phases = [
        Phase("hover", t_hover, near(t_hover), timeout=120, pos_cap=0.035),
        # Reach the calibrated board-level fingertip target before closing.
        # A loose z gate advances above the pad contact band and misses M4/M8.
        Phase("descend", t_descend, near(t_descend, 0.003, z_tol=0.0008),
              timeout=80, gate=0.004, pos_cap=0.03, zcap=0.5, fail_on_timeout=False),
        Phase("grasp", t_grasp, grasped, timeout=55, on_enter=enter_grasp,
              fail_on_timeout=False),
        Phase("microlift", t_microlift, microlifted, timeout=45, zcap=0.12),
        Phase("lift", t_lift, lifted, timeout=90, pos_cap=0.025, zcap=0.25,
              on_enter=enter_lift),
        Phase("transport", t_transport, lambda m: xy_err(nut_base(), bolt_tip()) < 0.002,
              timeout=120, pos_cap=0.035),
        # M20: slow the tip approach — a fast zcap slams the hex onto the
        # lead and leaves a vibrating 0.9 mm XY residual that never gates.
        Phase("align", t_align, aligned, timeout=180 if size >= 20 else 140,
              pos_cap=0.03, zcap=0.08 if size >= 20 else 0.4),
        Phase("engage", t_engage, engaged, timeout=80, zcap=0.1,
              on_enter=enter_engage),
        # Catch the first flank before ratcheting: sweep the wrist under a
        # bounded downward preload until the nut drops below the bolt tip.
        # Without this the nut spins on the lead-in cone and never descends.
        Phase("lead", t_lead, lead_aligned, timeout=120, zcap=0.05,
              on_enter=enter_lead, fail_on_timeout=False),
    ]
    for cycle in range(THREAD_CYCLES):
        phases.extend([
            Phase(f"thread_{cycle + 1}", t_thread, thread_done, timeout=90, zcap=0.02,
                  on_enter=enter_thread),
            Phase(f"release_{cycle + 1}", t_release, released, timeout=20,
                  on_enter=enter_release, fail_on_timeout=False),
        ])
        if cycle < THREAD_CYCLES - 1:
            phases.extend([
                # Unwind before the loosely engaged nut can tip inside the open
                # jaw cage. Joint velocity limits still bound the actual motion.
                Phase(f"reset_{cycle + 1}", t_reset, reset, timeout=40, rot_cap=0.5),
                Phase(f"realign_{cycle + 1}", t_realign, realigned, timeout=35,
                      pos_cap=0.02, zcap=0.08, fail_on_timeout=False),
                Phase(f"regrasp_{cycle + 1}", t_regrasp, regrasped, timeout=55,
                      on_enter=enter_regrasp),
                Phase(f"center_{cycle + 1}", t_center, centered,
                      timeout=30, pos_cap=0.02),
                Phase(f"press_{cycle + 1}", t_press, pressed,
                      timeout=40, zcap=0.05, fail_on_timeout=False),
            ])
    # The final open hold lets a correctly threaded nut settle through the same
    # three-step success debounce as every intermediate release.
    phases.append(Phase("hold", t_hold, never, timeout=10**9))
    machine = Machine(base, servo, phases)
    machine.guide = (
        f"M{size} pitch={pitch * 1e3:.2f}mm nut_h={nut_h * 1e3:.1f}mm: "
        f"grasp preload {grasp_preload * 1e3:+.1f}mm, regrasp sink "
        f"{regrasp_sink * 1e3:.1f}mm; each {turn_angle:.1f}rad clockwise "
        f"stroke descends by the measured yaw, then release/caged-unwind/regrasp"
    )

    def diagnostics():
        """Small, host-printable snapshot for expert iteration."""
        base_pos = nut_base()
        tip_pos = bolt_tip()
        values = {
            "xy_mm": float(xy_err(base_pos, tip_pos)[0] * 1e3),
            "tip_gap_mm": float((base_pos[0, 2] - tip_pos[0, 2]) * 1e3),
            "seat_gap_mm": float((base_pos[0, 2] - (tip_pos[0, 2] - 1.5 * pitch)) * 1e3),
            "yaw_rad": float(nut_yaw()[0]),
            "upright": float(nut_upright()[0]),
            "lin_mps": float(nut_speed()[0]),
            "ang_rps": float(nut_ang_speed()[0]),
        }
        phase = machine.phases[int(machine.phase[0])].name
        if phase.startswith("thread_"):
            progress = turn_progress()[0]
            actual_off = quat_apply(machine.servo.ee_quat(), inhand)
            values.update({
                "cmd_deg": float(torch.rad2deg(turn_command[0])),
                "stroke_deg": float(torch.rad2deg(progress)),
                "stroke_drop_mm": float((turn_base[0, 2] - base_pos[0, 2]) * 1e3),
                "helix_mm": float(pitch * progress / (2.0 * torch.pi) * 1e3),
                "depth_err_mm": float(
                    (pitch * progress / (2.0 * torch.pi) - (turn_base[0, 2] - base_pos[0, 2]))
                    * 1e3
                ),
                "force_dN": float(force_delta[0]),
                "peak_force_dN": float(peak_force_delta[0]),
                "track_mm": float(torch.norm(base_pos[0] - (machine.servo.ee()[0] + actual_off[0])) * 1e3),
            })
        return values

    def sweep_metrics():
        base_pos = nut_base()
        return {
            "xy_mm": xy_err(base_pos, bolt_tip()) * 1e3,
            "tip_gap_mm": (base_pos[:, 2] - bolt_tip()[:, 2]) * 1e3,
            "stroke_drop_mm": (turn_base[:, 2] - base_pos[:, 2]) * 1e3,
            "upright": nut_upright(),
        }

    machine.diagnostics = diagnostics
    machine.sweep_metrics = sweep_metrics
    return machine

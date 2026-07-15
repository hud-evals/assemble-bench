"""Privileged scripted expert for the M4--M16 nut-thread tasks.

The expert grasps the nut flat from the board, places its bore over the bolt,
and follows the thread helix with coupled negative wrist yaw and axial motion.
The required 1.5+ turns exceed one comfortable Franka wrist sweep, so each
short turn is followed by a physical release, yaw reset, and regrasp while the
nut remains engaged on the bolt.
"""

import torch

import warp as wp
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_error_magnitude

from assembly_bench.environments.assembly.variants import NUTBOLT

from .base import Machine, Phase, home_quat, pos_of, quat_of, rolled


OPEN, CLOSE = 0.0, 1.0

# Generated asset body heights, measured from their authored bounds.
NUT_HEIGHT = {4: 0.0032, 8: 0.0065, 12: 0.010, 16: 0.013}
SAFE_BASE_Z = 0.070
ALIGN_CLEARANCE = 0.004
ENGAGE_GAP = 0.0002

# Factory's nut task permits negative yaw only. Seven 2 rad strokes provide
# 2.23 turns; success normally terminates during the fifth release (1.59 turns).
THREAD_CYCLES = 7
TURN_ANGLE = 2.0
TURN_RATE = 0.06


def make_machine(base, servo, *, size, seed=None):
    """Build a pick, engage, and release-regrip threading phase machine."""
    if size not in NUT_HEIGHT:
        raise ValueError(f"unsupported nut size: M{size}")

    N, dev = base.num_envs, base.device
    dims = NUTBOLT[size]
    head_h, shank, pitch = dims["head_h"], dims["shank"], dims["pitch"]
    nut_h = NUT_HEIGHT[size]

    hold_quat = home_quat(servo)
    grasp_anchor = torch.zeros((N, 3), device=dev)
    lift_anchor = torch.zeros((N, 3), device=dev)
    release_anchor = torch.zeros((N, 3), device=dev)
    release_quat = hold_quat.clone()
    inhand = torch.zeros((N, 3), device=dev)
    turn_base = torch.zeros((N, 3), device=dev)
    turn_yaw_start = torch.zeros(N, device=dev)
    thread_pending = torch.zeros(N, dtype=torch.bool, device=dev)

    aim_off = torch.zeros((N, 3), device=dev)
    hover_extra = torch.zeros(N, device=dev)
    safe_base_z = torch.full((N,), SAFE_BASE_Z, device=dev)
    grasp_dwell = torch.full((N,), 20, dtype=torch.long, device=dev)
    release_dwell = torch.full((N,), 10, dtype=torch.long, device=dev)
    turn_rate = torch.full((N,), TURN_RATE, device=dev)

    if seed is not None:
        g = torch.Generator(device=dev).manual_seed(seed)
        u = lambda lo, hi: lo + (hi - lo) * torch.rand(N, generator=g, device=dev)
        # Keep the pads near hex flats, but vary the captured thread phase.
        hold_quat = rolled(hold_quat, u(-0.12, 0.12))
        aim_off[:, 0], aim_off[:, 1] = u(-0.0005, 0.0005), u(-0.0005, 0.0005)
        aim_off[:, 2] = u(-0.08 * nut_h, 0.08 * nut_h)
        hover_extra = u(0.0, 0.025)
        safe_base_z += u(-0.006, 0.012)
        grasp_dwell = u(18.0, 30.0).long()
        release_dwell = u(9.0, 15.0).long()
        turn_rate = u(0.052, 0.068)

    turn_steps = torch.ceil(TURN_ANGLE / turn_rate).long() + 6
    thread_xy_tol = min(0.0003, 0.025 * size / 1000.0)
    thread_preload = min(0.0001, 0.1 * pitch)
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

    def nut_upright():
        z = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(N, 3)
        return quat_apply(quat_of(base, "held_part"), z)[:, 2]

    def nut_yaw():
        x = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(N, 3)
        axis = quat_apply(quat_of(base, "held_part"), x)
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
        p[:, 2] = nut_rest[:, 2] + head_h + 0.5 * nut_h + aim_off[:, 2]
        return p

    def t_hover(m):
        p = grasp_tcp()
        p[:, 2] += 0.05 + hover_extra
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
        p = m.servo.ee() + (bolt_tip() - nut_base())
        p[:, 2] = m.servo.ee()[:, 2] + (
            bolt_tip()[:, 2] + ALIGN_CLEARANCE - nut_base()[:, 2]
        )
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
        dz = target[:, 2] - m.servo.ee()[:, 2]
        target[:, 2] = m.servo.ee()[:, 2] + torch.where(
            tracking, dz, torch.zeros_like(dz)
        )
        return target, hold_quat, CLOSE

    def enter_thread(m, ids):
        # Machine enters a phase before the preceding phase's final action is
        # simulated. Defer the capture to the first thread target evaluation.
        thread_pending[ids] = True

    def turn_progress():
        delta = nut_yaw() - turn_yaw_start
        delta -= 2.0 * torch.pi * torch.round(delta / (2.0 * torch.pi))
        return torch.clamp(-delta, 0.0, TURN_ANGLE)

    def t_thread(m):
        ids = thread_pending.nonzero().squeeze(-1)
        if ids.numel():
            # Re-capture after every regrasp. Each stroke starts from the
            # achieved depth, so a blocked stroke cannot build a ram target.
            turn_base[ids] = nut_base()[ids]
            turn_yaw_start[ids] = nut_yaw()[ids]
            inhand[ids] = quat_apply_inverse(
                m.servo.ee_quat()[ids], nut_base()[ids] - m.servo.ee()[ids]
            )
            thread_pending[ids] = False
        command_angle = torch.minimum(
            m.timer.to(turn_rate.dtype) * turn_rate,
            torch.full_like(turn_rate, TURN_ANGLE),
        )
        progress = turn_progress()
        target_quat = rolled(hold_quat, -command_angle)
        desired = turn_base.clone()
        desired[:, :2] = bolt_tip()[:, :2]
        # Axial progress follows the ACTUAL nut rotation, not elapsed time. A
        # stalled wrist therefore applies only a tiny preload, never a blind ram.
        desired[:, 2] -= pitch * progress / (2.0 * torch.pi) + thread_preload
        target = desired - quat_apply(target_quat, inhand)

        # Couple the axial target to yaw, but bound contact preload and stop
        # descent immediately if alignment or in-hand tracking is lost.
        actual_off = quat_apply(m.servo.ee_quat(), inhand)
        aligned = xy_err(nut_base(), bolt_tip()) < thread_xy_tol
        tracking = torch.norm(nut_base() - (m.servo.ee() + actual_off), dim=-1) < 0.006
        dz = torch.clamp(target[:, 2] - m.servo.ee()[:, 2], -0.0008, 0.0002)
        target[:, 2] = m.servo.ee()[:, 2] + torch.where(
            aligned & tracking, dz, torch.zeros_like(dz)
        )
        return target, target_quat, CLOSE

    def enter_release(m, ids):
        release_anchor[ids] = m.servo.ee()[ids]
        release_quat[ids] = m.servo.ee_quat()[ids]

    def t_release(m):
        # Opening laterally leaves the nut supported by the real threads.
        return release_anchor, release_quat, OPEN

    def t_reset(m):
        # Reset yaw in place while the fingers are wide open, then reclose from
        # the same axial level. This preserves thread engagement between strokes.
        return release_anchor, hold_quat, OPEN

    def enter_regrasp(m, ids):
        grasp_anchor[ids] = m.servo.ee()[ids]

    def t_regrasp(m):
        return grasp_anchor, hold_quat, CLOSE

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
            & (nut_speed() < 0.01)
            & (nut_ang_speed() < 0.5)
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
        return (
            (xy_err(nut_base(), bolt_tip()) < thread_xy_tol)
            & ((gap - ALIGN_CLEARANCE).abs() < 0.0015)
            & (nut_upright() > 0.95)
        )

    def engaged(m):
        gap = nut_base()[:, 2] - bolt_tip()[:, 2]
        return (
            (xy_err(nut_base(), bolt_tip()) < thread_xy_tol)
            & ((gap - ENGAGE_GAP).abs() < 0.0008)
            & (nut_upright() > 0.95)
        )

    def thread_done(m):
        final_quat = rolled(hold_quat, torch.full((N,), -TURN_ANGLE, device=dev))
        return (
            (m.timer >= turn_steps)
            & (turn_progress() > 0.95 * TURN_ANGLE)
            & (quat_error_magnitude(m.servo.ee_quat(), final_quat) < 0.12)
            & (xy_err(nut_base(), bolt_tip()) < 2.0 * thread_xy_tol)
            & (nut_upright() > 0.9)
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
            & (quat_error_magnitude(m.servo.ee_quat(), hold_quat) < 0.08)
        )

    def regrasped(m):
        return (
            (m.timer >= grasp_dwell)
            & (nut_speed() < 0.01)
            & (nut_ang_speed() < 0.5)
        )

    never = lambda m: torch.zeros(N, dtype=torch.bool, device=dev)
    phases = [
        Phase("hover", t_hover, near(t_hover), timeout=150),
        Phase("descend", t_descend, near(t_descend, 0.009, z_tol=0.003),
              timeout=110, gate=0.004, zcap=0.3, fail_on_timeout=False),
        Phase("grasp", t_grasp, grasped, timeout=55, on_enter=enter_grasp,
              fail_on_timeout=False),
        Phase("microlift", t_microlift, microlifted, timeout=60, zcap=0.1),
        Phase("lift", t_lift, lifted, timeout=110, zcap=0.2, on_enter=enter_lift),
        Phase("transport", t_transport, lambda m: xy_err(nut_base(), bolt_tip()) < 0.002,
              timeout=150),
        Phase("align", t_align, aligned, timeout=120, zcap=0.25),
        Phase("engage", t_engage, engaged, timeout=120, zcap=0.04,
              on_enter=enter_engage),
    ]
    for cycle in range(THREAD_CYCLES):
        phases.extend([
            Phase(f"thread_{cycle + 1}", t_thread, thread_done, timeout=80, zcap=0.04,
                  on_enter=enter_thread),
            Phase(f"release_{cycle + 1}", t_release, released, timeout=35,
                  on_enter=enter_release, fail_on_timeout=False),
        ])
        if cycle < THREAD_CYCLES - 1:
            phases.extend([
                Phase(f"reset_{cycle + 1}", t_reset, reset, timeout=45),
                Phase(f"regrasp_{cycle + 1}", t_regrasp, regrasped, timeout=55,
                      on_enter=enter_regrasp, fail_on_timeout=False),
            ])
    # The final open hold lets a correctly threaded nut settle through the same
    # three-step success debounce as every intermediate release.
    phases.append(Phase("hold", t_hold, never, timeout=10**9))
    return Machine(base, servo, phases)

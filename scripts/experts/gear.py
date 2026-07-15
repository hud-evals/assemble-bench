"""Privileged scripted expert for the small, medium, and large gear tasks.

The gear lies flat on the table. The expert grasps low around its body, lifts
it level, aligns its bore over the true shaft center, then searches the tooth
phase with a smooth one-pitch yaw wiggle under a gentle downward bias. Pressing
is disabled whenever XY alignment or grasp tracking is lost, so a tooth clash
rests instead of becoming a stiff-PD ram.
"""

import torch

import warp as wp
from isaaclab.utils.math import quat_apply

from .base import Machine, Phase, home_quat, pos_of, quat_of, rolled


OPEN, CLOSE = 0.0, 1.0

# The generated gears share a 25 mm body spanning local z=[5, 30] mm. Aim the
# fingertip plane into the lower body, not at the top rim, for a level grasp.
GRASP_ROOT_Z = 0.012
SAFE_ROOT_Z = 0.075
ALIGN_GAP = 0.035

# Module-2 gears: pitch radius 10/20/30 mm, hence 10/20/30 teeth. A yaw range
# of +/-pi/teeth covers one complete tooth pitch.
GEAR_TEETH = {"small": 10, "medium": 20, "large": 30}
WIGGLE_PERIOD = 48.0
PRESS_BIAS = 0.003


def make_machine(base, servo, *, size, seat_off, seed=None):
    """Build the gear pick-and-mesh phase machine after reset and settling."""
    if size not in GEAR_TEETH:
        raise ValueError(f"unsupported gear size: {size}")

    N, dev = base.num_envs, base.device
    hold_quat = home_quat(servo)
    grasp_anchor = torch.zeros((N, 3), device=dev)
    lift_anchor = torch.zeros((N, 3), device=dev)
    release_anchor = torch.zeros((N, 3), device=dev)
    inhand = torch.zeros((N, 3), device=dev)
    mesh_yaw = torch.zeros(N, device=dev)
    mesh_locked = torch.zeros(N, dtype=torch.bool, device=dev)

    hover_extra = torch.zeros(N, device=dev)
    safe_root_z = torch.full((N,), SAFE_ROOT_Z, device=dev)
    grasp_dwell = torch.full((N,), 18, dtype=torch.long, device=dev)
    wiggle_period = torch.full((N,), WIGGLE_PERIOD, device=dev)
    wiggle_sign = torch.ones(N, device=dev)
    aim_off = torch.zeros((N, 3), device=dev)

    pitch_half = torch.pi / GEAR_TEETH[size]
    if seed is not None:
        g = torch.Generator(device=dev).manual_seed(seed)
        u = lambda lo, hi: lo + (hi - lo) * torch.rand(N, generator=g, device=dev)
        # The complete pitch search recovers this small approach-yaw variation.
        hold_quat = rolled(hold_quat, u(-0.25 * pitch_half, 0.25 * pitch_half))
        aim_off[:, 0], aim_off[:, 1] = u(-0.001, 0.001), u(-0.001, 0.001)
        hover_extra = u(0.0, 0.025)
        safe_root_z += u(-0.008, 0.015)
        grasp_dwell = u(16.0, 30.0).long()
        wiggle_period = u(42.0, 54.0)
        wiggle_sign = torch.where(u(0.0, 1.0) < 0.5, -1.0, 1.0)

    gear = lambda: pos_of(base, "held_part")

    def shaft():
        off = torch.tensor(seat_off, device=dev).expand(N, 3)
        return pos_of(base, "fixed_part") + quat_apply(quat_of(base, "fixed_part"), off)

    def gear_speed():
        return torch.norm(wp.to_torch(base.scene["held_part"].data.root_lin_vel_w), dim=-1)

    def gear_upright():
        z = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(N, 3)
        return quat_apply(quat_of(base, "held_part"), z)[:, 2]

    # Follow only a gear that is at rest. Once contact moves it, freeze the
    # approach point so the arm never chases a displaced dynamic object.
    gear_rest = gear().clone()

    def grasp_tcp():
        settled = gear_speed() < 0.01
        gear_rest[settled] = gear()[settled]
        p = gear_rest + aim_off
        p[:, 2] = gear_rest[:, 2] + GRASP_ROOT_Z
        return p

    def toward(target, m):
        return m.servo.ee() + (target - m.servo.tcp())

    def xy_err(a, b):
        return torch.norm(a[:, :2] - b[:, :2], dim=-1)

    def t_hover(m):
        p = grasp_tcp()
        p[:, 2] += 0.055 + hover_extra
        return toward(p, m), hold_quat, OPEN

    def t_descend(m):
        return toward(grasp_tcp(), m), hold_quat, OPEN

    def enter_grasp(m, ids):
        # Close at the achieved pose. A fixed ideal target can push the flat
        # gear into the table when the open fingertips contact early.
        grasp_anchor[ids] = m.servo.ee()[ids]

    def t_grasp(m):
        return grasp_anchor, hold_quat, CLOSE

    def t_microlift(m):
        p = grasp_anchor.clone()
        p[:, 2] += 0.012
        return p, hold_quat, CLOSE

    def enter_lift(m, ids):
        lift_anchor[ids] = grasp_anchor[ids]
        lift_anchor[ids, 2] += safe_root_z[ids] - gear()[ids, 2]

    def t_lift(m):
        return lift_anchor, hold_quat, CLOSE

    def t_transport(m):
        # In free space, live error feedback corrects small in-grip shifts.
        p = m.servo.ee() + (shaft() - gear())
        p[:, 2] = m.servo.ee()[:, 2] + (safe_root_z - gear()[:, 2])
        return p, hold_quat, CLOSE

    def t_align(m):
        p = m.servo.ee() + (shaft() - gear())
        p[:, 2] = m.servo.ee()[:, 2] + (shaft()[:, 2] + ALIGN_GAP - gear()[:, 2])
        return p, hold_quat, CLOSE

    def enter_mesh(m, ids):
        # Freeze the in-hand offset before contact. A live gear-relative target
        # becomes an unreachable moving target as soon as teeth clash.
        inhand[ids] = gear()[ids] - m.servo.ee()[ids]
        mesh_yaw[ids] = 0.0
        mesh_locked[ids] = False

    def t_mesh(m):
        target = shaft() - inhand
        aligned = xy_err(gear(), shaft()) < 0.001
        tracking = torch.norm(gear() - (m.servo.ee() + inhand), dim=-1) < 0.012
        gap = gear()[:, 2] - shaft()[:, 2]
        # Keep a few millimetres of downward position error so the compliant
        # arm develops real preload instead of resetting to a nearly zero-force
        # target every step. The XY/tracking gates still stop an off-axis ram.
        dz = torch.clamp(-gap, min=-PRESS_BIAS, max=0.0)
        target[:, 2] = m.servo.ee()[:, 2] + torch.where(
            aligned & tracking, dz, torch.zeros_like(dz)
        )

        # Smoothly scan exactly one tooth pitch. The bounded target and XY gate
        # let a bad phase rest on the neighbours until a valley lines up.
        phase = 2.0 * torch.pi * m.timer / wiggle_period
        candidate = wiggle_sign * pitch_half * torch.sin(phase)
        newly_locked = gap < 0.012
        searching = ~mesh_locked & ~newly_locked
        mesh_yaw[searching] = candidate[searching]
        mesh_locked.logical_or_(newly_locked)
        return target, rolled(hold_quat, mesh_yaw), CLOSE

    def t_hold(m):
        return m.servo.ee(), m.servo.ee_quat(), CLOSE

    def enter_release(m, ids):
        release_anchor[ids] = m.servo.ee()[ids]

    def t_release(m):
        return release_anchor, hold_quat, OPEN

    def t_retreat(m):
        p = release_anchor.clone()
        p[:, 2] += 0.04
        return p, hold_quat, OPEN

    def t_settle(m):
        return m.servo.ee(), hold_quat, OPEN

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
        return (m.timer >= grasp_dwell) & (gear_speed() < 0.01)

    def microlifted(m):
        return gear()[:, 2] > gear_rest[:, 2] + 0.006

    def lifted(m):
        return (
            near(t_lift)(m)
            & (gear()[:, 2] > 0.045)
            & (gear_upright() > 0.9)
        )

    def aligned(m):
        return (
            (xy_err(gear(), shaft()) < 0.0008)
            & ((gear()[:, 2] - shaft()[:, 2] - ALIGN_GAP).abs() < 0.004)
            & (gear_upright() > 0.9)
        )

    def seated(m):
        gap = gear()[:, 2] - shaft()[:, 2]
        return (
            (xy_err(gear(), shaft()) < 0.002)
            & (gap < 0.003)
            & (gear_speed() < 0.05)
        )

    def engaged(m):
        gap = gear()[:, 2] - shaft()[:, 2]
        return (xy_err(gear(), shaft()) < 0.002) & (gap < 0.012)

    never = lambda m: torch.zeros(N, dtype=torch.bool, device=dev)

    return Machine(base, servo, [
        Phase("hover", t_hover, near(t_hover), timeout=150),
        # If the open fingers meet the table/gear slightly early, close at the
        # achieved pose instead of forcing the ideal z and ramming the part.
        Phase("descend", t_descend, near(t_descend, 0.009, z_tol=0.003),
              timeout=100, gate=0.004, zcap=0.35, fail_on_timeout=False),
        Phase("grasp", t_grasp, grasped, timeout=50, on_enter=enter_grasp,
              fail_on_timeout=False),
        Phase("microlift", t_microlift, microlifted, timeout=60, zcap=0.1),
        Phase("lift", t_lift, lifted, timeout=100, zcap=0.2, on_enter=enter_lift),
        Phase("transport", t_transport, lambda m: xy_err(gear(), shaft()) < 0.002,
              timeout=150),
        Phase("align", t_align, aligned, timeout=120, zcap=0.3),
        Phase("mesh", t_mesh, engaged, timeout=420, zcap=0.2, on_enter=enter_mesh),
        Phase("release", t_release, lambda m: m.timer >= 12, timeout=20,
              on_enter=enter_release, fail_on_timeout=False),
        Phase("retreat", t_retreat, near(t_retreat), timeout=60, zcap=0.5),
        Phase("settle", t_settle, seated, timeout=100),
        Phase("hold", t_hold, never, timeout=10**9),
    ])

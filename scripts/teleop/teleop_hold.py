"""Keyboard teleop with an ABSOLUTE end-effector hold (drift-free).

Arena's stock teleop.py drives the differential-IK action in relative mode:
every step the controller sets ``target = apply_delta_pose(measured_pose,
delta)``. With ``delta = 0`` (no key pressed) the target re-anchors to the
*current measured* pose, so any external wrench -- e.g. the weight of a grasped
peg -- slowly walks the arm and is never corrected. Zeroing the command cannot
help: the zero command IS what re-anchors.

This loop keeps a persistent absolute target frame that only moves when a key is
pressed, and each step sends the CORRECTION delta ``(target - measured)`` as the
6-DoF command. Idle -> the arm is actively driven back to the held pose. The
action shape (6 pose deltas + gripper) is unchanged, so the recorder / Mimic
reconstruction (target = measured (+) delta) stays consistent.

Keyboard (click the streamed viewport first): W/S A/D Q/E translate,
Z/X T/G C/V rotate, K gripper toggle, R reset, L zero-hold to current pose.
"""

import gymnasium as gym

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli

parser = get_isaaclab_arena_cli_parser()
parser.add_argument("--sensitivity", type=float, default=1.0, help="Unused; kept for CLI compat.")
add_example_environments_cli_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import contextlib

import torch

import omni.log
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.utils.math import compute_pose_error


def _find_ik_term(env):
    """Return the differential-IK arm action term (has _compute_frame_pose)."""
    for name in env.action_manager.active_terms:
        term = env.action_manager.get_term(name)
        if hasattr(term, "_compute_frame_pose"):
            return term
    raise RuntimeError("No differential-IK action term found; is the embodiment droid_differential_ik?")


def main() -> None:
    arena_builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg, env_kwargs = arena_builder.build_registered()
    env_cfg.terminations.time_out = None

    env = gym.make(env_name, cfg=env_cfg, **env_kwargs)
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import reapply_viewer_cfg

    reapply_viewer_cfg(env)
    env = env.unwrapped

    if not (hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices):
        raise RuntimeError(
            f"teleop device '{args_cli.teleop_device}' not configured on the environment; "
            "this hold loop supports the keyboard/spacemouse SE(3) devices only."
        )

    should_reset = False
    zero_hold = False

    def reset_cb():
        nonlocal should_reset
        should_reset = True

    def zero_hold_cb():
        # Snap the held target back to the current measured pose (clears any
        # accumulated operator offset without resetting the episode).
        nonlocal zero_hold
        zero_hold = True

    teleop = create_teleop_device(
        args_cli.teleop_device, env_cfg.teleop_devices.devices, {"R": reset_cb, "RESET": reset_cb, "L": zero_hold_cb}
    )
    arm = _find_ik_term(env)

    def capture_target():
        pos, quat = arm._compute_frame_pose()  # noqa: SLF001 (stable internal API)
        return pos.clone(), quat.clone()

    env.reset()
    teleop.reset()
    hold_pos, hold_quat = capture_target()
    print("Absolute-hold teleop started. R reset, L re-zero hold, K gripper.")

    # Threshold above which a keyboard command counts as "actively moving".
    move_eps = 1e-6
    was_moving = True  # forces a fresh hold capture on the first idle step

    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        while simulation_app.is_running():
            command = teleop.advance()
            if command is None:
                env.sim.render()
                continue
            command = command.to(env.device)
            delta = command[:6].unsqueeze(0)  # (1, 6) operator-commanded increment
            gripper = command[6:].unsqueeze(0)  # (1, 1)

            cur_pos, cur_quat = arm._compute_frame_pose()  # noqa: SLF001
            moving = bool(torch.linalg.norm(delta) > move_eps)

            if moving or zero_hold:
                # Actively commanding (or re-zeroing): pass the delta straight
                # through -- identical responsiveness to stock relative mode, no
                # accumulated lead -> no overshoot.
                corr = delta
                zero_hold = False
                was_moving = True
            else:
                # First idle step after motion: freeze the hold at the CURRENT
                # (post-motion) pose so there is nothing to spring back to.
                if was_moving:
                    hold_pos, hold_quat = cur_pos.clone(), cur_quat.clone()
                    was_moving = False
                # Drive measured -> frozen pose (no deadband).
                pos_err, rot_err = compute_pose_error(
                    cur_pos, cur_quat, hold_pos, hold_quat, rot_error_type="axis_angle"
                )
                corr = torch.cat([pos_err, rot_err], dim=-1)

            action = torch.cat([corr, gripper], dim=-1).repeat(env.num_envs, 1)
            env.step(action)

            if should_reset:
                env.reset()
                teleop.reset()
                hold_pos, hold_quat = capture_target()
                should_reset = False
                print("Environment reset; hold target re-captured.")

            if env.sim.is_stopped():
                break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

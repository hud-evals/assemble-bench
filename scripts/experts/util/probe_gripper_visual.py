"""Render the gripper close up in both binary-action states, tool-down.

Settles the arm at a tool-down pose over empty table, commands grip 0 then 1,
and saves full-res front+wrist frames plus pad-frame telemetry for each state.
The one authoritative answer to 'which command opens the pads, and where are
the tips'. Run: python scripts/experts/util/probe_gripper_visual.py --headless
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # scripts/

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

parser = get_isaaclab_arena_cli_parser()
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

with SimulationAppContext(args_cli):
    import torch

    import warp as wp
    from PIL import Image
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from assembly_bench.environments.assembly.assembly import AssemblyBenchEnvironment

    from experts.base import Servo, home_quat

    AssemblyBenchEnvironment.add_cli_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True

    env = ArenaEnvBuilder(AssemblyBenchEnvironment().get_env(args_cli), args_cli).make_registered(
        render_mode="rgb_array")
    env.reset()
    base = env.unwrapped
    servo = Servo(base)
    down = home_quat(servo)
    robot = base.scene["robot"]

    # Park tool-down at a clear spot, 12 cm up, in the front camera's view.
    target = torch.tensor([0.42, 0.10, 0.12], device=base.device).expand(base.num_envs, 3)

    def settle(grip_cmd, steps):
        g = torch.full((base.num_envs,), grip_cmd, device=base.device)
        for _ in range(steps):
            obs, *_ = env.step(servo.act(target.contiguous(), down, g))
        return obs

    os.makedirs("/tmp/grip_probe", exist_ok=True)
    for cmd in (0.0, 1.0):
        obs = settle(cmd, 140 if cmd == 0.0 else 40)
        fj = float(wp.to_torch(robot.data.joint_pos)[0, 7])
        names = robot.data.body_names
        li, ri = names.index("left_inner_finger"), names.index("right_inner_finger")
        bp = wp.to_torch(robot.data.body_pos_w)[0] - base.scene.env_origins[0]
        ee = servo.ee()[0]
        print(f"[grip] cmd={cmd} finger_joint={fj:.3f} flange=({ee[0]:.4f},{ee[1]:.4f},{ee[2]:.4f}) "
              f"L_inner=({bp[li][0]:.4f},{bp[li][1]:.4f},{bp[li][2]:.4f}) "
              f"R_inner=({bp[ri][0]:.4f},{bp[ri][1]:.4f},{bp[ri][2]:.4f})", flush=True)
        for cam in ("front_cam_rgb", "wrist_camera_rgb"):
            frame = obs["camera_obs"][cam][0].to(torch.uint8).cpu().numpy()[..., :3]
            Image.fromarray(frame).save(f"/tmp/grip_probe/cmd{int(cmd)}_{cam.split('_')[0]}.png")
        # Runtime AABBs of the fingertip COLLISION prims: the physical pad gap
        # and tip plane in this state (frames/cranks lie; boxes don't).
        from pxr import Usd, UsdGeom, UsdPhysics
        cachebb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        stage = base.scene.stage
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if "env_0/Robot" in path and "fingertips" in path and prim.HasAPI(UsdPhysics.CollisionAPI):
                r = cachebb.ComputeWorldBound(prim).ComputeAlignedRange()
                mn, mx = r.GetMin(), r.GetMax()
                side = "L" if "left" in path else "R"
                print(f"[grip]   tip_{side} AABB x[{mn[0]:.4f},{mx[0]:.4f}] "
                      f"y[{mn[1]:.4f},{mx[1]:.4f}] z[{mn[2]:.4f},{mx[2]:.4f}]", flush=True)
    print("[grip] saved /tmp/grip_probe/*", flush=True)
    env.close()

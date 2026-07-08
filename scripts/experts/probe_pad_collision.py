"""Decisive pad-collision test: place the peg IN the closed grasp channel.

Parks the gripper tool-down mid-air, closes it, then kinematically teleports
the peg so its shaft crosses the pinch point, releases it, and watches. If pad
collision exists the peg is held/expelled; if it falls straight through, the
pad collision geometry is absent/misplaced. Also probes 8 lateral offsets to
map where the channel actually blocks. Run:

    python scripts/experts/probe_pad_collision.py --headless
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    peg = base.scene["held_part"]
    N, dev = base.num_envs, base.device
    ids = torch.arange(N, device=dev)

    park = torch.tensor([0.42, 0.10, 0.20], device=dev).expand(N, 3).contiguous()

    def step(grip, n=1):
        obs = None
        for _ in range(n):
            obs, *_ = env.step(servo.act(park, down, torch.full((N,), grip, device=dev)))
        return obs

    step(0.0, 160)     # settle tool-down at the park pose, open
    step(1.0, 30)      # close fully in air
    tcp = servo.tcp()[0]
    print(f"[pad] closed; flange=({servo.ee()[0][0]:.4f},{servo.ee()[0][1]:.4f},{servo.ee()[0][2]:.4f}) "
          f"tip-plane tcp=({tcp[0]:.4f},{tcp[1]:.4f},{tcp[2]:.4f})", flush=True)

    # Candidate pinch points: the modeled tip midpoint and a grid around it
    # (up along the fingers and both lateral axes).
    offsets = [(0, 0, 0.010), (0, 0, 0.020), (0, 0, 0.030),
               (0.010, 0, 0.020), (-0.010, 0, 0.020), (0, 0.010, 0.020), (0, -0.010, 0.020)]
    os.makedirs("/tmp/pad_probe", exist_ok=True)
    for k, (dx, dy, dz) in enumerate(offsets):
        step(1.0, 5)
        tcp = servo.tcp()  # live tip plane
        pos = tcp + torch.tensor([dx, dy, dz], device=dev)
        # Peg root = base; put the SHAFT MIDDLE at the probe point.
        pos[:, 2] -= 0.025
        pose = torch.cat([pos + base.scene.env_origins,
                          torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev).expand(N, 4)], dim=-1)
        peg.write_root_pose_to_sim(pose, env_ids=ids)
        peg.write_root_velocity_to_sim(torch.zeros((N, 6), device=dev), env_ids=ids)
        z0 = (wp.to_torch(peg.data.root_pos_w) - base.scene.env_origins)[0, 2]
        obs = step(1.0, 30)   # 2 seconds of gravity
        p = (wp.to_torch(peg.data.root_pos_w) - base.scene.env_origins)[0]
        drop = float(z0 - p[2])
        verdict = "FELL THROUGH" if drop > 0.05 else ("HELD/RESTING" if drop < 0.01 else "slid")
        print(f"[pad] probe {k} off=({dx:+.3f},{dy:+.3f},{dz:+.3f}): drop={drop:.3f} m -> {verdict} "
              f"peg=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})", flush=True)
        frame = obs["camera_obs"]["front_cam_rgb"][0].to(torch.uint8).cpu().numpy()[..., :3]
        Image.fromarray(frame).save(f"/tmp/pad_probe/probe{k}.png")

    env.close()

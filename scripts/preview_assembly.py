"""Render stills of an assembly variant and sanity-check poses + image quality.

Builds the environment exactly like the policy runner, steps zero actions to
let physics settle, then saves the policy cameras (front + wrist, 224x224) and
prints part positions and per-channel color stats (catches black/grayscale
renders):

    python scripts/preview_assembly.py --task peg_round_8mm_tight --out /tmp/peg
"""

import argparse

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

    AssemblyBenchEnvironment.add_cli_args(parser)
    parser.add_argument("--out", type=str, default="/tmp/assembly_preview",
                        help="output stem; saves <out>_front.png and <out>_wrist.png")
    parser.add_argument("--settle_steps", type=int, default=40)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True  # re-parse resets it

    env_def = AssemblyBenchEnvironment()
    arena_env = env_def.get_env(args_cli)
    env = ArenaEnvBuilder(arena_env, args_cli).make_registered(render_mode="rgb_array")
    env.reset()

    # Zero actions; let the parts settle on their stands.
    actions = torch.zeros(
        env.unwrapped.num_envs, env.unwrapped.action_manager.total_action_dim, device=env.unwrapped.device)
    for _ in range(args_cli.settle_steps):
        obs, *_ = env.step(actions)

    scene = env.unwrapped.scene
    for name, asset in {**scene.rigid_objects, **scene.articulations}.items():
        pos = wp.to_torch(asset.data.root_pos_w)[0].tolist()
        print(f"  {name}: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")

    # Save the policy camera observations (DROID-native 1280x720).
    for key, img in obs["camera_obs"].items():
        frame = img[0].to(torch.uint8).cpu().numpy()[..., :3]
        # Color sanity: a grayscale/black render has (near-)identical channels.
        ch = frame.reshape(-1, 3).astype(float)
        chroma = abs(ch[:, 0] - ch[:, 1]).mean() + abs(ch[:, 1] - ch[:, 2]).mean()
        print(f"{key}: shape={frame.shape} mean={frame.mean():.1f} std={frame.std():.1f} "
              f"chroma={chroma:.2f} ({'COLOR OK' if chroma > 1.0 else 'LOOKS GRAYSCALE/BLACK'})", flush=True)
        path = f"{args_cli.out}_{key.removesuffix('_rgb').removesuffix('_cam').removesuffix('_camera')}.png"
        Image.fromarray(frame).save(path)
        print("saved", path, flush=True)

    env.close()

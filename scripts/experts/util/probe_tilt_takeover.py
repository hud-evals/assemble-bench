"""Isaac smoke: tip the peg ≥30° and check ExpertTakeover latches + moves.

Run (isaac env):
  cd assembly_bench && python scripts/experts/util/probe_tilt_takeover.py --headless
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

parser = get_isaaclab_arena_cli_parser()
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.num_envs = 1
args_cli.task = "peg_round_8mm"

with SimulationAppContext(args_cli):
    import torch
    import warp as wp
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    from assembly_bench.environments.assembly.assembly import (
        AssemblyBenchEnvironment,
        make_assembly_env,
    )

    # Build with takeover (both + tilt).
    env = make_assembly_env(
        task="peg_round_8mm",
        num_envs=1,
        expert_takeover=True,
        takeover_mode="both",
    )
    env.reset()
    base = env.unwrapped
    peg = base.scene["held_part"]
    N, dev = base.num_envs, base.device
    ids = torch.arange(N, device=dev)

    # Tip peg ~45° about x, slightly above table near hole (off stand).
    hole = wp.to_torch(base.scene["fixed_part"].data.root_pos_w) - base.scene.env_origins
    pos = hole.clone()
    pos[:, 0] += 0.02
    pos[:, 2] = 0.04
    a = math.radians(45.0) * 0.5
    # Isaac root quat wxyz
    quat = torch.tensor([[math.cos(a), math.sin(a), 0.0, 0.0]], device=dev).expand(N, 4)
    pose = torch.cat([pos + base.scene.env_origins, quat], dim=-1)
    peg.write_root_pose_to_sim(pose, env_ids=ids)
    peg.write_root_velocity_to_sim(torch.zeros((N, 6), device=dev), env_ids=ids)

    # Hold arm still, grip open — tilt gate does not require closed.
    q = wp.to_torch(base.scene["robot"].data.joint_pos)[:, :7].clone()
    act = torch.cat([q, torch.zeros((N, 1), device=dev)], dim=-1)

    latched_at = None
    for t in range(30):
        obs, *_ = env.step(act)
        ea = float(obs["policy"]["expert_active"][0, 0])
        if latched_at is None and ea > 0.5:
            latched_at = t
            print(f"[probe] latched at step {t}", flush=True)
        if latched_at is not None and t > latched_at + 20:
            break

    # Expert should be driving (executed_action differs from hold once latched).
    ok = latched_at is not None and latched_at <= 10  # debounce=5 + margin
    print(f"[probe] tilt-takeover {'PASS' if ok else 'FAIL'} latched_at={latched_at}", flush=True)
    env.close()
    raise SystemExit(0 if ok else 1)

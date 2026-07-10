"""HUD environment for the assembly benchmark — declarative, one file.

The gym capability, contract, and serving are derived by `env.gym(...)`;
this file builds one Arena scene and serves it. Serve (isaac6 env):

    OMNI_KIT_ACCEPT_EULA=YES python -m hud.environment.server env.py --port 8765

Scene config is fixed at process launch via env vars (defaults in parentheses):
``ASSEMBLY_TASK`` (peg_round_8mm_tight), ``ASSEMBLY_NUM_ENVS`` (1),
``ASSEMBLY_EMBODIMENT``, ``ASSEMBLY_REWARD`` (none). Episodic args (seed) go
through ``sim.reset``. Switching task/num_envs needs a new served process.
"""

import os

# Isaac must own the process main thread and be up before any isaaclab import.
from isaaclab.app import AppLauncher

_app = AppLauncher(headless=True, enable_cameras=True).app

from hud import Environment

from assembly_bench.environments.assembly.assembly import make_assembly_env

assembly_env = make_assembly_env(
    task=os.environ.get("ASSEMBLY_TASK", "peg_round_8mm_tight"),
    num_envs=int(os.environ.get("ASSEMBLY_NUM_ENVS", "1")),
    embodiment=os.environ.get("ASSEMBLY_EMBODIMENT", "droid_abs_joint_pos"),
    reward=os.environ.get("ASSEMBLY_REWARD", "none"),
)

env = Environment(name="assembly-bench")
sim = env.gym(assembly_env)


@env.template(id="assembly")
async def assembly(seed: int = 0):
    """One assembly episode on the built scene."""
    yield {"prompt": await sim.reset(seed=seed)}
    yield await sim.result()

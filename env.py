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
from functools import partial

# Isaac must own the process main thread and be up before any isaaclab import.
from isaaclab.app import AppLauncher

_app = AppLauncher(headless=True, enable_cameras=True).app

from hud import Environment

from assembly_bench.environments.assembly.assembly import make_assembly_env

make_env = partial(
    make_assembly_env,
    num_envs=int(os.environ.get("ASSEMBLY_NUM_ENVS", "1")),
    embodiment=os.environ.get("ASSEMBLY_EMBODIMENT", "droid_abs_joint_pos_softmimic"),
    reward=os.environ.get("ASSEMBLY_REWARD", "none"),
)

env = Environment(name="assembly-bench")
sim = env.gym(make_env)


@env.template(id="assembly")
async def assembly(
    task: str = "peg_round_8mm_tight",
    seed: int = 0,
    num_envs: int | None = None,
    embodiment: str | None = None,
    reward: str | None = None,
):
    """One assembly episode. Env-defining args rebuild the scene when they change."""
    # Only forward overrides; None keeps the process-launch defaults (ASSEMBLY_*).
    reset_kw = {"task": task, "seed": seed}
    if num_envs is not None:
        reset_kw["num_envs"] = num_envs
    if embodiment is not None:
        reset_kw["embodiment"] = embodiment
    if reward is not None:
        reset_kw["reward"] = reward
    yield {"prompt": await sim.reset(**reset_kw)}
    yield await sim.result()

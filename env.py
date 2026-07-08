"""HUD environment for the assembly benchmark — declarative, one file.

The gym capability, contract, and serving are derived by `env.gym(make_assembly_env)`;
this file only declares the sim factory and the task template. Serve (isaac6 env):

    OMNI_KIT_ACCEPT_EULA=YES python -m hud.environment.server env.py --port 8765

The server runs ALL 29 variants in sequence from one process: every factory
parameter is env-defining, so when a reset asks for a new task instance (or a
different num_envs / embodiment) the bridge closes the old sim env and rebuilds
it through the factory — the Isaac app itself stays up across the whole run.
"""

# Isaac must own the process main thread and be up before any isaaclab import.
from isaaclab.app import AppLauncher

_app = AppLauncher(headless=True, enable_cameras=True).app

from hud import Environment

from assembly_bench.environments.assembly.assembly import make_assembly_env
from assembly_bench.environments.assembly.variants import VARIANTS

env = Environment(name="assembly-bench")
sim = env.gym(make_assembly_env)


@env.template(id="assembly")
async def assembly(task: str = "peg_round_8mm_tight", seed: int = 0, num_envs: int = 1,
                   embodiment: str = "droid_abs_joint_pos"):
    """One assembly variant episode; `task` selects among the 29 variants and
    `num_envs` slots run as one vectorized batch (N graded traces)."""
    assert task in VARIANTS, f"unknown variant {task!r} (choose from {sorted(VARIANTS)})"
    yield {"prompt": await sim.reset(task=task, seed=seed, num_envs=num_envs,
                                     embodiment=embodiment)}
    yield await sim.result()

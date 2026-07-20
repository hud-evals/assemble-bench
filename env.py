"""HUD environment for the assembly benchmark — declarative, one file.

The gym capability, contract, and serving are derived by `env.gym(...)`;
this file declares the sim, and the sim child process runs it. Serve
(isaac6 env):

    OMNI_KIT_ACCEPT_EULA=YES python -m hud.environment.server env.py --port 8765

Scene config is the factory signature: ``task`` / ``num_envs`` / ``embodiment``
/ ``reward`` are build params (GymBridge rebuilds the scene when one changes),
with ``ASSEMBLY_*`` env vars as deploy-time defaults. Episodic args (seed) go
through ``sim.reset``.
"""

import os
from pathlib import Path

from hud import Environment

# One Kit app per process, booted on first env build. Isaac lives ONLY in the
# sim child (env.gym spawns `env.py:make_env` in its own process); the server
# process imports this module too and must stay light — no Isaac at import.
_app = None


def make_env(
    task: str = "peg_round_8mm_tight",
    num_envs: int = int(os.environ.get("ASSEMBLY_NUM_ENVS", "1")),
    embodiment: str = os.environ.get("ASSEMBLY_EMBODIMENT", "droid_abs_joint_pos_softmimic"),
    reward: str = os.environ.get("ASSEMBLY_REWARD", "none"),
):
    """Module-level factory the sim child re-imports by source path."""
    global _app
    if _app is None:
        # Isaac must be up before any isaaclab import; the sim child calls
        # this on its main thread, which Kit requires to own.
        from isaaclab.app import AppLauncher

        _app = AppLauncher(headless=True, enable_cameras=True).app

    from assembly_bench.environments.assembly.assembly import make_assembly_env

    return make_assembly_env(task=task, num_envs=num_envs, embodiment=embodiment, reward=reward)


env = Environment(name="assembly-bench")
# The authored contract by absolute path (a cwd-relative default would silently
# derive a fresh one); the task default as a build arg, so the scene built at
# start is the one the first episode uses — no throwaway Isaac rebuild.
sim = env.gym(
    make_env,
    contract=Path(__file__).parent / "contract.json",
    task="peg_round_8mm_tight",
)


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
    # reset → {prompt, token}; token scopes the episode for result().
    ep = await sim.reset(**reset_kw)
    yield {"prompt": ep["prompt"], "robot": {"token": ep["token"]}}
    yield await sim.result(token=ep["token"])

"""HUD environment for the assembly benchmark — declarative, one file.

The gym capability, contract, and serving are derived by `env.gym(...)`;
this file declares the sim, and the sim child process runs it. Serve
(isaac6 env):

    OMNI_KIT_ACCEPT_EULA=YES python -m hud.environment.server env.py --port 8765

Scene config is the factory signature: ``task`` / ``num_envs`` / ``embodiment``
/ ``reward`` are build params (GymBridge rebuilds the scene when one changes),
with ``ASSEMBLY_*`` env vars as deploy-time defaults. Optional CG-DAgger
(``expert_takeover``) wraps the bare Arena env via ``scripts/experts/rl``.
Episodic args (seed) go through ``sim.reset``.

Two agent surfaces share one sim process (see ``agents/``):
- ``openpi/0`` (``robot``) – VLA joint control via the ``assembly`` template
- ``mcp`` (``tools``) – LLM end-effector tools via the ``assembly_agent`` template
  (in development; not ready for use yet)
"""

import os
import sys
from pathlib import Path

from agents import AssemblyToolBridge
from agents.prompt import agent_prompt
from hud import Environment

# One Kit app per process, booted on first env build. Isaac lives ONLY in the
# sim child (env.gym spawns `env.py:make_env` in its own process); the server
# process imports this module too and must stay light — no Isaac at import.
_app = None

# Honor ASSEMBLY_TASK so docker eval warms the same variant the suite starts on.
_DEFAULT_TASK = os.environ.get("ASSEMBLY_TASK", "peg_round_8mm")

# LLM path always rebuilds on the differential-IK embodiment (7-D EEF deltas).
_EEF_EMBODIMENT = "droid_differential_ik"

# scripts/ on path so ``experts.rl.takeover`` imports cleanly.
_SCRIPTS = Path(__file__).resolve().parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def make_env(
    task: str = _DEFAULT_TASK,
    num_envs: int = int(os.environ.get("ASSEMBLY_NUM_ENVS", "1")),
    embodiment: str = os.environ.get("ASSEMBLY_EMBODIMENT", "droid_abs_joint_pos_softmimic"),
    reward: str = os.environ.get("ASSEMBLY_REWARD", "none"),
    expert_takeover: bool | str = _env_flag("ASSEMBLY_EXPERT_TAKEOVER"),
    takeover_mode: str | None = None,
):
    """Module-level factory the sim child re-imports by source path."""
    global _app
    if _app is None:
        # Isaac must be up before any isaaclab import; the sim child calls
        # this on its main thread, which Kit requires to own.
        from isaaclab.app import AppLauncher

        _app = AppLauncher(headless=True, enable_cameras=True).app

    from assembly_bench.environments.assembly.assembly import make_assembly_env

    # GymBridge may pass build args as strings from the wire.
    if isinstance(expert_takeover, str):
        expert_takeover = expert_takeover.strip().lower() in ("1", "true", "yes", "on")
    mode = takeover_mode or os.environ.get("ASSEMBLY_TAKEOVER_MODE", "grasp")

    # Bare Arena env; CG-DAgger is an optional outer wrap (not part of the package).
    env = make_assembly_env(
        task=task, num_envs=num_envs, embodiment=embodiment, reward=reward,
    )
    if expert_takeover:
        from experts.rl.takeover import ExpertTakeover

        env = ExpertTakeover(env, task=task, takeover_mode=str(mode))
        print(f"[env] expert_takeover ON for {task} (mode={mode})", flush=True)
    return env


env = Environment(name="assembly-bench")
# Tool bridge publishes both the openpi wire and the MCP tools from the sim
# process; env.gym publishes every capability the bridge declares.
# Docker image hud may predate bridge= (treats it as a JSON build default and
# crashes on ABCMeta) — only pass it when gym_command accepts the kwarg.
_gym_kw: dict = {
    "contract": str(Path(__file__).parent / "contract.json"),
    "task": _DEFAULT_TASK,
}
try:
    from hud.environment.robot.gym import gym_command as _gym_command
    import inspect as _inspect

    if "bridge" in _inspect.signature(_gym_command).parameters:
        _gym_kw["bridge"] = AssemblyToolBridge
except Exception:
    pass
sim = env.gym(make_env, **_gym_kw)


@env.template(id="assembly")
async def assembly(
    task: str = _DEFAULT_TASK,
    seed: int = 0,
    num_envs: int | None = None,
    embodiment: str | None = None,
    reward: str | None = None,
    expert_takeover: bool | None = None,
    takeover_mode: str | None = None,
):
    """One assembly episode for a VLA (joint-position openpi wire)."""
    # Only forward overrides; None keeps the process-launch defaults (ASSEMBLY_*).
    reset_kw = {"task": task, "seed": seed}
    if num_envs is not None:
        reset_kw["num_envs"] = num_envs
    if embodiment is not None:
        reset_kw["embodiment"] = embodiment
    if reward is not None:
        reset_kw["reward"] = reward
    if expert_takeover is not None:
        reset_kw["expert_takeover"] = expert_takeover
    if takeover_mode is not None:
        reset_kw["takeover_mode"] = takeover_mode
    # reset → {prompt, token}; token scopes the episode for result().
    ep = await sim.reset(**reset_kw)
    yield {"prompt": ep["prompt"], "robot": {"token": ep["token"]}}
    yield await sim.result(token=ep["token"])


@env.template(id="assembly_agent")
async def assembly_agent(
    task: str = _DEFAULT_TASK,
    seed: int = 0,
    guided: bool = True,
    episode_length_s: float = 80.0,
):
    """One assembly episode for an LLM (MCP end-effector tools).

    Default ``episode_length_s=80`` → ~1200 ticks @ 15 Hz (2× the VLA peg
    timeout). VLA ``assembly`` does not pass this, so peg variants stay at 40 s.
    """
    # Differential IK rebuilds the scene when the prior episode was joint-control.
    ep = await sim.reset(
        task=task,
        seed=seed,
        embodiment=_EEF_EMBODIMENT,
        guided=guided,
        episode_length_s=episode_length_s,
    )
    yield agent_prompt(ep["prompt"], guided=guided)
    yield await sim.result(token=ep["token"])

"""HUD environment for the assembly benchmark — declarative, one file.

The gym capability, contract, and serving are derived by `env.gym(...)`;
this file declares the sim, and the sim child process runs it. Serve
(isaac6 env):

    OMNI_KIT_ACCEPT_EULA=YES python -m hud.environment.server env.py --port 8765

Scene config is the factory signature: ``task`` / ``num_envs`` / ``embodiment``
/ ``reward`` / ``expert_takeover`` are build params (GymBridge rebuilds the
scene when one changes), with ``ASSEMBLY_*`` env vars as deploy-time defaults.
Episodic args (seed) go through ``sim.reset``.

Two agent surfaces share one sim process (see ``tool_bridge.py``):
- ``openpi/0`` (``robot``) — VLA joint control via the ``assembly`` template
- ``mcp`` (``tools``) — LLM end-effector tools via the ``assembly_agent`` template
"""

import os
from pathlib import Path

from hud import Environment
from tool_bridge import AssemblyToolBridge

# One Kit app per process, booted on first env build. Isaac lives ONLY in the
# sim child (env.gym spawns `env.py:make_env` in its own process); the server
# process imports this module too and must stay light — no Isaac at import.
_app = None

# Honor ASSEMBLY_TASK so docker eval warms the same variant the suite starts on.
_DEFAULT_TASK = os.environ.get("ASSEMBLY_TASK", "peg_round_8mm")

# LLM path always rebuilds on the differential-IK embodiment (7-D EEF deltas).
_EEF_EMBODIMENT = "droid_differential_ik"


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

    return make_assembly_env(
        task=task, num_envs=num_envs, embodiment=embodiment, reward=reward,
        expert_takeover=bool(expert_takeover),
        takeover_mode=str(mode),
    )


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


def _agent_prompt(instruction: str, *, guided: bool) -> str:
    """Prompt for the MCP tool surface: same task, fingertip waypoints instead of joints."""
    parts = (
        "You control a Franka arm with a Robotiq gripper over a tabletop assembly task.\n"
        f"Task: {instruction}\n\n"
        "Tools: look, move_to, grasp, release, get_state. Coordinates are millimetres in the "
        "robot base frame: +x away from the robot, +y to its left, +z up, z=0 at the tabletop. "
        "move_to aims the fingertip plane (not the flange). The motion stalls on contact — "
        "compare the reply pose to what you asked for; a gap means something is in the way.\n\n"
        "Typical plan: look → hover above the loose part → descend → grasp → lift clear of the "
        "stand → move above the target → lower to seat → release. Look after every move. "
        "Keep z above ~20 mm unless you are grasping or inserting. Budget is tight; avoid "
        "long open-loop drifts.\n"
    )
    if guided:
        parts += (
            "\nGuided mode: every tool reply includes the true loose-part and target poses "
            "in millimetres — use them; still look to confirm contact and grasp.\n"
        )
    else:
        parts += (
            "\nVision mode: part poses are NOT given. Localize from look(front) / look(wrist) "
            "and the fingertip pose in each reply.\n"
        )
    return parts


@env.template(id="assembly_agent")
async def assembly_agent(
    task: str = _DEFAULT_TASK,
    seed: int = 0,
    guided: bool = True,
):
    """One assembly episode for an LLM (MCP end-effector tools)."""
    # Differential IK rebuilds the scene when the prior episode was joint-control.
    ep = await sim.reset(
        task=task, seed=seed, embodiment=_EEF_EMBODIMENT, guided=guided,
    )
    yield _agent_prompt(ep["prompt"], guided=guided)
    yield await sim.result(token=ep["token"])

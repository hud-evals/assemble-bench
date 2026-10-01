"""HUD environment for AssembleBench — declarative, one file.

The gym capability, contract, and serving are derived by `env.gym(...)`;
this file declares the sim, and the sim child process runs it. Serve
(isaac6 env):

    OMNI_KIT_ACCEPT_EULA=YES python -m hud.environment.server env.py --port 8765

Scene config is the factory signature: ``task`` / ``num_envs`` / ``embodiment``
/ ``reward`` are build params (GymBridge rebuilds the scene when one changes),
with ``ASSEMBLY_*`` env vars as deploy-time defaults. Optional CG-DAgger
(``expert_takeover``) wraps the bare Arena env via ``scripts/experts/rl``.
Episodic args (seed) go through ``sim.reset``.

Two agent surfaces share one sim process:
- ``openpi/0`` (``robot``) – VLA joint control via the ``assembly`` template
- ``mcp`` (``control``) – LLM joint targets via ``assembly_direct`` (``move_joints``)

There is no end-effector absolute action. ``droid_differential_ik`` is 7-D
deltas, so it is not the direct-control embodiment. Fingertip tools in
``agents/`` are not published. Part poses and expert channels are absent from
``contract.json``, which is what the motion tool labels.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

from agents.peg_control import GRIPPER_MAX_STEP, NOTES, SETTLE_S, joint_reference
from hud import Environment
from hud.environment.robot import DirectControl

# One Kit app per process, booted on first env build. Isaac lives ONLY in the
# sim child (env.gym spawns `env.py:make_env` in its own process); the server
# process imports this module too and must stay light — no Isaac at import.
_app = None

# Honor ASSEMBLY_TASK so docker eval warms the same variant the suite starts on.
_DEFAULT_TASK = os.environ.get("ASSEMBLY_TASK", "peg_round_8mm")

# Direct control steps this embodiment: 8-D absolute joint targets.
_JOINT_EMBODIMENT = "droid_abs_joint_pos_softmimic"

# ~1000 control steps at 15 Hz for the LLM template. Peg variants stay at 40 s
# (600 ticks) unless a caller passes episode_length_s. ceil(s / step_dt) is 1001.
_DIRECT_EPISODE_S = 1000 / 15

# First scene build includes the PhysX step that can spin inside fabric with
# the GIL held. A Python timer never runs then; a side process kills the build.
_BUILD_BUDGET_S = 600.0

# scripts/ on path so ``experts.rl.takeover`` imports cleanly.
_SCRIPTS = Path(__file__).resolve().parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _arm_build_watchdog() -> Path:
    """SIGKILL this process if the scene build does not finish within the budget."""
    flag = Path(f"/tmp/assemble-bench-build-{time.time_ns()}")
    code = (
        "import os, signal, sys, time\n"
        "pid, budget, flag = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]\n"
        "time.sleep(budget)\n"
        "if not os.path.exists(flag):\n"
        "    print(f'[env] build exceeded {budget:.0f}s; killing {pid}', flush=True)\n"
        "    os.kill(pid, signal.SIGKILL)\n"
    )
    subprocess.Popen(
        [sys.executable, "-c", code, str(os.getpid()), str(_BUILD_BUDGET_S), str(flag)],
        start_new_session=True,
    )
    print(f"[env] build watchdog armed ({_BUILD_BUDGET_S:.0f}s)", flush=True)
    return flag


def _clear_build_watchdog(flag: Path) -> None:
    flag.write_text("ok")
    print("[env] build watchdog cleared", flush=True)


def make_env(
    task: str = _DEFAULT_TASK,
    num_envs: int = int(os.environ.get("ASSEMBLY_NUM_ENVS", "1")),
    embodiment: str = os.environ.get("ASSEMBLY_EMBODIMENT", "droid_abs_joint_pos_softmimic"),
    reward: str = os.environ.get("ASSEMBLY_REWARD", "none"),
    expert_takeover: bool | str = _env_flag("ASSEMBLY_EXPERT_TAKEOVER"),
    takeover_mode: str | None = None,
    episode_length_s: float | str | None = None,
):
    """Module-level factory the sim child re-imports by source path."""
    global _app
    if _app is None:
        # Isaac must be up before any isaaclab import; the sim child calls
        # this on its main thread, which Kit requires to own.
        from isaaclab.app import AppLauncher

        _app = AppLauncher(headless=True, enable_cameras=True).app

    from assemble_bench.environments.assembly.assembly import make_assembly_env

    # GymBridge may pass build args as strings from the wire.
    if isinstance(expert_takeover, str):
        expert_takeover = expert_takeover.strip().lower() in ("1", "true", "yes", "on")
    mode = takeover_mode or os.environ.get("ASSEMBLY_TAKEOVER_MODE", "grasp")

    # Bare Arena env; CG-DAgger is an optional outer wrap (not part of the package).
    length_s = None if episode_length_s in (None, "") else float(episode_length_s)
    watchdog = _arm_build_watchdog()
    try:
        env = make_assembly_env(
            task=task,
            num_envs=num_envs,
            embodiment=embodiment,
            reward=reward,
            episode_length_s=length_s,
        )
    finally:
        _clear_build_watchdog(watchdog)
    if expert_takeover:
        from experts.rl.takeover import ExpertTakeover

        env = ExpertTakeover(env, task=task, takeover_mode=str(mode))
        print(f"[env] expert_takeover ON for {task} (mode={mode})", flush=True)
    return env


env = Environment(name="assembly-bench")
sim = env.gym(
    make_env,
    contract=str(Path(__file__).parent / "contract.json"),
    task=_DEFAULT_TASK,
)
# Stock motion tool on the joint contract. reference= is the 8-D command
# (7 measured joints + gripper 0 open); auto-bind would need a matching obs.
DirectControl(
    notes=NOTES,
    max_step=GRIPPER_MAX_STEP,
    settle=SETTLE_S,
    reference=joint_reference,
).attach(sim)


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


@env.template(id="assembly_direct")
async def assembly_direct(task: str = _DEFAULT_TASK, seed: int = 0):
    """One assembly episode for an LLM (``move_joints`` on the sole slot).

    The reset claim is what the motion tool binds. Yielding a robot token
    would hand that slot to a policy client. The horizon is ~1000 control
    steps; the VLA ``assembly`` template keeps the variant default (pegs 40 s).
    """
    ep = await sim.reset(
        task=task,
        seed=seed,
        num_envs=1,
        embodiment=_JOINT_EMBODIMENT,
        reward="none",
        expert_takeover=False,
        episode_length_s=_DIRECT_EPISODE_S,
    )
    yield {"prompt": ep["prompt"]}
    yield await sim.result()

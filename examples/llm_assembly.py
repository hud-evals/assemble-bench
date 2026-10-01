"""Peg insertion through stock joint direct control, on Modal.

The env serves ``control`` / ``move_joints`` (8-D absolute joint targets).
There is no end-effector absolute action, so this is not ``move_to``. One
scripted call holds the gripper open, then ``EPISODES`` (default 1) of
``gpt-6-astra`` on ``peg_round_8mm`` at medium effort. Astra does not start
if that call errors. ``MAX_STEPS`` (default 20) is the tool-call budget.

Publish the image first (``modal run docker/modal_deploy.py``), then::

    python examples/llm_assembly.py

``ASSEMBLE_RUN_LOG``, when set, receives a ``modal-job:`` line per sandbox
and the HUD job URL.
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager

from hud.agents import create_agent
from hud.agents.base import Agent
from hud.eval import Task, Taskset
from hud.eval.runtime import ModalRuntime
from hud.eval.runtime.core import RuntimeConfig, RuntimeGPU, RuntimeLimits, RuntimeResources
from hud.eval.run import Run
from hud.settings import settings

IMAGE_NAME = "hud-assemble-bench-env"
PORT = 8765
WORKDIR = "/opt/assemble-bench"
COMMAND = (
    "python",
    "-m",
    "hud.environment.server",
    "env.py",
    "--host",
    "0.0.0.0",
    "--port",
    str(PORT),
)
TASK = os.environ.get("TASK", "peg_round_8mm")
EPISODES = int(os.environ.get("EPISODES", "1"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "20"))
MODEL = os.environ.get("HUD_LLM_MODEL", "gpt-6-astra")

# These must not appear in a move_joints result. The contract omits them.
_HIDDEN = (
    "policy/held_part_pose",
    "policy/fixed_part_pose",
    "policy/expert_active",
    "policy/executed_action",
)

SYSTEM_PROMPT = (
    "You control a Franka Panda arm with a Robotiq gripper in a tabletop assembly "
    "simulation through the move_joints tool. Targets are absolute joint radians. "
    "gripper.open_close is 0 open and 1 closed; values above 0.5 close. Each call "
    "plays a short motion and returns camera frames and joint state. "
    "camera_obs/front_cam_rgb sees the table; camera_obs/wrist_camera_rgb looks from "
    "the wrist. The first call has no prior frame: start from the task text, then "
    "re-check the returned frames after every move. Every call must include a note "
    "saying what you see and why you chose the motion. Part poses are not in the "
    "state. Reply without a tool call when the peg is seated, or when the episode "
    "says it has ended."
)


def _append_job(line: str) -> None:
    path = os.environ.get("ASSEMBLE_RUN_LOG")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line if line.endswith("\n") else line + "\n")


class _LoggedRuntime(ModalRuntime):
    @asynccontextmanager
    async def __call__(self, task: Task):
        async with super().__call__(task) as runtime:
            sandbox_id = (runtime.params or {}).get("instance_id")
            if sandbox_id:
                print(f"modal-job: {sandbox_id}", flush=True)
                _append_job(f"modal-job: {sandbox_id}")
            yield runtime


def _runtime() -> ModalRuntime:
    return _LoggedRuntime(
        IMAGE_NAME,
        command=COMMAND,
        workdir=WORKDIR,
        port=PORT,
        env_vars={
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ACCEPT_EULA": "Y",
            "PRIVACY_CONSENT": "Y",
            "OMNI_KIT_ALLOW_ROOT": "1",
            "NVIDIA_DRIVER_CAPABILITIES": "all",
        },
        runtime_config=RuntimeConfig(
            resources=RuntimeResources(
                cpu=8,
                memory_mb=65536,
                gpu=RuntimeGPU(type="L40S", count=1),
            ),
            limits=RuntimeLimits(startup_timeout_s=1800, run_timeout_s=7200),
        ),
    )


def _task(seed: int) -> Task:
    return Task(
        env="assembly-bench",
        id="assembly_direct",
        slug=f"{TASK}-seed{seed}",
        args={"task": TASK, "seed": seed},
    )


class HoldGripper(Agent):
    """One move_joints call: gripper stays open. Checks the result text."""

    async def __call__(self, run: Run) -> None:
        control = await run.client.open("control")
        result = await control.call_tool(
            "move_joints",
            {
                "targets": [{"name": "gripper.open_close", "value": 0.0}],
                "note": "Hold the gripper open and read the cameras before any arm motion.",
            },
        )
        texts = [block.text for block in result.content if getattr(block, "text", None)]
        text = "\n".join(texts)
        print(text[:800], flush=True)
        if result.isError:
            raise RuntimeError(f"move_joints failed: {text}")
        leaked = [key for key in _HIDDEN if key in text]
        if leaked:
            raise RuntimeError(f"tool result labeled privileged keys: {leaked}")
        print("privileged_keys_in_tool_result=no", flush=True)


def _print_job(label: str, job, started: float) -> None:
    elapsed = time.perf_counter() - started
    url = f"{settings.hud_web_url.rstrip('/')}/jobs/{job.id}"
    print(f"modal-job: {url}", flush=True)
    _append_job(f"modal-job: {url}")
    for run in job.runs:
        print(
            f"[{label}] reward={run.reward} success={run.evaluation.get('success')} "
            f"trace={run.trace_id} wall_s={elapsed:.1f} job={url}",
            flush=True,
        )


async def main() -> None:
    print(
        f"[llm] task={TASK} episodes={EPISODES} max_steps={MAX_STEPS} "
        f"model={MODEL} image={IMAGE_NAME} gpu=L40S",
        flush=True,
    )
    started = time.perf_counter()
    smoke = await Taskset(f"assemble-{TASK}-move_joints", [_task(0)]).run(
        HoldGripper(),
        runtime=_runtime(),
        max_concurrent=1,
    )
    _print_job("scripted", smoke, started)
    scripted = smoke.runs[0]
    if scripted.grade.is_error or EPISODES < 1:
        print("[llm] skipping astra", flush=True)
        return

    agent = create_agent(
        MODEL,
        system_prompt=SYSTEM_PROMPT,
        max_steps=MAX_STEPS,
        reasoning={"effort": "medium"},
    )
    tasks = [_task(seed) for seed in range(EPISODES)]
    started = time.perf_counter()
    job = await Taskset(f"{MODEL} x {TASK}", tasks).run(
        agent,
        runtime=_runtime(),
        max_concurrent=1,
    )
    _print_job("astra", job, started)


if __name__ == "__main__":
    asyncio.run(main())

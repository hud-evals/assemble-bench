"""Peg insertion by an LLM through ``move_joints``, on Modal or a local GPU.

The env serves one motion tool, ``move_joints`` (8-D absolute joint targets in
radians). A call plays the motion until the arm reaches the target or stops, then
returns both cameras and the joint state. Part poses are never returned.

A scripted ``move_joints`` call checks the stack first and the agent only starts if
it passes. ``RUNTIME`` picks where the env runs:

- ``modal`` (default): publish the image once (``modal run docker/modal_deploy.py``).
- ``tcp://127.0.0.1:8765``: attach to an env already served on a local GPU
  (see the README, "Run on a local GPU").

Then::

    python examples/llm_assembly.py

``TASK`` (default ``peg_round_8mm``), ``EPISODES`` (default 3; 0 runs only the scripted
check), ``MAX_STEPS`` (tool calls per episode, default 100) and ``HUD_LLM_MODEL``
(default ``gpt-6-astra``) override the defaults. An episode succeeds when the peg seats
before the 1000-tick (66.7 s) horizon.
"""

from __future__ import annotations

import asyncio
import os

from hud.agents import create_agent
from hud.agents.base import Agent
from hud.eval import Task, Taskset
from hud.eval.run import Run
from hud.eval.runtime import ModalRuntime, Runtime
from hud.eval.runtime.core import RuntimeConfig, RuntimeGPU, RuntimeLimits, RuntimeResources
from hud.settings import settings

IMAGE_NAME = "hud-assemble-bench-env"
PORT = 8765
RUNTIME = os.environ.get("RUNTIME", "modal")
TASK = os.environ.get("TASK", "peg_round_8mm")
EPISODES = int(os.environ.get("EPISODES", "3"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "100"))
MODEL = os.environ.get("HUD_LLM_MODEL", "gpt-6-astra")

# Keys the contract omits; none may appear in a tool result.
PRIVILEGED_KEYS = ("policy/held_part_pose", "policy/fixed_part_pose", "policy/expert_active")

SYSTEM_PROMPT = (
    "You control a Franka arm with a Robotiq gripper in a tabletop assembly simulation "
    "through move_joints. Each call moves to absolute joint targets, then returns both "
    "camera frames and the joint state. The first call has no prior frame: start from the "
    "task text, then re-check the frames after every move. Part poses are not in the state. "
    "Reply without a tool call when the peg is seated or the episode has ended."
)


def runtime() -> Runtime | ModalRuntime:
    if RUNTIME.startswith("tcp://"):
        return Runtime(RUNTIME)
    if RUNTIME != "modal":
        raise ValueError(f"RUNTIME must be 'modal' or a tcp:// url, got {RUNTIME!r}")
    return ModalRuntime(
        IMAGE_NAME,
        command=("python", "-m", "hud.environment.server", "env.py", "--host", "0.0.0.0", "--port", str(PORT)),
        workdir="/opt/assemble-bench",
        port=PORT,
        runtime_config=RuntimeConfig(
            resources=RuntimeResources(cpu=8, memory_mb=65536, gpu=RuntimeGPU(type="L40S", count=1)),
            limits=RuntimeLimits(startup_timeout_s=1800, run_timeout_s=7200),
        ),
    )


def task(seed: int) -> Task:
    return Task(env="assembly-bench", id="assembly_direct", slug=f"{TASK}-seed{seed}", args={"task": TASK, "seed": seed})


class HoldGripper(Agent):
    """One scripted ``move_joints`` call that keeps the gripper open."""

    async def __call__(self, run: Run) -> None:
        control = await run.client.open("control")
        result = await control.call_tool(
            "move_joints",
            {
                "target": {"name": "gripper.open_close", "value": 0.0},
                "others": [],
                "note": "Scripted check: hold the gripper open.",
            },
        )
        text = "\n".join(block.text for block in result.content if getattr(block, "text", None))
        print(text[:800], flush=True)
        if result.isError:
            raise RuntimeError(f"move_joints failed: {text}")
        if leaked := [key for key in PRIVILEGED_KEYS if key in text]:
            raise RuntimeError(f"tool result labeled privileged keys: {leaked}")


def report(label: str, job) -> None:
    web = settings.hud_web_url.rstrip("/")
    print(f"[{label}] job={web}/jobs/{job.id}", flush=True)
    for run in job.runs:
        print(
            f"[{label}] reward={run.reward} success={run.evaluation.get('success')} "
            f"trace={web}/trace/{run.trace_id}",
            flush=True,
        )
        steps = run.trace.steps
        calls = sum(len(getattr(step, "tool_calls", None) or []) for step in steps)
        print(
            f"[{label}] status={run.trace.status} stop_reason={run.trace.stop_reason} "
            f"steps={len(steps)} tool_calls={calls}",
            flush=True,
        )
        for step in steps:
            if step.error:
                print(f"[{label}] step {step.step_id} error: {step.error[:400]}", flush=True)
        if run.trace.content:
            print(f"[{label}] final: {run.trace.content[:400]}", flush=True)


async def main() -> None:
    check = await Taskset(f"assemble-{TASK}-move_joints", [task(0)]).run(HoldGripper(), runtime=runtime(), max_concurrent=1)
    report("scripted", check)
    scripted = check.runs[0]
    # Pre-launch failures set the trace status without grade.is_error.
    if scripted.trace.is_error or scripted.grade.is_error:
        print("[llm] scripted check failed; skipping the agent", flush=True)
        return
    if EPISODES < 1:
        print("[llm] scripted check passed; EPISODES=0, skipping the agent", flush=True)
        return

    agent = create_agent(MODEL, system_prompt=SYSTEM_PROMPT, max_steps=MAX_STEPS, reasoning={"effort": "medium"})
    job = await Taskset(f"{MODEL} x {TASK}", [task(seed) for seed in range(EPISODES)]).run(
        agent, runtime=runtime(), max_concurrent=1
    )
    report("agent", job)
    seated = sum(run.evaluation.get("success") is True for run in job.runs)
    print(f"[agent] success={seated}/{len(job.runs)}", flush=True)


if __name__ == "__main__":
    if os.environ.get("VERBOSE"):
        import logging

        logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())

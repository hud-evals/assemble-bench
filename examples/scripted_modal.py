"""One peg episode on Modal: hold the measured joints, gripper open.

The policy reads ``policy/joint_pos`` only. Part poses are not in the
contract. This runner uses the robot wire; ``examples/llm_assembly.py`` is
the ``move_joints`` path.

Publish the image first (``modal run docker/modal_deploy.py``), then::

    python examples/scripted_modal.py

``TASK`` defaults to ``peg_round_8mm``. ``STEPS`` is the control-tick cap
(default 100). The sim's own peg timeout is 40 s; this stops earlier.
``ASSEMBLE_RUN_LOG``, when set, receives a ``modal-job:`` line.
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np

from hud.agents.robot.agent import RobotAgent
from hud.agents.robot.model import Model
from hud.eval import Task, Taskset
from hud.eval.runtime import ModalRuntime
from hud.eval.runtime.core import RuntimeConfig, RuntimeGPU, RuntimeLimits, RuntimeResources
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


class HoldJoints(Model):
    """Repeat the measured arm pose and command the gripper open."""

    def infer(self, batch: Any) -> np.ndarray:
        data = batch["data"]
        joints = np.asarray(data["policy/joint_pos"], dtype=np.float32).reshape(-1)
        if joints.size < 7:
            raise ValueError(f"policy/joint_pos needs 7 values, got {joints.shape}")
        action = np.concatenate([joints[:7], np.zeros(1, dtype=np.float32)])
        return action.reshape(1, 1, -1)


class HoldAgent(RobotAgent):
    max_steps = int(os.environ.get("STEPS", "100"))
    log_every = 20

    def __init__(self) -> None:
        self.model = HoldJoints()
        self.adapter = None


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
            "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
            "VK_DRIVER_FILES": "/etc/vulkan/icd.d/nvidia_icd.json",
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


async def main() -> None:
    task_id = os.environ.get("TASK", "peg_round_8mm")
    agent = HoldAgent()
    task = Task(
        env="assembly-bench",
        id="assembly",
        slug=task_id,
        args={"task": task_id, "seed": 0, "num_envs": 1},
    )
    print(
        f"[episode] {task_id} hold-joints steps={agent.max_steps} image={IMAGE_NAME} gpu=L40S",
        flush=True,
    )
    started = time.perf_counter()
    job = await Taskset(f"assemble-{task_id}", [task]).run(agent, runtime=_runtime(), max_concurrent=1)
    elapsed = time.perf_counter() - started
    url = f"{settings.hud_web_url.rstrip('/')}/jobs/{job.id}"
    print(f"modal-job: {url}", flush=True)
    _append_job(f"modal-job: {url}")
    rewards = [run.reward for run in job.runs]
    print(f"[episode] reward={rewards} wall_s={elapsed:.1f} job={url}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

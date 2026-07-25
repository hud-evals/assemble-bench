"""Smoke-test the assembly env through HUD across placements.

One short zero-action episode (peg_round_8mm) per selected runtime:

    python scripts/smoke_runtimes.py local    # LocalRuntime spawns env.py under the isaac6 python
    python scripts/smoke_runtimes.py docker   # DockerRuntime boots hud-assembly-env with --gpus
    python scripts/smoke_runtimes.py modal    # ModalRuntime sandbox (needs a Modal token + pushed image)

Run from the agent conda env (only needs hud + numpy).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # the workspace root
sys.path.insert(0, str(ROOT))

from inventory.agents.zero_action import ZeroActionAgent

from hud import Task, Taskset

ISAAC_PY = "~/miniconda3/envs/isaac6/bin/python"
IMAGE = "hud-assembly-env"


def make_runtime(kind: str):
    from hud.eval.runtime import DockerRuntime, LocalRuntime, ModalRuntime

    if kind == "local":
        return LocalRuntime(ROOT / "assembly_bench/env.py", python=ISAAC_PY, ready_timeout=900)
    if kind == "docker":
        return DockerRuntime(
            IMAGE,
            run_args=["--gpus", "all", "-e", "NVIDIA_DRIVER_CAPABILITIES=all"],
            ready_timeout=900,  # cold Isaac boot inside the container
        )
    # Build cloud-side from the same Dockerfile (nothing local to push); the
    # ignore list mirrors docker/Dockerfile.dockerignore.
    import modal

    image = modal.Image.from_dockerfile(
        ROOT / "assembly_bench/docker/Dockerfile",
        context_dir=ROOT,
        ignore=[
            "*",
            "!assembly_bench/**",
            "!hud-python/**",
            "!env/IsaacLab-Arena/**",
            "assembly_bench/submodules/**",
            "assembly_bench/eval/**",
            "assembly_bench/notes/**",
            "env/IsaacLab-Arena/submodules/Isaac-GR00T/**",
            "env/IsaacLab-Arena/docs/**",
            "**/__pycache__/**",
            "**/.git/**",
            "**/.venv/**",
            "hud-python/docs/**",
        ],
    )
    return ModalRuntime(
        image=image,
        command=["/isaac-sim/python.sh", "-m", "hud.environment.server",
                 "env.py", "--host", "0.0.0.0", "--port", "8765"],
        workdir="/app/assembly_bench",
        runtime_config={
            "resources": {"gpu": {"type": "L40S", "count": 1}},
            "limits": {"startup_timeout_s": 1800, "run_timeout_s": 3600},
        },
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime", choices=["local", "docker", "modal"])
    parser.add_argument("--task", default="peg_round_8mm")
    args = parser.parse_args()

    task = Task(env="assembly-bench", id="assembly", slug=args.task,
                args={"task": args.task, "seed": 0})
    start = time.time()
    job = await Taskset(f"smoke-{args.runtime}", [task]).run(
        ZeroActionAgent(), runtime=make_runtime(args.runtime), max_concurrent=1
    )
    run = job.runs[0]
    print(f"\n[smoke:{args.runtime}] status={run.trace.status} reward={run.reward:.2f} "
          f"runtime={run.runtime} elapsed={time.time() - start:.0f}s")
    if run.trace.status != "completed":
        raise SystemExit(f"[smoke:{args.runtime}] FAILED")
    print(f"[smoke:{args.runtime}] OK")


if __name__ == "__main__":
    asyncio.run(main())

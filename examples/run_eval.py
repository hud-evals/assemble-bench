"""Evaluate a pi0.5 checkpoint on AssembleBench through HUD.

Serve the environment first (repo README, Path B), then run this from the repo root::

    OMNI_KIT_ACCEPT_EULA=YES python -m hud.environment.server env.py --port 8765
    python examples/run_eval.py --task peg_round_16mm --num-envs 4

``--num-envs`` is the Isaac vectorization width: one sim process, N parallel
episodes, N traces. The writeup's protocol is 30 episodes per task
(``--num-envs 15 --waves 2``).

Set ``HUD_API_KEY`` to stream traces to hud.ai; without it everything runs and
grades locally.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pi05_agent import DEFAULT_CHECKPOINT, PI05AssemblyAgent

from hud import Job, Runtime, Task, Taskset
from hud.agents.robot import BatchedAgent
from hud.eval.runtime import Shared
from hud.settings import settings


def _reward(run) -> float:
    graded = getattr(run, "evaluation", None) or {}
    if isinstance(graded, dict) and "reward" in graded:
        return float(graded["reward"])
    return float(getattr(run, "reward", 0.0) or 0.0)


def _success(run) -> bool:
    graded = getattr(run, "evaluation", None) or {}
    return bool(graded.get("success", False)) if isinstance(graded, dict) else False


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="peg_round_16mm", help="benchmark task id")
    p.add_argument("--num-envs", type=int, default=4, help="parallel episodes per wave")
    p.add_argument("--waves", type=int, default=1, help="sequential batches of --num-envs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runtime", default="tcp://127.0.0.1:8765", help="served env url")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="HF repo id or local dir")
    p.add_argument("--max-steps", type=int, default=None, help="override the family step cap")
    p.add_argument("--job-name", default=None)
    args = p.parse_args()

    n = args.num_envs
    job_name = args.job_name or f"pi05-{args.task}"
    # settings also reads ~/.hud/.env, so this is not just os.environ.
    streaming = bool(settings.api_key)

    policy = PI05AssemblyAgent(args.checkpoint, step_cap=args.max_steps)
    agent = BatchedAgent(policy, batch_size=n)

    job = await Job.start(job_name, group=n)
    where = (
        f"{settings.hud_web_url.rstrip('/')}/jobs/{job.id}"
        if streaming
        else "HUD_API_KEY unset — grading locally, no traces uploaded"
    )
    print(f"[eval] {n * args.waves} episodes on {args.task} — {where}", flush=True)

    # One shared sim connection, fanned out to n concurrent rollouts.
    shared = Shared(Runtime(args.runtime), width=n)
    for wave in range(args.waves):
        task = Task(
            env="assembly-bench",
            id="assembly",
            slug=args.task,
            args={"task": args.task, "seed": args.seed + wave, "num_envs": n},
        )
        done = len(job.runs)
        await Taskset(f"{job_name}-w{wave}", [task]).run(
            agent, runtime=shared, group=1, max_concurrent=n, job=job
        )
        batch = job.runs[done:]
        print(f"[eval] wave {wave}: {sum(map(_success, batch))}/{len(batch)} seated", flush=True)

    wins = sum(map(_success, job.runs))
    rewards = [_reward(r) for r in job.runs]
    mean = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"\n[eval] {args.task}: {wins}/{len(job.runs)} succeeded, mean reward {mean:.1f}")
    if streaming:
        print(f"[eval] traces: {settings.hud_web_url.rstrip('/')}/jobs/{job.id}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

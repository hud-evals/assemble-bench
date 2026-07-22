"""Eval Cosmos3-Policy-DROID against a running assembly env.

Requires a cosmos-framework policy server::

    python -m cosmos_framework.scripts.action_policy_server_robolab --port 8000

Then::

    COSMOS_HOST=127.0.0.1 COSMOS_PORT=8000 \
    RUNTIME=tcp://127.0.0.1:8765 \
    conda run -n vla python scripts/eval_cosmos.py --tasks eval/debug.json --group 1
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hud.eval import Runtime, Taskset  # noqa: E402
from hud.eval.job import Job  # noqa: E402
from hud.settings import settings  # noqa: E402

from cosmos_droid import Agent, CosmosDroidAgent  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=os.path.join(ROOT, "eval", "debug.json"))
    ap.add_argument("--runtime", default=os.environ.get("RUNTIME", "tcp://127.0.0.1:8765"))
    ap.add_argument("--group", type=int, default=1, help="repeats per task")
    ap.add_argument(
        "--max_concurrent",
        type=int,
        default=1,
        help="parallel rollouts (keep 1 — RemoteModel is not batchable)",
    )
    ap.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="cap control ticks (default: RobotAgent.max_steps=520)",
    )
    ap.add_argument(
        "--name",
        default=os.environ.get("JOB_NAME", "cosmos-droid-assembly-debug"),
        help="HUD job name shown on the platform",
    )
    a = ap.parse_args()

    ts = Taskset.from_file(a.tasks)
    if a.max_steps is not None:
        CosmosDroidAgent.max_steps = a.max_steps
    agent = Agent()
    print(
        f"[eval] cosmos host={os.environ.get('COSMOS_HOST', 'localhost')}:"
        f"{os.environ.get('COSMOS_PORT', '8000')}  "
        f"{len(list(ts))} tasks × group={a.group} @ {a.runtime}",
        flush=True,
    )
    print(
        f"[eval] telemetry={'on' if (settings.telemetry_enabled and settings.api_key) else 'OFF'} "
        f"web={settings.hud_web_url}",
        flush=True,
    )
    job = await Job.start(a.name, group=a.group)
    print(f"[eval] streaming → {settings.hud_web_url}/jobs/{job.id}", flush=True)
    job = await ts.run(
        agent,
        runtime=Runtime(a.runtime),
        group=a.group,
        max_concurrent=a.max_concurrent,
        job=job,
    )
    by: dict[str, list[float]] = {}
    for run in job.runs:
        slug = run.slug or "?"
        r = float(run.reward)
        by.setdefault(slug, []).append(r)
        print(f"  {slug}: reward={r}", flush=True)
    print("[eval] summary:", flush=True)
    for slug, rs in by.items():
        ok = sum(1 for x in rs if x == 1.0 or x > 0.5)
        print(f"  {slug}: {ok}/{len(rs)}  mean={sum(rs) / len(rs):.3f}", flush=True)
    print(f"[eval] job={job.id} url={settings.hud_web_url}/jobs/{job.id}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

"""Eval a merged assembly SFT checkpoint on the canonical 4-task set.

    CHECKPOINT=~/checkpoints/pi05_assembly_bench_2_sft_20k \
    RUNTIME=tcp://127.0.0.1:32768 \
    conda run -n vla python scripts/eval_sft.py --group 8
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

# scripts/ is not a package; allow importing the agent beside us.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hud.eval import Runtime, Taskset  # noqa: E402

from pi05_assembly_sft import Agent  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=os.path.join(ROOT, "eval", "canonical4.json"))
    ap.add_argument("--runtime", default=os.environ.get("RUNTIME", "tcp://127.0.0.1:32768"))
    ap.add_argument("--group", type=int, default=8, help="repeats per task")
    ap.add_argument("--max_concurrent", type=int, default=8,
                    help="parallel rollouts (match group / ASSEMBLY_NUM_ENVS for 8-env baseline)")
    a = ap.parse_args()

    ts = Taskset.from_file(a.tasks)
    agent = Agent()
    print(f"[eval] {len(list(ts))} tasks × group={a.group} @ {a.runtime}", flush=True)
    job = await ts.run(
        agent,
        runtime=Runtime(a.runtime),
        group=a.group,
        max_concurrent=a.max_concurrent,
    )
    # Summarize rewards per task slug.
    by: dict[str, list[float]] = {}
    for run in job.runs:
        slug = run.slug or "?"
        r = float(run.reward)
        by.setdefault(slug, []).append(r)
        print(f"  {slug}: reward={r}", flush=True)
    print("[eval] summary:", flush=True)
    for slug, rs in by.items():
        ok = sum(1 for x in rs if x == 1.0 or x > 0.5)
        print(f"  {slug}: {ok}/{len(rs)}  mean={sum(rs)/len(rs):.3f}", flush=True)
    print(f"[eval] job={job.id} url={getattr(job, 'url', None)}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

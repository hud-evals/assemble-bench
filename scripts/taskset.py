"""Regenerate HUD run suites under ``tasks/`` from ``variants.py``.

Source of truth for *what exists* is ``environments/assembly/variants.py``.
These JSON files are only *which rows to run* for ``hud eval``.

    python scripts/taskset.py

Then run a VLA via ``examples/run_eval.py`` (see the repo README). The LLM tool
path under ``tasks/agent/`` is in development.
"""

from __future__ import annotations

import json
from pathlib import Path

from assemble_bench.environments.assembly.variants import VARIANTS

ROOT = Path(__file__).resolve().parents[1]
ENV = "assembly-bench"

# Small first-run subset: one variant per NIST family (excludes smoke debug).
SMOKE = ("peg_round_8mm", "gear_medium", "nut_M16")


def _write(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} tasks -> {path}")
    return path


def _vla_row(name: str, seed: int = 0) -> dict:
    return {
        "env": ENV,
        "id": "assembly",
        "slug": name,
        "args": {"task": name, "seed": seed},
    }


def export_vla(*, seed: int = 0) -> None:
    """VLA template ``assembly`` — joint-position openpi wire."""
    _write(ROOT / "tasks" / "vla" / "all.json",
           [_vla_row(name, seed) for name in sorted(VARIANTS)])
    _write(ROOT / "tasks" / "vla" / "smoke.json",
           [_vla_row(name, seed) for name in SMOKE if name in VARIANTS])
    # Hello-world pick-place (not NIST).
    if "debug" in VARIANTS:
        _write(ROOT / "tasks" / "vla" / "debug.json", [_vla_row("debug", seed)])


def export_agent(*, seed: int = 0) -> None:
    """MCP template ``assembly_agent`` — EE tools; guided vs vision-only."""
    # Peg smokes with both prompt modes (see env.py assembly_agent).
    pegs = ("peg_round_8mm", "peg_round_4mm")
    rows = []
    for task in pegs:
        if task not in VARIANTS:
            continue
        for guided, tag in ((True, "guided"), (False, "vision")):
            rows.append(
                {
                    "env": ENV,
                    "id": "assembly_agent",
                    "slug": f"{task}_{tag}",
                    "args": {"task": task, "seed": seed, "guided": guided},
                }
            )
    _write(ROOT / "tasks" / "agent" / "pegs.json", rows)


if __name__ == "__main__":
    export_vla()
    export_agent()

"""Regenerate HUD run suites under ``tasks/`` from ``variants.py``.

Source of truth for *what exists* is ``environments/assembly/variants.py``.
These JSON files are only *which rows to run* for ``hud eval``.

    python scripts/taskset.py

Then run a VLA via ``examples/run_eval.py`` or an LLM via ``examples/llm_assembly.py``
(see the repo README).
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


def export_llm(*, seed: int = 0) -> None:
    """LLM template ``assembly_direct`` — ``move_joints`` on the peg variants."""
    _write(ROOT / "tasks" / "llm" / "pegs.json",
           [
               {
                   "env": ENV,
                   "id": "assembly_direct",
                   "slug": name,
                   "args": {"task": name, "seed": seed},
               }
               for name in sorted(VARIANTS)
               if name.startswith("peg_")
           ])


if __name__ == "__main__":
    export_vla()
    export_llm()

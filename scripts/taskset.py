"""Export the full assembly benchmark taskset to ``tasks.json``.

One row per variant, generated from the pure-data manifest so the JSON can
never drift from ``variants.py``. Regenerate::

    python taskset.py

The JSON is import-free, so ``hud eval`` can load it from any conda env while
the sim serves from ``isaac6`` (``--full`` runs all 16 variants)::

    hud eval assembly_bench/tasks.json inventory/agents/pi05_droid.py \\
        --full --runtime tcp://127.0.0.1:8765
"""

from __future__ import annotations

import json
from pathlib import Path

from assembly_bench.environments.assembly.variants import VARIANTS

ENV = "assembly-bench"
TEMPLATE = "assembly"
OUTPUT = Path(__file__).resolve().parents[1] / "tasks.json"


def export(path: Path = OUTPUT, seed: int = 0) -> Path:
    rows = [
        {"env": ENV, "id": TEMPLATE, "slug": name, "args": {"task": name, "seed": seed}}
        for name in sorted(VARIANTS)
    ]
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} tasks -> {path}")
    return path


if __name__ == "__main__":
    export()

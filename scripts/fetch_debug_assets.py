"""Fetch RoboLab apple/bowl USDs into ``assembly_bench/assets/parts/debug/``.

The debug variant is gitignored (local sanity-check only). Run once after clone::

    python scripts/fetch_debug_assets.py
    # or point at an existing RoboLab checkout:
    ROBOLAB_ROOT=/path/to/RoboLab python scripts/fetch_debug_assets.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "assembly_bench" / "assets" / "parts" / "debug"
_CANDIDATES = [
    Path(os.environ["ROBOLAB_ROOT"]) if "ROBOLAB_ROOT" in os.environ else None,
    ROOT.parent / "RoboLab",  # sibling: project/RoboLab
]


def _find_robolab() -> Path:
    for c in _CANDIDATES:
        if c is not None and (c / "assets" / "objects").is_dir():
            return c
    raise SystemExit(
        "RoboLab checkout not found. Clone https://github.com/NVlabs/RoboLab "
        "and set ROBOLAB_ROOT, or place it at project/RoboLab."
    )


def _lfs_pull(repo: Path, include: str) -> None:
    subprocess.run(
        ["git", "lfs", "pull", f"--include={include}"],
        cwd=repo,
        check=False,
    )


def main() -> None:
    repo = _find_robolab()
    print(f"[fetch] RoboLab={repo}")
    include = ",".join(
        [
            "assets/objects/objaverse/apple_01.usd",
            "assets/objects/objaverse/textures/apple_01.jpg",
            "assets/objects/ycb/bowl.usd",
            "assets/objects/ycb/textures/obj_000013.png",
        ]
    )
    _lfs_pull(repo, include)

    src_apple = repo / "assets/objects/objaverse/apple_01.usd"
    src_apple_tex = repo / "assets/objects/objaverse/textures/apple_01.jpg"
    src_bowl = repo / "assets/objects/ycb/bowl.usd"
    src_bowl_tex = repo / "assets/objects/ycb/textures/obj_000013.png"
    for p in (src_apple, src_apple_tex, src_bowl, src_bowl_tex):
        if not p.is_file() or p.stat().st_size < 1024:
            raise SystemExit(f"missing/incomplete LFS file: {p} (run git lfs pull)")

    DST.mkdir(parents=True, exist_ok=True)
    (DST / "textures").mkdir(exist_ok=True)
    shutil.copy2(src_apple, DST / "apple_01.usd")
    shutil.copy2(src_apple_tex, DST / "textures" / "apple_01.jpg")
    shutil.copy2(src_bowl, DST / "bowl.usd")
    shutil.copy2(src_bowl_tex, DST / "textures" / "obj_000013.png")
    print(f"[fetch] wrote {DST}")
    for p in sorted(DST.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(DST)}  ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
    sys.exit(0)

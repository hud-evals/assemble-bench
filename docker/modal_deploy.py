"""Build the AssembleBench image on Modal from the NGC Isaac Sim base and publish it.

The pip ``isaacsim`` builds (6.0.0.1, 6.1.0.0) render an empty tiled camera
buffer (Warp CUDA error 700) on this stack; the NGC ``isaac-sim:6.0.0-dev2``
base does not. So this image is ``docker/Dockerfile`` replayed on top of the NGC
base. ``nvcr.io`` needs a login, supplied as a Modal secret holding
``REGISTRY_USERNAME`` / ``REGISTRY_PASSWORD`` (NGC username ``$oauthtoken`` and
an NGC API key)::

    modal secret create ngc-registry REGISTRY_USERNAME='$oauthtoken' REGISTRY_PASSWORD=...
    modal run docker/modal_deploy.py

Then the LLM run (L40S; A100 and H100 cannot render)::

    RUNTIME=modal python examples/llm_assembly.py

Requires Modal credentials (``modal token new`` or ``MODAL_TOKEN_ID`` /
``MODAL_TOKEN_SECRET``). Override the secret name with ``NGC_SECRET``.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

IMAGE_NAME = "hud-assemble-bench-env"
APP_NAME = "hud-envs"
PORT = 8765
NGC_BASE = "nvcr.io/nvidia/isaac-sim:6.0.0-dev2"
NGC_SECRET = os.environ.get("NGC_SECRET", "ngc-registry")
REPO_ROOT = Path(__file__).resolve().parents[1]

# RTX-class. A100 / H100 cannot render Isaac Sim 6.
GPU = "L40S"

_TOP = {"assemble_bench", "scripts", "env.py", "direct_control.py", "contract.json", "submodules"}
_SKIP = {".git", "__pycache__", ".venv", "Isaac-GR00T", "docs"}


def _ignore(path: Path) -> bool:
    parts = path.parts
    if parts[0] not in _TOP or any(part in _SKIP for part in parts):
        return True
    return parts[0] == "submodules" and len(parts) > 1 and parts[1] != "IsaacLab-Arena"


def _dockerfile_steps() -> list[str]:
    """``docker/Dockerfile`` minus the base-image lines (Modal supplies the base)."""
    steps = []
    for line in (REPO_ROOT / "docker" / "Dockerfile").read_text().splitlines():
        if line.startswith(("FROM ", "ARG BASE_IMAGE")):
            continue
        steps.append(line)
    return steps


image = modal.Image.from_registry(
    NGC_BASE, secret=modal.Secret.from_name(NGC_SECRET)
).dockerfile_commands(_dockerfile_steps(), context_dir=REPO_ROOT, ignore=_ignore)

app = modal.App(APP_NAME)


@app.local_entrypoint()
def main() -> None:
    """Build the image and publish it under ``IMAGE_NAME``."""
    sb_app = modal.App.lookup(APP_NAME, create_if_missing=True)
    image.build(app=sb_app)
    image.publish(IMAGE_NAME)
    print(f"published image: {IMAGE_NAME}  ->  ModalRuntime({IMAGE_NAME!r}, gpu={GPU})")

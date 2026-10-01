"""Build the AssembleBench image on Modal and publish it by name.

``nvcr.io/nvidia/isaac-sim`` returns 401 without an NGC login, so this image
follows the README's host install: public ``isaacsim[all,extscache]==6.0.0.1``
on a CUDA image, then ``docker/modal_image.sh``. The NGC Dockerfile is unchanged.

    modal run docker/modal_deploy.py

Then a hold-joint episode, or the LLM smoke (L40S; A100 and H100 cannot render)::

    python examples/scripted_modal.py
    python examples/llm_assembly.py

Requires ``MODAL_TOKEN_ID`` and ``MODAL_TOKEN_SECRET``.
"""

from __future__ import annotations

from pathlib import Path

import modal

IMAGE_NAME = "hud-assemble-bench-env"
APP_NAME = "hud-envs"
PORT = 8765
REPO_ROOT = Path(__file__).resolve().parents[1]
ISAACSIM = "isaacsim[all,extscache]==6.0.0.1"

# RTX-class. A100 / H100 cannot render Isaac Sim 6.
GPU = "L40S"

_SKIP = {".git", "__pycache__", ".venv", "Isaac-GR00T"}


def _ignore(path: Path) -> bool:
    return any(part in _SKIP for part in path.parts)


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-runtime-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install(
        "git",
        "cmake",
        "build-essential",
        "ffmpeg",
        "libglu1-mesa",
        "libgl1",
        "libglib2.0-0",
        "libvulkan1",
    )
    .pip_install(ISAACSIM, extra_index_url="https://pypi.nvidia.com")
    .env(
        {
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ACCEPT_EULA": "Y",
            "PRIVACY_CONSENT": "Y",
            "OMNI_KIT_ALLOW_ROOT": "1",
            "NVIDIA_DRIVER_CAPABILITIES": "all",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_dir(REPO_ROOT, remote_path="/opt/assemble-bench", copy=True, ignore=_ignore)
    .run_commands("bash /opt/assemble-bench/docker/modal_image.sh")
)

app = modal.App(APP_NAME)


@app.local_entrypoint()
def main() -> None:
    """Build the image and publish it under ``IMAGE_NAME``."""
    sb_app = modal.App.lookup(APP_NAME, create_if_missing=True)
    image.build(app=sb_app)
    image.publish(IMAGE_NAME)
    print(f"published image: {IMAGE_NAME}  ->  ModalRuntime({IMAGE_NAME!r}, gpu={GPU})")

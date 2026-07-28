# Assembly bench in Docker

Containerizes the bench's `env.py` behind the standard HUD serving entry point.
One port (8765, the control channel) is all the host needs: the robot WebSocket
binds container-loopback and rides the control channel's capability tunnel.

**GPU required.** Isaac renders with RTX; run with `--gpus all`.

## Build

One self-contained image from the Isaac Sim base — Isaac Lab (Arena's pinned
checkout), the isaaclab_arena runtime, hud-python, and this bench. Arena's dev
stack (Isaac-GR00T submodule, isaacteleop, gh tooling) is deliberately left out.

```bash
# context = workspace root with hud-python/, assembly_bench/, env/IsaacLab-Arena/
docker build -f assembly_bench/docker/Dockerfile -t hud-assembly-env .
```

hud-python installs into kit's python with a constraints freeze (shadow-install
the few packages hud needs newer, never upgrade kit-owned ones).

## Run

```bash
docker run -d --name assembly-env --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --publish 127.0.0.1::8765 \
  hud-assembly-env
```

Boot takes minutes; watch `docker logs -f assembly-env` for `HUD_SERVE_PORT=8765`.

## Eval against it

Attach to the running container (`docker port assembly-env 8765` gives the host port):

```bash
hud eval assembly_bench/tasks/vla/all.json inventory/agents/pi05_droid.py \
    --full --runtime tcp://127.0.0.1:<host-port> --num-envs 4
```

Or let a provider own the lifecycle — fresh container per rollout
(pays the Isaac boot each task; fine for smoke tests, not sweeps):

```python
from hud.eval.runtime import DockerRuntime, ModalRuntime

DockerRuntime("hud-assembly-env", run_args=["--gpus", "all", "-e", "NVIDIA_DRIVER_CAPABILITIES=all"])
ModalRuntime("hud-assembly-env", runtime_config={"resources": {"gpu": {"type": "L40S", "count": 1}},
                                                 "limits": {"startup_timeout_s": 1200}})
```

ModalRuntime builds the image cloud-side from this Dockerfile
(`modal.Image.from_dockerfile`, see `scripts/smoke_runtimes.py`) — but Isaac
Sim **cannot currently run on Modal**: their sandboxes execute under gVisor,
whose GPU proxy does not implement the CUDA<->graphics interop Isaac needs
(`cudaErrorNotSupported` from carb/physx at boot, even with
`NVIDIA_DRIVER_CAPABILITIES=all`). Use raw-Docker GPU hosts for cloud fan-out.

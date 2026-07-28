# Assembly Bench in Docker

Serves `env.py` behind the HUD entry point. One published port (8765) is enough —
the robot WebSocket rides the control channel's capability tunnel.

**GPU required.** Isaac renders with RTX; run with `--gpus all`.

## Build

Build from the **assembly_bench repo root** (no sibling repos needed). Initialize
Arena + its nested IsaacLab first so the `COPY` paths exist:

```bash
# once, from the repo root:
git submodule update --init submodules/IsaacLab-Arena
git -C submodules/IsaacLab-Arena config url."https://github.com/".insteadOf "git@github.com:"
git -C submodules/IsaacLab-Arena submodule update --init submodules/IsaacLab

docker build -f docker/Dockerfile -t hud-assembly-env .
```

Or just run `./scripts/setup_sim.sh` on a host with Isaac (it fetches the same
submodules), then build.

The image installs `hud-python` from PyPI into kit's Python (constraint-frozen so
kit-owned packages are never upgraded).

## Run

First boot downloads Omniverse assets and can take **5–15 minutes**. Mount caches
so restarts are ~1–2 minutes:

```bash
mkdir -p ~/.cache/isaac/{kit,ov,glcache,computecache,warp,logs}

docker run -d --name assembly-env --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e OMNI_KIT_ACCEPT_EULA=YES \
  -p 127.0.0.1:8765:8765 \
  -v ~/.cache/isaac/kit:/isaac-sim/kit/cache \
  -v ~/.cache/isaac/ov:/root/.cache/ov \
  -v ~/.cache/isaac/glcache:/root/.cache/nvidia/GLCache \
  -v ~/.cache/isaac/computecache:/root/.nv/ComputeCache \
  -v ~/.cache/isaac/warp:/root/.cache/warp \
  -v ~/.cache/isaac/logs:/root/.nvidia-omniverse/logs \
  hud-assembly-env
```

Watch for ready:

```bash
docker logs -f assembly-env   # wait for HUD_SERVE_PORT=8765
```

If the first boot times out before that line appears, restart the container —
warm caches usually come up in under two minutes.

## Eval against it

On the host (agent environment from `./scripts/setup_agent.sh`):

```bash
python examples/run_eval.py --task peg_round_16mm --num-envs 4 \
    --runtime tcp://127.0.0.1:8765
```

If you published a different host port, pass that in `--runtime`.

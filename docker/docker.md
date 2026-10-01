# AssembleBench in Docker

Serves `env.py` behind the HUD entry point. One published port (8765) is enough —
the robot WebSocket rides the control channel's capability tunnel.

**GPU required.** Isaac renders with RTX; run with `--gpus all`.

## Build

Build from the **assemble-bench repo root** (no sibling repos needed):

```bash
./scripts/setup_sim.sh --submodules-only   # Arena + IsaacLab over HTTPS (no SSH keys)
docker build -f docker/Dockerfile -t assemble-bench-env .
```

Needs an NVIDIA NGC login to pull the Isaac Sim base image
(`docker login nvcr.io`).

The image installs `hud` (the HUD SDK, formerly `hud-python`) from
`014a43f6` into kit's Python (constraint-frozen so kit-owned packages are never
upgraded). That pin is required for Path B (`GymBridge` / `env.gym`) and for
`DirectControl`. PyPI `0.6.10`–`0.6.12` export neither.

## Run

First boot downloads Omniverse assets and can take **5–15 minutes**. Mount caches
so restarts are ~1–2 minutes:

```bash
mkdir -p ~/.cache/isaac/{kit,ov,glcache,computecache,warp,logs}

docker run -d --name assemble-bench --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e OMNI_KIT_ACCEPT_EULA=YES \
  -p 127.0.0.1:8765:8765 \
  -v ~/.cache/isaac/kit:/isaac-sim/kit/cache \
  -v ~/.cache/isaac/ov:/root/.cache/ov \
  -v ~/.cache/isaac/glcache:/root/.cache/nvidia/GLCache \
  -v ~/.cache/isaac/computecache:/root/.nv/ComputeCache \
  -v ~/.cache/isaac/warp:/root/.cache/warp \
  -v ~/.cache/isaac/logs:/root/.nvidia-omniverse/logs \
  assemble-bench-env
```

Watch for ready:

```bash
docker logs -f assemble-bench   # wait for HUD_SERVE_PORT=8765
```

If the first boot times out before that line appears, restart the container —
warm caches usually come up in under two minutes.

## Eval against it

On the host (agent environment from the README's Install section):

```bash
python examples/run_eval.py --task peg_round_16mm --num-envs 4 \
    --runtime tcp://127.0.0.1:8765
```

If you published a different host port, pass that in `--runtime`.

## Path A inside the container

The image can also run the sim directly (README Path A) instead of serving. Everything
is installed against kit's Python at `/isaac-sim/python.sh`; Arena lives at `/workspace`:

```bash
docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all -e OMNI_KIT_ACCEPT_EULA=YES \
  assemble-bench-env /isaac-sim/python.sh \
    /workspace/isaaclab_arena/evaluation/policy_runner.py \
    --policy_type zero_action --num_episodes 1 --headless \
    --external_environment_class_path \
    assemble_bench.environments.assembly.assembly:AssembleBenchEnvironment \
    assemble_bench --task peg_round_8mm
```

The same cache mounts as above apply (first run is a cold boot otherwise).

## LLM control on a local GPU

Serve the container as above, then `RUNTIME=tcp://127.0.0.1:8765 python examples/llm_assembly.py`
(`EPISODES=0` for the scripted check only). Full steps are in the README, "Run on a local GPU".
`scripts/check_local_gpu.sh` checks the driver, Docker GPU access, and the image beforehand.

## Modal

`nvcr.io` answers 401 without an NGC login. `docker/modal_deploy.py` follows the
README host install instead: public `isaacsim[all,extscache]==6.0.0.1` on a CUDA
image, then `docker/modal_image.sh`. The NGC Dockerfile above is unchanged.
Isaac Sim 6 needs an RTX GPU. Use `L40S`. A100 and H100 cannot render.

```bash
modal run docker/modal_deploy.py
python examples/llm_assembly.py
```

`examples/llm_assembly.py` is the LLM path: one scripted `move_joints` call,
then `EPISODES` (default 3) of `gpt-6-astra` on `assembly_direct`. The agent does
not start if that call errors. `MAX_STEPS` (default 100) is the tool-call budget.
`assembly_direct` sets the sim horizon to 1000 control ticks (15 Hz); the VLA
`assembly` template keeps the peg default of 40 s.

The Modal image installs `h5py` (Arena metrics) and `hydra-core` (`isaaclab_tasks`
config loading), which the `--no-deps` installs skip, and `libxt6` (MaterialX loads
`libXt.so.6` at Kit startup). For Vulkan it follows the RoboLab and RoboDojo
images: no Mesa ICDs (apt recommends install them and they win over NVIDIA),
`libegl1`, and NVIDIA ICD and EGL manifests under `/usr/share`. With that, Kit
reports the L40S and driver 580. `env.py` kills the sim if a build, reset or step
overruns its budget, so a hang fails the rollout instead of holding the sandbox.

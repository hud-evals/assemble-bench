# assembly-bench

NIST-taskboard assembly benchmark built as an external environment package on
[Isaac Lab Arena](https://github.com/isaac-sim/IsaacLab-Arena). See `README.md`
for the task space (27 variants), layout, and the full install/run commands, and
`docker/docker.md` for the containerized env server.

## Cursor Cloud specific instructions

### Hardware reality: the simulator needs an NVIDIA RTX GPU (absent here)

The core product is an Isaac Sim 6.x / Isaac Lab Arena robotics simulation that
serves assembly episodes over the HUD SDK. **Isaac Sim renders with RTX and has
no CPU fallback** (see `docker/docker.md`: "GPU REQUIRED"; `notes/modal_isaac_gvisor_report.md`
for why even a virtualized GPU without CUDA↔graphics interop fails). The default
Cursor Cloud VM has **no NVIDIA GPU, driver, or Vulkan** (`nvidia-smi`, `/dev/nvidia*`,
`vulkaninfo` all absent), so the following **cannot run here** and are expected to
fail on this VM, not because of a code/setup bug:

- `env.py` (`AppLauncher(...)` → Isaac kit boot)
- `scripts/preview_assembly.py`, `scripts/smoke_runtimes.py local`
- Arena's `policy_runner.py` / `eval_runner.py` (see `README.md`, `eval/README.md`)
- Building/running `docker/Dockerfile` (needs `--gpus all` + `nvcr.io/nvidia/isaac-sim` base)

To actually run the simulator you need a raw-Docker host with an NVIDIA RTX GPU
and NGC access to `nvcr.io/nvidia/isaac-sim:6.0.0-dev2`; follow `docker/docker.md`.

### What works CPU-only (and is set up by the update script)

The `assembly_bench` package installs standalone. The benchmark **manifest is
deliberately import-free** (`assembly_bench/environments/assembly/variants.py`) —
this is the surface `hud eval` and `tasks.json` consume. A `.venv` is created at
the repo root by the update script; use it directly (no `activate` needed):

```bash
# list all 27 task variants
.venv/bin/python -c "from assembly_bench.environments.assembly.variants import VARIANTS; print(' '.join(sorted(VARIANTS)))"

# regenerate tasks.json from the manifest (idempotent)
cd scripts && ../.venv/bin/python taskset.py
```

The Isaac-dependent modules (`assembly.py`, `scene.py`, `tasks.py`, `cameras.py`,
`observations.py`, `rewards.py`, `recorders.py`) import `isaaclab*` /
`isaaclab_arena*`, which are only present inside the GPU Docker image — importing
them on this VM raises `ModuleNotFoundError`, by design.

### Submodule note

`.gitmodules` declares `submodules/IsaacLab-Arena`, but it is **not committed as a
gitlink** in the tree, so `git submodule update --init` is a no-op. The real Arena
checkout only matters inside the GPU Docker build (`docker/Dockerfile` COPYs it in).

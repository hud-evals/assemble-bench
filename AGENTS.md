# AssembleBench

NIST-taskboard assembly benchmark built as an external environment package on
[Isaac Lab Arena](https://github.com/isaac-sim/IsaacLab-Arena). `README.md` has the task
list, install options, and run commands; `docker/docker.md` covers the container.

## Cursor Cloud specific instructions

### The simulator needs an NVIDIA RTX GPU

Isaac Sim 6 renders with RTX and has no CPU fallback. The default Cloud VM has no NVIDIA
GPU or driver, so none of these run there; failures are expected, not setup bugs:

- `env.py` and `python -m hud.environment.server env.py` (Kit boot)
- Arena's `policy_runner.py` (README, Path A)
- `docker build -f docker/Dockerfile` and `docker run --gpus all` (needs an NGC login for
  `nvcr.io/nvidia/isaac-sim:6.0.0-dev2`)
- `examples/llm_assembly.py` and `modal run docker/modal_deploy.py` (need `HUD_API_KEY`;
  Modal needs credentials and an `ngc-registry` Modal secret for the NGC base)

To run an episode, use a GPU host and follow the README sections "Path B - HUD" and
"Run on a local GPU" (`scripts/check_local_gpu.sh` preflights the host). The LLM path is
the `move_joints` tool (`direct_control.py`, template `assembly_direct`).

### What works CPU-only

The task catalog `assemble_bench/environments/assembly/variants.py` imports no Isaac
modules (15 entries: 14 benchmark tasks plus `debug`). Install it into a venv at the
repo root (git-ignored) and regenerate the suites under `tasks/`; the output is
unchanged when `variants.py` is:

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e assemble_bench
.venv/bin/python -c "from assemble_bench.environments.assembly.variants import VARIANTS; print(sorted(VARIANTS))"
cd scripts && ../.venv/bin/python taskset.py
```

The other modules in `assemble_bench/environments/assembly/` import `isaaclab*` and
`isaaclab_arena*`, which exist only in the GPU image or a host Isaac install. Importing
them elsewhere raises `ModuleNotFoundError`.

### Submodule

`submodules/IsaacLab-Arena` is a committed gitlink but is empty after a plain clone. Fetch
it, and Arena's pinned IsaacLab, with `./scripts/setup_sim.sh --submodules-only` (HTTPS
remotes, skips LFS). Do not use `git clone --recursive`.

# Cosmos3-Policy-DROID on assembly_bench

Zero-shot eval of NVIDIA’s DROID Cosms policies against this bench’s Franka +
Robotiq `droid_abs_joint_pos` contract (8-D absolute joints + gripper @ 15 Hz).

The env side stays unchanged. The agent is a thin HUD `RemoteModel` client that
talks to a separate [cosmos-framework](https://github.com/NVIDIA/cosmos-framework)
OpenPI WebSocket policy server.

## What you need

| Piece | Role |
|---|---|
| **assembly env** | Isaac Docker image `hud-assembly-env` serving HUD on `tcp://127.0.0.1:8765` (see [`docker/docker.md`](docker/docker.md)) |
| **`vla` conda env** | Agent side: `hud-python`, `openpi-client`, `av` (video streaming), torch |
| **cosmos-framework** | Host or container checkout with `uv sync --all-extras --group=cu130-train --group=policy-server` (use `cu128-train` on CUDA 12.x) |
| **GPU headroom** | Env ~10 GB + Cosms Nano ~30–40 GB; a 48 GB+ card is comfortable |
| **HF access** | `nvidia/Cosmos3-Nano-Policy-DROID` (and optionally Edge). Guardrail is gated — the server can run with `guardrails=False` for policy-only serving |
| **HUD telemetry** (optional) | `~/.hud/.env` pointing at `https://api.hud.ai` to stream jobs |

Supported checkpoints:

- `nvidia/Cosmos3-Nano-Policy-DROID` (default)
- `nvidia/Cosmos3-Edge-Policy-DROID` (pass `--checkpoint-path` + `--format-prompt-as-json True`)

## Start the policy server

```bash
cd /path/to/cosmos-framework
export HF_TOKEN=… HF_HOME=~/.cache/huggingface LD_LIBRARY_PATH=
uv sync --all-extras --group=cu130-train --group=policy-server
source .venv/bin/activate

python -m cosmos_framework.scripts.action_policy_server_robolab \
  --checkpoint-path nvidia/Cosmos3-Nano-Policy-DROID \
  --port 8000 --host 0.0.0.0
```

Health check: `curl http://127.0.0.1:8000/healthz` → `OK`.

If startup fails downloading `nvidia/Cosmos-Guardrail1` (gated), disable
guardrails in the server’s OmniSetup overrides (`guardrails=False`) — policy
serving only needs action inference.

## Useful scripts

| Script | Purpose |
|---|---|
| [`scripts/cosmos_droid.py`](scripts/cosmos_droid.py) | Agent: packs obs → Cosms request, `RemoteModel(response_key="action")`. Export `Agent` for `hud eval` / `eval_cosmos.py` |
| [`scripts/eval_cosmos.py`](scripts/eval_cosmos.py) | Taskset runner → HUD job. Supports `--tasks`, `--group`, `--max_steps`, `--name` |
| [`scripts/mock_cosmos_server.py`](scripts/mock_cosmos_server.py) | OpenPI-protocol hold-pose mock on `:8000` for adapter/env smoke tests (arm will not move) |
| [`scripts/fetch_debug_assets.py`](scripts/fetch_debug_assets.py) | Pull RoboLab `apple_01` + YCB `bowl` into `assets/parts/debug/` (gitignored) |
| [`eval/debug.json`](eval/debug.json) | Apple→bowl sanity task |
| [`eval/pegs.json`](eval/pegs.json) | All 8 peg-insert variants |
| [`eval/canonical4.json`](eval/canonical4.json) / [`eval/baseline_3task.json`](eval/baseline_3task.json) | Broader sweeps |

### Agent packing notes

RoboLab Cosms expects one concat image: **wrist (360×640) over left|right half-res
exteriors** → 540×640, plus `observation/joint_position` (7) and
`observation/gripper_position` (1). Response key is **`action`** (not `actions`),
chunk horizon **32**.

This bench only has front + wrist (no dual over-shoulder). The adapter **black-
masks both exterior tiles** (zeros). Duplicating `front` into those tiles looks
worse empirically. Optional: `COSMOS_MASK_WRIST=1` for a proprio-only ablation.

Env vars: `COSMOS_HOST`, `COSMOS_PORT` (default `localhost:8000`),
`COSMOS_CHUNK_SIZE` (default `32`).

`RemoteModel` is **not batchable** — keep `--max_concurrent 1` (one agent per
rollout).

## Eval examples

```bash
# Env (separate terminal / already running)
docker run -d --name assembly-env --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e OMNI_KIT_ACCEPT_EULA=YES \
  -p 127.0.0.1:8765:8765 \
  hud-assembly-env

# Debug apple→bowl (needs fetch_debug_assets.py once)
conda activate vla
set -a && source ~/.hud/.env && set +a   # optional: stream to hud.ai
COSMOS_HOST=127.0.0.1 COSMOS_PORT=8000 RUNTIME=tcp://127.0.0.1:8765 \
  python scripts/eval_cosmos.py \
    --tasks eval/debug.json --group 1 --max_concurrent 1 \
    --name cosmos-nano-debug

# All pegs
COSMOS_HOST=127.0.0.1 COSMOS_PORT=8000 RUNTIME=tcp://127.0.0.1:8765 \
  python scripts/eval_cosmos.py \
    --tasks eval/pegs.json --group 1 --max_concurrent 1 \
    --name cosmos-nano-pegs
```

Adapter-only smoke without Cosms weights:

```bash
python scripts/mock_cosmos_server.py --port 8000
# then eval_cosmos.py as above — expects a still arm (hold pose)
```

## Known gaps

- **Cameras:** restoring RoboLab-style left/right over-shoulder views would close
  the biggest zero-shot gap; black exteriors are the current best workaround.
- **Scene OOD:** NIST tabletop + prompts differ from RoboLab kitchen demos.
- **Debug assets:** `assembly_bench/assets/parts/debug/` is gitignored; run
  `python scripts/fetch_debug_assets.py` with a RoboLab checkout (`ROBOLAB_ROOT`).

# Assembly Bench

**Contact-rich robot assembly tasks for evaluating and training VLAs.**
Tested headless on L40S, RTX 6000 Ada, and RTX PRO 6000 Blackwell GPUs.

14 tabletop assembly tasks – insert pegs, mesh gears, thread nuts – modelled on the
[NIST ATB-1 assembly taskboard](https://www.nist.gov/el/intelligent-systems-division-73500/robotic-grasping-and-manipulation-assembly/assembly).
The robot is the [DROID](https://arxiv.org/abs/2403.12945) platform (Franka Panda 7-DoF +
Robotiq 2F-85), so any DROID checkpoint plugs in without retargeting. Every task is scored
on real assembly geometry rather than a proximity heuristic: the part has to actually seat,
and a gear that clashes teeth or a nut that cross-threads cannot descend.

- Blog post: [Benchmarking Robot Models on Contact-Rich Assembly](https://www.hud.ai/blog/assembly-benchmark)
- Demonstration dataset (1355 episodes): [`hud-evals/AssemblyBench`](https://huggingface.co/datasets/hud-evals/AssemblyBench)
- Reference checkpoints: [`pi05-AssemblyBench-12k`](https://huggingface.co/hud-evals/pi05-AssemblyBench-12k) (BC) · [`pi05-AssemblyBench-cgdagger-r3`](https://huggingface.co/hud-evals/pi05-AssemblyBench-cgdagger-r3) (final)

## Tasks

| | Family | Tasks | Skills |
|---|---|---|---|
| 🟦 | Round peg insertion | `peg_round_4mm` `peg_round_8mm` `peg_round_12mm` `peg_round_16mm` | insertion |
| 🟪 | Square peg insertion | `peg_square_4mm` `peg_square_8mm` `peg_square_12mm` `peg_square_16mm` | alignment, insertion |
| 🟧 | Gear meshing | `gear_small` `gear_medium` `gear_large` | alignment, insertion, fitting |
| 🟩 | Nut threading | `nut_M8` `nut_M12` `nut_M16` | alignment, threading |

Peg names are stem diameters in mm. Pegs start upright in a presentation bore beside their
hole; gears and nuts start flat on the table. Square pegs are rectangular, so they are not
yaw-symmetric – the hole is re-clocked every episode and the policy has to match it. Part
poses are jittered every episode.

A 15th task, `debug`, puts an apple in a bowl. It is a hello-world check that the
plumbing works, not part of the benchmark.

## Requirements

- NVIDIA GPU with RT cores – 16 GB+ for the simulator, 24 GB+ if a VLA shares the same GPU
  (datacenter cards without RT cores, such as A100 and H100, cannot render)
- Ubuntu 22.04+ (Isaac Sim 6 needs GLIBC >= 2.35)
- [Isaac Sim 6.x](https://docs.isaacsim.omniverse.nvidia.com/current/installation/download.html)
  (NGC image `nvcr.io/nvidia/isaac-sim:6.0.0` or a local install)
- `export OMNI_KIT_ACCEPT_EULA=YES` in every shell that launches Isaac

Path B's agent side also needs a normal Python 3.10+ env (separate from Isaac) and a
Hugging Face login that has accepted
[`google/paligemma-3b-pt-224`](https://huggingface.co/google/paligemma-3b-pt-224)
(the pi0.5 tokenizer is gated; the checkpoints themselves are public).

## Install

```bash
git clone https://github.com/hud-evals/assembly_bench.git
cd assembly_bench

# Inside your Isaac Sim environment (conda env, NGC container, or /isaac-sim/python.sh):
export OMNI_KIT_ACCEPT_EULA=YES
./scripts/setup_sim.sh              # Arena submodule + Isaac Lab + this package
./scripts/setup_sim.sh --with-hud   # same, plus hud-python for Path B
```

`setup_sim.sh` fetches the [Isaac Lab Arena](https://github.com/isaac-sim/IsaacLab-Arena)
submodule (and its pinned IsaacLab), installs both editable, then installs
`assembly_bench`. Arena stays unmodified; this repo plugs in through its registration API.

For Path B's VLA agent, in a **separate** Python 3.10+ environment:

```bash
./scripts/setup_agent.sh            # lerobot, hud-python, torch, …
hf auth login                       # after accepting the PaliGemma gate above
```

No Isaac Sim on the host? Skip `setup_sim.sh` and serve via Docker instead – see Path B
and [`docker/docker.md`](docker/docker.md).

## Running the benchmark

Two paths. **Path A** drives Isaac Sim directly – the usual route if you already have
Isaac Sim on the machine. **Path B (HUD)** serves the sim over the network for evaluating
VLAs, parallel envs, and optional per-episode traces.

### Path A – Isaac Sim directly

Smoke-test that the scene builds (zero actions, no model weights). From the repo root,
with the Isaac env active:

```bash
cd submodules/IsaacLab-Arena
OMNI_KIT_ACCEPT_EULA=YES python isaaclab_arena/evaluation/policy_runner.py \
    --policy_type zero_action --num_episodes 1 --headless \
    --external_environment_class_path \
    assembly_bench.environments.assembly.assembly:AssemblyBenchEnvironment \
    assembly_bench --task peg_round_8mm
```

| Flag | Default | Notes |
|---|---|---|
| `--task` | `peg_round_8mm` | any task id from the table above, or `debug` |
| `--embodiment` | `droid_abs_joint_pos_softmimic` | DROID with this benchmark's contact tuning; `droid_abs_joint_pos` is stock, any `franka_*` also works |
| `--reward` | `none` | `staged` / `potential` add dense reward for RL |
| `--hdr` | `asm_machine_shop` | any Arena HDR registry name, or `none` |
| `--num_envs` | `1` | parallel envs on one GPU |

Or render stills of both policy cameras:

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/preview_assembly.py --task peg_round_8mm --out /tmp/peg
# → /tmp/peg_front.png, /tmp/peg_wrist.png  (log should say COLOR OK)
```

### Path B – HUD

Serve the environment once, then attach a policy over TCP. The simulator and the policy
can live in different environments (or on different machines).

**1. Serve the env** (pick one)

Host Isaac (after `./scripts/setup_sim.sh --with-hud`):

```bash
OMNI_KIT_ACCEPT_EULA=YES python -m hud.environment.server env.py --port 8765
```

Or Docker (no host Isaac install – build from this repo alone):

```bash
# details + cache mounts: docker/docker.md
docker build -f docker/Dockerfile -t hud-assembly-env .
docker run -d --name assembly-env --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e OMNI_KIT_ACCEPT_EULA=YES \
  -p 127.0.0.1:8765:8765 hud-assembly-env
```

Wait for `HUD_SERVE_PORT=8765` in the logs (first boot can take 5–15 minutes).

**2. Run a VLA** (agent env from `./scripts/setup_agent.sh`)

The wire is DROID: front + wrist RGB at 640×360, joint positions, 8-D action (7 joint
targets + binary gripper) at 15 Hz. [`examples/`](examples/) loads our final pi0.5
checkpoint from Hugging Face:

```bash
python examples/run_eval.py --task peg_round_16mm --num-envs 4
```

First run downloads ~9 GB of weights.

| Flag | Default | Notes |
|---|---|---|
| `--task` | `peg_round_16mm` | any task id from the table above, or `debug` |
| `--num-envs` | `4` | parallel episodes in one sim process |
| `--waves` | `1` | sequential batches (`15 × 2` = the writeup's 30-ep protocol) |
| `--checkpoint` | `hud-evals/pi05-AssemblyBench-cgdagger-r3` | HF repo id or local dir |
| `--runtime` | `tcp://127.0.0.1:8765` | where the env is serving |

**3. Stream traces (optional)**

Everything grades locally with no account. Set `HUD_API_KEY` to also upload each episode's
video, actions, and reward to [hud.ai](https://hud.ai):

```bash
export HUD_API_KEY=sk-hud-...
python examples/run_eval.py --task peg_round_16mm --num-envs 4
```

Without the key, nothing is sent anywhere.

> An LLM tool-use path (`agents/`, `tasks/agent/`) is **in development** and not ready
> for use yet.

## Test your install

| Check | Command | Expect |
|---|---|---|
| Quick (Path A, no weights) | `python scripts/preview_assembly.py --task peg_round_8mm --out /tmp/peg` | two PNGs, `COLOR OK` |
| Full (Path B + pi0.5) | serve env, then `python examples/run_eval.py --task peg_round_16mm --num-envs 2` | episodes grade; optional job URL if `HUD_API_KEY` is set |

## Additional information

### Layout

```
assembly_bench/
├── scripts/setup_sim.sh     install Arena + this package into Isaac Sim Python
├── scripts/setup_agent.sh   install Path B VLA deps into a normal Python
├── requirements-agent.txt   agent-side pip deps (Path B)
├── assembly_bench/          the pip-installable Arena environment package
│   ├── environments/assembly/   variants.py (the task catalog), scene, tasks, rewards
│   └── assets/parts/            pegs, gears, nuts, NIST board
├── examples/                pi0.5 VLA + eval runner
├── tasks/                   HUD run lists
├── docker/                  self-contained Isaac + HUD image (build from this repo)
├── env.py, contract.json    HUD entry point and its observation/action wire
└── submodules/IsaacLab-Arena    unmodified Arena (git submodule)
```

### Adding your own task

Tasks are pure Python data, not USD scenes. Add an entry to `VARIANTS` in
[`variants.py`](assembly_bench/environments/assembly/variants.py) – which parts to spawn,
where, and the seat geometry that defines success – then run `python scripts/taskset.py`
to refresh the HUD run lists. See [`tasks/README.md`](tasks/README.md).

### Dense rewards for RL

Evaluation uses sparse success (`--reward none`). For training, `--reward staged` gives
milestone rewards (lift, engage, thread start, success) plus best-so-far progress, and
`--reward potential` uses signed potential differences so backsliding is penalised. Both
are also available as `make_assembly_env(..., reward=...)`.

### Scripted experts

The demonstration dataset was generated without teleoperation, by scripted experts that
read privileged simulator state:

```bash
python scripts/experts/run_expert.py --headless --task peg_round_8mm --num_envs 4
```

See the docstring in [`run_expert.py`](scripts/experts/run_expert.py) for recording
recipes. `scripts/experts/rl/` holds the code-gated DAgger loop from the writeup, enabled
on the HUD path with `ASSEMBLY_EXPERT_TAKEOVER=1`.

### Credits

Built on [Isaac Lab Arena](https://github.com/isaac-sim/IsaacLab-Arena) and Isaac Sim 6
(NVIDIA). Task design follows NIST's
[ATB-1 assembly benchmarking procedure](https://www.nist.gov/el/intelligent-systems-division-73500/robotic-grasping-and-manipulation-assembly/assembly),
and the contact-rich assembly setup builds on
[Factory](https://arxiv.org/abs/2205.03532) and [FORGE](https://arxiv.org/abs/2408.04587).
The robot platform and reference checkpoints come from
[DROID](https://arxiv.org/abs/2403.12945).

# Examples

Run a real VLA against the benchmark. [`pi05_agent.py`](pi05_agent.py) loads a pi0.5
DROID checkpoint from the Hugging Face Hub and [`run_eval.py`](run_eval.py) evaluates it
on a task, optionally streaming every rollout to the HUD platform.

Nothing here depends on the rest of the repo, so you can copy these two files as the
starting point for your own agent.

## Setup

The agent and the simulator are separate processes that talk over TCP, so they need
separate environments – and can live on separate machines. From the repo root, in any
Python 3.10+ environment (not Isaac):

```bash
./scripts/setup_agent.sh        # or: pip install -r requirements-agent.txt
```

The pi0.5 tokenizer is gated even though the checkpoints are public, so accept
[`google/paligemma-3b-pt-224`](https://huggingface.co/google/paligemma-3b-pt-224) once
and log in:

```bash
hf auth login
```

## Run

Serve the environment (see the repo README for the Isaac Sim install, or use
[`docker/`](../docker/docker.md) to skip it), and wait for `HUD_SERVE_PORT=8765`:

```bash
OMNI_KIT_ACCEPT_EULA=YES python -m hud.environment.server env.py --port 8765
```

Then, in the agent environment:

```bash
python examples/run_eval.py --task peg_round_16mm --num-envs 4
```

First run downloads ~9 GB of weights. `--num-envs` is how many episodes run in parallel
inside the one simulator; `--waves` repeats batches sequentially. The writeup's protocol
is 30 episodes per task:

```bash
python examples/run_eval.py --task peg_round_16mm --num-envs 15 --waves 2
```

Useful flags: `--checkpoint` (any HF repo id or local directory), `--runtime` (if the
simulator is not on `127.0.0.1:8765`), `--max-steps`, `--seed`.

## Streaming to the HUD platform

By default everything runs and grades on your machine and nothing is uploaded. Set an
API key from [hud.ai](https://hud.ai) to also record each episode's video, actions, and
reward as a trace you can replay and compare across checkpoints:

```bash
export HUD_API_KEY=sk-hud-...
python examples/run_eval.py --task peg_round_16mm --num-envs 4
```

The run prints a job URL either way; it only resolves to something when a key is set.

## Checkpoints

Reference pi0.5 checkpoints from the [writeup](https://www.hud.ai/blog/assembly-benchmark),
all finetuned from [`DAVIAN-Robotics/pi05_droid_jointpos`](https://huggingface.co/DAVIAN-Robotics/pi05_droid_jointpos)
on [`hud-evals/AssemblyBench`](https://huggingface.co/datasets/hud-evals/AssemblyBench):

| Checkpoint | What it is |
|---|---|
| [`hud-evals/pi05-AssemblyBench-12k`](https://huggingface.co/hud-evals/pi05-AssemblyBench-12k) | behavior-cloning baseline, trained on all 14 tasks |
| [`hud-evals/pi05-AssemblyBench-cgdagger-r3`](https://huggingface.co/hud-evals/pi05-AssemblyBench-cgdagger-r3) | BC + 3 rounds of code-gated DAgger – the default here |

The CG-DAgger checkpoint was trained only on `peg_round_8mm` corrections, so the other
round-peg sizes measure whether the recovery transfers. Base DROID checkpoints score 0%
here without task finetuning – the writeup covers why.

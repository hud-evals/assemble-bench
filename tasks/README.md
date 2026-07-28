# Task suites (HUD run lists)

**Define** a task in
[`environments/assembly/variants.py`](../assembly_bench/environments/assembly/variants.py)
(`held` / `fixed` / seat geometry / instruction). That catalog is the source of truth.

**Run** a batch by pointing HUD at a suite under this folder. Suites only list task
names — they do not define new scenes.

```
tasks/
├── vla/                 # template id "assembly" (joint-position / openpi)
│   ├── all.json         # every task (14 benchmark + debug)
│   ├── smoke.json       # one peg + gear + nut
│   └── debug.json       # apple → bowl hello-world (not part of the benchmark)
└── agent/               # template id "assembly_agent" (MCP end-effector tools)
    └── pegs.json        # peg smokes × guided / vision
```

`vla/` and `agent/` are separate because they hit different `env.py` templates
(`assembly` vs `assembly_agent`) and different args (`guided` only on agent).

## Running a suite

LLM agents run straight from the `hud eval` CLI:

```bash
hud eval tasks/agent/pegs.json claude --runtime tcp://127.0.0.1:8765
```

VLAs run through the Python SDK — the `hud eval` CLI only accepts built-in LLM agent
types, not a policy checkpoint. [`examples/run_eval.py`](../examples/run_eval.py) is a
complete runner:

```bash
python examples/run_eval.py --task peg_round_8mm --num-envs 4
```

To sweep a whole suite, loop over its slugs:

```bash
python -c "import json;[print(r['slug']) for r in json.load(open('tasks/vla/all.json'))]" \
  | xargs -I{} python examples/run_eval.py --task {} --num-envs 15 --waves 2
```

## Regenerate from variants

```bash
python scripts/taskset.py
```

## Add your own task

1. Add an entry to `VARIANTS` in `variants.py` (copy a sibling in the same family).
2. Run `python scripts/taskset.py` so `vla/all.json` picks it up.
3. Or hand-write a one-off suite JSON that sets `"args": {"task": "<your_name>"}`.

Single-task check straight through Isaac Sim, no HUD involved:

```bash
python isaaclab_arena/evaluation/policy_runner.py \
    --policy_type zero_action --num_episodes 1 \
    --external_environment_class_path \
    assembly_bench.environments.assembly.assembly:AssemblyBenchEnvironment \
    assembly_bench --task peg_round_8mm
```

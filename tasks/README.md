# Task suites (HUD run lists)

**Define** a task in
[`environments/assembly/variants.py`](../assemble_bench/environments/assembly/variants.py)
(`held` / `fixed` / seat geometry / instruction). That catalog is the source of truth.

**Run** a batch by pointing HUD at a suite under this folder. Suites only list task
names – they do not define new scenes.

```
tasks/
├── vla/                 # template id "assembly" (joint-position / openpi)
│   ├── all.json         # every task (14 benchmark + debug)
│   ├── smoke.json       # one peg + gear + nut
│   └── debug.json       # apple → bowl hello-world (not part of the benchmark)
└── agent/               # LLM tool path (in development) – not ready yet
    └── pegs.json
```

## Running a suite

VLAs run through the Python SDK. [`examples/run_eval.py`](../examples/run_eval.py) is a
complete runner:

```bash
python examples/run_eval.py --task peg_round_8mm --num-envs 4
```

To sweep a whole suite, loop over its slugs:

```bash
python -c "import json;[print(r['slug']) for r in json.load(open('tasks/vla/all.json'))]" \
  | xargs -I{} python examples/run_eval.py --task {} --num-envs 15 --waves 2
```

> The LLM tool-use path (`tasks/agent/`, `env` template `assembly_agent`) is **in
> development** and not ready for use yet.

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
    assemble_bench.environments.assembly.assembly:AssembleBenchEnvironment \
    assemble_bench --task peg_round_8mm
```

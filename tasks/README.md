# Task suites (HUD run lists)

**Define** a scene variant in
[`environments/assembly/variants.py`](../assembly_bench/environments/assembly/variants.py)
(`held` / `fixed` / seat geometry / instruction). That catalog is the source of
truth.

**Run** a batch with HUD by pointing `hud eval` at a suite under this folder.
Suites only list variant names — they do not define new scenes.

```
tasks/
├── vla/                 # template id "assembly" (joint-position / openpi)
│   ├── all.json         # every variant (NIST + debug)
│   ├── smoke.json       # one peg + gear + nut
│   └── debug.json       # apple → bowl hello-world (not NIST)
└── agent/               # template id "assembly_agent" (MCP EE tools)
    └── pegs.json        # peg smokes × guided / vision
```

`vla/` and `agent/` are separate because they hit different `env.py` templates
(`assembly` vs `assembly_agent`) and different args (`guided` only on agent).

## Regenerate from variants

```bash
python scripts/taskset.py
```

## Examples

```bash
# Hello-world pick-place (not NIST)
hud eval tasks/vla/debug.json <agent> --runtime tcp://127.0.0.1:8765

# Quick NIST smoke (3 variants)
hud eval tasks/vla/smoke.json <agent> --runtime tcp://127.0.0.1:8765

# Full VLA suite
hud eval tasks/vla/all.json <agent> --full --runtime tcp://127.0.0.1:8765

# LLM tool path
hud eval tasks/agent/pegs.json <agent> --runtime tcp://127.0.0.1:8765
```

## Add your own variant

1. Add an entry to `VARIANTS` in `variants.py` (copy a sibling in the same family).
2. Run `python scripts/taskset.py` so `vla/all.json` picks it up.
3. Or hand-write a one-off suite JSON that sets `"args": {"task": "<your_name>"}`.

Single-variant Arena smoke (no HUD suite needed):

```bash
python isaaclab_arena/evaluation/policy_runner.py \
    --policy_type zero_action --num_episodes 1 \
    --external_environment_class_path \
    assembly_bench.environments.assembly.assembly:AssemblyBenchEnvironment \
    assembly_bench --task peg_round_8mm
```

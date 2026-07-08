# Batch evaluation

`assembly_all_variants.json` runs the whole benchmark — one job per variant,
all 29 — through Arena's eval jobs runner
(`isaaclab_arena/evaluation/eval_runner.py`), instead of 29 manual
`policy_runner.py` invocations.

From `submodules/IsaacLab-Arena`:

```bash
python isaaclab_arena/evaluation/eval_runner.py \
    --external_environment_class_path \
    assembly_bench.environments.assembly.assembly:AssemblyBenchEnvironment \
    --eval_jobs_config ../../eval/assembly_all_variants.json \
    --chunk_size 1
```

Swap `policy_type` (and `policy_config_dict`) per job to evaluate a real
policy; `zero_action` just exercises the scenes. Add
`--output_base_dir <dir>` if the default `/eval/output` isn't writable.

`--chunk_size 1` is required: `--external_environment_class_path` is
registered once per job inside the runner process and the registry asserts on
duplicate keys, so each job must run in a fresh subprocess.

Per-variant fallback without the jobs runner:

```bash
for t in $(python -c "from assembly_bench.environments.assembly.variants import VARIANTS; print(' '.join(VARIANTS))"); do
    python isaaclab_arena/evaluation/policy_runner.py \
        --policy_type zero_action --num_episodes 1 \
        --external_environment_class_path \
        assembly_bench.environments.assembly.assembly:AssemblyBenchEnvironment \
        assembly_bench --task "$t"
done
```

#!/usr/bin/env bash
# Install the assembly_bench peg task into an RLinf checkout (run in the RLinf venv/container).
#
#   rl/install_into_rlinf.sh /path/to/RLinf
#
# What it does:
#   1. makes assembly_bench importable from the RLinf root;
#   2. copies the adapter into rlinf/envs/isaaclab/tasks/;
#   3. registers "AssemblyBench-Peg-v0" -> IsaaclabAssemblyBenchEnv (idempotent);
#   4. copies the env + training configs into examples/embodiment/config/.
#
# Assets/USDs come from the assembly_bench package + Arena's Nucleus registry; the
# Arena submodule (Isaac Sim 6.x) must be installed in the same env.
set -euo pipefail

RLINF="${1:?usage: install_into_rlinf.sh /path/to/RLinf}"
RL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH_DIR=$(cd "$RL_DIR/.." && pwd)

[[ -d "$RLINF/rlinf/envs/isaaclab/tasks" ]] || { echo "not an RLinf checkout: $RLINF" >&2; exit 1; }

# 1. assembly_bench importable (RLinf launches from its repo root).
ln -sfn "$BENCH_DIR/assembly_bench" "$RLINF/assembly_bench"

# 2. adapter
cp "$RL_DIR/assembly_rlinf.py" "$RLINF/rlinf/envs/isaaclab/tasks/assembly_bench.py"

# 3. registration (append once)
INIT="$RLINF/rlinf/envs/isaaclab/__init__.py"
if ! grep -q "IsaaclabAssemblyBenchEnv" "$INIT"; then
  python3 - "$INIT" <<'EOF'
import sys
path = sys.argv[1]
src = open(path).read()
src = src.replace(
    "from .tasks.stack_cube import IsaaclabStackCubeEnv",
    "from .tasks.assembly_bench import IsaaclabAssemblyBenchEnv\n"
    "from .tasks.stack_cube import IsaaclabStackCubeEnv",
)
src = src.replace(
    "REGISTER_ISAACLAB_ENVS = {",
    'REGISTER_ISAACLAB_ENVS = {\n    "AssemblyBench-Peg-v0": IsaaclabAssemblyBenchEnv,',
)
open(path, "w").write(src)
print(f"[install] registered AssemblyBench-Peg-v0 in {path}")
EOF
else
  echo "[install] registration already present"
fi

# 4. configs
cp "$RL_DIR/config/env/isaaclab_assembly_bench.yaml" "$RLINF/examples/embodiment/config/env/"
cp "$RL_DIR/config/assembly_bench_ppo_openpi_pi05.yaml" "$RLINF/examples/embodiment/config/"

echo "[install] done. Train with:"
echo "  cd $RLINF && bash examples/embodiment/run_embodiment.sh assembly_bench_ppo_openpi_pi05"
echo "  (set rollout/actor model_path to your RLinf-Pi05-Polaris-droid_jointpos checkout first)"

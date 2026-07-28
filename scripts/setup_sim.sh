#!/usr/bin/env bash
# Install Assembly Bench + Isaac Lab Arena into the *currently active* Isaac Sim
# Python. Activate that env first (conda, NGC container, or kit's python.sh).
#
#   export OMNI_KIT_ACCEPT_EULA=YES
#   ./scripts/setup_sim.sh
#   ./scripts/setup_sim.sh --with-hud    # also install hud-python (Path B)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WITH_HUD=0
for arg in "$@"; do
  case "$arg" in
    --with-hud) WITH_HUD=1 ;;
    -h|--help)
      sed -n '2,8p' "$0"
      exit 0
      ;;
    *)
      echo "unknown flag: $arg" >&2
      exit 1
      ;;
  esac
done

# Prefer kit's python when running inside the NGC Isaac Sim image.
if [[ -x /isaac-sim/python.sh ]]; then
  PY=(/isaac-sim/python.sh)
elif [[ -n "${ISAAC_PY:-}" ]]; then
  # shellcheck disable=SC2206
  PY=($ISAAC_PY)
else
  PY=(python)
fi

echo "[setup] python: ${PY[*]} ($(${PY[@]} -c 'import sys; print(sys.executable)'))"

# --- Arena + its pinned IsaacLab (skip Isaac-GR00T; this bench does not need it) ---
echo "[setup] fetching IsaacLab-Arena submodule..."
git submodule update --init submodules/IsaacLab-Arena
# Arena pins nested remotes as git@github.com; rewrite to HTTPS for clone without SSH keys.
git -C submodules/IsaacLab-Arena config url."https://github.com/".insteadOf "git@github.com:"
git -C submodules/IsaacLab-Arena submodule update --init submodules/IsaacLab

# Smoke-check that Isaac is importable before we spend time on pip.
if ! ${PY[@]} -c "import isaacsim" 2>/dev/null; then
  cat >&2 <<'EOF'
[setup] ERROR: this Python cannot import isaacsim.

Install Isaac Sim 6.x first, then re-run this script *inside* that environment:
  - NGC container:  nvcr.io/nvidia/isaac-sim:6.0.0  (use /isaac-sim/python.sh)
  - or follow:      https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

Also set:  export OMNI_KIT_ACCEPT_EULA=YES
EOF
  exit 1
fi

echo "[setup] installing Isaac Lab (Arena pin, --no-deps)..."
for d in submodules/IsaacLab-Arena/submodules/IsaacLab/source/isaaclab*/; do
  ${PY[@]} -m pip install --no-deps -e "$d"
done

echo "[setup] installing Isaac Lab Arena..."
${PY[@]} -m pip install -e submodules/IsaacLab-Arena

echo "[setup] installing assembly_bench..."
${PY[@]} -m pip install -e assembly_bench

if [[ "$WITH_HUD" -eq 1 ]]; then
  echo "[setup] installing hud-python (Path B)..."
  ${PY[@]} -m pip install "hud-python>=0.6.10"
fi

cat <<EOF

[setup] done.

Quick smoke (Path A):
  cd submodules/IsaacLab-Arena
  OMNI_KIT_ACCEPT_EULA=YES ${PY[*]} isaaclab_arena/evaluation/policy_runner.py \\
      --policy_type zero_action --num_episodes 1 --headless \\
      --external_environment_class_path \\
      assembly_bench.environments.assembly.assembly:AssemblyBenchEnvironment \\
      assembly_bench --task peg_round_8mm

Or:  OMNI_KIT_ACCEPT_EULA=YES ${PY[*]} scripts/preview_assembly.py --task peg_round_8mm --out /tmp/peg
EOF

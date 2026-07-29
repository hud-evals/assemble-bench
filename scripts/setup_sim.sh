#!/usr/bin/env bash
# Install AssembleBench + Isaac Lab Arena into the *currently active* Isaac Sim
# Python. Activate that env first (conda, NGC container, or kit's python.sh).
# Runs exactly the manual steps from the README's "Host Isaac Sim" section,
# plus the submodule fetch (HTTPS rewrite + LFS skip).
#
#   export OMNI_KIT_ACCEPT_EULA=YES
#   ./scripts/setup_sim.sh                     # submodules + pip into Isaac Python
#   ./scripts/setup_sim.sh --submodules-only   # just fetch Arena/IsaacLab (e.g. before docker build)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SUBMODULES_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --submodules-only) SUBMODULES_ONLY=1 ;;
    -h|--help)
      sed -n '2,9p' "$0"
      exit 0
      ;;
    *)
      echo "unknown flag: $arg (try --help)" >&2
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

fetch_submodules() {
  # Arena LFS is docs/media only — assembly assets live in this repo. Skipping
  # smudge makes cold clones much faster and avoids needing a working git-lfs pull.
  export GIT_LFS_SKIP_SMUDGE=1

  # Arena pins nested remotes as git@github.com:… — rewrite to HTTPS for the
  # duration of this process (no global git config change; works without SSH keys).
  export GIT_CONFIG_COUNT=1
  export GIT_CONFIG_KEY_0="url.https://github.com/.insteadof"
  export GIT_CONFIG_VALUE_0="git@github.com:"

  echo "[setup] fetching IsaacLab-Arena…"
  git submodule update --init submodules/IsaacLab-Arena

  # Belt-and-suspenders: rewrite .gitmodules on disk in case a git older than
  # 2.31 ignores GIT_CONFIG_* during submodule clone.
  if grep -q 'git@github.com:' submodules/IsaacLab-Arena/.gitmodules 2>/dev/null; then
    sed -i.bak 's|git@github.com:|https://github.com/|g' \
      submodules/IsaacLab-Arena/.gitmodules
    git -C submodules/IsaacLab-Arena submodule sync submodules/IsaacLab
  fi

  echo "[setup] fetching Arena's pinned IsaacLab (skipping Isaac-GR00T)…"
  git -C submodules/IsaacLab-Arena submodule update --init submodules/IsaacLab

  if [[ ! -d submodules/IsaacLab-Arena/submodules/IsaacLab/source/isaaclab ]]; then
    echo "[setup] ERROR: IsaacLab did not land under submodules/IsaacLab-Arena/submodules/IsaacLab" >&2
    exit 1
  fi
  echo "[setup] submodules ready."
}

fetch_submodules

if [[ "$SUBMODULES_ONLY" -eq 1 ]]; then
  echo "[setup] --submodules-only: done (no pip)."
  exit 0
fi

echo "[setup] python: ${PY[*]} ($(${PY[@]} -c 'import sys; print(sys.executable)'))"

# Smoke-check that Isaac is importable before we spend time on pip.
if ! ${PY[@]} -c "import isaacsim" 2>/dev/null; then
  cat >&2 <<'EOF'
[setup] ERROR: this Python cannot import isaacsim.

Install Isaac Sim 6.x first, then re-run this script *inside* that environment:
  - NGC container:  nvcr.io/nvidia/isaac-sim:6.0.0-dev2  (the tag docker/Dockerfile builds on;
                    uses /isaac-sim/python.sh automatically)
  - Local install:  https://docs.isaacsim.omniverse.nvidia.com/current/installation/download.html

Also set:  export OMNI_KIT_ACCEPT_EULA=YES

Only need the git checkouts (e.g. before docker build)?
  ./scripts/setup_sim.sh --submodules-only
EOF
  exit 1
fi

echo "[setup] installing Isaac Lab (Arena pin, --no-deps)…"
shopt -s nullglob
labs=(submodules/IsaacLab-Arena/submodules/IsaacLab/source/isaaclab*/)
if [[ ${#labs[@]} -eq 0 ]]; then
  echo "[setup] ERROR: no isaaclab* packages under IsaacLab/source" >&2
  exit 1
fi
for d in "${labs[@]}"; do
  ${PY[@]} -m pip install --no-deps -e "$d"
done

echo "[setup] installing Isaac Lab Arena…"
${PY[@]} -m pip install -e submodules/IsaacLab-Arena
# Arena's registries import these at registration time but don't declare them
# (same pins as docker/Dockerfile).
${PY[@]} -m pip install "pin-pink==3.1.0" "rsl-rl-lib==5.0.1"

echo "[setup] installing assemble_bench…"
${PY[@]} -m pip install -e assemble_bench

echo "[setup] installing HUD serving stack (Path B)…"
# GymBridge / Shared live on this commit (hud-python#481) until they ship on PyPI.
${PY[@]} -m pip install \
  "hud @ git+https://github.com/hud-evals/hud-python.git@a08d8d83fe56c9427bcba53536c548410dedd330" \
  msgpack
# openpi-client pins numpy<2 but only its msgpack codec is used; av just for wheels.
${PY[@]} -m pip install --no-deps "av>=12" "openpi-client==0.1.2"

cat <<EOF

[setup] done.

Quick smoke (Path A):
  cd submodules/IsaacLab-Arena
  OMNI_KIT_ACCEPT_EULA=YES ${PY[*]} isaaclab_arena/evaluation/policy_runner.py \\
      --policy_type zero_action --num_episodes 1 --headless \\
      --external_environment_class_path \\
      assemble_bench.environments.assembly.assembly:AssembleBenchEnvironment \\
      assemble_bench --task peg_round_8mm

Or:  OMNI_KIT_ACCEPT_EULA=YES ${PY[*]} scripts/preview_assembly.py --task peg_round_8mm --out /tmp/peg
EOF

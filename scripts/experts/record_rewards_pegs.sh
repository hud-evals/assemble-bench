#!/usr/bin/env bash
# Record loose round-peg demos (dense reward + privileged poses, keep_failures)
# for the given sizes. Default reward is staged (legacy dataset); set
# REWARD=potential for the Markov Φ-diff seed used by PLD Stage 1 v10+.
#
# Usage:
#   SIZES="4mm 12mm 16mm" bash scripts/experts/record_rewards_pegs.sh
#   REWARD=potential SIZES="8mm" bash scripts/experts/record_rewards_pegs.sh
#   MODE=smoke REWARD=potential SIZES="8mm" bash scripts/experts/record_rewards_pegs.sh
#
# IMPORTANT: --record must be under /app/assembly_bench (the docker mount); a
# host path is a different filesystem inside the container and is wiped on exit.
set -euo pipefail

CACHE=/home/ubuntu/docker/isaac-expert-cache
HOST_DIR=/home/ubuntu/project/assembly_bench/data/hdf5
LOGDIR=/home/ubuntu/outputs/reward_demos
mkdir -p "$LOGDIR" "$HOST_DIR" "$CACHE"/{kit,ov,glcache,computecache,pip,warp,logs}

SIZES="${SIZES:-4mm 12mm 16mm}"
MODE="${MODE:-bulk}"                 # bulk | smoke
REWARD="${REWARD:-staged}"           # staged | potential
NUM_ENVS="${NUM_ENVS:-16}"           # "batch size 16": parallel envs (96GB VRAM)
MAX_DEMOS="${MAX_DEMOS:-100}"
# Bank successes/failures to this mix (requires --keep_failures). 0.7 -> 70/30.
TARGET_SUCCESS_RATE="${TARGET_SUCCESS_RATE:-0.7}"
WAVES="${WAVES:-40}"                 # 16 envs * 40 waves = 640 episodes cap >> 100

if [[ "$REWARD" != "staged" && "$REWARD" != "potential" ]]; then
  echo "REWARD must be staged|potential (got $REWARD)" >&2
  exit 2
fi

# Suffix host/container hdf5 so potential and staged demos don't clobber each other.
SUFFIX=""
if [[ "$REWARD" == "potential" ]]; then
  SUFFIX="_potential"
fi

if [[ "$MODE" == "smoke" ]]; then
  NUM_ENVS="${NUM_ENVS_SMOKE:-8}"
  WAVES=1
fi

docker rm -f assembly-expert assembly-env smoke-rewards 2>/dev/null || true

for size in $SIZES; do
  task="peg_round_${size}"
  ctr_hdf5="/app/assembly_bench/data/hdf5/${task}_rewards${SUFFIX}.hdf5"
  host_hdf5="$HOST_DIR/${task}_rewards${SUFFIX}.hdf5"

  if [[ "$MODE" == "smoke" ]]; then
    echo "======== SMOKE $task reward=$REWARD n=$NUM_ENVS ========"
    docker run --name smoke-rewards --rm \
      --gpus all --network host \
      -e NVIDIA_DRIVER_CAPABILITIES=all -e OMNI_KIT_ACCEPT_EULA=YES -e PYTHONUNBUFFERED=1 \
      -v /home/ubuntu/project/assembly_bench:/app/assembly_bench \
      -v "$CACHE/kit:/isaac-sim/kit/cache" -v "$CACHE/ov:/root/.cache/ov" \
      -v "$CACHE/glcache:/root/.cache/nvidia/GLCache" -v "$CACHE/computecache:/root/.nv/ComputeCache" \
      -v "$CACHE/pip:/root/.cache/pip" -v "$CACHE/warp:/root/.cache/warp" \
      -v "$CACHE/logs:/root/.nvidia-omniverse/logs" \
      -w /app/assembly_bench hud-assembly-env \
      /isaac-sim/python.sh scripts/experts/smoke_rewards.py \
        --headless --task "$task" --num_envs "$NUM_ENVS" --waves 1 \
        --disable_cameras --no_stream --reward "$REWARD" \
        --reset_warmup_steps 8 --reset_rt_subframes 1 \
      2>&1 | tee "$LOGDIR/${task}_smoke_${REWARD}.log"
    docker rm -f smoke-rewards 2>/dev/null || true
    continue
  fi

  rm -f "$host_hdf5"
  echo "======== RECORD rewards $task reward=$REWARD n=$MAX_DEMOS envs=$NUM_ENVS ========"
  echo "record path (container): $ctr_hdf5"
  echo "host mount:              $host_hdf5"

  docker run --name assembly-expert \
    --gpus all --network host \
    -e NVIDIA_DRIVER_CAPABILITIES=all -e OMNI_KIT_ACCEPT_EULA=YES -e PYTHONUNBUFFERED=1 \
    -v /home/ubuntu/project/assembly_bench:/app/assembly_bench \
    -v "$CACHE/kit:/isaac-sim/kit/cache" -v "$CACHE/ov:/root/.cache/ov" \
    -v "$CACHE/glcache:/root/.cache/nvidia/GLCache" -v "$CACHE/computecache:/root/.nv/ComputeCache" \
    -v "$CACHE/pip:/root/.cache/pip" -v "$CACHE/warp:/root/.cache/warp" \
    -v "$CACHE/logs:/root/.nvidia-omniverse/logs" \
    -w /app/assembly_bench hud-assembly-env \
    /isaac-sim/python.sh scripts/experts/run_expert.py \
      --headless --task "$task" \
      --num_envs "$NUM_ENVS" --waves "$WAVES" --max_demos "$MAX_DEMOS" \
      --keep_failures --target_success_rate "$TARGET_SUCCESS_RATE" \
      --reward "$REWARD" --no_stream \
      --record "$ctr_hdf5" \
      --arm_stiffness 150 --arm_damping 40 \
      --reset_warmup_steps 8 --reset_rt_subframes 1 \
      2>&1 | tee "$LOGDIR/${task}_rewards_${REWARD}.log"

  if [[ -f "$host_hdf5" ]]; then
    ls -lh "$host_hdf5"
  else
    echo "WARN: mount path empty — docker cp rescue"
    docker cp assembly-expert:"$ctr_hdf5" "$host_hdf5"
    ls -lh "$host_hdf5"
  fi
  docker rm -f assembly-expert 2>/dev/null || true
  echo "done: $host_hdf5"
done

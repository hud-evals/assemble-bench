#!/usr/bin/env bash
# Four taskset staged-reward smokes via hud-assembly-env + scripted expert.
set -euo pipefail

CACHE=/home/ubuntu/docker/isaac-expert-cache
LOGDIR=/home/ubuntu/outputs/reward_smokes
mkdir -p "$LOGDIR" "$CACHE"/{kit,ov,glcache,computecache,pip,warp,logs}

# Stop anything holding the GPU.
docker rm -f assembly-expert assembly-env smoke-rewards 2>/dev/null || true

TASKS=(
  peg_round_8mm_loose
  peg_square_8mm_loose
  gear_medium
  nut_m16_loose
)

NUM_ENVS="${NUM_ENVS:-4}"
PASS=0
FAIL=0

for task in "${TASKS[@]}"; do
  log="$LOGDIR/${task}.log"
  echo "======== SMOKE $task ========"
  # Nuts need a longer horizon; square may need a second wave for a success.
  waves=1
  extra=()
  case "$task" in
    peg_square_*) waves=2 ;;
    nut_*) extra+=(--episode_length_s 150 --max_steps 900) ;;
  esac

  set +e
  docker run --name smoke-rewards --rm \
    --gpus all \
    --network host \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e OMNI_KIT_ACCEPT_EULA=YES \
    -e PYTHONUNBUFFERED=1 \
    -v /home/ubuntu/project/assembly_bench:/app/assembly_bench \
    -v "$CACHE/kit:/isaac-sim/kit/cache" \
    -v "$CACHE/ov:/root/.cache/ov" \
    -v "$CACHE/glcache:/root/.cache/nvidia/GLCache" \
    -v "$CACHE/computecache:/root/.nv/ComputeCache" \
    -v "$CACHE/pip:/root/.cache/pip" \
    -v "$CACHE/warp:/root/.cache/warp" \
    -v "$CACHE/logs:/root/.nvidia-omniverse/logs" \
    -w /app/assembly_bench \
    hud-assembly-env \
    /isaac-sim/python.sh scripts/experts/smoke_rewards.py \
      --headless --task "$task" --num_envs "$NUM_ENVS" --waves "$waves" \
      --disable_cameras --no_stream --reward staged \
      --reset_warmup_steps 8 --reset_rt_subframes 1 \
      "${extra[@]}" \
    2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  set -e
  docker rm -f smoke-rewards 2>/dev/null || true

  if [[ "$rc" -eq 0 ]] && grep -q '^PASS:' "$log"; then
    echo "OK $task"
    PASS=$((PASS + 1))
  else
    echo "BAD $task (rc=$rc) — see $log"
    FAIL=$((FAIL + 1))
  fi
done

echo "======== DONE pass=$PASS fail=$FAIL ========"
exit "$FAIL"

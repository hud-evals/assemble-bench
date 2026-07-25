#!/usr/bin/env bash
# Record peg_round_8mm demos with staged reward + privileged poses
# (success + failure) for PA-RL critic pretrain → assembly_bench_rewards.
#
# IMPORTANT: --record must be under /app/assembly_bench (the docker mount).
# A host path like /home/ubuntu/project/... is a different filesystem inside
# the container and is wiped when the container exits.
set -euo pipefail

CACHE=/home/ubuntu/docker/isaac-expert-cache
HOST_DIR=/home/ubuntu/project/assembly_bench/data/hdf5
HOST_HDF5="$HOST_DIR/peg_round_8mm_rewards.hdf5"
# In-container path on the bind mount (literal — do not pass a host path).
CTR_HDF5=/app/assembly_bench/data/hdf5/peg_round_8mm_rewards.hdf5
LOGDIR=/home/ubuntu/outputs/reward_demos
mkdir -p "$LOGDIR" "$HOST_DIR" "$CACHE"/{kit,ov,glcache,computecache,pip,warp,logs}

NUM_ENVS="${NUM_ENVS:-8}"
MAX_DEMOS="${MAX_DEMOS:-100}"
WAVES="${WAVES:-40}"

docker rm -f assembly-expert assembly-env smoke-rewards 2>/dev/null || true
rm -f "$HOST_HDF5"

echo "======== RECORD rewards peg_round_8mm n=$MAX_DEMOS ========"
echo "record path (container): $CTR_HDF5"
echo "host mount:              $HOST_HDF5"

# No --rm: keep the container so we can rescue the file if the mount misbehaves.
docker run --name assembly-expert \
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
  /isaac-sim/python.sh scripts/experts/run_expert.py \
    --headless --task peg_round_8mm \
    --num_envs "$NUM_ENVS" --waves "$WAVES" --max_demos "$MAX_DEMOS" \
    --keep_failures --reward staged --no_stream \
    --record /app/assembly_bench/data/hdf5/peg_round_8mm_rewards.hdf5 \
    --arm_stiffness 150 --arm_damping 40 \
    --reset_warmup_steps 8 --reset_rt_subframes 1 \
    2>&1 | tee "$LOGDIR/peg_round_8mm_rewards.log"

# Prefer the bind-mounted file; fall back to docker cp from either path.
if [[ -f "$HOST_HDF5" ]]; then
  ls -lh "$HOST_HDF5"
else
  echo "WARN: mount path empty — docker cp rescue"
  docker cp assembly-expert:/app/assembly_bench/data/hdf5/peg_round_8mm_rewards.hdf5 "$HOST_HDF5" \
    || docker cp assembly-expert:/home/ubuntu/project/assembly_bench/data/hdf5/peg_round_8mm_rewards.hdf5 "$HOST_HDF5"
  ls -lh "$HOST_HDF5"
fi
docker rm -f assembly-expert 2>/dev/null || true
echo "done: $HOST_HDF5"

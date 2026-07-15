#!/usr/bin/env bash
# Launch a streamed teleop RECORDING session in the container runtime.
# Same working livestream formula as start_teleop_container.sh
# (isaaclab.python.kit + present pass enabled + --viz kit + --livestream 1),
# but runs the assembly_bench record wrapper, which layers the canonical 8-D
# abs_joint_action label stream on top of Arena's record_demos.
#
# Usage:
#   scripts/teleop/start_record_container.sh [task] [num_demos] [dataset_file]
# Defaults: peg_round_8mm_loose, 10 demos,
#           data/teleop/<task>_dik_<UTC timestamp>.hdf5 (repo mount -> host).
#
# Connect the Isaac Sim WebRTC client to this host's public IP. Keyboard:
# W/S A/D Q/E translate, Z/X T/G C/V rotate, K gripper, R discard+reset.
# Successful episodes only are exported; the session auto-stops after 4 h.

set -euo pipefail

TASK="${1:-peg_round_8mm_loose}"
NUM_DEMOS="${2:-10}"
DATASET="${3:-data/teleop/${TASK}_dik_$(date -u +%Y%m%d_%H%M%S).hdf5}"
MAX_SECONDS="${MAX_SECONDS:-14400}"
PUBLIC_IP="${PUBLIC_IP:-$(curl -s --max-time 5 ifconfig.me)}"
# Spawn the DROID wrist + front cameras (the recorded observations) so they can
# be inspected in extra viewports while driving. Set ENABLE_CAMERAS=0 to skip.
ENABLE_CAMERAS="${ENABLE_CAMERAS:-1}"
# With cameras, the .rendering variant is REQUIRED: it carries the
# isaaclab.cameras_enabled=true setting that camera sensors check at spawn
# (an explicit --experience bypasses AppLauncher's automatic selection of it).
if [ "$ENABLE_CAMERAS" = "1" ]; then
  EXPERIENCE="${EXPERIENCE:-isaaclab.python.rendering.kit}"
else
  EXPERIENCE="${EXPERIENCE:-isaaclab.python.kit}"
fi
KIT_ARGS="${KIT_ARGS:---/exts/omni.kit.renderer.core/present/enabled=true}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CACHE=/home/ubuntu/docker/isaac-teleop-cache

mkdir -p "$CACHE"/{kit,ov,glcache,computecache,pip,warp,logs} "$REPO/data/teleop"

docker rm -f assembly-teleop >/dev/null 2>&1 || true

docker run -d --name assembly-teleop \
  --gpus all --network host \
  -v "$REPO":/app/assembly_bench \
  -v "$CACHE/kit":/isaac-sim/kit/cache \
  -v "$CACHE/ov":/root/.cache/ov \
  -v "$CACHE/glcache":/root/.cache/nvidia/GLCache \
  -v "$CACHE/computecache":/root/.nv/ComputeCache \
  -v "$CACHE/pip":/root/.cache/pip \
  -v "$CACHE/warp":/root/.cache/warp \
  -v "$CACHE/logs":/root/.nvidia-omniverse/logs \
  -e OMNI_KIT_ACCEPT_EULA=YES -e PYTHONUNBUFFERED=1 \
  -e PUBLIC_IP="$PUBLIC_IP" \
  -w /root/.nvidia-omniverse/logs \
  hud-assembly-env \
  timeout "$MAX_SECONDS" /isaac-sim/python.sh \
    /app/assembly_bench/scripts/teleop/record_demos.py \
    --experience "$EXPERIENCE" \
    --kit_args="$KIT_ARGS" \
    --viz kit \
    $( [ "$ENABLE_CAMERAS" = "1" ] && echo --enable_cameras ) \
    --livestream 1 --num_envs 1 --step_hz 15 \
    --dataset_file "/app/assembly_bench/$DATASET" \
    --num_demos "$NUM_DEMOS" \
    --external_environment_class_path \
      assembly_bench.environments.assembly.assembly:AssemblyBenchEnvironment \
    assembly_bench --task "$TASK" \
    --embodiment droid_differential_ik --teleop_device keyboard

echo "recording session started (task=$TASK, demos=$NUM_DEMOS, out=$DATASET)"
echo "Readiness: docker logs -f assembly-teleop 2>&1 | grep -i 'recording started'"

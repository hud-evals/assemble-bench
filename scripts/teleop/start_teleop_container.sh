#!/usr/bin/env bash
# Launch the assembly_bench keyboard-teleop session in the proven container
# runtime (the host `isaac` conda env cannot boot the sim; the hud-assembly-env
# image can). Persists Kit's shader/compute/asset caches across restarts so
# boots after the first take ~2-3 min instead of ~10+.
#
# Usage:
#   scripts/teleop/start_teleop_container.sh [task] [max_seconds]
# Defaults: peg_round_8mm_loose, 14400 (4 h auto-stop).
#
# Connect from the Isaac Sim WebRTC Streaming Client at the server's public IP
# (TCP 49100 + UDP 47998 must be reachable). Keyboard, after clicking the
# viewport: W/S A/D Q/E translate, Z/X T/G C/V rotate, K gripper, R reset.

set -euo pipefail

TASK="${1:-peg_round_8mm_loose}"
MAX_SECONDS="${2:-14400}"
PUBLIC_IP="${PUBLIC_IP:-$(curl -s --max-time 5 ifconfig.me)}"
# Livestream needs (a) a UI framebuffer to capture -> the full isaaclab.python
# experience (headless profiles have no window/viewport extensions), and (b) the
# present pass, which isaaclab.python.kit explicitly disables ("Fixes MGPU
# stability issue", single-GPU here so safe to re-enable). Without present the
# NVENC session opens but encodes 0 fps -- the "connects but black screen" bug.
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

mkdir -p "$CACHE"/{kit,ov,glcache,computecache,pip,warp,logs}

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
    /app/assembly_bench/scripts/teleop/teleop_hold.py \
    --experience "$EXPERIENCE" \
    --kit_args="$KIT_ARGS" \
    --viz kit \
    $( [ "$ENABLE_CAMERAS" = "1" ] && echo --enable_cameras ) \
    --livestream 1 --num_envs 1 \
    --external_environment_class_path \
      assembly_bench.environments.assembly.assembly:AssemblyBenchEnvironment \
    assembly_bench --task "$TASK" \
    --embodiment droid_differential_ik --teleop_device keyboard

echo "assembly-teleop started (task=$TASK, public_ip=$PUBLIC_IP, auto-stop=${MAX_SECONDS}s)"
echo "Follow readiness with: docker logs -f assembly-teleop 2>&1 | grep -i 'Teleoperation started'"

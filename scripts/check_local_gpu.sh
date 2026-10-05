#!/usr/bin/env bash
# Preflight for running AssembleBench on a local NVIDIA GPU with Docker.
#
#   ./scripts/check_local_gpu.sh              # host GPU + Docker + image
#   ./scripts/check_local_gpu.sh --served     # also require the env server on :8765
#
# Exits non-zero on the first failed check. It does not start Isaac Sim; the
# scripted move_joints check in the README does that.
set -euo pipefail

IMAGE="${IMAGE:-assemble-bench-env}"
PORT="${PORT:-8765}"
MIN_DRIVER_MAJOR=580
MIN_VRAM_MIB=16000

fail() { echo "[check] FAIL: $*" >&2; exit 1; }
ok() { echo "[check] ok: $*"; }

command -v nvidia-smi >/dev/null || fail "nvidia-smi not found; install the NVIDIA driver"
IFS=, read -r gpu driver vram < <(nvidia-smi --query-gpu=name,driver_version,memory.total \
  --format=csv,noheader,nounits | head -n1)
gpu="${gpu# }"; driver="${driver# }"; vram="${vram# }"
(( ${driver%%.*} >= MIN_DRIVER_MAJOR )) || fail "driver $driver; need $MIN_DRIVER_MAJOR or newer"
(( vram >= MIN_VRAM_MIB )) || fail "$gpu has $vram MiB; need 16 GB+ (24 GB+ if a model shares the GPU)"
case "$gpu" in
  *A100*|*H100*|*H200*|*B200*) fail "$gpu has no RT cores; Isaac Sim cannot render on it" ;;
esac
ok "$gpu, driver $driver, $vram MiB"

command -v docker >/dev/null || fail "docker not found"
docker info --format '{{json .Runtimes}}' | grep -q nvidia \
  || fail "Docker has no nvidia runtime; install nvidia-container-toolkit and run 'sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker'"
ok "docker nvidia runtime"

docker image inspect "$IMAGE" >/dev/null 2>&1 \
  || fail "image '$IMAGE' not built; run 'docker build -f docker/Dockerfile -t $IMAGE .' (needs 'docker login nvcr.io')"
docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all "$IMAGE" nvidia-smi -L >/dev/null \
  || fail "container cannot see the GPU"
ok "container sees the GPU"

if [[ "${1:-}" == "--served" ]]; then
  (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null \
    || fail "nothing listening on 127.0.0.1:$PORT; see 'docker logs -f assemble-bench' (wait for HUD_SERVE_PORT=$PORT)"
  ok "env server answers on 127.0.0.1:$PORT"
fi

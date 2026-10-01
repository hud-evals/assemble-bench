#!/bin/bash
# Install AssembleBench into a Python that already has Isaac Sim 6 (pip).
# nvcr.io/nvidia/isaac-sim requires an NGC login; this is the host install
# from the README, run inside the Modal image. Kit's python.sh is not here.
set -euo pipefail

ROOT=/opt/assemble-bench
cd "$ROOT"

python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
python -c 'import isaacsim; print("isaacsim", isaacsim.__file__)'

shopt -s nullglob
labs=(submodules/IsaacLab-Arena/submodules/IsaacLab/source/isaaclab*/)
if [[ ${#labs[@]} -eq 0 ]]; then
  echo "IsaacLab source packages are missing" >&2
  exit 1
fi
for dir in "${labs[@]}"; do
  python -m pip install --no-cache-dir --no-deps -e "$dir"
done
# --no-deps keeps Kit's torch/warp pins. isaaclab.utils and the teleop
# package import these directly; they are not in the isaacsim wheels.
python -m pip install --no-cache-dir "lazy_loader>=0.4" "toml" "prettytable==3.3.0" "gymnasium==1.2.1"

python -m pip install --no-cache-dir "rsl-rl-lib==5.0.1"
python -m pip install --force-reinstall --no-cache-dir "daqp==0.8.5"
python -m pip install --no-cache-dir "pin-pink==3.1.0"
python -m pip install --no-cache-dir -e submodules/IsaacLab-Arena
python -m pip install --no-cache-dir --no-deps -e assemble_bench

# Same constraint install as docker/Dockerfile: hud may add packages, not
# upgrade the ones Kit already owns.
HUD_GIT_REF=7dd1e3c14d8118605ecf0f2fd10ae4fede85e80c
python -m pip install --no-cache-dir --ignore-installed \
  "packaging>=24.0" "pydantic>=2.11.7" "pyperclip>=1.9.0" "uvicorn>=0.35" "websockets>=15.0.1" \
  "openai==2.44.0"
python -m pip list --format=freeze --exclude-editable \
  | grep -v -E "\+|@ file|prompt.toolkit|^(packaging|pydantic|pyperclip|uvicorn|websockets|openai)==" \
  > /tmp/kit-constraints.txt
python -m pip install --no-cache-dir -c /tmp/kit-constraints.txt \
  "hud @ git+https://github.com/hud-evals/hud-python.git@${HUD_GIT_REF}" msgpack
python -m pip install --no-cache-dir --no-deps "av==18.0.0" "openpi-client==0.1.2"
python -c "from openpi_client import msgpack_numpy; from hud.environment.robot import DirectControl, GymBridge; print('hud robot stack OK')"

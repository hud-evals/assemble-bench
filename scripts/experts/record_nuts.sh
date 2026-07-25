#!/usr/bin/env bash
# Record factory nut-thread demos (M8/M12/M16/M20 loose only) per data.md.
# One Isaac process at a time; append-only HDF5; success-filtered.
set -euo pipefail
cd /app/assembly_bench
mkdir -p data/hdf5
# Most reliable first so we bank demos while iterating flaky tiers.
TASKS=(
  nut_M16
  nut_M12
  nut_M20
  nut_M8
)
MAX_DEMOS="${MAX_DEMOS:-50}"
NUM_ENVS="${NUM_ENVS:-8}"
WAVES="${WAVES:-80}"

for task in "${TASKS[@]}"; do
  out="data/hdf5/${task}.hdf5"
  if [[ -f "$out" ]]; then
    # h5py attr num_demos via python if present
    n=$(/isaac-sim/python.sh - <<PY
import h5py
from pathlib import Path
p=Path("$out")
if not p.is_file():
    print(0)
else:
    with h5py.File(p,"r") as f:
        print(int(f["data"].attrs.get("num_demos", len(f["data"]))))
PY
)
    if [[ "$n" -ge "$MAX_DEMOS" ]]; then
      echo "[record_nuts] skip $task — already $n demos"
      continue
    fi
    echo "[record_nuts] resume $task — have $n / $MAX_DEMOS"
  else
    echo "[record_nuts] start $task → $out"
  fi
  /isaac-sim/python.sh scripts/experts/run_expert.py \
    --headless --task "$task" \
    --num_envs "$NUM_ENVS" --waves "$WAVES" --max_demos "$MAX_DEMOS" \
    --episode_length_s 150 --max_steps 900 \
    --record "$out" \
    --reset_warmup_steps 120 --reset_rt_subframes 32 \
    2>&1 | tee "data/hdf5/${task}.log"
  echo "[record_nuts] finished $task"
done
echo "[record_nuts] ALL DONE"

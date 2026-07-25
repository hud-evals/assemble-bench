"""Offline: replay insert/tilt gates on a HUD trace's privileged poses.

Usage (vla env):
  python assembly_bench/scripts/experts/util/replay_tilt_gate_on_trace.py \\
      --trace-json /tmp/trace_a930f3d3.json
"""

from __future__ import annotations

import argparse
import ast
import json
import math

import torch

from assembly_bench.environments.assembly.insert_fail import (
    PROGRESS_EPS,
    STALL_STEPS as INSERT_STALL,
    at_seat,
    half_radius_m,
    insert_fail_fire,
    peg_diameter_mm,
    peg_hole_xy,
    seat_cost,
    update_insert_stall,
)
from assembly_bench.environments.assembly.tilt_fail import (
    STALL_STEPS as TILT_STALL,
    TILT_COS,
    TILT_COS_INSERT,
    nearly_seated,
    peg_upright_cos,
    tilt_fail_fire,
    update_tilt_stall,
)


def _state(e):
    s = e.get("state")
    return s if isinstance(s, dict) else ast.literal_eval(s)


def _vals(st, key):
    v = st.get(key)
    return v.get("values") if isinstance(v, dict) else v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace-json", required=True)
    p.add_argument("--task", default="peg_round_8mm")
    args = p.parse_args()

    events = json.load(open(args.trace_json))["events"]
    obs = [e for e in events if e.get("kind") == "robot_observation"]

    xy_ok = half_radius_m(peg_diameter_mm(args.task))
    insert_stall = torch.zeros(1, dtype=torch.long)
    tilt_stall = torch.zeros(1, dtype=torch.long)
    prev_cost = float("nan")
    first_insert = first_tilt = None

    for e in obs:
        st = _state(e)
        tick = e.get("tick")
        held = _vals(st, "policy/held_part_pose")
        fixed = _vals(st, "policy/fixed_part_pose")
        grip = float(_vals(st, "policy/gripper_pos")[0])
        peg = torch.tensor([[held[0], held[1], held[2]]])
        hole = torch.tensor([[fixed[0], fixed[1], fixed[2]]])
        # held_part_pose is xyzw (qx,qy,qz,qw) — same as Isaac root_quat_w.
        quat = torch.tensor([[held[3], held[4], held[5], held[6]]])

        closed = torch.tensor([grip >= 0.5])
        seated = at_seat(peg, hole)
        off_center = peg_hole_xy(peg, hole) > xy_ok
        cost = float(seat_cost(peg, hole)[0])
        improving = torch.tensor(
            [math.isfinite(prev_cost) and cost < prev_cost - PROGRESS_EPS]
        )
        insert_stall = update_insert_stall(
            insert_stall,
            closed=closed,
            at_seat_now=seated,
            off_center=off_center,
            improving=improving,
        )
        prev_cost = cost
        if first_insert is None and bool(
            insert_fail_fire(insert_stall, window=INSERT_STALL)[0]
        ):
            first_insert = tick

        cos = peg_upright_cos(quat)
        tipped = torch.where(
            closed & seated, cos < TILT_COS_INSERT, cos < TILT_COS
        )
        tipped = tipped & (peg[:, 2] > 0.005)
        nearly = nearly_seated(peg, hole)
        tilt_stall = update_tilt_stall(tilt_stall, tipped=tipped, seated=nearly)
        if first_tilt is None and bool(tilt_fail_fire(tilt_stall, window=TILT_STALL)[0]):
            first_tilt = tick

    ea = [_vals(_state(e), "policy/expert_active")[0] for e in obs]
    print(f"ticks={len(obs)}  expert_active_any={any(v > 0.5 for v in ea)}")
    print(f"xy_ok (half radius)={xy_ok*1000:.2f} mm")
    print(f"insert_fail would fire at tick: {first_insert}")
    print(f"tilt_fail   would fire at tick: {first_tilt}")


if __name__ == "__main__":
    main()

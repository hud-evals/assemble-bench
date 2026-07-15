"""Diversity gate for success-filtered expert HDF5s (what LeRobot doctor misses).

Computes cross-episode start-state std, episode-length std, and pairwise
trajectory L2 in joint space. Near-zero values mean a single-template dataset.

    conda run -n vla python scripts/experts/util/diversity_check.py data/peg_round_8mm_tight.hdf5
"""

import argparse
import sys

import h5py
import numpy as np

# DROID home arm pose (7 joints) -- every episode must START here, or the demo
# conditions the policy on states eval (which resets to home) never sees. This
# catches the settle-reset bug where the arm was dragged off home before frame 0.
HOME_JOINTS = np.array([0.0, -0.2 * np.pi, 0.0, -0.8 * np.pi, 0.0, 0.6 * np.pi, 0.0])
HOME_TOL = 0.20  # per-episode L2 over 7 arm joints (rad); ~small reset noise only


def check(path, traj_pairs=24, traj_frames=32, seed=0):
    """Return (ok, metrics dict) for one HDF5."""
    with h5py.File(path, "r") as f:
        keys = sorted((k for k in f["data"] if k.startswith("demo_")),
                      key=lambda s: int(s.split("_")[1]))
        if not keys:
            return False, {"error": "0 demos"}
        starts, lens, trajs = [], [], []
        for k in keys:
            st = f["data"][k]["obs"]["state"][:].astype(np.float64)
            starts.append(st[0])
            lens.append(len(st))
            idx = np.linspace(0, len(st) - 1, min(traj_frames, len(st))).astype(int)
            trajs.append(st[idx])

    starts, lens = np.stack(starts), np.asarray(lens)
    start_std = starts.std(axis=0)
    N = len(keys)

    # Frame-0 distance from DROID home per episode (settle-reset bug detector).
    home_err = np.linalg.norm(starts[:, :7] - HOME_JOINTS, axis=1)
    off_home = int((home_err > HOME_TOL).sum())

    # Pairwise mean per-frame L2 on arm joints (subsample episodes).
    rng = np.random.default_rng(seed)
    M = min(N, traj_pairs)
    pick = rng.choice(N, size=M, replace=False) if N > M else np.arange(N)
    dists = []
    for i in range(len(pick)):
        for j in range(i + 1, len(pick)):
            a, b = trajs[pick[i]][:, :7], trajs[pick[j]][:, :7]
            T = min(len(a), len(b))
            dists.append(np.linalg.norm(a[:T] - b[:T], axis=1).mean())
    dists = np.asarray(dists) if dists else np.zeros(1)

    metrics = {
        "demos": N,
        "ep_len_mean": float(lens.mean()),
        "ep_len_std": float(lens.std()),
        "ep_len_min": int(lens.min()),
        "ep_len_max": int(lens.max()),
        "start_joint_std_mean": float(start_std[:7].mean()),
        "start_grip_std": float(start_std[7]),
        "pairwise_traj_L2_mean": float(dists.mean()),
        "pairwise_traj_L2_min": float(dists.min()),
        "start_home_err_mean": float(home_err.mean()),
        "start_home_err_max": float(home_err.max()),
        "off_home": off_home,
    }

    # Heuristic fails: near-zero diversity = single template; off-home = the
    # settle-reset bug leaked non-home starts into the demos.
    ok = True
    fails = []
    if metrics["start_joint_std_mean"] < 1e-3:
        ok, fails = False, fails + ["start-state joint std ~0"]
    if metrics["ep_len_std"] < 1.0:
        ok, fails = False, fails + ["episode-length std ~0"]
    if metrics["pairwise_traj_L2_mean"] < 0.01:
        ok, fails = False, fails + ["pairwise traj L2 ~0"]
    if off_home > 0:
        ok, fails = False, fails + [f"{off_home}/{N} start off-home (>{HOME_TOL} rad)"]
    metrics["fails"] = fails
    return ok, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hdf5", nargs="+", help="success-filtered expert HDF5 path(s)")
    args = ap.parse_args()
    all_ok = True
    for path in args.hdf5:
        ok, m = check(path)
        all_ok &= ok
        print(f"\n{path}")
        if "error" in m:
            print(f"  ERROR: {m['error']}")
            all_ok = False
            continue
        print(f"  demos={m['demos']}  ep_len mean/std/min/max="
              f"{m['ep_len_mean']:.1f}/{m['ep_len_std']:.1f}/{m['ep_len_min']}/{m['ep_len_max']}")
        print(f"  start_joint_std_mean={m['start_joint_std_mean']:.4f}  "
              f"start_grip_std={m['start_grip_std']:.4f}")
        print(f"  pairwise_traj_L2 mean/min="
              f"{m['pairwise_traj_L2_mean']:.4f}/{m['pairwise_traj_L2_min']:.4f}")
        print(f"  start_home_err mean/max={m['start_home_err_mean']:.3f}/"
              f"{m['start_home_err_max']:.3f}  off_home={m['off_home']}/{m['demos']}")
        print(f"  DIVERSITY_GATE: {'PASS' if ok else 'FAIL'} "
              + (f"({'; '.join(m['fails'])})" if m["fails"] else ""))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

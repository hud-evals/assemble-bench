"""Convert recorded expert HDF5s into ONE multi-task LeRobot v3.0 dataset.

The AssembleBench dataset: action = the native DROID 8-D joint-position
command (7 absolute arm joint targets + binary gripper) the IK expert emitted;
state = [joint_pos(7), gripper_pos(1)] -- the exact pi0.5-DROID contract --
plus joint_vel(7), eef_pos(3, base frame) and eef_quat(4, world wxyz) as separate
obs keys; images at DROID-RLDS 320x180. Every recorded variant HDF5 becomes tasks
in one dataset, keyed by its per-episode instruction. Runs in the ``vla`` env
(no Isaac needed).

**Frame alignment:** LeRobot rows are written as pre-step ``(s_t, a_t, r_t)``
(state before action). Legacy HDF5 with missing/``post_step`` ``sa_align`` is
realigned via ``inventory.recording.sa_align`` before ``add_frame``.

    conda run -n vla python scripts/experts/util/convert_lerobot.py --push
"""

import argparse
import glob
import os
import shutil
import sys

# Accelerated LFS uploaders silently TRUNCATE large parquet -- disable pre-import.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import h5py
import numpy as np

# util/ -> experts/ -> scripts/ -> assemble_bench/
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# project root (parent of assemble_bench) for inventory.recording.sa_align
_PROJECT = os.path.dirname(ROOT)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)
from inventory.recording.sa_align import maybe_realign_episode  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--glob", default=os.path.join(ROOT, "data", "hdf5", "*.hdf5"))
parser.add_argument("--repo_id", default="hud-evals/AssembleBench")
parser.add_argument("--fps", type=int, default=15)
parser.add_argument("--crf", type=int, default=20)
parser.add_argument("--push", action="store_true")
# Append into an existing Hub/local LeRobot dataset instead of creating fresh
# (needed when peg/gear tasks already live on the Hub and only nuts are new).
parser.add_argument("--append", action="store_true",
                    help="resume writing into existing repo_id (download Hub → local, then append)")
args = parser.parse_args()

out_root = os.path.join(ROOT, "data", "lerobot", args.repo_id.split("/")[-1])
# Fresh create needs a clean root; --append downloads into it instead.
if not args.append and os.path.exists(out_root):
    shutil.rmtree(out_root)

from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

H, W = 180, 320   # DROID-RLDS resolution (native 1280x720 downscaled 1/4, 16:9)
features = {
    "observation.images.front": {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channel"]},
    "observation.images.wrist": {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channel"]},
    "observation.state": {"dtype": "float32", "shape": (8,), "names": None},      # joint(7)+gripper(1)
    "observation.joint_vel": {"dtype": "float32", "shape": (7,), "names": None},  # arm joint vel (rad/s)
    "observation.eef_pos": {"dtype": "float32", "shape": (3,), "names": None},    # base-frame XYZ (m)
    "observation.eef_quat": {"dtype": "float32", "shape": (4,), "names": None},   # world WXYZ
    "action": {"dtype": "float32", "shape": (8,), "names": None},                 # 7 joint targets + gripper
}

# Probe first HDF5 for optional PA-RL fields (reward + privileged poses).
_probe_files = sorted(glob.glob(args.glob))
_has_reward = _has_poses = False
if _probe_files:
    with h5py.File(_probe_files[0], "r") as _h:
        _d0 = next((k for k in _h["data"] if k.startswith("demo_")), None)
        if _d0 is not None:
            _has_reward = "reward" in _h["data"][_d0]
            _has_poses = "held_part_pose" in _h["data"][_d0]["obs"]
if _has_poses:
    features["observation.held_part_pose"] = {"dtype": "float32", "shape": (7,), "names": None}
    features["observation.fixed_part_pose"] = {"dtype": "float32", "shape": (7,), "names": None}
if _has_reward:
    features["next.reward"] = {"dtype": "float32", "shape": (1,), "names": None}
print(f"[convert] optional fields: reward={_has_reward} privileged_poses={_has_poses}", flush=True)

rgb_encoder = RGBEncoderConfig(vcodec="h264", crf=args.crf)
if args.append:
    from huggingface_hub import snapshot_download
    print(f"downloading {args.repo_id} → {out_root} for append ...", flush=True)
    snapshot_download(args.repo_id, repo_type="dataset", local_dir=out_root)
    ds = LeRobotDataset.resume(
        args.repo_id, root=out_root, force_cache_sync=True, rgb_encoder=rgb_encoder,
    )
    print(f"resumed: episodes={ds.meta.total_episodes} frames={ds.meta.total_frames} "
          f"tasks={ds.meta.total_tasks}", flush=True)
else:
    # H.264 at a sharper CRF than the AV1 default (lerobot 0.6.0 RGBEncoderConfig).
    ds = LeRobotDataset.create(args.repo_id, fps=args.fps, features=features, root=out_root,
                               use_videos=True, metadata_buffer_size=1,
                               rgb_encoder=rgb_encoder)

files = sorted(glob.glob(args.glob))
tasks_written = []
for hdf5 in files:
    with h5py.File(hdf5, "r") as h:
        data = h["data"]
        demo_keys = sorted((k for k in data if k.startswith("demo_")), key=lambda s: int(s.split("_")[1]))
        if not demo_keys:
            print(f"skip {os.path.basename(hdf5)} (0 demos)", flush=True)
            continue
        task = str(data.attrs["language"])
        tasks_written.append((os.path.basename(hdf5), task, len(demo_keys)))
        for k in demo_keys:
            o = data[k]["obs"]
            state = o["state"][:].astype(np.float32)
            joint_vel = o["joint_vel"][:].astype(np.float32)
            eef_pos = o["eef_pos"][:].astype(np.float32)
            eef_quat = o["eef_quat"][:].astype(np.float32)
            action = data[k]["action"][:].astype(np.float32)
            wrist, front = o["wrist_rgb"][:], o["front_rgb"][:]
            held = o["held_part_pose"][:].astype(np.float32) if "held_part_pose" in o else None
            fixed = o["fixed_part_pose"][:].astype(np.float32) if "fixed_part_pose" in o else None
            reward = data[k]["reward"][:].astype(np.float32) if "reward" in data[k] else None
            # Ensure pre_step (s_t, a_t, r_t); no-op if HDF5 already tagged.
            extras = {"joint_vel": joint_vel, "eef_pos": eef_pos, "eef_quat": eef_quat,
                      "wrist": wrist, "front": front}
            if held is not None:
                extras["held"] = held
                extras["fixed"] = fixed
            state, action, reward, extras, align = maybe_realign_episode(
                state, action, reward, data[k].attrs, data.attrs, **extras)
            joint_vel, eef_pos, eef_quat = extras["joint_vel"], extras["eef_pos"], extras["eef_quat"]
            wrist, front = extras["wrist"], extras["front"]
            held = extras.get("held")
            fixed = extras.get("fixed")
            for t in range(len(state)):
                frame = {
                    "observation.images.front": front[t],
                    "observation.images.wrist": wrist[t],
                    "observation.state": state[t],
                    "observation.joint_vel": joint_vel[t],
                    "observation.eef_pos": eef_pos[t],
                    "observation.eef_quat": eef_quat[t],
                    "action": action[t],
                    "task": task,
                }
                if held is not None:
                    frame["observation.held_part_pose"] = held[t]
                    frame["observation.fixed_part_pose"] = fixed[t]
                if reward is not None:
                    frame["next.reward"] = np.asarray([reward[t]], dtype=np.float32)
                ds.add_frame(frame)
            ds.save_episode()
        print(f"wrote {len(demo_keys)} episodes from {os.path.basename(hdf5)} "
              f"(last demo sa_align={align})", flush=True)

ds.finalize()
print(f"LeRobot dataset at {out_root}; tasks={len(tasks_written)} "
      f"episodes={ds.num_episodes} frames={ds.num_frames}", flush=True)
for name, task, n in tasks_written:
    print(f"  {name}: {n} demos | \"{task}\"", flush=True)


def _task_rows():
    """Prefer full post-finalize task table (covers append + prior Hub tasks)."""
    try:
        from collections import Counter

        import pandas as pd
        tasks_path = os.path.join(out_root, "meta", "tasks.parquet")
        if not os.path.isfile(tasks_path):
            raise FileNotFoundError(tasks_path)
        t = pd.read_parquet(tasks_path)
        counts = Counter()
        # Episode meta may span multiple chunk files after append.
        ep_root = os.path.join(out_root, "meta", "episodes")
        for dp, _, fns in os.walk(ep_root):
            for fn in fns:
                if not fn.endswith(".parquet"):
                    continue
                ep = pd.read_parquet(os.path.join(dp, fn), columns=["tasks"])
                for cell in ep["tasks"]:
                    for s in (cell if isinstance(cell, (list, tuple)) else [cell]):
                        counts[s] += 1
        return [(str(task_str), counts.get(task_str, "?")) for task_str, _ in t.iterrows()]
    except Exception as e:
        print(f"[card] task table fallback ({e})", flush=True)
        return [(t, n) for _, t, n in tasks_written]


def _dataset_card():
    user, name = args.repo_id.split("/")
    viz = f"https://huggingface.co/spaces/lerobot/visualize_dataset?path={user}%2F{name}"
    rows = "\n".join(f"| `{t}` | {n} |" for t, n in _task_rows())
    return f"""---
license: apache-2.0
task_categories:
- robotics
tags:
- LeRobot
- robotics
- isaac-sim
- assembly
- manipulation
configs:
- config_name: default
  data_files: data/*/*.parquet
---

This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).

<a class="flex" href="{viz}">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/badges/resolve/main/visualize-this-dataset-xl.svg"/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/badges/resolve/main/visualize-this-dataset-dark-xl.svg"/>
</a>

# {name}

Contact-rich **assembly** demonstrations (NIST peg/gear/nut) generated by a privileged scripted IK
expert in NVIDIA Isaac Lab (Arena). Actions are the **native DROID joint-position** command
(7 absolute arm joint targets + binary gripper), so pi0.5/openpi/GR00T consume them directly.

- **Robot:** Franka Panda + Robotiq 2F-85 (DROID platform); expert drives a DifferentialIKController
  (dls) into stiff PD joint-position control.
- **Control / record rate:** **{args.fps} Hz**.

## Tasks (per-episode instruction)

| task | demos |
|------|-------|
{rows}

## Observation / action / state layout

| key | dtype | shape | meaning / units |
|-----|-------|-------|-----------------|
| `observation.images.front` | video (H.264, CRF {args.crf}) | (180,320,3) | third-person RGB, uint8, DROID-RLDS resolution |
| `observation.images.wrist` | video (H.264, CRF {args.crf}) | (180,320,3) | eye-in-hand RGB, uint8, DROID-RLDS resolution |
| `observation.state` | float32 | (8,) | joint_position (7, rad) + gripper_position (1, 0=open 1=closed) |
| `observation.joint_vel` | float32 | (7,) | arm joint velocities (rad/s, no gripper) |
| `observation.eef_pos` | float32 | (3,) | end-effector position, robot base frame (m) |
| `observation.eef_quat` | float32 | (4,) | end-effector orientation, world frame, **wxyz** |
| `action` | float32 | (8,) | 7 absolute arm joint-position targets (rad) + gripper (1: 1=close, 0=open) |
"""


if args.push:
    import subprocess
    import time

    from huggingface_hub import HfApi
    api = HfApi()
    print(f"pushing to https://huggingface.co/datasets/{args.repo_id} (public) ...", flush=True)
    api.create_repo(args.repo_id, repo_type="dataset", private=False, exist_ok=True)
    with open(os.path.join(out_root, "README.md"), "w") as f:
        f.write(_dataset_card())

    def served(rel):
        """Download served bytes; return (size, sha256, parquet_tail)."""
        import hashlib
        url = f"https://huggingface.co/datasets/{args.repo_id}/resolve/main/{rel}"
        subprocess.run(["curl", "-sL", url, "-o", "/tmp/served.bin"], check=False)
        try:
            data = open("/tmp/served.bin", "rb").read()
            return len(data), hashlib.sha256(data).hexdigest(), data[-4:]
        except OSError:
            return -1, "", b""

    def local_sha(path):
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    # Skip HF download cache left by --append (hub rejects uploads under .cache/).
    paths = sorted(
        os.path.relpath(os.path.join(dp, fn), out_root)
        for dp, _, fns in os.walk(out_root) for fn in fns
        if ".cache" not in os.path.relpath(os.path.join(dp, fn), out_root).split(os.sep)
    )
    for rel in paths:
        local = os.path.join(out_root, rel)
        size = os.path.getsize(local)
        lsha = local_sha(local)
        is_parquet = rel.endswith(".parquet")
        for _ in range(8):
            hsz, hsha, tail = served(rel)
            # Size alone is not enough (info.json stayed 3719B while content changed).
            if hsz == size and hsha == lsha and (not is_parquet or tail == b"PAR1"):
                break
            api.upload_file(path_or_fileobj=local, path_in_repo=rel, repo_id=args.repo_id,
                            repo_type="dataset", commit_message=f"upload {rel}")
            time.sleep(2)
        hsz, hsha, tail = served(rel)
        assert hsz == size and hsha == lsha and (not is_parquet or tail == b"PAR1"), \
            f"{rel} mismatch (served {hsz}/{size}, sha {hsha[:8]}!={lsha[:8]}, tail={tail})"
        print(f"verified {rel} ({size} bytes)", flush=True)

    main = [b.target_commit for b in api.list_repo_refs(args.repo_id, repo_type="dataset").branches
            if b.name == "main"][0]
    try:
        api.delete_tag(args.repo_id, tag="v3.0", repo_type="dataset")
    except Exception:
        pass
    api.create_tag(args.repo_id, tag="v3.0", revision=main, repo_type="dataset")
    print(f"done: https://huggingface.co/datasets/{args.repo_id}", flush=True)

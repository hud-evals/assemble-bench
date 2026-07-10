"""Success-filtered HDF5 recorder for the scripted experts.

Accumulates the VLA modality per control tick (wrist + front RGB at 224, DROID
proprio, the 8-D joint action), one buffer per env. An env stops recording the
instant it terminates; only SUCCESS episodes are written, trimmed to the frame
the success fired. Matches the pi0.5-DROID contract: state = [joint_pos(7),
gripper_pos(1)], action = [7 joint targets, gripper]. Consumed by
convert_lerobot.py (in the ``vla`` env).
"""

import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import warp as wp

IMG = 224


def _imgs(obs_cam, key):
    """(N,H,W,3) uint8 tensor -> (N,IMG,IMG,3) uint8 numpy (plain resize)."""
    x = obs_cam[key].to(torch.float32).permute(0, 3, 1, 2)     # N,3,H,W
    x = F.interpolate(x, size=(IMG, IMG), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).cpu().numpy()


class Recorder:
    """One HDF5 file, appended across waves. Call ``step`` after every env.step
    and ``flush`` at each wave end to write that wave's successful episodes."""

    def __init__(self, base, out_path, task_id, instruction):
        self.base = base
        self.out = out_path
        self.task_id = task_id
        self.instruction = instruction
        self.N = base.num_envs
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        self.h = h5py.File(out_path, "w")
        self.g = self.h.create_group("data")
        self.g.attrs["task"] = task_id
        self.g.attrs["language"] = instruction
        self.g.attrs["source"] = "scripted_peg"
        self.n_demos = 0
        self._new_wave()

    def _new_wave(self):
        self.buf = [[] for _ in range(self.N)]      # per-env list of frame dicts
        self.active = [True] * self.N
        self.ep_success = [False] * self.N

    def step(self, obs, action, done, success):
        """Append one recorded frame per still-active env; freeze on done."""
        cam = obs["camera_obs"]
        wrist = _imgs(cam, "wrist_camera_rgb")
        front = _imgs(cam, "front_cam_rgb")
        joint = obs["policy"]["joint_pos"].cpu().numpy().astype(np.float32)
        grip = obs["policy"]["gripper_pos"].cpu().numpy().astype(np.float32)
        state = np.concatenate([joint, grip], axis=1)           # (N,8)
        act = action.detach().cpu().numpy().astype(np.float32)
        done = done.cpu().numpy()
        success = success.cpu().numpy()
        for i in range(self.N):
            if not self.active[i]:
                continue
            self.buf[i].append({"wrist_rgb": wrist[i], "front_rgb": front[i],
                                 "state": state[i], "action": act[i]})
            if done[i]:
                self.active[i] = False
                self.ep_success[i] = bool(success[i])

    def flush(self):
        """Write this wave's successful episodes, then reset for the next wave."""
        kept = 0
        for i in range(self.N):
            if not self.ep_success[i] or not self.buf[i]:
                continue
            frames = self.buf[i]
            d = self.g.create_group(f"demo_{self.n_demos}")
            d.attrs["success"] = True
            o = d.create_group("obs")
            for k in ("wrist_rgb", "front_rgb", "state"):
                o.create_dataset(k, data=np.stack([f[k] for f in frames]),
                                 compression="gzip", compression_opts=4)
            d.create_dataset("action", data=np.stack([f["action"] for f in frames]))
            self.n_demos += 1
            kept += 1
        self.g.attrs["num_demos"] = self.n_demos
        self._new_wave()
        return kept

    def close(self):
        self.h.close()
        print(f"[record] wrote {self.n_demos} demos -> {self.out}", flush=True)

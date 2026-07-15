"""Success-filtered HDF5 recorder for the scripted experts.

Accumulates the VLA modality per control tick (wrist + front RGB at DROID-RLDS
320x180, DROID proprio, the 8-D joint action), one buffer per env. On terminate
(IsaacLab auto-reset), a successful buffer is written immediately and recording
re-arms for the next episode in the same wave. state = [joint_pos(7),
gripper_pos(1)] (pi0.5-DROID contract), plus joint_vel(7), eef_pos(3, world XYZ)
and eef_quat(4, world WXYZ); action = [7 joint targets, gripper]. Consumed by
util/convert_lerobot.py (in the ``vla`` env).
"""

import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F

# DROID-native cameras render 1280x720; store the DROID-RLDS 320x180 (exact 1/4,
# 16:9 preserved). Model-agnostic: each VLA resizes/pads from here itself.
IMG_H, IMG_W = 180, 320


def _imgs(obs_cam, key):
    """(N,H,W,C) uint8 tensor -> (N,180,320,3) uint8 numpy (area downscale)."""
    x = obs_cam[key][..., :3].to(torch.float32).permute(0, 3, 1, 2)   # N,3,H,W
    x = F.interpolate(x, size=(IMG_H, IMG_W), mode="area")
    return x.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8).cpu().numpy()


class Recorder:
    """One HDF5 file, appended across waves. Call ``step`` after every env.step
    and ``flush`` at each wave end to write that wave's successful episodes."""

    def __init__(self, base, out_path, task_id, instruction, max_demos=None):
        self.base = base
        self.out = out_path
        self.task_id = task_id
        self.instruction = instruction
        self.max_demos = max_demos
        self.N = base.num_envs
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        self.h = h5py.File(out_path, "a")
        if "data" in self.h:
            self.g = self.h["data"]
            if self.g.attrs.get("task") != task_id:
                raise ValueError(f"{out_path} contains task {self.g.attrs.get('task')!r}, not {task_id!r}")
            self.n_demos = int(self.g.attrs.get("num_demos", 0))
        else:
            self.g = self.h.create_group("data")
            self.g.attrs["task"] = task_id
            self.g.attrs["language"] = instruction
            self.g.attrs["source"] = "scripted_privileged"
            self.g.attrs["num_demos"] = 0
            self.n_demos = 0
        self._new_wave()

    def _new_wave(self):
        self.buf = [[] for _ in range(self.N)]      # per-env list of frame dicts
        self.wave_demos = 0

    def step(self, obs, action, done, success):
        """Append one recorded frame per env; on done commit + re-arm.

        IsaacLab auto-resets a slot on terminate (obs is already post-reset) and
        the expert restarts its phase machine for a new episode in the same
        wave. The done-step obs belongs to the NEXT episode -- don't glue it
        onto the finished demo (same boundary rule as ``hud.wrap``).
        """
        cam = obs["camera_obs"]
        wrist = _imgs(cam, "wrist_camera_rgb")
        front = _imgs(cam, "front_cam_rgb")
        pol = obs["policy"]

        def f32(key):
            return pol[key].cpu().numpy().astype(np.float32)

        state = np.concatenate([f32("joint_pos"), f32("gripper_pos")], axis=1)  # (N,8)
        joint_vel = f32("joint_vel")                                            # (N,7)
        eef_pos, eef_quat = f32("eef_pos"), f32("eef_quat")   # world XYZ / WXYZ
        act = action.detach().cpu().numpy().astype(np.float32)
        done = done.cpu().numpy()
        success = success.cpu().numpy()
        for i in range(self.N):
            frame = {"wrist_rgb": wrist[i], "front_rgb": front[i],
                     "state": state[i], "joint_vel": joint_vel[i],
                     "eef_pos": eef_pos[i], "eef_quat": eef_quat[i],
                     "action": act[i]}
            if done[i]:
                below_limit = self.max_demos is None or self.n_demos < self.max_demos
                if bool(success[i]) and self.buf[i] and below_limit:
                    self._write_demo(self.buf[i])
                self.buf[i] = [frame]  # post-reset frame starts the next episode
            else:
                self.buf[i].append(frame)

    def _write_demo(self, frames):
        d = self.g.create_group(f"demo_{self.n_demos}")
        d.attrs["success"] = True
        o = d.create_group("obs")
        for k in ("wrist_rgb", "front_rgb", "state", "joint_vel", "eef_pos", "eef_quat"):
            o.create_dataset(k, data=np.stack([f[k] for f in frames]),
                             compression="gzip", compression_opts=4)
        d.create_dataset("action", data=np.stack([f["action"] for f in frames]))
        self.n_demos += 1
        self.wave_demos += 1
        self.g.attrs["num_demos"] = self.n_demos

    def flush(self):
        """Return demos committed this wave, then reset buffers for the next."""
        kept = self.wave_demos
        self._new_wave()
        return kept

    def close(self):
        self.h.close()
        print(f"[record] wrote {self.n_demos} demos -> {self.out}", flush=True)

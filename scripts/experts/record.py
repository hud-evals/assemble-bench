"""HDF5 recorder for the scripted experts.

Accumulates the VLA modality per control tick (wrist + front RGB at DROID-RLDS
320x180, DROID proprio, the 8-D joint action), plus optional staged reward and
privileged part poses for PA-RL critic pretrain. One buffer per env. On
terminate (IsaacLab auto-reset), a finished buffer is written (success-only by
default; ``keep_failures`` keeps every completed episode).
"""

import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F

# Store the DROID-RLDS 320x180 (16:9), model-agnostic: each VLA resizes/pads
# from here itself. Cameras render 640x360 (see cameras.py) -> clean 2x area
# downscale; the downscale also anti-aliases residual RTX grain.
IMG_H, IMG_W = 180, 320

# Obs keys always written (images + proprio + action).
_BASE_OBS = ("wrist_rgb", "front_rgb", "state", "joint_vel", "eef_pos", "eef_quat")
# Optional privileged / reward fields (written when present on the frame).
_OPT_OBS = ("held_part_pose", "fixed_part_pose")


def _imgs(obs_cam, key):
    """(N,H,W,C) uint8 tensor -> (N,180,320,3) uint8 numpy (area downscale)."""
    x = obs_cam[key][..., :3].to(torch.float32).permute(0, 3, 1, 2)   # N,3,H,W
    x = F.interpolate(x, size=(IMG_H, IMG_W), mode="area")
    return x.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8).cpu().numpy()


class Recorder:
    """One HDF5 file, appended across waves. Call ``step`` after every env.step
    and ``flush`` at each wave end to write that wave's finished episodes."""

    def __init__(self, base, out_path, task_id, instruction, max_demos=None,
                 keep_failures=False):
        self.base = base
        self.out = out_path
        self.task_id = task_id
        self.instruction = instruction
        self.max_demos = max_demos
        self.keep_failures = keep_failures
        self.N = base.num_envs
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
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

    def step(self, obs, action, done, success, reward=None):
        """Append one recorded frame per env; on done commit + re-arm.

        IsaacLab auto-resets a slot on terminate (obs is already post-reset) and
        the expert restarts its phase machine for a new episode in the same
        wave. The done-step obs belongs to the NEXT episode -- don't glue it
        onto the finished demo (same boundary rule as ``hud.wrap``). Terminal
        reward is folded into the last pre-reset frame when present.
        """
        cam = obs["camera_obs"]
        wrist = _imgs(cam, "wrist_camera_rgb")
        front = _imgs(cam, "front_cam_rgb")
        pol = obs["policy"]

        def f32(key):
            return pol[key].cpu().numpy().astype(np.float32)

        state = np.concatenate([f32("joint_pos"), f32("gripper_pos")], axis=1)  # (N,8)
        joint_vel = f32("joint_vel")                                            # (N,7)
        # ee_pos is body_pos_w (raw world), so with N parallel envs on a grid it
        # carries the per-env origin offset (~+-15 m). Subtract env_origins to
        # store it env-local -- i.e. relative to the robot base at the origin,
        # the DROID "eef pose in base frame" convention, and consistent with
        # every other env-local quantity. Orientation is offset-invariant.
        eef_pos = (pol["eef_pos"] - self.base.scene.env_origins).cpu().numpy().astype(np.float32)
        eef_quat = f32("eef_quat")                            # world WXYZ (frame-invariant)
        act = action.detach().cpu().numpy().astype(np.float32)
        done = done.cpu().numpy()
        success = success.cpu().numpy()
        rew = None if reward is None else np.asarray(
            reward.detach().cpu() if torch.is_tensor(reward) else reward, dtype=np.float32)
        # Privileged poses (env-local xyz + xyzw) when the embodiment exposes them.
        held = f32("held_part_pose") if "held_part_pose" in pol else None
        fixed = f32("fixed_part_pose") if "fixed_part_pose" in pol else None

        for i in range(self.N):
            frame = {"wrist_rgb": wrist[i], "front_rgb": front[i],
                     "state": state[i], "joint_vel": joint_vel[i],
                     "eef_pos": eef_pos[i], "eef_quat": eef_quat[i],
                     "action": act[i]}
            if held is not None:
                frame["held_part_pose"] = held[i]
            if fixed is not None:
                frame["fixed_part_pose"] = fixed[i]
            if rew is not None:
                frame["reward"] = np.float32(rew[i])
            if done[i]:
                # Terminal reward belongs to the last pre-reset action.
                if self.buf[i] and rew is not None:
                    self.buf[i][-1]["reward"] = np.float32(
                        self.buf[i][-1].get("reward", 0.0) + float(rew[i]))
                below_limit = self.max_demos is None or self.n_demos < self.max_demos
                keep = bool(success[i]) or self.keep_failures
                if keep and self.buf[i] and below_limit:
                    self._write_demo(self.buf[i], success=bool(success[i]))
                self.buf[i] = [frame]  # post-reset frame starts the next episode
            else:
                self.buf[i].append(frame)

    def _write_demo(self, frames, *, success):
        d = self.g.create_group(f"demo_{self.n_demos}")
        d.attrs["success"] = bool(success)
        o = d.create_group("obs")
        for k in _BASE_OBS:
            o.create_dataset(k, data=np.stack([f[k] for f in frames]),
                             compression="gzip", compression_opts=4)
        for k in _OPT_OBS:
            if k in frames[0]:
                o.create_dataset(k, data=np.stack([f[k] for f in frames]),
                                 compression="gzip", compression_opts=4)
        d.create_dataset("action", data=np.stack([f["action"] for f in frames]))
        if "reward" in frames[0]:
            d.create_dataset("reward", data=np.asarray(
                [f.get("reward", 0.0) for f in frames], dtype=np.float32))
        self.n_demos += 1
        self.wave_demos += 1
        self.g.attrs["num_demos"] = self.n_demos
        self.h.flush()  # kill-safe: land demos on disk each commit

    def commit_open(self):
        """Write any still-open buffers as failures (wave ended without env done)."""
        if not self.keep_failures:
            return
        for i in range(self.N):
            below_limit = self.max_demos is None or self.n_demos < self.max_demos
            if self.buf[i] and below_limit and len(self.buf[i]) > 1:
                self._write_demo(self.buf[i], success=False)
            self.buf[i] = []

    def flush(self):
        """Return demos committed this wave, then reset buffers for the next."""
        kept = self.wave_demos
        self._new_wave()
        return kept

    def close(self):
        self.h.close()
        print(f"[record] wrote {self.n_demos} demos -> {self.out}", flush=True)

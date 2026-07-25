"""HDF5 recorder for the scripted experts.

Accumulates the VLA modality per control tick (wrist + front RGB at DROID-RLDS
320x180, DROID proprio, the 8-D joint action), plus optional staged reward and
privileged part poses for PA-RL critic pretrain. One buffer per env. On
terminate (IsaacLab auto-reset), a finished buffer is written (success-only by
default; ``keep_failures`` keeps every completed episode).

**Recording convention (``sa_align="pre_step"``):** each row is
``(s_t, a_t, r_t)`` — observation *before* the action, action taken *from*
that state, reward for the transition. Callers must pass pre-step ``obs`` into
``step`` (see ``run_expert``). Matches online PLD / textbook Q(s, a). Legacy
files without this tag stored post-step ``(s_{t+1}, a_t, r_t)``; loaders use
``inventory.recording.sa_align`` to recover.
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
    """One HDF5 file, appended across waves (pre-step ``(s_t, a_t, r_t)``).

    Call ``step(pre_obs, action, …)`` with the observation **from which**
    ``action`` was taken, after ``env.step`` has returned that transition's
    reward / done. ``flush`` at each wave end banks open buffers.
    """

    def __init__(self, base, out_path, task_id, instruction, max_demos=None,
                 keep_failures=False, target_success_rate=None):
        self.base = base
        self.out = out_path
        self.task_id = task_id
        self.instruction = instruction
        self.max_demos = max_demos
        self.keep_failures = keep_failures or target_success_rate is not None
        # Optional mix: bank until max_success + max_fail == max_demos.
        self.max_success = None
        self.max_fail = None
        if target_success_rate is not None:
            if max_demos is None:
                raise ValueError("target_success_rate requires max_demos")
            if not 0.0 < target_success_rate < 1.0:
                raise ValueError("target_success_rate must be in (0, 1)")
            self.max_success = int(round(target_success_rate * max_demos))
            self.max_fail = max_demos - self.max_success
        self.N = base.num_envs
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        self.h = h5py.File(out_path, "a")
        if "data" in self.h:
            self.g = self.h["data"]
            if self.g.attrs.get("task") != task_id:
                raise ValueError(f"{out_path} contains task {self.g.attrs.get('task')!r}, not {task_id!r}")
            self.n_demos = int(self.g.attrs.get("num_demos", 0))
            # Resume counts from demos already on disk.
            self.n_success = sum(
                1 for k in self.g.keys() if k.startswith("demo_") and bool(self.g[k].attrs.get("success"))
            )
            self.n_fail = self.n_demos - self.n_success
        else:
            self.g = self.h.create_group("data")
            self.g.attrs["task"] = task_id
            self.g.attrs["language"] = instruction
            self.g.attrs["source"] = "scripted_privileged"
            self.g.attrs["num_demos"] = 0
            # pre_step: (s_t, a_t, r_t) — state before action (see sa_align.py).
            self.g.attrs["sa_align"] = "pre_step"
            self.n_demos = 0
            self.n_success = 0
            self.n_fail = 0
        if self.max_success is not None:
            print(
                f"[record] mix target: {self.max_success} success / {self.max_fail} fail "
                f"(have {self.n_success}/{self.n_fail})",
                flush=True,
            )
        self._new_wave()

    def _want(self, success: bool) -> bool:
        """Whether this finished episode should be banked under current quotas."""
        if self.max_success is not None:
            if success:
                return self.n_success < self.max_success
            return self.n_fail < self.max_fail
        return bool(success) or self.keep_failures

    def _new_wave(self):
        self.buf = [[] for _ in range(self.N)]      # per-env list of frame dicts
        # Slot finished once this wave: run_expert holds pose and must NOT start
        # episode 2 — keep buf empty so commit_open cannot bank frozen stubs.
        self.closed = [False] * self.N
        self.wave_demos = 0

    @staticmethod
    def _is_real_trajectory(frames) -> bool:
        """Reject frozen post-done hold stubs (no motion / absurdly short)."""
        if len(frames) < 80:  # <~5 s @ 15 Hz — not a full pick attempt
            return False
        eef = np.stack([f["eef_pos"] for f in frames])
        path = float(np.sum(np.linalg.norm(np.diff(eef, axis=0), axis=1)))
        return path >= 0.02  # ≥2 cm EE travel

    @staticmethod
    def _fail_is_bankable(frames) -> bool:
        """Fail half: insert near-miss (ran, cleared stand, approached hole)."""
        if not Recorder._is_real_trajectory(frames):
            return False
        if "held_part_pose" not in frames[0] or "fixed_part_pose" not in frames[0]:
            return True
        held = np.stack([f["held_part_pose"] for f in frames])
        fixed = np.stack([f["fixed_part_pose"] for f in frames])
        # Match analysis QC: LIFT_Z=0.03 and HOLE_XY_MM=8.
        if float(held[:, 2].max()) < 0.03:
            return False
        min_xy = float(np.linalg.norm(held[:, :2] - fixed[:, :2], axis=1).min())
        return min_xy <= 0.008

    def step(self, obs, action, done, success, reward=None):
        """Bank one ``(s_t, a_t, r_t)`` frame per env; on done commit and close.

        ``obs`` = s_t (before ``action``); ``action`` = a_t taken from s_t;
        ``reward`` / ``done`` = outcome of that transition. After terminate the
        slot is held frozen — do not open a second buffer this wave.
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
            if self.closed[i]:
                continue  # held after first finish — not a real episode
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
            # pre_step row — including the terminal transition.
            self.buf[i].append(frame)
            if done[i]:
                below_limit = self.max_demos is None or self.n_demos < self.max_demos
                ok = bool(success[i])
                bankable = bool(self.buf[i]) and (ok or self._fail_is_bankable(self.buf[i]))
                if self._want(ok) and bankable and below_limit:
                    self._write_demo(self.buf[i], success=ok)
                self.buf[i] = []
                self.closed[i] = True  # no episode 2 this wave

    def _write_demo(self, frames, *, success):
        d = self.g.create_group(f"demo_{self.n_demos}")
        d.attrs["success"] = bool(success)
        d.attrs["sa_align"] = "pre_step"  # (s_t, a_t, r_t); not post-step
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
        if success:
            self.n_success += 1
        else:
            self.n_fail += 1
        self.g.attrs["num_demos"] = self.n_demos
        self.h.flush()  # kill-safe: land demos on disk each commit

    def commit_open(self):
        """Bank still-open buffers as failures (wave ended without env done).

        Only real attempts: closed slots are empty; frozen stubs fail the
        motion/length gate.
        """
        if not self.keep_failures:
            return
        for i in range(self.N):
            below_limit = self.max_demos is None or self.n_demos < self.max_demos
            if (self.buf[i] and below_limit and self._want(False)
                    and self._fail_is_bankable(self.buf[i])):
                self._write_demo(self.buf[i], success=False)
            self.buf[i] = []

    def flush(self):
        """Return demos committed this wave, then reset buffers for the next."""
        kept = self.wave_demos
        self._new_wave()
        return kept

    def close(self):
        self.h.close()
        mix = f" ({self.n_success} success / {self.n_fail} fail)" if self.n_demos else ""
        print(f"[record] wrote {self.n_demos} demos{mix} -> {self.out}", flush=True)

"""Extra policy observations for the assembly benchmark.

Mirrors the DROID embodiment's observation helpers (isaaclab_arena
embodiments/droid/observations.py): env-local reads off the articulation,
Panda arm joints only, in joint-buffer order (asserted panda_joint1..7
first on this asset by the expert servo).
"""

import torch

import warp as wp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

_PANDA_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]


def arm_joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Angular velocities (rad/s) of the 7 Panda arm joints."""
    robot = env.scene[asset_cfg.name]
    joint_indices = [i for i, name in enumerate(robot.data.joint_names) if name in _PANDA_JOINTS]
    return wp.to_torch(robot.data.joint_vel)[:, joint_indices]


def part_pose(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Privileged root pose: env-local xyz + world xyzw quat -> (N, 7).

    For the PA-RL critic only (not the VLA). Pos is origin-subtracted so it
    matches recorded eef_pos; quat is Isaac's native xyzw.
    """
    asset = env.scene[asset_cfg.name]
    pos = wp.to_torch(asset.data.root_pos_w) - env.scene.env_origins
    quat = wp.to_torch(asset.data.root_quat_w)  # xyzw
    return torch.cat([pos, quat], dim=-1)

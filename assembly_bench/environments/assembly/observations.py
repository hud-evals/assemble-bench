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

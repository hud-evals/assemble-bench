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
    """Env-local root pose of a scene asset: position(3) + quat xyzw(4).

    Privileged observation for the PA-RL critic (same convention as the
    reward-labeled demo datasets: env-origin-relative position, xyzw quat).
    """
    asset = env.scene[asset_cfg.name]
    pos = wp.to_torch(asset.data.root_pos_w) - env.scene.env_origins
    return torch.cat([pos, wp.to_torch(asset.data.root_quat_w)], dim=-1)


# Alias used by assembly.py privileged ObsTerm wiring.
asset_root_pose = part_pose


def expert_active(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-env 0/1: scripted expert has latched (set by ExpertTakeover wrapper)."""
    n, dev = env.num_envs, env.device
    flag = getattr(env, "_expert_takeover_active", None)
    if flag is None or not isinstance(flag, torch.Tensor) or flag.shape[0] != n:
        return torch.zeros(n, 1, device=dev)
    return flag.to(device=dev, dtype=torch.float32).view(n, 1)


def executed_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Action that ``env.step`` actually applied (expert override when latched).

    Set by ``ExpertTakeover``; zeros before the first step. HG-DAgger recording
    uses this so labels are expert actions, not the discarded policy chunk.
    """
    n, dev = env.num_envs, env.device
    act = getattr(env, "_expert_executed_action", None)
    if act is None or not isinstance(act, torch.Tensor) or act.shape[0] != n:
        return torch.zeros(n, 8, device=dev)
    return act.to(device=dev, dtype=torch.float32).view(n, -1)[:, :8]

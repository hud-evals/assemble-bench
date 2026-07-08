"""Staged assembly reward for RL fine-tuning (opt-in; eval leaves it off).

The source benchmark's ``RewardBook`` ported into one Arena reward term: a
once-fired milestone staircase (grasped < lifted < engaged < success) plus a
continuous bonus for every new best insertion depth. Sparse task success alone
is too weak a signal for PPO, so this hands out credit for each sub-skill and
for partial insertion progress.

Only ``get_rewards_cfg()`` (via ``build_rewards_cfg``) wires this in, and only
when the task is built with ``reward_mode="staged"`` -- the default eval path
returns no reward terms, so success/termination semantics are unchanged.

The milestone predicates reuse the exact seat geometry the success check
(``tasks.part_seated``) computes, so "engaged"/"success" here track the same
target the benchmark grades on. State (per-env fired flags + running best
depth) lives on a ``ManagerTermBase`` -- the same stateful-term pattern as
``tasks.hold_success`` -- so it persists across steps and resets per episode.
"""

import torch

import warp as wp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from assembly_bench.environments.assembly.tasks import part_seated
from assembly_bench.environments.assembly.variants import TABLE_TOP_Z, AssemblyVariant

# Milestone payouts (source benchmark RewardBook weights).
W_GRASP = 0.1
W_LIFT = 0.2
W_ENGAGE = 0.4
W_SUCCESS = 1.0
W_PARTIAL = 0.5

# Grasp geometry (source benchmark constants, base_link flange EE frame).
GRASP_XY_TOL = 0.03
GRASP_Z_OFF = 0.16          # flange -> fingertip pad along the approach axis (m)
GRASP_Z_TOL = 0.06
GRASP_CLOSED_FINGER = 0.2   # gripper_pos (finger_joint/(pi/4)) past this => closed
MILESTONES = ("grasped", "lifted", "engaged", "success")


def _align_and_gap(env, held_cfg, fixed_cfg, seat_off, held_base_z_off):
    """Shared seat geometry: (xy distance to target, seat gap). Mirrors
    ``part_seated`` -- target = fixed root + seat_off rotated into the fixed
    frame; gap = held base height above the target (<=0 fully seated)."""
    held = env.scene[held_cfg.name]
    fixed = env.scene[fixed_cfg.name]
    held_pos = wp.to_torch(held.data.root_pos_w) - env.scene.env_origins
    fixed_pos = wp.to_torch(fixed.data.root_pos_w) - env.scene.env_origins
    fixed_quat = wp.to_torch(fixed.data.root_quat_w)
    off = torch.tensor(seat_off, device=env.device).expand(env.num_envs, 3)
    target = fixed_pos + quat_apply(fixed_quat, off)
    xy = torch.norm(held_pos[:, :2] - target[:, :2], dim=-1)
    gap = (held_pos[:, 2] + held_base_z_off) - target[:, 2]
    return xy, gap


def _grasped(env, held_cfg, robot_cfg, ee_body, gripper_joint) -> torch.Tensor:
    """Held part centered under the closed gripper (kinematic outcome, not a
    grasp-quality score). EE is the flange, so a grasped part sits ~GRASP_Z_OFF
    below it."""
    held = env.scene[held_cfg.name]
    robot = env.scene[robot_cfg.name]
    held_pos = wp.to_torch(held.data.root_pos_w) - env.scene.env_origins
    ee_idx = robot.data.body_names.index(ee_body)
    ee_pos = wp.to_torch(robot.data.body_pos_w)[:, ee_idx] - env.scene.env_origins
    gj_idx = robot.data.joint_names.index(gripper_joint)
    grip = wp.to_torch(robot.data.joint_pos)[:, gj_idx] / (torch.pi / 4)
    near_xy = torch.norm(held_pos[:, :2] - ee_pos[:, :2], dim=-1) < GRASP_XY_TOL
    near_z = (ee_pos[:, 2] - held_pos[:, 2] - GRASP_Z_OFF).abs() < GRASP_Z_TOL
    return near_xy & near_z & (grip > GRASP_CLOSED_FINGER)


class staged_assembly_reward(ManagerTermBase):
    """Once-fired milestone staircase + new-best-depth bonus.

    Per env: ``_fired`` (a bool per milestone) and ``_best_pc`` (running best
    partial credit). ``reset`` clears them for resetting envs; ``__call__`` pays
    each milestone the first time it holds and rewards any gain in best depth.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._fired = {k: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
                       for k in MILESTONES}
        self._best_pc = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None) -> None:
        idx = slice(None) if env_ids is None else env_ids
        for v in self._fired.values():
            v[idx] = False
        self._best_pc[idx] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        held_cfg: SceneEntityCfg,
        fixed_cfg: SceneEntityCfg,
        robot_cfg: SceneEntityCfg,
        seat_off: tuple[float, float, float],
        held_base_z_off: float,
        align_tol: float,
        seat_tol: float,
        engage_gap: float,
        lift_clear: float,
        partial_socket_h: float,
        stand_z: float,
        ee_body: str,
        gripper_joint: str,
    ) -> torch.Tensor:
        held = env.scene[held_cfg.name]
        held_z = (wp.to_torch(held.data.root_pos_w)[:, 2] - env.scene.env_origins[:, 2])
        xy, gap = _align_and_gap(env, held_cfg, fixed_cfg, seat_off, held_base_z_off)

        states = {
            "grasped": _grasped(env, held_cfg, robot_cfg, ee_body, gripper_joint),
            "lifted": (held_z - stand_z) > lift_clear,
            "engaged": (xy < align_tol) & (gap < engage_gap),
            "success": part_seated(env, held_cfg, fixed_cfg, seat_off, held_base_z_off,
                                   align_tol, seat_tol),
        }
        weights = {"grasped": W_GRASP, "lifted": W_LIFT, "engaged": W_ENGAGE, "success": W_SUCCESS}

        r = torch.zeros(env.num_envs, device=env.device)
        for k in MILESTONES:
            newly = states[k] & ~self._fired[k]
            r += weights[k] * newly.float()
            self._fired[k] |= states[k]

        # Continuous partial credit: reward only positive gains in best depth.
        pc = torch.clamp(1.0 - gap / partial_socket_h, 0.0, 1.0)
        r += W_PARTIAL * torch.clamp(pc - self._best_pc, min=0.0)
        self._best_pc = torch.maximum(self._best_pc, pc)

        # The RewardManager scales every term by weight*dt; undo dt so the
        # once-fired payouts are absolute amounts, not per-second rates.
        return r / env.step_dt


@configclass
class AssemblyRewardsCfg:
    staged: RewardTermCfg = None


def build_rewards_cfg(variant: AssemblyVariant, held, fixed, robot_name: str = "robot",
                      ee_body: str = "base_link", gripper_joint: str = "finger_joint"
                      ) -> AssemblyRewardsCfg:
    """The staged reward term wired to a variant's seat geometry. ``weight=1.0``
    (per-milestone weights live in the term; dt is undone inside it)."""
    return AssemblyRewardsCfg(staged=RewardTermCfg(
        func=staged_assembly_reward,
        weight=1.0,
        params={
            "held_cfg": SceneEntityCfg(held.name),
            "fixed_cfg": SceneEntityCfg(fixed.name),
            "robot_cfg": SceneEntityCfg(robot_name),
            "seat_off": variant.seat_off,
            "held_base_z_off": variant.held_base_z_off,
            "align_tol": variant.align_tol,
            "seat_tol": variant.seat_tol,
            "engage_gap": variant.engage_gap,
            "lift_clear": variant.lift_clear,
            "partial_socket_h": variant.partial_socket_h,
            "stand_z": TABLE_TOP_Z,
            "ee_body": ee_body,
            "gripper_joint": gripper_joint,
        },
    ))

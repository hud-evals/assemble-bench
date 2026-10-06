"""Assembly RL rewards (opt-in; eval leaves them off).

Two modes:

- ``staged`` — once-fired milestones + new-best potentials (pay only progress).
- ``potential`` — same milestones + signed Φ(s')−Φ(s) (Markov in pose state;
  nonzero local slope every step; needed for residual SAC inside a ξ-ball).

Phases (both modes):
  pre-lift  -- Φ_grasp: EE approaches the held part
  post-lift -- Φ_xy / Φ_depth: held part approaches the seat
  nut only  -- after engage: once-fired first-turn + Φ_thread

Wired when ``reward_mode`` is ``"staged"`` or ``"potential"``. Seat geometry
matches ``tasks.part_seated``. State lives on ``ManagerTermBase``.
"""

import math

import torch

import warp as wp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from assemble_bench.environments.assembly.scoring import (
    W_ALIGN_PC,
    W_DEPTH_PC,
    W_ENGAGE,
    W_GRASP_PC,
    W_LIFT,
    W_SUCCESS,
    W_THREAD_PC,
    W_THREAD_START,
)
from assemble_bench.environments.assembly.tasks import part_seated, peg_upright_cos
from assemble_bench.environments.assembly.variants import TABLE_TOP_Z, AssemblyVariant

# Grasp approach (flange EE -> held). Falloff radius for Φ_grasp.
GRASP_Z_OFF = 0.16
GRASP_DIST0 = 0.15
# Seat xy approach falloff (m); depth uses variant.partial_socket_h.
ALIGN_DIST0 = 0.10
# Nut thread: first-turn threshold + target cumulative |Δyaw| (~1.5 turns).
THREAD_START_RAD = 0.3
THREAD_TARGET_RAD = 1.5 * 2.0 * math.pi

MILESTONES = ("lifted", "engaged", "thread_start", "success")


def _align_and_gap(env, held_cfg, fixed_cfg, seat_off, held_base_z_off):
    """Seat geometry: (xy distance to target, seat gap). Mirrors ``part_seated``."""
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


def _yaw_z(q: torch.Tensor) -> torch.Tensor:
    """World-z yaw from xyzw quaternions [N,4]."""
    x, y, z, w = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class staged_assembly_reward(ManagerTermBase):
    """Phase-gated new-best potentials + once-fired milestones."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        n, dev = env.num_envs, env.device
        self._fired = {k: torch.zeros(n, dtype=torch.bool, device=dev) for k in MILESTONES}
        self._best_grasp = torch.zeros(n, device=dev)
        self._best_xy = torch.zeros(n, device=dev)
        self._best_depth = torch.zeros(n, device=dev)
        self._best_thread = torch.zeros(n, device=dev)
        self._prev_yaw = torch.zeros(n, device=dev)
        self._yaw_valid = torch.zeros(n, dtype=torch.bool, device=dev)
        self._turns = torch.zeros(n, device=dev)

    def reset(self, env_ids=None) -> None:
        idx = slice(None) if env_ids is None else env_ids
        for v in self._fired.values():
            v[idx] = False
        self._best_grasp[idx] = 0.0
        self._best_xy[idx] = 0.0
        self._best_depth[idx] = 0.0
        self._best_thread[idx] = 0.0
        self._prev_yaw[idx] = 0.0
        self._yaw_valid[idx] = False
        self._turns[idx] = 0.0

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
        family: str = "peg_insert",
        min_upright_cos: float | None = None,
        seat_overshoot_tol: float | None = None,
        min_thread_rad: float | None = None,
    ) -> torch.Tensor:
        held = env.scene[held_cfg.name]
        robot = env.scene[robot_cfg.name]
        origins = env.scene.env_origins
        held_pos = wp.to_torch(held.data.root_pos_w) - origins
        held_quat = wp.to_torch(held.data.root_quat_w)
        held_z = held_pos[:, 2]
        xy, gap = _align_and_gap(env, held_cfg, fixed_cfg, seat_off, held_base_z_off)

        # Φ_grasp: flange approaches the held part (pre-lift only).
        ee_idx = robot.data.body_names.index(ee_body)
        ee_pos = wp.to_torch(robot.data.body_pos_w)[:, ee_idx] - origins
        grasp_d = torch.norm(held_pos - (ee_pos - torch.tensor(
            [0.0, 0.0, GRASP_Z_OFF], device=env.device)), dim=-1)
        phi_grasp = torch.clamp(1.0 - grasp_d / GRASP_DIST0, 0.0, 1.0)

        phi_xy = torch.clamp(1.0 - xy / ALIGN_DIST0, 0.0, 1.0)
        phi_depth = torch.clamp(1.0 - gap / partial_socket_h, 0.0, 1.0)

        lifted = (held_z - stand_z) > lift_clear
        # Nuts: engage/thread only while upright (no credit for tip-spin beside bolt).
        upright = (
            peg_upright_cos(held_quat) >= min_upright_cos
            if min_upright_cos is not None
            else torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        )
        engaged = (xy < align_tol) & (gap < engage_gap) & upright
        success = part_seated(
            env, held_cfg, fixed_cfg, seat_off, held_base_z_off, align_tol, seat_tol,
            min_upright_cos=min_upright_cos, seat_overshoot_tol=seat_overshoot_tol)

        # Nut: accumulate |Δyaw| while engaged -> Φ_thread + first-turn milestone.
        is_nut = family == "nut_thread"
        thread_start = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        if is_nut:
            yaw = _yaw_z(held_quat)
            delta = (yaw - self._prev_yaw + torch.pi) % (2.0 * torch.pi) - torch.pi
            self._turns = self._turns + torch.where(
                engaged & self._yaw_valid, delta.abs(), torch.zeros_like(delta))
            self._prev_yaw = yaw
            self._yaw_valid = self._yaw_valid | engaged
            thread_start = self._turns > THREAD_START_RAD
            phi_thread = torch.clamp(self._turns / THREAD_TARGET_RAD, 0.0, 1.0)
            # Same on-bolt turn floor as the success termination.
            if min_thread_rad is not None:
                success = success & (self._turns >= min_thread_rad)
        else:
            phi_thread = torch.zeros(env.num_envs, device=env.device)

        states = {
            "lifted": lifted,
            "engaged": engaged,
            "thread_start": thread_start if is_nut else torch.zeros_like(lifted),
            "success": success,
        }
        weights = {
            "lifted": W_LIFT,
            "engaged": W_ENGAGE,
            "thread_start": W_THREAD_START,
            "success": W_SUCCESS,
        }

        r = torch.zeros(env.num_envs, device=env.device)
        for k in MILESTONES:
            newly = states[k] & ~self._fired[k]
            r += weights[k] * newly.float()
            self._fired[k] |= states[k]

        # Continuous: grasp approach until lift fires; seat approach after.
        pre_lift = ~self._fired["lifted"]
        dg = torch.clamp(phi_grasp - self._best_grasp, min=0.0)
        r += W_GRASP_PC * dg * pre_lift.float()
        self._best_grasp = torch.where(pre_lift, torch.maximum(self._best_grasp, phi_grasp),
                                       self._best_grasp)

        post_lift = self._fired["lifted"]
        dxy = torch.clamp(phi_xy - self._best_xy, min=0.0)
        dd = torch.clamp(phi_depth - self._best_depth, min=0.0)
        r += (W_ALIGN_PC * dxy + W_DEPTH_PC * dd) * post_lift.float()
        self._best_xy = torch.where(post_lift, torch.maximum(self._best_xy, phi_xy),
                                    self._best_xy)
        self._best_depth = torch.where(post_lift, torch.maximum(self._best_depth, phi_depth),
                                       self._best_depth)

        # Nut thread densify after engage (turns on bolt; depth already covered above).
        if is_nut:
            on_bolt = self._fired["engaged"]
            dt = torch.clamp(phi_thread - self._best_thread, min=0.0)
            r += W_THREAD_PC * dt * on_bolt.float()
            self._best_thread = torch.where(
                on_bolt, torch.maximum(self._best_thread, phi_thread), self._best_thread)

        # RewardManager scales by weight*dt; undo dt so payouts are absolute.
        return r / env.step_dt


class potential_assembly_reward(ManagerTermBase):
    """Milestones + signed Φ(s')−Φ(s) (Markov; nonzero slope every step).

    Same phase-gated potentials as ``staged``, but pays the potential *difference*
    instead of new-best deltas — so reversing away from the peg is penalized and
    the critic sees an action gradient inside the residual ξ-ball. Phase switches
    (lift / engage for nuts) re-anchor ``_prev_phi`` so the Φ definition change
    does not inject a spurious jump; the milestone already covers that transition.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        n, dev = env.num_envs, env.device
        self._fired = {k: torch.zeros(n, dtype=torch.bool, device=dev) for k in MILESTONES}
        self._prev_phi = torch.zeros(n, device=dev)
        self._prev_yaw = torch.zeros(n, device=dev)
        self._yaw_valid = torch.zeros(n, dtype=torch.bool, device=dev)
        self._turns = torch.zeros(n, device=dev)
        # Smoke / diagnostics: peak |potential| seen this episode (not used in r).
        self._peak_grasp = torch.zeros(n, device=dev)
        self._peak_xy = torch.zeros(n, device=dev)
        self._peak_depth = torch.zeros(n, device=dev)
        self._peak_thread = torch.zeros(n, device=dev)

    def reset(self, env_ids=None) -> None:
        idx = slice(None) if env_ids is None else env_ids
        for v in self._fired.values():
            v[idx] = False
        self._prev_phi[idx] = 0.0
        self._prev_yaw[idx] = 0.0
        self._yaw_valid[idx] = False
        self._turns[idx] = 0.0
        self._peak_grasp[idx] = 0.0
        self._peak_xy[idx] = 0.0
        self._peak_depth[idx] = 0.0
        self._peak_thread[idx] = 0.0

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
        family: str = "peg_insert",
        min_upright_cos: float | None = None,
        seat_overshoot_tol: float | None = None,
        min_thread_rad: float | None = None,
    ) -> torch.Tensor:
        held = env.scene[held_cfg.name]
        robot = env.scene[robot_cfg.name]
        origins = env.scene.env_origins
        held_pos = wp.to_torch(held.data.root_pos_w) - origins
        held_quat = wp.to_torch(held.data.root_quat_w)
        held_z = held_pos[:, 2]
        xy, gap = _align_and_gap(env, held_cfg, fixed_cfg, seat_off, held_base_z_off)

        ee_idx = robot.data.body_names.index(ee_body)
        ee_pos = wp.to_torch(robot.data.body_pos_w)[:, ee_idx] - origins
        grasp_d = torch.norm(held_pos - (ee_pos - torch.tensor(
            [0.0, 0.0, GRASP_Z_OFF], device=env.device)), dim=-1)
        phi_grasp = torch.clamp(1.0 - grasp_d / GRASP_DIST0, 0.0, 1.0)
        phi_xy = torch.clamp(1.0 - xy / ALIGN_DIST0, 0.0, 1.0)
        phi_depth = torch.clamp(1.0 - gap / partial_socket_h, 0.0, 1.0)

        lifted = (held_z - stand_z) > lift_clear
        upright = (
            peg_upright_cos(held_quat) >= min_upright_cos
            if min_upright_cos is not None
            else torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        )
        engaged = (xy < align_tol) & (gap < engage_gap) & upright
        success = part_seated(
            env, held_cfg, fixed_cfg, seat_off, held_base_z_off, align_tol, seat_tol,
            min_upright_cos=min_upright_cos, seat_overshoot_tol=seat_overshoot_tol)

        is_nut = family == "nut_thread"
        thread_start = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        if is_nut:
            yaw = _yaw_z(held_quat)
            delta = (yaw - self._prev_yaw + torch.pi) % (2.0 * torch.pi) - torch.pi
            self._turns = self._turns + torch.where(
                engaged & self._yaw_valid, delta.abs(), torch.zeros_like(delta))
            self._prev_yaw = yaw
            self._yaw_valid = self._yaw_valid | engaged
            thread_start = self._turns > THREAD_START_RAD
            phi_thread = torch.clamp(self._turns / THREAD_TARGET_RAD, 0.0, 1.0)
            if min_thread_rad is not None:
                success = success & (self._turns >= min_thread_rad)
        else:
            phi_thread = torch.zeros(env.num_envs, device=env.device)

        states = {
            "lifted": lifted,
            "engaged": engaged,
            "thread_start": thread_start if is_nut else torch.zeros_like(lifted),
            "success": success,
        }
        weights = {
            "lifted": W_LIFT,
            "engaged": W_ENGAGE,
            "thread_start": W_THREAD_START,
            "success": W_SUCCESS,
        }

        r = torch.zeros(env.num_envs, device=env.device)
        # Track which milestones newly fire — phase-switch re-anchor uses these.
        newly = {}
        for k in MILESTONES:
            newly[k] = states[k] & ~self._fired[k]
            r += weights[k] * newly[k].float()
            self._fired[k] |= states[k]

        # Phase-gated Φ (same pieces as staged's continuous terms).
        pre_lift = ~self._fired["lifted"]
        post_lift = self._fired["lifted"]
        phi = W_GRASP_PC * phi_grasp * pre_lift.float()
        phi = phi + (W_ALIGN_PC * phi_xy + W_DEPTH_PC * phi_depth) * post_lift.float()
        if is_nut:
            phi = phi + W_THREAD_PC * phi_thread * self._fired["engaged"].float()

        # Re-anchor Φ on phase switches so the definition change isn't a fake payout.
        phase_switch = newly["lifted"]
        if is_nut:
            phase_switch = phase_switch | newly["engaged"]
        prev = torch.where(phase_switch, phi, self._prev_phi)
        r = r + (phi - prev)
        self._prev_phi = phi

        # Peaks for smoke diagnostics (mirror staged's best-* checks).
        self._peak_grasp = torch.where(
            pre_lift, torch.maximum(self._peak_grasp, phi_grasp), self._peak_grasp)
        self._peak_xy = torch.where(
            post_lift, torch.maximum(self._peak_xy, phi_xy), self._peak_xy)
        self._peak_depth = torch.where(
            post_lift, torch.maximum(self._peak_depth, phi_depth), self._peak_depth)
        if is_nut:
            on_bolt = self._fired["engaged"]
            self._peak_thread = torch.where(
                on_bolt, torch.maximum(self._peak_thread, phi_thread), self._peak_thread)

        return r / env.step_dt


@configclass
class AssemblyRewardsCfg:
    staged: RewardTermCfg = None
    potential: RewardTermCfg = None


def build_rewards_cfg(variant: AssemblyVariant, held, fixed, robot_name: str = "robot",
                      ee_body: str = "base_link",
                      mode: str = "staged") -> AssemblyRewardsCfg:
    """Reward wired to a variant's seat geometry. ``mode`` is staged|potential."""
    params = {
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
        "family": variant.family,
        "min_upright_cos": variant.min_upright_cos,
        "seat_overshoot_tol": variant.seat_overshoot_tol,
        "min_thread_rad": variant.min_thread_rad,
    }
    if mode == "potential":
        return AssemblyRewardsCfg(potential=RewardTermCfg(
            func=potential_assembly_reward, weight=1.0, params=params))
    if mode != "staged":
        raise ValueError(f"unknown reward mode {mode!r}; expected staged|potential")
    return AssemblyRewardsCfg(staged=RewardTermCfg(
        func=staged_assembly_reward, weight=1.0, params=params))

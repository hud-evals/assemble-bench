"""NIST assembly task for Isaac Lab Arena.

One task class covers all families (peg insert / gear mesh / nut thread);
only the seat geometry differs, carried by the variant. Success
is the source benchmark's outcome check: the held part's base point reaches
the seat target (fixed root + family offset, rotated into the fixed frame)
within the alignment/seat tolerances AND the part is at rest — debounced by
``hold_success`` (N consecutive true steps). Nuts also require upright pose,
no seat overshoot, and enough on-bolt yaw (side-squeeze / forced tip push
used to false-trigger on xy+depth alone).
"""

import numpy as np
import torch
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
import warp as wp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

import isaaclab_arena_environments.mdp as mdp
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.object_moved import ObjectMovedRateMetric
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object

from assembly_bench.environments.assembly.variants import TABLE_TOP_Z, AssemblyVariant

# Success must hold this many consecutive steps (debounces one-frame trues).
SUCCESS_HOLD_STEPS = 3
# The presentation stand only presents the part upright -- near-frictionless
# so the part lifts off cleanly instead of jamming in the bore.
STAND_FRICTION = 0.01
# Robotiq fingertip pads (rubber). With friction_combine_mode=max this μ wins
# over a metal-ish held part (~0.5) at the grasp contact.
PAD_FRICTION = 1.3
PAD_BODIES = ("left_inner_finger", "right_inner_finger")
# Failure: the held part fell below the tabletop.
DROP_HEIGHT = TABLE_TOP_Z - 0.05
# Nut success gates (also authored on nut variants in variants.py).
NUT_UPRIGHT_COS = 0.9  # ~25° from vertical
NUT_MIN_THREAD_RAD = 0.3  # min |Δyaw| on the bolt (matches reward THREAD_START)


def peg_upright_cos(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Cosine of peg/nut tilt from vertical (1 = upright). ``quat`` is xyzw."""
    qx, qy = quat_xyzw[:, 0], quat_xyzw[:, 1]
    return 1.0 - 2.0 * (qx * qx + qy * qy)


def check_seated(
    held_pos: torch.Tensor,
    held_quat_xyzw: torch.Tensor,
    held_lin_vel: torch.Tensor,
    target: torch.Tensor,
    held_base_z_off: float,
    align_tol: float,
    seat_tol: float,
    max_speed: float = 0.05,
    min_upright_cos: float | None = None,
    seat_overshoot_tol: float | None = None,
) -> torch.Tensor:
    """True when the held base is at the seat target and nearly at rest.

    ``target`` is already in the env frame (fixed root + rotated seat_off).
    Nuts also pass ``min_upright_cos`` / ``seat_overshoot_tol`` so a tip-over
    against the shank cannot count as threaded.
    """
    aligned = torch.norm(held_pos[:, :2] - target[:, :2], dim=-1) < align_tol
    seat_gap = (held_pos[:, 2] + held_base_z_off) - target[:, 2]
    seated = seat_gap < seat_tol
    if seat_overshoot_tol is not None:
        # Reject nuts that fell past the seat (beside / through the bolt).
        seated = seated & (seat_gap > -seat_overshoot_tol)
    stable = torch.norm(held_lin_vel, dim=-1) < max_speed
    ok = aligned & seated & stable
    if min_upright_cos is not None:
        ok = ok & (peg_upright_cos(held_quat_xyzw) >= min_upright_cos)
    return ok


# ---------------------------------------------------------------------------
# Success predicate
# ---------------------------------------------------------------------------


def _yaw_z(q: torch.Tensor) -> torch.Tensor:
    """World-z yaw from xyzw quaternions [N,4]."""
    x, y, z, w = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def part_seated(
    env: ManagerBasedRLEnv,
    held_cfg: SceneEntityCfg,
    fixed_cfg: SceneEntityCfg,
    seat_off: tuple[float, float, float],
    held_base_z_off: float,
    align_tol: float,
    seat_tol: float,
    max_speed: float = 0.05,
    min_upright_cos: float | None = None,
    seat_overshoot_tol: float | None = None,
) -> torch.Tensor:
    """Held base at the seat target (aligned in xy, descended past the seat
    depth) and the part nearly at rest. seat_off is rotated into the fixed
    asset's frame so the target tracks any base yaw (rect-peg clocking)."""
    held = env.scene[held_cfg.name]
    fixed = env.scene[fixed_cfg.name]
    held_pos = wp.to_torch(held.data.root_pos_w) - env.scene.env_origins
    fixed_pos = wp.to_torch(fixed.data.root_pos_w) - env.scene.env_origins
    fixed_quat = wp.to_torch(fixed.data.root_quat_w)

    off = torch.tensor(seat_off, device=env.device).expand(env.num_envs, 3)
    target = fixed_pos + quat_apply(fixed_quat, off)
    return check_seated(
        held_pos,
        wp.to_torch(held.data.root_quat_w),
        wp.to_torch(held.data.root_lin_vel_w),
        target,
        held_base_z_off,
        align_tol,
        seat_tol,
        max_speed=max_speed,
        min_upright_cos=min_upright_cos,
        seat_overshoot_tol=seat_overshoot_tol,
    )


class hold_success(ManagerTermBase):
    """Success wrapper: the raw predicate ``func(env, **params)`` must hold for
    ``SUCCESS_HOLD_STEPS`` consecutive steps. Optional ``min_thread_rad``
    (nuts) also requires cumulative |Δyaw| while engaged and upright."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        n, dev = env.num_envs, env.device
        self._streak = torch.zeros(n, dtype=torch.long, device=dev)
        self._turns = torch.zeros(n, device=dev)
        self._prev_yaw = torch.zeros(n, device=dev)
        self._yaw_valid = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids=None) -> None:
        idx = env_ids if env_ids is not None else slice(None)
        self._streak[idx] = 0
        self._turns[idx] = 0.0
        self._prev_yaw[idx] = 0.0
        self._yaw_valid[idx] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        func,
        params: dict,
        min_thread_rad: float | None = None,
        engage_gap: float | None = None,
    ) -> torch.Tensor:
        ok = func(env, **params)
        # Nut: accumulate |Δyaw| only while on-bolt and upright (not tip-spin).
        if min_thread_rad is not None:
            held = env.scene[params["held_cfg"].name]
            fixed = env.scene[params["fixed_cfg"].name]
            origins = env.scene.env_origins
            held_pos = wp.to_torch(held.data.root_pos_w) - origins
            fixed_pos = wp.to_torch(fixed.data.root_pos_w) - origins
            fixed_quat = wp.to_torch(fixed.data.root_quat_w)
            held_quat = wp.to_torch(held.data.root_quat_w)
            off = torch.tensor(params["seat_off"], device=env.device).expand(env.num_envs, 3)
            target = fixed_pos + quat_apply(fixed_quat, off)
            xy = torch.norm(held_pos[:, :2] - target[:, :2], dim=-1)
            gap = (held_pos[:, 2] + params["held_base_z_off"]) - target[:, 2]
            upright = peg_upright_cos(held_quat) >= (
                params["min_upright_cos"]
                if params.get("min_upright_cos") is not None
                else NUT_UPRIGHT_COS)
            engaged = (xy < params["align_tol"]) & (gap < float(engage_gap)) & upright
            yaw = _yaw_z(held_quat)
            delta = (yaw - self._prev_yaw + torch.pi) % (2.0 * torch.pi) - torch.pi
            self._turns = self._turns + torch.where(
                engaged & self._yaw_valid, delta.abs(), torch.zeros_like(delta))
            self._prev_yaw = yaw
            self._yaw_valid = self._yaw_valid | engaged
            ok = ok & (self._turns >= min_thread_rad)
        self._streak = torch.where(ok, self._streak + 1, 0)
        return self._streak >= SUCCESS_HOLD_STEPS


# ---------------------------------------------------------------------------
# Reset randomization (the source benchmark's jitter scheme)
# ---------------------------------------------------------------------------


def reset_assembly_workspace(
    env,
    env_ids: torch.Tensor,
    held_group: list[tuple[str, tuple[float, float, float]]],
    fixed_group: list[tuple[str, tuple[float, float, float]]],
    rand_held_xy: float,
    rand_fixed_xy: float,
    rand_fixed_yaw: float,
) -> None:
    """Teleport the workspace to jittered poses: the held group (part + its
    presentation stand) shares one xy jitter so the part stays presented, the
    fixed group (socket + rigidly-attached extras) shares an independent one.
    The FIRST fixed-group asset optionally gets a yaw jitter (hole clocking)."""
    if env_ids is None:
        return
    n = len(env_ids)
    device = env.device
    origins = env.scene.env_origins[env_ids]
    # NOTE: this IsaacLab stack uses XYZW quaternions throughout (math utils,
    # spawners, root states) -- identity is (0,0,0,1), NOT the wxyz (1,0,0,0).
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).expand(n, 4)

    def place(group, mag, yaw_range=0.0):
        jitter = torch.zeros((n, 3), device=device)
        jitter[:, :2] = mag * (2 * torch.rand((n, 2), device=device) - 1)
        for i, (name, base) in enumerate(group):
            pos = torch.tensor(base, device=device) + jitter + origins
            quat = identity
            if i == 0 and yaw_range > 0.0:
                half = 0.5 * yaw_range * (2 * torch.rand(n, device=device) - 1)
                quat = torch.stack(
                    [torch.zeros_like(half), torch.zeros_like(half), torch.sin(half), torch.cos(half)], dim=-1)
            asset = env.scene[name]
            asset.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
            asset.write_root_velocity_to_sim(torch.zeros((n, 6), device=device), env_ids=env_ids)

    place(held_group, rand_held_xy)
    place(fixed_group, rand_fixed_xy, rand_fixed_yaw)


def settle_and_render(env, env_ids, steps: int = 120, rt_subframes: int = 32) -> None:
    """Post-reset warmup: settle parts, push fabric transforms to the renderer,
    then flush RTX temporal state (DLAA) via ``rep.orchestrator.step``.

    IsaacLab ``reset()`` alone can return camera obs of the previous state;
    one-render-per-step does not clear teleport ghosts. Replicator's subframe
    pump re-renders the same frame N times across all render products.

    Raw ``sim.step`` bypasses ``write_data_to_sim``, so re-flush the home joint
    target each settle step or the arm drifts off during warmup."""
    # Native 1280x720 dual-camera renders are intentionally expensive. Keep the
    # production defaults above, but let expert-development runs shorten this
    # reset-only anti-ghosting pass without changing task configuration.
    import os

    steps = int(os.environ.get("ASSEMBLY_RESET_WARMUP_STEPS", steps))
    rt_subframes = int(os.environ.get("ASSEMBLY_RESET_RT_SUBFRAMES", rt_subframes))
    print(
        f"[reset] settle warmup: physics_frames={steps} rt_subframes={rt_subframes}",
        flush=True,
    )
    del env_ids
    for _ in range(steps):
        for art in env.scene.articulations.values():
            art.write_data_to_sim()
        env.sim.step(render=True)
        for sensor in env.scene.sensors.values():
            sensor.update(dt=0.0, force_recompute=True)
    # DLAA history survives re-renders; FXAA→DLAA tears it down. Must *render*
    # under FXAA (app.update alone left a 1-tick front_cam ghost). Skip when
    # cameras/Replicator aren't loaded (--disable_cameras). Don't pause
    # /app/player/playSimulations — headless Kit can exit 0.
    try:
        import omni.kit.app
        import omni.replicator.core as rep
    except ModuleNotFoundError:
        return
    app = omni.kit.app.get_app()
    rep.settings.set_render_rtx_realtime(antialiasing="FXAA")
    for _ in range(4):
        for art in env.scene.articulations.values():
            art.write_data_to_sim()  # raw sim.step skips manager write
        env.sim.step(render=True)
        app.update()
    for sensor in env.scene.sensors.values():
        sensor.update(dt=0.0, force_recompute=True)
    rep.settings.set_render_rtx_realtime(antialiasing="DLAA")
    for _ in range(rt_subframes):
        app.update()
    # Post-flush render so tick-0 obs is clean on every camera product.
    for art in env.scene.articulations.values():
        art.write_data_to_sim()
    env.sim.step(render=True)
    for sensor in env.scene.sensors.values():
        sensor.update(dt=0.0, force_recompute=True)


def _friction_term(
    asset_name: str,
    friction: float,
    body_names: tuple[str, ...] | None = None,
) -> EventTermCfg:
    """Startup term pinning an asset's contact friction (source: set_friction).

    Optional ``body_names`` scopes the write (e.g. Robotiq pads only); omit to
    set every shape on the asset.
    """
    asset_cfg = (
        SceneEntityCfg(asset_name, body_names=list(body_names))
        if body_names is not None
        else SceneEntityCfg(asset_name)
    )
    return EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (friction, friction),
            "dynamic_friction_range": (friction, friction),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "asset_cfg": asset_cfg,
        },
    )


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


@configclass
class AssemblyTerminationsCfg:
    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    success: TerminationTermCfg = MISSING
    part_dropped: TerminationTermCfg = MISSING


@configclass
class AssemblyEventsCfg:
    reset_all: EventTermCfg = MISSING
    randomize_workspace: EventTermCfg = MISSING
    friction_held: EventTermCfg | None = None
    friction_fixed: EventTermCfg | None = None
    friction_stand: EventTermCfg | None = None
    friction_pads: EventTermCfg | None = None
    friction_extra_0: EventTermCfg | None = None
    friction_extra_1: EventTermCfg | None = None
    # LAST field: terms run in declaration order, and the warmup must follow
    # every pose-reset above. 120 frames clears DLAA teleport-ghosts (50 did not).
    settle_warmup: EventTermCfg = EventTermCfg(
        func=settle_and_render, mode="reset", params={"steps": 120})


@register_task
class NISTAssemblyTask(TaskBase):
    """Pick -> transport -> align -> seat (insert / mesh / thread / place)."""

    def __init__(
        self,
        variant: AssemblyVariant,
        held: Object,
        fixed: Object,
        stand: Object | None = None,
        extras: list[Object] = [],
        reward_mode: str | None = None,
    ):
        super().__init__(episode_length_s=variant.episode_length_s, task_description=variant.instruction)
        self.variant = variant
        self.held = held
        self.fixed = fixed
        self.stand = stand
        self.extras = list(extras)
        # "staged" / "potential" turn on dense RL shaping (rewards.py); None/other
        # leaves the env reward-free (eval default -- terminations unchanged).
        self.reward_mode = reward_mode
        # Our reset event owns all part poses.
        for asset in [held, fixed, stand, *self.extras]:
            if asset is not None:
                asset.disable_reset_pose()

    def get_termination_cfg(self):
        v = self.variant
        raw = TerminationTermCfg(
            func=part_seated,
            params={
                "held_cfg": SceneEntityCfg(self.held.name),
                "fixed_cfg": SceneEntityCfg(self.fixed.name),
                "seat_off": v.seat_off,
                "held_base_z_off": v.held_base_z_off,
                "align_tol": v.align_tol,
                "seat_tol": v.seat_tol,
                "min_upright_cos": v.min_upright_cos,
                "seat_overshoot_tol": v.seat_overshoot_tol,
            },
        )
        hold_params = {"func": raw.func, "params": raw.params}
        if v.min_thread_rad is not None:
            hold_params["min_thread_rad"] = v.min_thread_rad
            hold_params["engage_gap"] = v.engage_gap
        return AssemblyTerminationsCfg(
            success=TerminationTermCfg(func=hold_success, params=hold_params),
            part_dropped=TerminationTermCfg(
                func=mdp_isaac_lab.root_height_below_minimum,
                params={"minimum_height": DROP_HEIGHT, "asset_cfg": SceneEntityCfg(self.held.name)},
            ),
        )

    def get_events_cfg(self):
        v = self.variant
        # Held part + stand share one jitter; fixed + extras get an independent
        # one (extras keep their rigid offsets). The stand bore sits on the table.
        held_group = [(self.held.name, v.held_pos)]
        if self.stand is not None:
            held_group.insert(0, (self.stand.name, (v.held_pos[0], v.held_pos[1], TABLE_TOP_Z)))
        fixed_group = [(self.fixed.name, v.fixed_pos)] + [
            (asset.name, tuple(p + o for p, o in zip(v.fixed_pos, off)))
            for asset, (_, off) in zip(self.extras, v.extras)
        ]
        events = AssemblyEventsCfg(
            reset_all=EventTermCfg(
                func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True}
            ),
            randomize_workspace=EventTermCfg(
                func=reset_assembly_workspace,
                mode="reset",
                params={
                    "held_group": held_group,
                    "fixed_group": fixed_group,
                    "rand_held_xy": v.rand_xy,
                    "rand_fixed_xy": v.rand_xy if v.rand_fixed_xy is None else v.rand_fixed_xy,
                    "rand_fixed_yaw": v.rand_fixed_yaw,
                },
            ),
            friction_held=_friction_term(self.held.name, v.held_friction),
            friction_fixed=_friction_term(self.fixed.name, v.fixed_friction),
            # Pad μ only (not the whole robot) so grasp grip ≠ arm/table contacts.
            friction_pads=_friction_term("robot", PAD_FRICTION, PAD_BODIES),
        )
        if self.stand is not None:
            events.friction_stand = _friction_term(self.stand.name, STAND_FRICTION)
        for i, asset in enumerate(self.extras):
            setattr(events, f"friction_extra_{i}", _friction_term(asset.name, v.fixed_friction))
        return events

    def get_rewards_cfg(self):
        """Dense RL reward when reward_mode is staged|potential, else none."""
        if self.reward_mode not in ("staged", "potential"):
            return None
        from assembly_bench.environments.assembly.rewards import build_rewards_cfg
        return build_rewards_cfg(self.variant, self.held, self.fixed,
                                 mode=self.reward_mode)

    def get_scene_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode):
        raise NotImplementedError

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric(), ObjectMovedRateMetric(self.held)]

    def get_viewer_cfg(self) -> ViewerCfg:
        # In front of and above the workspace, opposite the robot.
        return get_viewer_cfg_look_at_object(lookat_object=self.held, offset=np.array([0.75, -0.55, 0.55]))

"""NIST assembly task for Isaac Lab Arena.

One task class covers all families (peg insert / gear mesh / nut thread);
only the seat geometry differs, carried by the variant. Success
is the source benchmark's outcome check: the held part's base point reaches
the seat target (fixed root + family offset, rotated into the fixed frame)
within the alignment/seat tolerances AND the part is at rest — debounced by
``hold_success`` (N consecutive true steps). The hard contact skill lives in
the policy, not the check: a mis-clocked peg, clashing gear, or cross-threaded
nut cannot descend, so the seat gap stays large.
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
# Failure: the held part fell below the tabletop.
DROP_HEIGHT = TABLE_TOP_Z - 0.05

# ---------------------------------------------------------------------------
# Success predicate
# ---------------------------------------------------------------------------


def part_seated(
    env: ManagerBasedRLEnv,
    held_cfg: SceneEntityCfg,
    fixed_cfg: SceneEntityCfg,
    seat_off: tuple[float, float, float],
    held_base_z_off: float,
    align_tol: float,
    seat_tol: float,
    max_speed: float = 0.05,
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

    aligned = torch.norm(held_pos[:, :2] - target[:, :2], dim=-1) < align_tol
    seat_gap = (held_pos[:, 2] + held_base_z_off) - target[:, 2]
    seated = seat_gap < seat_tol
    stable = torch.norm(wp.to_torch(held.data.root_lin_vel_w), dim=-1) < max_speed
    return aligned & seated & stable


class hold_success(ManagerTermBase):
    """Success wrapper: the raw predicate ``func(env, **params)`` must hold for
    ``SUCCESS_HOLD_STEPS`` consecutive steps. Class-based so it keeps a per-env
    streak counter; the TerminationManager resets it for envs that reset."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._streak = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None) -> None:
        self._streak[env_ids if env_ids is not None else slice(None)] = 0

    def __call__(self, env: ManagerBasedRLEnv, func, params: dict) -> torch.Tensor:
        self._streak = torch.where(func(env, **params), self._streak + 1, 0)
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
    """Post-reset warmup: settle the freshly jittered parts into contact, push
    the teleported transforms through fabric to the renderer (IsaacLab's
    reset() alone returns camera obs of the PREVIOUS state, see
    notes/ISSUE_stale_reset_camera_obs.md), then flush the RTX temporal state
    (DLAA history) with a Replicator-style subframe pump -- the step-loop's
    one-render-per-step did NOT clear teleport ghosts (near-solid ghosts of
    pre-reset poses survived 120 warmup frames and faded over ~300 recorded
    frames). ``rep.orchestrator.step(rt_subframes=N)`` is NVIDIA's documented
    flush for exactly this (SDG pipelines teleporting assets): it pauses the
    timeline and re-renders the SAME frame N times across all render products.
    Steps the whole sim, so it assumes benchmark-style global resets.

    The raw ``sim.step`` bypasses the manager's ``write_data_to_sim``, so the
    home joint-position TARGET staged by ``reset_scene_to_default`` never
    reaches PhysX -- the arm teleports home but the stale prior-episode target
    drags it back off during settle. Re-flush each step so the PD holds home."""
    del env_ids
    for _ in range(steps):
        for art in env.scene.articulations.values():
            art.write_data_to_sim()
        env.sim.step(render=True)
        for sensor in env.scene.sensors.values():
            sensor.update(dt=0.0, force_recompute=True)
    # Hard temporal-history reset: DLAA's static-pixel blend is too sticky to
    # flush by re-rendering alone (120 warmup renders + a 32-frame pump only
    # FADED teleport ghosts). Toggling the AA mode tears down the accumulation
    # buffers -- one FXAA frame has no history at all -- then DLAA rebuilds
    # them from the NEW scene; a short pump re-converges quality before the
    # first recorded frame. (rep.orchestrator.step is NOT usable here: it
    # blocks on Replicator's capture pipeline, which this workflow never runs.)
    import omni.kit.app
    import omni.replicator.core as rep
    app = omni.kit.app.get_app()
    env.sim.set_setting("/app/player/playSimulations", False)
    rep.settings.set_render_rtx_realtime(antialiasing="FXAA")
    for _ in range(2):
        app.update()
    rep.settings.set_render_rtx_realtime(antialiasing="DLAA")
    for _ in range(rt_subframes):
        app.update()
    env.sim.set_setting("/app/player/playSimulations", True)
    # Refetch so the first post-reset obs reads the pumped, ghost-free frame.
    for sensor in env.scene.sensors.values():
        sensor.update(dt=0.0, force_recompute=True)


def _friction_term(asset_name: str, friction: float) -> EventTermCfg:
    """Startup term pinning an asset's contact friction (source: set_friction)."""
    return EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (friction, friction),
            "dynamic_friction_range": (friction, friction),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "asset_cfg": SceneEntityCfg(asset_name),
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
        # "staged" turns on the RL shaping reward (rewards.py); None/other leaves
        # the env reward-free (eval default -- termination semantics unchanged).
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
            },
        )
        return AssemblyTerminationsCfg(
            success=TerminationTermCfg(func=hold_success, params={"func": raw.func, "params": raw.params}),
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
        )
        if self.stand is not None:
            events.friction_stand = _friction_term(self.stand.name, STAND_FRICTION)
        for i, asset in enumerate(self.extras):
            setattr(events, f"friction_extra_{i}", _friction_term(asset.name, v.fixed_friction))
        return events

    def get_rewards_cfg(self):
        """Staged RL reward when reward_mode="staged", else no reward terms."""
        if self.reward_mode != "staged":
            return None
        from assembly_bench.environments.assembly.rewards import build_rewards_cfg
        return build_rewards_cfg(self.variant, self.held, self.fixed)

    def get_scene_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode):
        raise NotImplementedError

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric(), ObjectMovedRateMetric(self.held)]

    def get_viewer_cfg(self) -> ViewerCfg:
        # In front of and above the workspace, opposite the robot.
        return get_viewer_cfg_look_at_object(lookat_object=self.held, offset=np.array([0.75, -0.55, 0.55]))

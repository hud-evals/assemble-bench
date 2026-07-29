"""Assembly-tuned DROID embodiment for Isaac Lab Arena.

Registers ``droid_abs_joint_pos_softmimic``: the stock Arena DROID absolute-
joint-position embodiment plus the AssembleBench's contact-stability
tuning, authored ONCE at construction instead of mutated imperatively inside
``AssembleBenchEnvironment.get_env`` on every env build.

Why the overlay exists
----------------------
The DROID hand is a Robotiq 2F-85 whose two fingers are mechanically linked;
PhysX models that linkage as a "mimic joint" -- a penalty spring with a
``naturalFrequency`` (stiffness) and ``dampingRatio``. The stock asset authors
it near-rigid and undamped (naturalFrequency 1e6, dampingRatio ~0). At the
benchmark's 60 Hz step that spring is numerically unstable (omega*dt >> 1), so a
hard ram self-amplifies and the gripper explodes (velocities -> NaN).

The calibrated DROID USD must not be edited (its wrist camera is mounted
frame-for-frame against the source demos), so we author a thin USD *overlay*
that references the calibrated asset and overrides only the five mimic springs
to a stiff, well-damped coupling (naturalFrequency 1000, dampingRatio 0.7).
PhysX parses joint params at spawn, so this has to live on the joint prim, not
an actuator.

Why a registered embodiment (not imperative mutation)
-----------------------------------------------------
The overlay is pure static data (it depends only on the source USD path), so it
is authored once per process and reused. ``Usd.Stage.CreateNew`` raises if the
layer identifier is already registered -- which is exactly what broke the served
flow, where the env is built twice (serve-time build + first ``reset`` rebuild).
Binding the tuning to a registered embodiment subclass makes the build path
declarative and rebuild-safe, per Arena's external-package pattern.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

from isaaclab.managers import ObservationTermCfg as ObsTerm, SceneEntityCfg

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment

from assembly_bench.environments.assembly.observations import arm_joint_vel, part_pose

# Robotiq 2F-85 mimic-joint prims (relative to the flattened DROID asset) and
# the PhysX penalty-spring axis each one couples.
_MIMIC_JOINTS = (
    ("right_outer_knuckle_joint", "rotZ"),
    ("right_inner_finger_joint", "rotX"),
    ("right_inner_finger_knuckle_joint", "rotX"),
    ("left_inner_finger_knuckle_joint", "rotX"),
    ("left_inner_finger_joint", "rotX"),
)
_MIMIC_NATURAL_FREQUENCY = 1000.0
_MIMIC_DAMPING_RATIO = 0.7

# Authored overlays, keyed by source USD path: the overlay is static, so build
# it once per process and reuse it across every env (re)build. A second
# Usd.Stage.CreateNew on the same layer identifier would raise.
_OVERLAY_CACHE: dict[str, str] = {}


def _softened_mimic_overlay(source_usd_path: str) -> str:
    """Path to a USD overlay referencing *source_usd_path* with the Robotiq
    mimic springs softened. Authored once per process, then reused."""
    cached = _OVERLAY_CACHE.get(source_usd_path)
    if cached is not None and os.path.exists(cached):
        return cached

    from pxr import Sdf, Usd

    overlay = os.path.join(tempfile.gettempdir(), "asm_droid_softened_mimic.usd")
    # USD caches layers by identifier; reuse an already-open layer (authored by
    # an earlier build in this process) instead of re-creating it.
    if Sdf.Layer.Find(overlay) is None:
        if os.path.exists(overlay):
            os.remove(overlay)
        stage = Usd.Stage.CreateNew(overlay)
        root = stage.OverridePrim("/panda")
        root.GetReferences().AddReference(source_usd_path)
        stage.SetDefaultPrim(root.GetPrim())
        for joint, axis in _MIMIC_JOINTS:
            prim = stage.OverridePrim(f"/panda/Gripper/Robotiq_2F_85/Joints/{joint}")
            prim.CreateAttribute(
                f"physxMimicJoint:{axis}:naturalFrequency", Sdf.ValueTypeNames.Float
            ).Set(_MIMIC_NATURAL_FREQUENCY)
            prim.CreateAttribute(
                f"physxMimicJoint:{axis}:dampingRatio", Sdf.ValueTypeNames.Float
            ).Set(_MIMIC_DAMPING_RATIO)
        stage.GetRootLayer().Save()

    _OVERLAY_CACHE[source_usd_path] = overlay
    return overlay


def apply_assembly_droid_tuning(embodiment: Any) -> None:
    """Apply the AssembleBench's contact-stability tuning to a DROID
    embodiment in place: softened-mimic spawn overlay, solver-iteration
    headroom, a compliant arm, and joint velocity in the policy observation.

    Kept as a module function so any DROID variant can opt into the same
    tuning without duplicating it.
    """
    robot = embodiment.scene_config.robot

    # Point the spawn at the softened-mimic overlay (keeps the calibrated asset,
    # overrides only the mimic springs). Mimic coupling -- not the finger drive --
    # was the NaN divergence root cause; finger PD is set explicitly below.
    robot.spawn.usd_path = _softened_mimic_overlay(robot.spawn.usd_path)

    # Ramming the table should stall, not explode: keep depenetration bounded,
    # add articulation velocity iterations (0 -> 8) so the solver damps contact
    # velocity, and raise position iterations (64 -> 192) to give the stiff
    # mimic spring the headroom to hold without diverging.
    robot.spawn.rigid_props.max_depenetration_velocity = 5.0
    robot.spawn.articulation_props.solver_velocity_iteration_count = 8
    robot.spawn.articulation_props.solver_position_iteration_count = 192

    # Compliant arm for pi0.5 grip ("arm gives, gripper stays"): soft PD so
    # off-center pad contact yields a few mm instead of knocking the peg away.
    # Scripted-expert insert seating needs firmer 150/40 — pass --arm_stiffness
    # 150 --arm_damping 40 from run_expert recording runs only.
    for name, vlim, stiff, damp in (
        ("panda_shoulder", 2.175, 100.0, 28.0),  # was 400/80 stock, then 150/40
        ("panda_forearm", 2.61, 50.0, 18.0),     # softer wrist for pad give
    ):
        robot.actuators[name].velocity_limit_sim = vlim
        robot.actuators[name].armature = 1e-3
        robot.actuators[name].stiffness = stiff
        robot.actuators[name].damping = damp

    # Gripper finger drive stays USD-native (stiff 100, damp ~0, maxForce 16.5).
    # Do not override with high PD/effort — that drove through the underactuated
    # linkage and broke scripted-expert microlift clamp on round pegs.

    # Joint velocities alongside joint positions in the policy obs (recorded to
    # HDF5 via the flat policy-obs recorder term).
    embodiment.observation_config.policy.joint_vel = ObsTerm(func=arm_joint_vel)
    # Privileged part poses for the PA-RL critic (not fed to the VLA).
    embodiment.observation_config.policy.held_part_pose = ObsTerm(
        func=part_pose, params={"asset_cfg": SceneEntityCfg("held_part")})
    embodiment.observation_config.policy.fixed_part_pose = ObsTerm(
        func=part_pose, params={"asset_cfg": SceneEntityCfg("fixed_part")})


@register_asset
class DroidAbsoluteJointPositionSoftMimicEmbodiment(DroidAbsoluteJointPositionEmbodiment):
    """DROID absolute-joint-position embodiment with the AssembleBench's
    contact-stability tuning baked in (softened Robotiq mimic + solver/PD).

    This is the benchmark's default embodiment; the pi0.5-DROID contract (8-D
    action: 7 joint targets + binary gripper) is unchanged from the stock
    absolute-joint-position embodiment it subclasses.
    """

    name = "droid_abs_joint_pos_softmimic"
    tags = ["embodiment", "assembly"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        apply_assembly_droid_tuning(self)

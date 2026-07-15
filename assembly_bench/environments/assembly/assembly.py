"""NIST assembly benchmark environment for Isaac Lab Arena (externally defined).

A Franka faces the NIST-taskboard workspace on the ``table`` background, with
one of 27 task variants (peg insert / gear mesh / nut thread)
selected via ``--task``. Run with, e.g.::

    python isaaclab_arena/evaluation/policy_runner.py \\
        --policy_type zero_action --num_episodes 1 \\
        --external_environment_class_path \\
        assembly_bench.environments.assembly.assembly:AssemblyBenchEnvironment \\
        assembly_bench --task peg_round_8mm_tight
"""

import argparse

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

from assembly_bench.environments.assembly.variants import TABLE_TOP_Z, VARIANTS


def assembly_bench_env_cfg_callback(env_cfg):
    """Arena's assembly sim settings, at the benchmark's 15 Hz control rate.

    Arena's callback sets 60 Hz physics with decimation 2 (30 Hz control);
    pi0.5-DROID runs at 15 Hz, so decimate by 4 and render once per policy
    step (camera observations and recorded videos are 15 fps).
    """
    from isaaclab.sim import RenderCfg
    from isaaclab_arena_environments import mdp

    env_cfg = mdp.assembly_env_cfg_callback(env_cfg)
    env_cfg.decimation = 4
    env_cfg.sim.render_interval = env_cfg.decimation
    # Grasp reliability: "max" friction combine takes the higher of the two
    # contacting materials' coefficients, so the grippy part (held_friction up to
    # 1.0) governs the pad contact regardless of the pad material -- the Isaac Lab
    # consensus fix for objects slipping out of a grasp (default "average" dilutes
    # a high part friction against a low pad friction).
    env_cfg.sim.physics_material.friction_combine_mode = "max"
    # Contact-velocity iterations: Arena ships max_velocity_iteration_count=1,
    # which under-resolves contact velocities -- a hard ram into the table then
    # diverges (joint vels blow to 1e30+). Raise it so TGS actually damps the
    # contact velocity and ramming stays bounded.
    env_cfg.sim.physics.max_velocity_iteration_count = 4
    # DLAA (native-res temporal AA): its history is also the denoiser -- FXAA
    # A/B measured temporal grain ~3.0 vs DLAA's ~0.15 (unusable boil), and
    # per-frame knobs (spp, DL denoiser) proved inert at runtime. The per-wave
    # teleport ghosts come from RTX geometry streaming holding stale transforms
    # (kit warning: readTransformsFromFabricInRenderDelegate + geometry
    # streaming "dynamic objects not streaming correctly"), so ghosting is
    # fixed at the source: run_expert disables geometry streaming via carb
    # right after app launch, before the scene loads.
    env_cfg.sim.render = RenderCfg(antialiasing_mode="DLAA")
    return env_cfg


class AssemblyBenchEnvironment(ExampleEnvironmentBase):

    name: str = "assembly_bench"

    def get_env(self, args_cli: argparse.Namespace):
        import os

        import isaaclab.sim as sim_utils
        import torch

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena_environments import mdp

        # Importing scene registers the asm_* assets with the AssetRegistry.
        import assembly_bench.environments.assembly.scene  # noqa: F401
        from assembly_bench.environments.assembly.cameras import make_assembly_camera_cfg
        from assembly_bench.environments.assembly.tasks import NISTAssemblyTask

        variant = VARIANTS[args_cli.task]

        # Background (the SeattleLabTable, as in the source benchmark), light and
        # embodiment from the registry. An HDR-registry HDRI becomes the dome
        # light's environment texture ("none" keeps a plain dome light).
        # Table pose as in the Isaac Lab Factory env (90-degree yaw; this
        # IsaacLab stack is xyzw end to end, so the tuple reads as authored).
        background = self.asset_registry.get_asset_by_name("table")()
        background.set_initial_pose(Pose(position_xyz=(0.55, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.70711, 0.70711)))
        light = self.asset_registry.get_asset_by_name("light")(
            spawner_cfg=sim_utils.DomeLightCfg(intensity=args_cli.light_intensity),
        )
        if args_cli.hdr != "none":
            light.add_hdr(self.hdr_registry.get_hdr_by_name(args_cli.hdr)())
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
            enable_cameras=args_cli.enable_cameras,
        )
        if args_cli.embodiment.startswith("droid"):
            # Keep Arena's calibrated DROID asset (franka_robotiq_2f_85_flattened.usd):
            # its wrist camera is mounted frame-for-frame against the source demos.
            from isaaclab.managers import ObservationTermCfg as ObsTerm

            from assembly_bench.environments.assembly.observations import arm_joint_vel

            # Keep the DROID USD's native firm finger drive (the Arena "gripper"
            # actuator: stiffness/damping = None -> USD-authored gains). It is
            # stiff, so the pad stops firmly at the part surface; the earlier soft
            # retune (stiffness 17) let the finger creep into the grasped part --
            # the collision regression. Explosion safety comes from the softened
            # mimic overlay below (the mimic coupling, not the finger drive, was
            # the divergence root cause per probe_ram), so the firm drive can stay
            # without reintroducing the blow-up.
            robot = embodiment.scene_config.robot

            # Soften the Robotiq mimic coupling. The USD authors it near-rigid and
            # undamped (naturalFrequency 1e6/1000, dampingRatio 0/0.05); at 60 Hz
            # that penalty spring is numerically unstable (omega*dt >> 1), so a hard
            # ram makes the coupling self-amplify and the gripper explodes. Author a
            # local overlay USD that references the calibrated asset and only
            # overrides the mimic spring to a stiff, well-damped coupling
            # (naturalFrequency 1000, dampingRatio 0.7). 1000 diverged natively at
            # 64 solver iters; it holds now that solver_position_iteration_count is
            # raised to 192 (below), which gives the penalty spring the headroom to
            # carry this stiffness -- verify with probe_ram. PhysX parses joint
            # params at spawn, so an actuator override can't reach them -- it must
            # be on the joint prim.
            import os
            import tempfile

            from pxr import Sdf, Usd

            _overlay = os.path.join(tempfile.gettempdir(), "droid_softened_mimic.usd")
            if os.path.exists(_overlay):
                os.remove(_overlay)
            _ov = Usd.Stage.CreateNew(_overlay)
            _root = _ov.OverridePrim("/panda")
            _root.GetReferences().AddReference(robot.spawn.usd_path)
            _ov.SetDefaultPrim(_root.GetPrim())
            for _jp, _axis in (
                ("right_outer_knuckle_joint", "rotZ"),
                ("right_inner_finger_joint", "rotX"),
                ("right_inner_finger_knuckle_joint", "rotX"),
                ("left_inner_finger_knuckle_joint", "rotX"),
                ("left_inner_finger_joint", "rotX"),
            ):
                _j = _ov.OverridePrim(f"/panda/Gripper/Robotiq_2F_85/Joints/{_jp}")
                _j.CreateAttribute(
                    f"physxMimicJoint:{_axis}:naturalFrequency", Sdf.ValueTypeNames.Float).Set(1000.0)
                _j.CreateAttribute(
                    f"physxMimicJoint:{_axis}:dampingRatio", Sdf.ValueTypeNames.Float).Set(0.7)
            _ov.GetRootLayer().Save()
            robot.spawn.usd_path = _overlay

            # Native "gripper" actuator (firm finger drive) kept as-is -- no retune.

            # Durability: make ramming the table stall instead of explode.
            # `velocity_limit` is a no-op on implicit actuators, so the arm ran
            # uncapped at the USD's 10 rad/s -- restore the real panda joint-speed
            # limits via velocity_limit_sim; armature (rotor inertia, base-panda
            # value) damps high-PD jitter under contact. The key stabilizer is the
            # articulation velocity iterations (0 -> 8): with 0 the solver never
            # damps contact velocity and a hard ram diverges; more iterations only
            # sharpen contact resolution, so keep them high. Do NOT starve the
            # robot's contact force (a low max_contact_impulse) or over-cap
            # depenetration -- that lets the arm tunnel through the fixed blocks.
            robot.spawn.rigid_props.max_depenetration_velocity = 5.0
            robot.spawn.articulation_props.solver_velocity_iteration_count = 8
            # Mimic-coupling stability headroom: a physxMimicJoint is a penalty
            # spring whose max stable stiffness (naturalFrequency) scales with the
            # articulation's position-solver iterations. Raise them 64 -> 192 so a
            # much higher mimic naturalFrequency can hold without diverging (i.e.
            # stiffer, less-giving pads). Iterations alone don't stiffen the pads --
            # they unlock room to push naturalFrequency up (see the mimic overlay).
            robot.spawn.articulation_props.solver_position_iteration_count = 192
            for _name, _vlim in (("panda_shoulder", 2.175), ("panda_forearm", 2.61)):
                robot.actuators[_name].velocity_limit_sim = _vlim
                robot.actuators[_name].armature = 1e-3
                # Compliant arm: soften PD (stiffness 400->150, damping 80->40) so
                # the arm yields under contact instead of shoving the now-rigid
                # gripper into the part -- the "arm gives, gripper stays" behavior.
                # Lower stiffness = the arm backs off more, so the gripper can still
                # close/grip while pressed against the table.
                robot.actuators[_name].stiffness = 150.0
                robot.actuators[_name].damping = 40.0

            # Do NOT override robot.spawn.collision_props. The calibrated DROID
            # USD's native gripper collision offsets grasp cleanly (verified
            # working before the sim-fix commits); blanket-applying a
            # CollisionPropertiesCfg here expands every robot collider (positive
            # rest_offset -> finger pads self-collide + pad-vs-part contact
            # mis-seats), which is what regressed collision. Keep native.

            # Joint velocities alongside joint positions in the policy obs (both
            # land in recorded HDF5s via the flat policy-obs recorder term).
            embodiment.observation_config.policy.joint_vel = ObsTerm(func=arm_joint_vel)
        if args_cli.embodiment == "droid_differential_ik":
            # Teleop embodiment fixes (Arena's DIK cfg is untested for DROID):
            # - body_name: upstream says "panda_link0" (the ARM BASE -- fixed-base
            #   jacobian indexing then wraps to the last body; the arm can't servo).
            #   Control the Robotiq "base_link" instead: same frame the eef_pos/
            #   eef_quat observations and the scripted expert's IK servo use.
            # - gripper: Se3Keyboard emits +1 open / -1 close; the ZeroToOne term
            #   (>0.5 = close) inverts that. Use the stock binary term (<0 = close),
            #   which matches the keyboard. NOTE: raw DIK gripper actions are thus
            #   +/-1, not the canonical 0/1 -- the canonical label stream is the
            #   recorded `abs_joint_action` (see recorders.py), which is derived
            #   from processed joint targets and convention-independent.
            from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg

            embodiment.action_config.arm_action.body_name = "base_link"
            embodiment.action_config.gripper_action = BinaryJointPositionActionCfg(
                asset_name="robot",
                joint_names=["finger_joint"],
                open_command_expr={"finger_joint": 0.0},
                close_command_expr={"finger_joint": torch.pi / 4},
            )
        if "franka" in args_cli.embodiment:
            # The Factory-tuned high-PD arm the Arena assembly examples use.
            embodiment.scene_config.robot = mdp.FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.replace(
                prim_path="{ENV_REGEX_NS}/Robot"
            )
        # Cameras: the embodiment's calibrated wrist camera + one frontal
        # exterior view, DROID-native 1280x720 (policy adapters resize).
        embodiment.camera_config = make_assembly_camera_cfg(embodiment)

        # Parts at explicit poses (the reset event re-jitters them per episode).
        # The peg's presentation stand is a second bore at the held position.
        held = self.asset_registry.get_asset_by_name(variant.held)(
            instance_name="held_part", initial_pose=Pose(position_xyz=variant.held_pos))
        fixed = self.asset_registry.get_asset_by_name(variant.fixed)(
            instance_name="fixed_part", initial_pose=Pose(position_xyz=variant.fixed_pos))
        stand = None
        if variant.stand is not None:
            # "peg_stand", not "stand": the DROID embodiment's robot stand
            # already occupies the "stand" scene entity name.
            stand = self.asset_registry.get_asset_by_name(variant.stand)(
                instance_name="peg_stand",
                initial_pose=Pose(position_xyz=(variant.held_pos[0], variant.held_pos[1], TABLE_TOP_Z)))
        extras = [
            self.asset_registry.get_asset_by_name(asset_name)(
                instance_name=f"extra_{i}",
                initial_pose=Pose(position_xyz=tuple(p + o for p, o in zip(variant.fixed_pos, off))))
            for i, (asset_name, off) in enumerate(variant.extras)
        ]

        parts = [a for a in (held, fixed, stand, *extras) if a is not None]
        scene = Scene(assets=[background, light, *parts])

        task = NISTAssemblyTask(variant=variant, held=held, fixed=fixed, stand=stand, extras=extras,
                                reward_mode=getattr(args_cli, "reward", None))

        # Optional teleoperation (demo collection). Only the differential-IK
        # embodiment can consume SE(3) devices; a keyboard cannot emit absolute
        # joint targets, so fail loudly instead of moving nothing.
        teleop_device = None
        teleop_device_name = getattr(args_cli, "teleop_device", None)
        if teleop_device_name is not None:
            if args_cli.embodiment != "droid_differential_ik":
                raise ValueError(
                    f"--teleop_device {teleop_device_name} requires --embodiment droid_differential_ik "
                    f"(got {args_cli.embodiment}): SE(3) teleop devices cannot drive absolute joint-position "
                    "actions."
                )
            # Arena's registry builds the device with a low default sensitivity
            # (0.05) and ignores teleop.py's --sensitivity flag. Raise it here so
            # each keypress moves the arm more per step; override per session with
            # TELEOP_POS_SENS / TELEOP_ROT_SENS. Keep rotation a touch lower than
            # translation -- roll/pitch/yaw at high gain make fine alignment jumpy.
            pos_sens = float(os.environ.get("TELEOP_POS_SENS", "0.1"))
            rot_sens = float(os.environ.get("TELEOP_ROT_SENS", "0.08"))
            # The streamed viewport looks at the workspace from the operator's
            # side, mirrored vs. the robot base frame, so W/S, A/D and Q/E all
            # read backwards. All three translation axes scale by pos_sensitivity,
            # so negating it flips them together (rotation unaffected). Set
            # TELEOP_INVERT_XLATE=0 to restore the raw base-frame directions.
            if os.environ.get("TELEOP_INVERT_XLATE", "1") == "1":
                pos_sens = -pos_sens
            teleop_device = self.device_registry.get_device_by_name(teleop_device_name)(
                pos_sensitivity=pos_sens, rot_sensitivity=rot_sens
            )

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=assembly_bench_env_cfg_callback,
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--task", type=str, default="peg_round_8mm_tight", choices=sorted(VARIANTS))
        # DROID platform (Franka + Robotiq 2F-85) with absolute joint-position
        # actions -- the source benchmark's droid_jointpos / pi0.5-DROID contract.
        parser.add_argument("--embodiment", type=str, default="droid_abs_joint_pos")
        parser.add_argument("--hdr", type=str, default="asm_machine_shop",
                            help='HDR name from the registry (e.g. "asm_machine_shop", '
                                 '"carpentry_shop_robolab"), or "none"')
        parser.add_argument("--light_intensity", type=float, default=1500.0)
        # "staged" enables the RL shaping reward (rewards.py); default "none"
        # keeps the env reward-free for eval.
        parser.add_argument("--reward", type=str, default="none", choices=["none", "staged"])
        # Teleop demo collection (Arena teleop.py / record_demos.py read this).
        # Requires --embodiment droid_differential_ik; default None keeps eval/
        # training paths teleop-free (no retargeter exists for abs joint pos).
        parser.add_argument("--teleop_device", type=str, default=None, choices=["keyboard", "spacemouse"],
                            help="SE(3) teleop device for demo collection "
                                 "(requires --embodiment droid_differential_ik)")


def make_assembly_env(
    task: str = "peg_round_8mm_tight",
    num_envs: int = 1,
    embodiment: str = "droid_abs_joint_pos",
    hdr: str = "asm_machine_shop",
    light_intensity: float = 1500.0,
    reward: str = "none",
):
    """Build the assembly Arena gym env for one variant (the Isaac app must be up).

    Shared factory for both run paths: the HUD server (``env.py``) and the RLinf
    training adapter. ``reward="staged"`` turns on the RL shaping reward; "none"
    (default) keeps the env reward-free for eval. ``num_envs`` is the vectorization
    width -- one build serves N lockstep slots.
    """
    import carb
    from isaaclab_arena.cli.isaaclab_arena_cli import (
        arena_env_builder_cfg_from_argparse,
        get_isaaclab_arena_cli_parser,
    )
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    # Rebuild guard: env construction calls rep.set_global_seed(cfg.seed), which
    # touches the replicator graph; pre-sync the carb setting to the Arena default
    # so a mid-run rebuild short-circuits instead of erroring.
    carb.settings.get_settings().set_int("/omni/replicator/globalSeed", 42)

    parser = get_isaaclab_arena_cli_parser()
    AssemblyBenchEnvironment.add_cli_args(parser)
    args, _ = parser.parse_known_args([])
    args.task, args.embodiment, args.hdr = task, embodiment, hdr
    args.light_intensity, args.enable_cameras = light_intensity, True
    args.num_envs = num_envs
    args.reward = reward
    arena_env = AssemblyBenchEnvironment().get_env(args)
    builder_cfg = arena_env_builder_cfg_from_argparse(args)
    return ArenaEnvBuilder(arena_env, builder_cfg).make_registered(render_mode="rgb_array")

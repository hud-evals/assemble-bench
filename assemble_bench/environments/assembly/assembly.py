"""NIST assembly benchmark environment for Isaac Lab Arena (externally defined).

A Franka faces the NIST-taskboard workspace on the ``table`` background, with
one of 14 benchmark variants (peg insert / gear mesh / nut thread), plus a
``debug`` pick-place smoke, selected via ``--task``. Run with, e.g.::

    python isaaclab_arena/evaluation/policy_runner.py \\
        --policy_type zero_action --num_episodes 1 \\
        --external_environment_class_path \\
        assemble_bench.environments.assembly.assembly:AssembleBenchEnvironment \\
        assemble_bench --task peg_round_8mm
"""

import argparse

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

from assemble_bench.environments.assembly.variants import TABLE_TOP_Z, VARIANTS


def assemble_bench_env_cfg_callback(env_cfg):
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
    # Grasp reliability: "max" takes the higher of the two contacting μ's, so
    # rubber pads (PAD_FRICTION≈1.2) win over a metal-ish peg (held≈0.7)
    # without inventing an unrealistic peg μ. Default "average" would dilute
    # the pad against the part.
    env_cfg.sim.physics_material.friction_combine_mode = "max"
    # Contact-velocity iterations: Arena ships max_velocity_iteration_count=1,
    # which under-resolves contact velocities -- a hard ram into the table then
    # diverges (joint vels blow to 1e30+). Raise it so TGS actually damps the
    # contact velocity and ramming stays bounded.
    env_cfg.sim.physics.max_velocity_iteration_count = 4
    # CCD: RoboLab pick-place default. Helps fast contacts / tunneling; kinematic
    # sockets ignore it (PhysX warning only). Pegs keep their thin object skin.
    env_cfg.sim.physics.enable_ccd = True
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


class AssembleBenchEnvironment(ExampleEnvironmentBase):

    name: str = "assemble_bench"

    def get_env(self, args_cli: argparse.Namespace):
        import isaaclab.sim as sim_utils
        import torch

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena_environments import mdp

        # Importing scene registers the asm_* assets with the AssetRegistry;
        # importing embodiments registers droid_abs_joint_pos_softmimic.
        import assemble_bench.environments.assembly.embodiments  # noqa: F401
        import assemble_bench.environments.assembly.scene  # noqa: F401
        from assemble_bench.environments.assembly.cameras import make_assembly_camera_cfg
        from assemble_bench.environments.assembly.tasks import NISTAssemblyTask

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
        # The benchmark's DROID contact-stability tuning (softened
        # Robotiq mimic overlay + solver/PD) lives in the registered
        # `droid_abs_joint_pos_softmimic` embodiment (see embodiments.py), the
        # default below -- no imperative USD authoring in the build path. The
        # calibrated DROID asset (wrist camera mounted frame-for-frame against
        # the source demos) is preserved via a referencing overlay.
        if args_cli.embodiment == "droid_differential_ik":
            # Arena's DIK cfg defaults are wrong for DROID: body_name must be
            # Robotiq base_link (not panda_link0), and gripper uses stock binary
            # (<0 = close) so EE-delta tooling matches Se3 devices.
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

        # Privileged part poses on the wire (policy/held_part_pose, policy/
        # fixed_part_pose) for the PA-RL critic; policy adapters ignore them.
        from isaaclab.managers import ObservationTermCfg as ObsTerm
        from isaaclab.managers import SceneEntityCfg

        from assemble_bench.environments.assembly.observations import asset_root_pose

        embodiment.observation_config.policy.held_part_pose = ObsTerm(
            func=asset_root_pose, params={"asset_cfg": SceneEntityCfg("held_part")})
        embodiment.observation_config.policy.fixed_part_pose = ObsTerm(
            func=asset_root_pose, params={"asset_cfg": SceneEntityCfg("fixed_part")})

        task = NISTAssemblyTask(variant=variant, held=held, fixed=fixed, stand=stand, extras=extras,
                                reward_mode=getattr(args_cli, "reward", None))

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            env_cfg_callback=assemble_bench_env_cfg_callback,
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--task", type=str, default="peg_round_8mm", choices=sorted(VARIANTS))
        # DROID platform (Franka + Robotiq 2F-85) with absolute joint-position
        # actions -- the source benchmark's droid_jointpos / pi0.5-DROID contract.
        # The _softmimic variant adds the benchmark's contact-stability tuning
        # (see embodiments.py); plain droid_abs_joint_pos is the untuned stock.
        parser.add_argument("--embodiment", type=str, default="droid_abs_joint_pos_softmimic")
        parser.add_argument("--hdr", type=str, default="asm_machine_shop",
                            help='HDR name from the registry (e.g. "asm_machine_shop", '
                                 '"carpentry_shop_robolab"), or "none"')
        parser.add_argument("--light_intensity", type=float, default=1500.0)
        # "staged" / "potential" enable dense RL shaping (rewards.py); default
        # "none" keeps the env reward-free for eval.
        parser.add_argument("--reward", type=str, default="none",
                            choices=["none", "staged", "potential"])


def make_assembly_env(
    task: str = "peg_round_8mm",
    num_envs: int = 1,
    embodiment: str = "droid_abs_joint_pos_softmimic",
    hdr: str = "asm_machine_shop",
    light_intensity: float = 1500.0,
    reward: str = "none",
):
    """Build the assembly Arena gym env for one variant (the Isaac app must be up).

    Shared factory for HUD / custom RL loops. ``reward="staged"|"potential"``
    turns on dense shaping; "none" (default) keeps the env reward-free for eval.
    ``num_envs`` is the vectorization width.
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
    AssembleBenchEnvironment.add_cli_args(parser)
    args, _ = parser.parse_known_args([])
    args.task, args.embodiment, args.hdr = task, embodiment, hdr
    args.light_intensity, args.enable_cameras = light_intensity, True
    args.num_envs = num_envs
    args.reward = reward
    arena_env = AssembleBenchEnvironment().get_env(args)
    builder_cfg = arena_env_builder_cfg_from_argparse(args)
    return ArenaEnvBuilder(arena_env, builder_cfg).make_registered(render_mode="rgb_array")

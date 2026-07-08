"""NIST assembly benchmark environment for Isaac Lab Arena (externally defined).

A Franka faces the NIST-taskboard workspace on the ``table`` background, with
one of 29 task variants (peg insert / gear mesh / nut thread)
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
    # DLAA instead of the default DLSS: native-resolution temporal AA. DLSS
    # renders low-res and upscales via temporal reprojection, which smears
    # moving/teleported geometry into ghosts and shimmers edges (aliasing).
    env_cfg.sim.render = RenderCfg(antialiasing_mode="DLAA")
    return env_cfg


class AssemblyBenchEnvironment(ExampleEnvironmentBase):

    name: str = "assembly_bench"

    def get_env(self, args_cli: argparse.Namespace):
        import isaaclab.sim as sim_utils

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

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
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
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
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
    return ArenaEnvBuilder(arena_env, args).make_registered(render_mode="rgb_array")

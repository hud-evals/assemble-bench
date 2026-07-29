"""Run a scripted expert on one variant, streaming every rollout to HUD.

Builds the Arena env exactly like the policy runner, wraps it with ``hud.wrap``
(each env slot's episode becomes a watchable trace with wrist+front video),
and drives the family's phase machine until every env terminates or times out.
Prints phase-population and pose diagnostics as it goes. Run (isaac6 env):

    python scripts/experts/run_expert.py --headless --task peg_round_8mm --num_envs 4

Common recording recipes (HDF5 under a path on the container bind mount)::

    # Peg demos with dense staged reward + failures (PA-RL critic seed)
    python scripts/experts/run_expert.py --headless --task peg_round_8mm \\
        --num_envs 8 --waves 40 --max_demos 100 --keep_failures \\
        --reward staged --record data/hdf5/peg_round_8mm_rewards.hdf5

    # Same with potential-shaped reward
    … --reward potential --record data/hdf5/peg_round_8mm_potential.hdf5

    # Nut-thread tiers (success-filtered; longer episodes)
    python scripts/experts/run_expert.py --headless --task nut_M16 \\
        --num_envs 8 --waves 80 --max_demos 50 --episode_length_s 150 \\
        --record data/hdf5/nut_M16.hdf5
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

parser = get_isaaclab_arena_cli_parser()
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = "--disable_cameras" not in sys.argv

with SimulationAppContext(args_cli):
    import gymnasium as gym
    import torch

    import hud
    import warp as wp
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from assembly_bench.environments.assembly.assembly import AssembleBenchEnvironment
    from assembly_bench.environments.assembly.variants import VARIANTS

    from experts import gear, nut, peg
    from experts.base import Servo, pos_of, quat_of
    from experts.record import Recorder

    class SuccessInfo(gym.Wrapper):
        """Surface the ``success`` termination into the step ``info`` dict.

        IsaacLab only exposes metrics under ``extras["log"]``, so hud.wrap's
        success probe (info["success"]/["is_success"]/["task_success"]) finds
        nothing and every trace closes UNLABELED. Read the term's pre-reset
        buffer (get_term returns the value that triggered the auto-reset) and
        expose it so streamed traces are marked success/fail and reward=success.
        """

        def step(self, action):
            obs, rew, term, trunc, info = self.env.step(action)
            try:
                info = {**info, "success": self.env.unwrapped.termination_manager.get_term("success")}
            except (KeyError, AttributeError):
                pass
            return obs, rew, term, trunc, info

    AssembleBenchEnvironment.add_cli_args(parser)
    parser.add_argument("--max_steps", type=int, default=None,
                        help="control steps per wave (default: task episode length at 15 Hz)")
    parser.add_argument("--episode_length_s", type=float, default=None,
                        help="override task timeout for long expert-development runs")
    parser.add_argument("--waves", type=int, default=1, help="global reset cycles to run")
    parser.add_argument("--job_name", type=str, default=None,
                        help="HUD job name override (useful for calibration sweeps)")
    parser.add_argument("--snap_every", type=int, default=0,
                        help="save env0 front+wrist frames every N steps to /tmp/expert_snaps")
    parser.add_argument("--sweep_aim", type=str, default=None,
                        help="axis,lo,hi (e.g. x,-0.014,0.007): per-env grasp aim offset sweep")
    parser.add_argument("--sweep_lead", type=str, default=None,
                        help="lo,hi: per-env nut thread lead-phase offset sweep in radians")
    parser.add_argument("--robot_usd", type=str, default=None,
                        help="override the robot USD (A/B-test gripper collision assets)")
    parser.add_argument("--grip_effort", type=float, default=None,
                        help="cap the finger drive effort (N*m). The USD default 16.5 is the "
                             "source bench's documented pathological config; 1.5 its validated fix")
    parser.add_argument("--arm_stiffness", type=float, default=None,
                        help="override arm PD stiffness for contact calibration")
    parser.add_argument("--arm_damping", type=float, default=None,
                        help="override arm PD damping for contact calibration")
    parser.add_argument("--language_instruction", type=str, default=None,
                        help="override the task's own description (Arena builder reads this; "
                             "None falls back to task.get_task_description())")
    parser.add_argument("--record", type=str, default=None,
                        help="write success-filtered episodes to this HDF5 (for LeRobot export)")
    parser.add_argument("--max_demos", type=int, default=None,
                        help="stop after this many demos are recorded (bulk data-gen)")
    parser.add_argument("--keep_failures", action="store_true",
                        help="record failed episodes too (default: success-filtered)")
    parser.add_argument("--target_success_rate", type=float, default=None,
                        help="with --max_demos, bank successes/failures to this mix "
                             "(e.g. 0.7 -> 70 success / 30 fail); implies keep_failures")
    parser.add_argument("--stream", action="store_true",
                        help="force HUD trace streaming even during a --record run (default: "
                             "streaming is ON for interactive runs, OFF for --record so bulk "
                             "data-gen never stalls on a dead telemetry endpoint)")
    parser.add_argument("--no_stream", action="store_true",
                        help="disable HUD streaming for fast policy-only diagnostics")
    parser.add_argument("--disable_cameras", action="store_true",
                        help="disable cameras for fast policy-only diagnostics (no stream/record)")
    parser.add_argument("--reset_warmup_steps", type=int, default=120,
                        help="rendered physics frames after reset; use 4-8 for fast expert iteration")
    parser.add_argument("--reset_rt_subframes", type=int, default=32,
                        help="render-only DLAA history flush after reset; use 1 for fast iteration")
    parser.add_argument("--camera_scale", type=float, default=1.0,
                        help="scale both camera resolutions (0.5 = 640x360 debug streams)")
    parser.add_argument("--held_friction", type=float, default=None,
                        help="override held-part friction for contact calibration")
    args_cli, _ = parser.parse_known_args()
    if args_cli.disable_cameras and (args_cli.record or args_cli.stream):
        raise ValueError("--disable_cameras cannot be combined with --record or --stream")
    args_cli.enable_cameras = not args_cli.disable_cameras
    os.environ["ASSEMBLY_RESET_WARMUP_STEPS"] = str(args_cli.reset_warmup_steps)
    os.environ["ASSEMBLY_RESET_RT_SUBFRAMES"] = str(args_cli.reset_rt_subframes)

    variant = VARIANTS[args_cli.task]
    episode_length_s = args_cli.episode_length_s or variant.episode_length_s
    max_steps = args_cli.max_steps or round(episode_length_s * 15)
    print(
        f"[expert] building {args_cli.task}: num_envs={args_cli.num_envs} "
        f"camera_scale={args_cli.camera_scale:g} warmup={args_cli.reset_warmup_steps}/"
        f"{args_cli.reset_rt_subframes}",
        flush=True,
    )
    arena_env = AssembleBenchEnvironment().get_env(args_cli)
    if args_cli.episode_length_s is not None:
        arena_env.task.episode_length_s = args_cli.episode_length_s
        print(f"[expert] episode timeout: {args_cli.episode_length_s:g}s", flush=True)
    if args_cli.held_friction is not None:
        from dataclasses import replace

        arena_env.task.variant = replace(
            arena_env.task.variant, held_friction=args_cli.held_friction
        )
        print(f"[expert] held-part friction: {args_cli.held_friction:g}", flush=True)
    if args_cli.camera_scale != 1.0:
        if not 0.1 <= args_cli.camera_scale <= 1.0:
            raise ValueError("--camera_scale must be in [0.1, 1.0]")
        camera_cfg = arena_env.embodiment.camera_config
        for name in getattr(camera_cfg, "__dataclass_fields__", {}):
            camera = getattr(camera_cfg, name)
            if hasattr(camera, "width") and hasattr(camera, "height"):
                camera.width = max(64, round(camera.width * args_cli.camera_scale))
                camera.height = max(64, round(camera.height * args_cli.camera_scale))
                print(f"[expert] camera {name}: {camera.width}x{camera.height}", flush=True)
    if args_cli.robot_usd:
        arena_env.embodiment.scene_config.robot.spawn.usd_path = args_cli.robot_usd
        print(f"[expert] robot USD override: {args_cli.robot_usd}", flush=True)
    if args_cli.grip_effort is not None:
        arena_env.embodiment.scene_config.robot.actuators["gripper"].effort_limit_sim = args_cli.grip_effort
        print(f"[expert] gripper effort cap: {args_cli.grip_effort} N*m", flush=True)
    if args_cli.arm_stiffness is not None or args_cli.arm_damping is not None:
        for name in ("panda_shoulder", "panda_forearm"):
            actuator = arena_env.embodiment.scene_config.robot.actuators[name]
            if args_cli.arm_stiffness is not None:
                actuator.stiffness = args_cli.arm_stiffness
            if args_cli.arm_damping is not None:
                actuator.damping = args_cli.arm_damping
        print(
            f"[expert] arm PD: stiffness={args_cli.arm_stiffness or 'default'} "
            f"damping={args_cli.arm_damping or 'default'}",
            flush=True,
        )
    print("[expert] creating Arena environment", flush=True)
    env = ArenaEnvBuilder(arena_env, args_cli).make_registered(render_mode="rgb_array")
    print("[expert] Arena environment ready; attaching HUD stream", flush=True)
    # Stream for interactive runs; skip it during bulk --record so a dead
    # telemetry endpoint can't stall generation (retries choke the step loop).
    if not args_cli.no_stream and (args_cli.stream or not args_cli.record):
        # demo_contract.json = scripted-expert HUD stream (not EnvHub contract.json).
        env = hud.wrap(SuccessInfo(env), job=args_cli.job_name or f"expert-{args_cli.task}",
                       task=variant.instruction,
                       contract="scripts/experts/demo_contract.json")
    base = env.unwrapped

    recorder = (
        Recorder(
            base, args_cli.record, args_cli.task, variant.instruction,
            max_demos=args_cli.max_demos,
            keep_failures=args_cli.keep_failures,
            target_success_rate=args_cli.target_success_rate,
        )
        if args_cli.record else None
    )

    for wave in range(args_cli.waves):
        print(f"[expert] wave {wave}: reset start", flush=True)
        reset_out = env.reset()
        # s_0 for pre-step recording (sa_align=pre_step). Gymnasium: (obs, info).
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        print(f"[expert] wave {wave}: reset complete", flush=True)
        # hud.wrap opens a fresh trace after every per-slot auto-reset. Restore
        # the four initial slots for each explicit wave; completed slots are
        # removed below so a fast success cannot create a short follow-on trace
        # while slower slots finish their first episode.
        if getattr(env, "_rec", None) is not None:
            if wave == 0:
                print(f"[hud] watch this run: {env._rec.job_url}", flush=True)
            env._rec.record_indices = list(range(min(base.num_envs, 4)))
        servo = Servo(base)
        aim_off = None
        if args_cli.sweep_aim:
            ax, lo, hi = args_cli.sweep_aim.split(",")
            aim_off = torch.zeros((base.num_envs, 3), device=base.device)
            aim_off[:, "xyz".index(ax)] = torch.linspace(
                float(lo), float(hi), base.num_envs, device=base.device)
        lead_phase_off = None
        if args_cli.sweep_lead:
            lo, hi = args_cli.sweep_lead.split(",")
            lead_phase_off = torch.linspace(
                float(lo), float(hi), base.num_envs, device=base.device
            )
        seed = (args_cli.seed or 0) + wave
        if variant.family == "peg_insert":
            if lead_phase_off is not None:
                raise ValueError("--sweep_lead only applies to nut-thread tasks")
            # Rect pegs get closed-loop yaw clocking; round pegs are symmetric.
            machine = peg.make_machine(
                base, servo, aim_off=aim_off, seed=seed,
                clock=variant.rand_fixed_yaw > 0.0,
            )
        elif variant.family == "gear_mesh":
            if aim_off is not None or lead_phase_off is not None:
                raise ValueError("--sweep_aim is currently a peg calibration option")
            machine = gear.make_machine(
                base, servo, size=args_cli.task.removeprefix("gear_"),
                seat_off=variant.seat_off, seed=seed,
            )
        elif variant.family == "nut_thread":
            size = int(args_cli.task.split("_")[1].lower().removeprefix("m"))
            machine = nut.make_machine(
                base, servo, size=size, aim_off=aim_off,
                lead_phase_off=lead_phase_off, seed=seed,
            )
        else:
            raise NotImplementedError(f"no scripted expert for family {variant.family!r}")
        if wave == 0 and getattr(machine, "guide", None):
            print(f"[expert-guide] {machine.guide}", flush=True)
        # The wave ends once every slot has completed its first episode.
        finished_once = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
        succ_ever = torch.zeros_like(finished_once)

        for step in range(max_steps):
            action = machine.action()
            # A completed slot was auto-reset inside env.step. Hold its current
            # reset pose while the other slots finish; do not start episode 2.
            hold = torch.cat([
                wp.to_torch(base.scene["robot"].data.joint_pos)[:, :7],
                action[:, 7:],
            ], dim=-1)
            action = torch.where(finished_once.unsqueeze(-1), hold, action)
            # pre_step convention: bank (obs=s_t, action=a_t, rew=r_t), then advance.
            next_obs, rew, terminated, truncated, _ = env.step(action)
            succ_now = base.termination_manager.get_term("success")
            done_now = terminated | truncated
            if recorder is not None:
                recorder.step(obs, action, done_now, succ_now, reward=rew)
            obs = next_obs
            if done_now.any() and getattr(env, "_rec", None) is not None:
                env._rec.record_indices = [
                    i for i in env._rec.record_indices if not bool(done_now[i])
                ]
            succ_ever |= succ_now
            finished_once |= done_now
            wave_complete = finished_once | machine.failed | machine.finished
            if args_cli.snap_every and step % args_cli.snap_every == 0:
                import os as _os

                from PIL import Image
                _os.makedirs("/tmp/expert_snaps", exist_ok=True)
                for cam in ("front_cam_rgb", "wrist_camera_rgb"):
                    frame = obs["camera_obs"][cam][0].to(torch.uint8).cpu().numpy()[..., :3]
                    # Native resolution: these snaps are the render-quality reference.
                    Image.fromarray(frame).save(
                        f"/tmp/expert_snaps/w{wave}_s{step:03d}_{cam.split('_')[0]}.png")
                view = base.render()   # third-person viewer (look-at-held-part)
                if view is not None:
                    Image.fromarray(view[..., :3]).save(
                        f"/tmp/expert_snaps/w{wave}_s{step:03d}_viewer.jpg")
            if step % 25 == 0 or bool(wave_complete.all()):
                p, h = pos_of(base, "held_part")[0], pos_of(base, "fixed_part")[0]
                t, q = servo.tcp()[0], servo.ee_quat()[0]
                finger = float(wp.to_torch(base.scene["robot"].data.joint_pos)[0, 7])
                failures = machine.failure_report()
                print(f"[expert] w{wave} s{step:3d} phases={machine.report()} "
                      f"succ={int(succ_ever.sum())}/{base.num_envs} "
                      f"finished={int(wave_complete.sum())} "
                      f"fail={int(machine.failed.sum())}{f' {failures}' if failures else ''} | env0 "
                      f"part=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
                      f"fixture=({h[0]:.3f},{h[1]:.3f}) "
                      f"tcp=({t[0]:.3f},{t[1]:.3f},{t[2]:.3f}) "
                      f"quat=({q[0]:.2f},{q[1]:.2f},{q[2]:.2f},{q[3]:.2f}) finger={finger:.2f}",
                      flush=True)
                diagnostics = getattr(machine, "diagnostics", None)
                if diagnostics is not None:
                    values = diagnostics()
                    print(
                        "[nut] "
                        + " ".join(f"{key}={value:.3f}" for key, value in values.items()),
                        flush=True,
                    )
            if bool(wave_complete.all()):
                break
        if recorder is not None:
            recorder.commit_open()  # keep_failures: bank slots that ended without env done
            kept = recorder.flush()
        else:
            kept = 0
        mix = ""
        if recorder is not None:
            mix = (f" recorded {kept} (total {recorder.n_demos}"
                   f" = {recorder.n_success}ok/{recorder.n_fail}fail)")
        print(f"[expert] wave {wave}: seated {int(succ_ever.sum())}/{base.num_envs}{mix}",
              flush=True)
        if machine.failure_report():
            print(f"[expert] timeout failure phases: {machine.failure_report()}", flush=True)
        # Stop early once enough demos are banked (bulk data-gen target).
        if recorder is not None and args_cli.max_demos and recorder.n_demos >= args_cli.max_demos:
            print(f"[expert] reached {recorder.n_demos} demos (>= {args_cli.max_demos}); stopping", flush=True)
            break
        if aim_off is not None or lead_phase_off is not None:
            fingers = wp.to_torch(base.scene["robot"].data.joint_pos)[:, 7]
            part_z = pos_of(base, "held_part")[:, 2]
            sweep_metrics = (
                machine.sweep_metrics() if getattr(machine, "sweep_metrics", None) else {}
            )
            for i in range(base.num_envs):
                off = [round(float(x), 4) for x in aim_off[i]] if aim_off is not None else "-"
                lead = round(float(lead_phase_off[i]), 4) if lead_phase_off is not None else "-"
                metrics = " ".join(
                    f"{key}={float(value[i]):.3f}" for key, value in sweep_metrics.items()
                )
                print(f"[sweep] env{i} aim_off={off} lead_off={lead} "
                      f"finger={float(fingers[i]):.3f} part_z={float(part_z[i]):.3f} "
                      f"phase={machine.phases[int(machine.phase[i])].name} "
                      f"failed={bool(machine.failed[i])} "
                      f"at={machine.phases[int(machine.failure_phase[i])].name if machine.failed[i] else '-'} "
                      f"{metrics}",
                      flush=True)

    if recorder is not None:
        recorder.close()
    env.close()

"""Run a scripted expert on one variant, streaming every rollout to HUD.

Builds the Arena env exactly like the policy runner, wraps it with ``hud.wrap``
(each env slot's episode becomes a watchable trace with wrist+front video),
and drives the family's phase machine until every env terminates or times out.
Prints phase-population and pose diagnostics as it goes. Run (isaac6 env):

    python scripts/experts/run_expert.py --headless --task peg_round_8mm_tight --num_envs 4
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
    from assembly_bench.environments.assembly.assembly import AssemblyBenchEnvironment
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

    AssemblyBenchEnvironment.add_cli_args(parser)
    parser.add_argument("--max_steps", type=int, default=None,
                        help="control steps per wave (default: task episode length at 15 Hz)")
    parser.add_argument("--episode_length_s", type=float, default=None,
                        help="override task timeout for long expert-development runs")
    parser.add_argument("--waves", type=int, default=1, help="global reset cycles to run")
    parser.add_argument("--job_name", type=str, default=None,
                        help="HUD job name override (useful for calibration sweeps)")
    parser.add_argument("--snap_every", type=int, default=0,
                        help="save env0 front+wrist frames every N steps to /tmp/expert_snaps")
    parser.add_argument("--debug_env0", action="store_true",
                        help="per-step env0 servo diagnostics (ori error, joint tracking)")
    parser.add_argument("--debug_aabb", action="store_true",
                        help="print env0 nut and fingertip collision AABBs during grasp/microlift")
    parser.add_argument("--calib_pregrasp", action="store_true",
                        help="servo uncapped to tool-down above the workspace and print joints")
    parser.add_argument("--sweep_aim", type=str, default=None,
                        help="axis,lo,hi (e.g. x,-0.014,0.007): per-env grasp aim offset sweep")
    parser.add_argument("--sweep_lead", type=str, default=None,
                        help="lo,hi: per-env nut thread lead-phase offset sweep in radians")
    parser.add_argument("--robot_usd", type=str, default=None,
                        help="override the robot USD (A/B-test gripper collision assets)")
    parser.add_argument("--factory_m16_assets", action="store_true",
                        help="use NVIDIA's curated Factory M16 nut/bolt USD pair")
    parser.add_argument("--curated_nut", action="store_true",
                        help="use curated Factory M16 nut only (A/B which SDF blocks threading)")
    parser.add_argument("--curated_bolt", action="store_true",
                        help="use curated Factory M16 bolt only (A/B which SDF blocks threading)")
    parser.add_argument("--nut_usd", type=str, default=None,
                        help="override held nut USD path (e.g. factory_nut_m16_loose.usd)")
    parser.add_argument("--bolt_usd", type=str, default=None,
                        help="override fixed bolt USD path (e.g. factory_bolt_m16_loose.usd)")
    parser.add_argument("--grip_effort", type=float, default=None,
                        help="cap the finger drive effort (N*m). The USD default 16.5 is the "
                             "source bench's documented pathological config; 1.5 its validated fix")
    parser.add_argument("--arm_stiffness", type=float, default=None,
                        help="override arm PD stiffness for contact calibration")
    parser.add_argument("--arm_damping", type=float, default=None,
                        help="override arm PD damping for contact calibration")
    parser.add_argument("--calib_toollen", type=str, default=None, choices=["open", "closed"],
                        help="descend the gripper (open|closed) onto the peg; flange z at "
                             "first contact - peg length = flange->tip length in that state")
    parser.add_argument("--language_instruction", type=str, default=None,
                        help="override the task's own description (Arena builder reads this; "
                             "None falls back to task.get_task_description())")
    parser.add_argument("--record", type=str, default=None,
                        help="write success-filtered episodes to this HDF5 (for LeRobot export)")
    parser.add_argument("--max_demos", type=int, default=None,
                        help="stop after this many demos are recorded (bulk data-gen)")
    parser.add_argument("--keep_failures", action="store_true",
                        help="record failed episodes too (default: success-filtered)")
    parser.add_argument("--stream", action="store_true",
                        help="force HUD trace streaming even during a --record run (default: "
                             "streaming is ON for interactive runs, OFF for --record so bulk "
                             "data-gen never stalls on a dead telemetry endpoint)")
    parser.add_argument("--no_stream", action="store_true",
                        help="disable HUD streaming for fast policy-only diagnostics")
    parser.add_argument("--disable_cameras", action="store_true",
                        help="disable cameras for fast policy-only diagnostics (no stream/record)")
    parser.add_argument("--debug_seat", action="store_true",
                        help="per-env success sub-metrics (xy/gap/speed vs tolerances) each "
                             "report tick -- diagnoses seated-looking pegs that miss success")
    parser.add_argument("--debug_fail", action="store_true",
                        help="per-env post-mortem at each wave end (phase/failed/yaw/xy/gap) "
                             "-- diagnoses which stage each failed episode died in")
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
    arena_env = AssemblyBenchEnvironment().get_env(args_cli)
    # Asset overrides. --factory_m16_assets / --curated_* swap nucleus curated;
    # --nut_usd / --bolt_usd point at local paths (e.g. converted factory_* stems).
    def _set_held_usd(path: str):
        arena_env.task.held.usd_path = path
        arena_env.scene.assets["held_part"].usd_path = path
        arena_env.scene.assets["held_part"].object_cfg.spawn.usd_path = path

    def _set_fixed_usd(path: str):
        arena_env.task.fixed.usd_path = path
        arena_env.scene.assets["fixed_part"].usd_path = path
        arena_env.scene.assets["fixed_part"].object_cfg.spawn.usd_path = path

    curate_nut = args_cli.factory_m16_assets or args_cli.curated_nut
    curate_bolt = args_cli.factory_m16_assets or args_cli.curated_bolt
    if curate_nut or curate_bolt:
        if args_cli.task not in {"nut_m16_loose", "nut_m16_tight"}:
            raise ValueError("curated M16 overrides require an M16 nut task")
        from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

        factory_dir = f"{ISAACLAB_NUCLEUS_DIR}/Factory"
        if curate_nut:
            _set_held_usd(f"{factory_dir}/factory_nut_m16.usd")
        if curate_bolt:
            _set_fixed_usd(f"{factory_dir}/factory_bolt_m16.usd")
        print(f"[expert] curated M16 overrides: nut={curate_nut} bolt={curate_bolt}", flush=True)
    if args_cli.nut_usd:
        _set_held_usd(args_cli.nut_usd)
        print(f"[expert] nut USD override: {args_cli.nut_usd}", flush=True)
    if args_cli.bolt_usd:
        _set_fixed_usd(args_cli.bolt_usd)
        print(f"[expert] bolt USD override: {args_cli.bolt_usd}", flush=True)
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
        env = hud.wrap(SuccessInfo(env), job=args_cli.job_name or f"expert-{args_cli.task}",
                       task=variant.instruction,
                       contract="scripts/experts/contract.json")
    base = env.unwrapped

    if args_cli.calib_pregrasp:
        from experts.base import home_quat
        from isaaclab.utils.math import quat_error_magnitude

        env.reset()
        servo = Servo(base)
        down = home_quat(servo)
        target = torch.tensor([0.40, 0.0, 0.25], device=base.device).expand(base.num_envs, 3)
        for step in range(220):
            a = servo.act(target.contiguous(), down, torch.zeros(base.num_envs, device=base.device),
                          rot_cap=10.0)
            env.step(a)
            if step % 40 == 0:
                e, o = servo.ee()[0], float(quat_error_magnitude(servo.ee_quat()[:1], down[:1])[0])
                print(f"[calib] s{step} ee=({e[0]:.4f},{e[1]:.4f},{e[2]:.4f}) ori_err={o:.4f}", flush=True)
        q = wp.to_torch(base.scene["robot"].data.joint_pos)[0, :7]
        e, o = servo.ee()[0], float(quat_error_magnitude(servo.ee_quat()[:1], down[:1])[0])
        print(f"[calib] FINAL ee=({e[0]:.4f},{e[1]:.4f},{e[2]:.4f}) ori_err={o:.4f} "
              f"q={[round(float(x), 4) for x in q]}", flush=True)
        env.close()
        raise SystemExit

    if args_cli.calib_toollen:
        from experts.base import home_quat
        env.reset()
        servo = Servo(base)
        down = home_quat(servo)
        peg0 = pos_of(base, "held_part").clone()
        grip = torch.full((base.num_envs,), 1.0 if args_cli.calib_toollen == "closed" else 0.0,
                          device=base.device)
        target = peg0.clone()
        target[:, 2] = 0.30
        for phase, steps in (("go_above", 200), ("descend", 400)):
            for step in range(steps):
                t = target.clone()
                if phase == "descend":
                    t[:, 2] = servo.ee()[:, 2] - 0.001   # 1 mm/step straight down
                env.step(servo.act(t, down, grip))
                moved = torch.norm(pos_of(base, "held_part") - peg0, dim=-1)
                if phase == "descend" and float(moved[0]) > 0.0008:
                    ee_z = float(servo.ee()[0, 2])
                    print(f"[toollen] {args_cli.calib_toollen} contact at flange z={ee_z:.4f}; "
                          f"peg top=0.050 -> TOOL_LEN = {ee_z - 0.050:.4f}", flush=True)
                    break
            else:
                continue
            break
        env.close()
        raise SystemExit

    recorder = (
        Recorder(
            base, args_cli.record, args_cli.task, variant.instruction,
            max_demos=args_cli.max_demos,
            keep_failures=args_cli.keep_failures,
        )
        if args_cli.record else None
    )

    for wave in range(args_cli.waves):
        print(f"[expert] wave {wave}: reset start", flush=True)
        env.reset()
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
            size = int(args_cli.task.split("_")[1].removeprefix("m"))
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
            obs, rew, terminated, truncated, _ = env.step(action)
            succ_now = base.termination_manager.get_term("success")
            done_now = terminated | truncated
            if recorder is not None:
                recorder.step(obs, action, done_now, succ_now, reward=rew)
            if done_now.any() and getattr(env, "_rec", None) is not None:
                env._rec.record_indices = [
                    i for i in env._rec.record_indices if not bool(done_now[i])
                ]
            if args_cli.debug_env0:
                from isaaclab.utils.math import quat_apply
                axis = quat_apply(servo.ee_quat()[:1], torch.tensor(
                    [[0.0, 0.0, 1.0]], device=base.device))[0]
                d = base.scene.sensors["ee_frame"].data
                li = d.target_frame_names.index("tool_leftfinger")
                ri = d.target_frame_names.index("tool_rightfinger")
                p = wp.to_torch(d.target_pos_w)[0] - base.scene.env_origins[0]
                print(f"[dbg] s{step:3d} ph={machine.phases[int(machine.phase[0])].name:9s} "
                      f"tool_z={axis[2]:.3f} ik_err={servo.dbg_pos_err:.4f} "
                      f"lim_margin={servo.dbg_lim_margin:.3f} "
                      f"Lpad=({p[li][0]:.4f},{p[li][1]:.4f},{p[li][2]:.4f}) "
                      f"Rpad=({p[ri][0]:.4f},{p[ri][1]:.4f},{p[ri][2]:.4f})", flush=True)
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
                phase_name = machine.phases[int(machine.phase[0])].name
                if args_cli.debug_aabb and phase_name in {"grasp", "microlift"}:
                    from pxr import Usd, UsdGeom, UsdPhysics

                    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
                    stage = base.scene.stage
                    nut_prim = stage.GetPrimAtPath("/World/envs/env_0/held_part")
                    nut_range = bbox.ComputeWorldBound(nut_prim).ComputeAlignedRange()
                    nut_min, nut_max = nut_range.GetMin(), nut_range.GetMax()
                    print(
                        f"[aabb] nut x[{nut_min[0]:.4f},{nut_max[0]:.4f}] "
                        f"y[{nut_min[1]:.4f},{nut_max[1]:.4f}] "
                        f"z[{nut_min[2]:.4f},{nut_max[2]:.4f}]",
                        flush=True,
                    )
                    for prim in stage.Traverse():
                        path = str(prim.GetPath())
                        if (
                            "env_0/Robot" in path
                            and "fingertips" in path
                            and prim.HasAPI(UsdPhysics.CollisionAPI)
                        ):
                            pad_range = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
                            pad_min, pad_max = pad_range.GetMin(), pad_range.GetMax()
                            side = "L" if "left" in path else "R"
                            print(
                                f"[aabb] pad_{side} x[{pad_min[0]:.4f},{pad_max[0]:.4f}] "
                                f"y[{pad_min[1]:.4f},{pad_max[1]:.4f}] "
                                f"z[{pad_min[2]:.4f},{pad_max[2]:.4f}]",
                                flush=True,
                            )
            if args_cli.debug_seat and (step % 25 == 0 or bool(wave_complete.all())):
                from isaaclab.utils.math import quat_apply
                held = base.scene["held_part"]
                fixed = base.scene["fixed_part"]
                hp = wp.to_torch(held.data.root_pos_w) - base.scene.env_origins
                fp = wp.to_torch(fixed.data.root_pos_w) - base.scene.env_origins
                fq = wp.to_torch(fixed.data.root_quat_w)
                off = torch.tensor(variant.seat_off, device=base.device).expand(base.num_envs, 3)
                tgt = fp + quat_apply(fq, off)
                xy = torch.norm(hp[:, :2] - tgt[:, :2], dim=-1)
                gap = (hp[:, 2] + variant.held_base_z_off) - tgt[:, 2]
                spd = torch.norm(wp.to_torch(held.data.root_lin_vel_w), dim=-1)
                def _yaw(name):
                    x = quat_apply(quat_of(base, name),
                                   torch.tensor([1.0, 0.0, 0.0], device=base.device).expand(base.num_envs, 3))
                    return torch.atan2(x[:, 1], x[:, 0])
                yerr = _yaw("fixed_part") - _yaw("held_part")
                yerr = yerr - torch.pi * torch.round(yerr / torch.pi)   # 2-fold (rect)
                for i in range(base.num_envs):
                    a = xy[i] < variant.align_tol
                    s = gap[i] < variant.seat_tol
                    v = spd[i] < 0.05
                    ph = machine.phases[int(machine.phase[i])].name
                    print(f"[seat] w{wave} s{step:3d} env{i} {ph:9s} xy={xy[i]*1e3:5.1f}mm "
                          f"gap={gap[i]*1e3:6.1f}mm yaw={float(yerr[i])*57.3:5.1f}deg spd={spd[i]:.3f} | "
                          f"align{'Y' if a else 'n'} seat{'Y' if s else 'n'} "
                          f"stbl{'Y' if v else 'n'} => {'SEATED' if (a and s and v) else '-'}",
                          flush=True)
            if bool(wave_complete.all()):
                break
        if recorder is not None:
            recorder.commit_open()  # keep_failures: bank slots that ended without env done
            kept = recorder.flush()
        else:
            kept = 0
        print(f"[expert] wave {wave}: seated {int(succ_ever.sum())}/{base.num_envs}"
              f"{f' recorded {kept} (total {recorder.n_demos})' if recorder else ''}", flush=True)
        if args_cli.debug_fail:
            # Per-env post-mortem: final phase, failed flag, and the seat/yaw
            # residuals -- shows WHY each env failed (bad clock vs bad seat).
            from isaaclab.utils.math import quat_apply
            def _yaw(name):
                x = quat_apply(quat_of(base, name),
                               torch.tensor([1.0, 0.0, 0.0], device=base.device).expand(base.num_envs, 3))
                return torch.atan2(x[:, 1], x[:, 0])
            ye = _yaw("fixed_part") - _yaw("held_part")
            ye = ye - torch.pi * torch.round(ye / torch.pi)   # 2-fold (rect)
            hp = pos_of(base, "held_part"); fp = pos_of(base, "fixed_part")
            xy = torch.norm(hp[:, :2] - fp[:, :2], dim=-1)
            gap = hp[:, 2] - fp[:, 2]
            for i in range(base.num_envs):
                print(f"[fail] w{wave} env{i} phase={machine.phases[int(machine.phase[i])].name:9s} "
                      f"failed={bool(machine.failed[i])} succ={bool(succ_ever[i])} "
                      f"yaw_err={float(ye[i])*57.3:5.1f}deg xy={float(xy[i])*1e3:5.1f}mm "
                      f"gap={float(gap[i])*1e3:6.1f}mm", flush=True)
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

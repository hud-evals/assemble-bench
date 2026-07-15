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
args_cli.enable_cameras = True

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
    parser.add_argument("--waves", type=int, default=1, help="global reset cycles to run")
    parser.add_argument("--snap_every", type=int, default=0,
                        help="save env0 front+wrist frames every N steps to /tmp/expert_snaps")
    parser.add_argument("--debug_env0", action="store_true",
                        help="per-step env0 servo diagnostics (ori error, joint tracking)")
    parser.add_argument("--calib_pregrasp", action="store_true",
                        help="servo uncapped to tool-down above the workspace and print joints")
    parser.add_argument("--sweep_aim", type=str, default=None,
                        help="axis,lo,hi (e.g. x,-0.014,0.007): per-env grasp aim offset sweep")
    parser.add_argument("--robot_usd", type=str, default=None,
                        help="override the robot USD (A/B-test gripper collision assets)")
    parser.add_argument("--grip_effort", type=float, default=None,
                        help="cap the finger drive effort (N*m). The USD default 16.5 is the "
                             "source bench's documented pathological config; 1.5 its validated fix")
    parser.add_argument("--calib_toollen", type=str, default=None, choices=["open", "closed"],
                        help="descend the gripper (open|closed) onto the peg; flange z at "
                             "first contact - peg length = flange->tip length in that state")
    parser.add_argument("--language_instruction", type=str, default=None,
                        help="override the task's own description (Arena builder reads this; "
                             "None falls back to task.get_task_description())")
    parser.add_argument("--record", type=str, default=None,
                        help="write success-filtered episodes to this HDF5 (for LeRobot export)")
    parser.add_argument("--max_demos", type=int, default=None,
                        help="stop after this many successful demos are recorded (bulk data-gen)")
    parser.add_argument("--stream", action="store_true",
                        help="force HUD trace streaming even during a --record run (default: "
                             "streaming is ON for interactive runs, OFF for --record so bulk "
                             "data-gen never stalls on a dead telemetry endpoint)")
    parser.add_argument("--debug_seat", action="store_true",
                        help="per-env success sub-metrics (xy/gap/speed vs tolerances) each "
                             "report tick -- diagnoses seated-looking pegs that miss success")
    parser.add_argument("--debug_fail", action="store_true",
                        help="per-env post-mortem at each wave end (phase/failed/yaw/xy/gap) "
                             "-- diagnoses which stage each failed episode died in")
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True

    variant = VARIANTS[args_cli.task]
    max_steps = args_cli.max_steps or round(variant.episode_length_s * 15)
    arena_env = AssemblyBenchEnvironment().get_env(args_cli)
    if args_cli.robot_usd:
        arena_env.embodiment.scene_config.robot.spawn.usd_path = args_cli.robot_usd
        print(f"[expert] robot USD override: {args_cli.robot_usd}", flush=True)
    if args_cli.grip_effort is not None:
        arena_env.embodiment.scene_config.robot.actuators["gripper"].effort_limit_sim = args_cli.grip_effort
        print(f"[expert] gripper effort cap: {args_cli.grip_effort} N*m", flush=True)
    env = ArenaEnvBuilder(arena_env, args_cli).make_registered(render_mode="rgb_array")
    # Stream for interactive runs; skip it during bulk --record so a dead
    # telemetry endpoint can't stall generation (retries choke the step loop).
    if args_cli.stream or not args_cli.record:
        env = hud.wrap(SuccessInfo(env), job=f"expert-{args_cli.task}", task=variant.instruction,
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
        )
        if args_cli.record else None
    )

    for wave in range(args_cli.waves):
        env.reset()
        # hud.wrap opens a fresh trace after every per-slot auto-reset. Restore
        # the four initial slots for each explicit wave; completed slots are
        # removed below so a fast success cannot create a short follow-on trace
        # while slower slots finish their first episode.
        if getattr(env, "_rec", None) is not None:
            env._rec.record_indices = list(range(min(base.num_envs, 4)))
        servo = Servo(base)
        aim_off = None
        if args_cli.sweep_aim:
            ax, lo, hi = args_cli.sweep_aim.split(",")
            aim_off = torch.zeros((base.num_envs, 3), device=base.device)
            aim_off[:, "xyz".index(ax)] = torch.linspace(
                float(lo), float(hi), base.num_envs, device=base.device)
        seed = (args_cli.seed or 0) + wave
        if variant.family == "peg_insert":
            # Rect pegs get closed-loop yaw clocking; round pegs are symmetric.
            machine = peg.make_machine(
                base, servo, aim_off=aim_off, seed=seed,
                clock=variant.rand_fixed_yaw > 0.0,
            )
        elif variant.family == "gear_mesh":
            if aim_off is not None:
                raise ValueError("--sweep_aim is currently a peg calibration option")
            machine = gear.make_machine(
                base, servo, size=args_cli.task.removeprefix("gear_"),
                seat_off=variant.seat_off, seed=seed,
            )
        elif variant.family == "nut_thread":
            if aim_off is not None:
                raise ValueError("--sweep_aim is currently a peg calibration option")
            size = int(args_cli.task.split("_")[1].removeprefix("m"))
            machine = nut.make_machine(base, servo, size=size, seed=seed)
        else:
            raise NotImplementedError(f"no scripted expert for family {variant.family!r}")
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
            obs, _, terminated, truncated, _ = env.step(action)
            succ_now = base.termination_manager.get_term("success")
            done_now = terminated | truncated
            if recorder is not None:
                recorder.step(obs, action, done_now, succ_now)
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
            if step % 25 == 0 or bool(finished_once.all()):
                p, h = pos_of(base, "held_part")[0], pos_of(base, "fixed_part")[0]
                t, q = servo.tcp()[0], servo.ee_quat()[0]
                finger = float(wp.to_torch(base.scene["robot"].data.joint_pos)[0, 7])
                print(f"[expert] w{wave} s{step:3d} phases={machine.report()} "
                      f"succ={int(succ_ever.sum())}/{base.num_envs} "
                      f"finished={int(finished_once.sum())} "
                      f"fail={int(machine.failed.sum())} | env0 "
                      f"part=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
                      f"fixture=({h[0]:.3f},{h[1]:.3f}) "
                      f"tcp=({t[0]:.3f},{t[1]:.3f},{t[2]:.3f}) "
                      f"quat=({q[0]:.2f},{q[1]:.2f},{q[2]:.2f},{q[3]:.2f}) finger={finger:.2f}",
                      flush=True)
            if args_cli.debug_seat and (step % 25 == 0 or bool(finished_once.all())):
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
            if bool(finished_once.all()):
                break
        kept = recorder.flush() if recorder is not None else 0
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
        # Stop early once enough demos are banked (bulk data-gen target).
        if recorder is not None and args_cli.max_demos and recorder.n_demos >= args_cli.max_demos:
            print(f"[expert] reached {recorder.n_demos} demos (>= {args_cli.max_demos}); stopping", flush=True)
            break
        if aim_off is not None:
            fingers = wp.to_torch(base.scene["robot"].data.joint_pos)[:, 7]
            peg_z = pos_of(base, "held_part")[:, 2]
            for i in range(base.num_envs):
                off = [round(float(x), 4) for x in aim_off[i]]
                print(f"[sweep] env{i} aim_off={off} "
                      f"finger={float(fingers[i]):.3f} peg_z={float(peg_z[i]):.3f} "
                      f"phase={machine.phases[int(machine.phase[i])].name} "
                      f"failed={bool(machine.failed[i])}", flush=True)

    if recorder is not None:
        recorder.close()
    env.close()

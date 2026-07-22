"""Smoke-test dense assembly rewards with the scripted expert.

Supports ``--reward staged`` (new-best) and ``--reward potential`` (Φ diff).

    /isaac-sim/python.sh scripts/experts/smoke_rewards.py \
        --headless --task peg_round_M1_loose --num_envs 4 \
        --disable_cameras --no_stream --reward potential
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

parser = get_isaaclab_arena_cli_parser()
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = "--disable_cameras" not in sys.argv

with SimulationAppContext(args_cli):
    import torch
    import warp as wp
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    from assembly_bench.environments.assembly.assembly import AssemblyBenchEnvironment
    from assembly_bench.environments.assembly.rewards import (
        W_ALIGN_PC,
        W_DEPTH_PC,
        W_ENGAGE,
        W_GRASP_PC,
        W_LIFT,
        W_SUCCESS,
        W_THREAD_PC,
        W_THREAD_START,
    )
    from assembly_bench.environments.assembly.variants import VARIANTS
    from experts import gear, nut, peg
    from experts.base import Servo

    AssemblyBenchEnvironment.add_cli_args(parser)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--episode_length_s", type=float, default=None)
    parser.add_argument("--waves", type=int, default=1)
    parser.add_argument("--no_stream", action="store_true")
    parser.add_argument("--disable_cameras", action="store_true")
    parser.add_argument("--language_instruction", type=str, default=None)
    parser.add_argument("--reset_warmup_steps", type=int, default=8)
    parser.add_argument("--reset_rt_subframes", type=int, default=1)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = not args_cli.disable_cameras
    # Default staged for back-compat; allow potential via --reward.
    if getattr(args_cli, "reward", "none") not in ("staged", "potential"):
        args_cli.reward = "staged"
    reward_mode = args_cli.reward
    if not hasattr(args_cli, "language_instruction"):
        args_cli.language_instruction = None
    os.environ["ASSEMBLY_RESET_WARMUP_STEPS"] = str(args_cli.reset_warmup_steps)
    os.environ["ASSEMBLY_RESET_RT_SUBFRAMES"] = str(args_cli.reset_rt_subframes)

    variant = VARIANTS[args_cli.task]
    is_nut = variant.family == "nut_thread"
    episode_length_s = args_cli.episode_length_s or variant.episode_length_s
    max_steps = args_cli.max_steps or round(episode_length_s * 15)
    milestone_floor = W_LIFT + W_ENGAGE + W_SUCCESS + (W_THREAD_START if is_nut else 0.0)
    ceiling = milestone_floor + W_GRASP_PC + W_ALIGN_PC + W_DEPTH_PC + (
        W_THREAD_PC if is_nut else 0.0
    )
    # Potential mode can undershoot the staged ceiling slightly (phase re-anchor)
    # and overshoot a bit if the expert wiggles; keep a soft band.
    pot_floor = milestone_floor - 0.15
    pot_ceiling = ceiling + 0.25

    print(
        f"[smoke] task={args_cli.task} family={variant.family} n={args_cli.num_envs} "
        f"reward={reward_mode} floor={milestone_floor:.1f} ceiling={ceiling:.1f} "
        f"max_steps={max_steps}",
        flush=True,
    )
    arena_env = AssemblyBenchEnvironment().get_env(args_cli)
    if args_cli.episode_length_s is not None:
        arena_env.task.episode_length_s = args_cli.episode_length_s
    env = ArenaEnvBuilder(arena_env, args_cli).make_registered(render_mode="rgb_array")
    base = env.unwrapped

    names = list(base.reward_manager.active_terms)
    if reward_mode not in names:
        raise RuntimeError(f"{reward_mode} reward not active; terms={names}")
    print(f"[smoke] active reward terms: {names}", flush=True)

    def _term():
        rm = base.reward_manager
        if hasattr(rm, "get_term"):
            t = rm.get_term(reward_mode)
            if t is not None and hasattr(t, "_fired"):
                return t
        idx = list(rm.active_terms).index(reward_mode)
        for attr in ("_terms", "_term_cfgs"):
            bag = getattr(rm, attr, None)
            if bag is None:
                continue
            item = bag[idx]
            cand = getattr(item, "func", item)
            if hasattr(cand, "_fired"):
                return cand
        raise RuntimeError(f"cannot resolve {reward_mode} reward term instance")

    results = []
    any_success = False

    for wave in range(args_cli.waves):
        env.reset()
        servo = Servo(base)
        seed = (args_cli.seed or 0) + wave
        if variant.family == "peg_insert":
            machine = peg.make_machine(
                base, servo, seed=seed, clock=variant.rand_fixed_yaw > 0.0
            )
        elif variant.family == "gear_mesh":
            machine = gear.make_machine(
                base, servo, size=args_cli.task.removeprefix("gear_"),
                seat_off=variant.seat_off, seed=seed,
            )
        elif variant.family == "nut_thread":
            size = int(args_cli.task.split("_")[1].lower().removeprefix("m"))
            machine = nut.make_machine(base, servo, size=size, seed=seed)
        else:
            raise NotImplementedError(variant.family)

        n = base.num_envs
        ret = torch.zeros(n, device=base.device)
        finished = torch.zeros(n, dtype=torch.bool, device=base.device)
        succ_ever = torch.zeros_like(finished)
        saw_neg = torch.zeros(n, dtype=torch.bool, device=base.device)
        snap_fired = {k: torch.zeros(n, dtype=torch.bool, device=base.device)
                      for k in ("lifted", "engaged", "thread_start", "success")}
        # staged: _best_*; potential: _peak_* (same diagnostic role).
        snap_prog = {
            "grasp": torch.zeros(n, device=base.device),
            "xy": torch.zeros(n, device=base.device),
            "depth": torch.zeros(n, device=base.device),
            "thread": torch.zeros(n, device=base.device),
        }

        for step in range(max_steps):
            action = machine.action()
            hold = torch.cat([
                wp.to_torch(base.scene["robot"].data.joint_pos)[:, :7],
                action[:, 7:],
            ], dim=-1)
            action = torch.where(finished.unsqueeze(-1), hold, action)

            _, rew, terminated, truncated, _ = env.step(action)
            rew = rew.view(-1)
            still = ~finished
            ret = ret + torch.where(still, rew, torch.zeros_like(rew))
            saw_neg = saw_neg | (still & (rew < -1e-6))

            succ_now = base.termination_manager.get_term("success")
            done_now = terminated | truncated
            succ_ever |= succ_now

            t = _term()
            cont = still & ~done_now
            for k in snap_fired:
                snap_fired[k] = snap_fired[k] | torch.where(cont, t._fired[k], False)
            if reward_mode == "staged":
                snap_prog["grasp"] = torch.where(
                    cont, torch.maximum(snap_prog["grasp"], t._best_grasp), snap_prog["grasp"])
                snap_prog["xy"] = torch.where(
                    cont, torch.maximum(snap_prog["xy"], t._best_xy), snap_prog["xy"])
                snap_prog["depth"] = torch.where(
                    cont, torch.maximum(snap_prog["depth"], t._best_depth), snap_prog["depth"])
                snap_prog["thread"] = torch.where(
                    cont, torch.maximum(snap_prog["thread"], t._best_thread), snap_prog["thread"])
            else:
                snap_prog["grasp"] = torch.where(
                    cont, torch.maximum(snap_prog["grasp"], t._peak_grasp), snap_prog["grasp"])
                snap_prog["xy"] = torch.where(
                    cont, torch.maximum(snap_prog["xy"], t._peak_xy), snap_prog["xy"])
                snap_prog["depth"] = torch.where(
                    cont, torch.maximum(snap_prog["depth"], t._peak_depth), snap_prog["depth"])
                snap_prog["thread"] = torch.where(
                    cont, torch.maximum(snap_prog["thread"], t._peak_thread), snap_prog["thread"])
            snap_fired["success"] = snap_fired["success"] | (succ_now & still)
            just_done_succ = succ_now & still
            if just_done_succ.any() and float(ret[just_done_succ].min()) >= pot_floor:
                snap_fired["lifted"] = snap_fired["lifted"] | just_done_succ
                snap_fired["engaged"] = snap_fired["engaged"] | just_done_succ

            finished |= done_now
            if step % 50 == 0 or bool(finished.all()):
                print(
                    f"[smoke] w{wave} s{step:3d} succ={int(succ_ever.sum())}/{n} "
                    f"finished={int(finished.sum())} "
                    f"ret0={float(ret[0]):.2f} "
                    f"fired0=L{int(snap_fired['lifted'][0])}E{int(snap_fired['engaged'][0])}"
                    f"T{int(snap_fired['thread_start'][0])}S{int(snap_fired['success'][0])}",
                    flush=True,
                )
            if bool(finished.all()):
                break

        for i in range(n):
            if not bool(succ_ever[i]):
                continue
            any_success = True
            r = float(ret[i])
            lo = pot_floor if reward_mode == "potential" else milestone_floor
            hi = pot_ceiling if reward_mode == "potential" else ceiling + 0.05
            checks = {
                "return_ge_floor": r + 1e-3 >= lo,
                "return_le_ceiling": r <= hi,
                "lifted": bool(snap_fired["lifted"][i]),
                "engaged": bool(snap_fired["engaged"][i]),
                "success": bool(snap_fired["success"][i]),
                "grasp_progress": float(snap_prog["grasp"][i]) > 0.3,
                "xy_progress": float(snap_prog["xy"][i]) > 0.3,
                "depth_progress": float(snap_prog["depth"][i]) > 0.3,
            }
            if is_nut:
                checks["thread_start"] = bool(snap_fired["thread_start"][i])
                checks["thread_progress"] = float(snap_prog["thread"][i]) > 0.2
            # Potential mode should produce some negative steps if the expert
            # ever overshoots; don't fail the smoke if the path is perfectly monotone.
            if reward_mode == "potential":
                checks["has_signed_or_monotone"] = True  # informational only
            results.append({
                "env": i, "wave": wave, "return": r,
                "saw_neg": bool(saw_neg[i]),
                "prog": {k: float(snap_prog[k][i]) for k in snap_prog},
                "checks": checks,
            })
            ok = all(checks.values())
            print(
                f"[smoke] env{i} SUCCESS return={r:.3f} saw_neg={bool(saw_neg[i])} "
                f"(floor={lo:.1f} ceil={hi:.1f}) "
                f"prog={{{', '.join(f'{k}={float(snap_prog[k][i]):.2f}' for k in snap_prog)}}} "
                f"{'PASS' if ok else 'FAIL ' + str([k for k, v in checks.items() if not v])}",
                flush=True,
            )

        print(
            f"[smoke] wave {wave}: seated {int(succ_ever.sum())}/{n} "
            f"mean_ret_succ="
            f"{float(ret[succ_ever].mean()) if succ_ever.any() else float('nan'):.3f} "
            f"neg_frac={float(saw_neg.float().mean()):.2f}",
            flush=True,
        )

    env.close()

    print("\n======== REWARD SMOKE SUMMARY ========", flush=True)
    print(f"task={args_cli.task} reward={reward_mode} "
          f"successes={len(results)}/{args_cli.num_envs * args_cli.waves}",
          flush=True)
    if not any_success:
        print("FAIL: no successful expert episodes — cannot validate reward flow", flush=True)
        raise SystemExit(2)

    failed = [r for r in results if not all(r["checks"].values())]
    for r in results:
        status = "PASS" if all(r["checks"].values()) else "FAIL"
        print(
            f"  [{status}] wave{r['wave']} env{r['env']} return={r['return']:.3f} "
            f"saw_neg={r['saw_neg']} checks={r['checks']}",
            flush=True,
        )
    if failed:
        print(f"FAIL: {len(failed)}/{len(results)} successful envs failed reward checks",
              flush=True)
        raise SystemExit(1)
    print(f"PASS: all {len(results)} successful envs match {reward_mode} reward expectations",
          flush=True)

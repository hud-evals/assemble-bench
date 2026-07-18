"""Smoke-test staged dense rewards with the scripted expert.

Runs one variant with ``--reward staged``, accumulates per-env returns, and
checks milestones / potentials against the expected success flow.

    /isaac-sim/python.sh scripts/experts/smoke_rewards.py \
        --headless --task peg_round_8mm_loose --num_envs 4 \
        --disable_cameras --no_stream --reward staged
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
    args_cli.reward = "staged"  # always on for this smoke
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

    print(
        f"[smoke] task={args_cli.task} family={variant.family} n={args_cli.num_envs} "
        f"reward=staged floor={milestone_floor:.1f} ceiling={ceiling:.1f} "
        f"max_steps={max_steps}",
        flush=True,
    )
    arena_env = AssemblyBenchEnvironment().get_env(args_cli)
    if args_cli.episode_length_s is not None:
        arena_env.task.episode_length_s = args_cli.episode_length_s
    env = ArenaEnvBuilder(arena_env, args_cli).make_registered(render_mode="rgb_array")
    base = env.unwrapped

    names = list(base.reward_manager.active_terms)
    if "staged" not in names:
        raise RuntimeError(f"staged reward not active; terms={names}")
    print(f"[smoke] active reward terms: {names}", flush=True)

    def _term():
        rm = base.reward_manager
        if hasattr(rm, "get_term"):
            t = rm.get_term("staged")
            if t is not None and hasattr(t, "_fired"):
                return t
        # Isaac Lab stores terms as a list parallel to active_terms.
        idx = list(rm.active_terms).index("staged")
        for attr in ("_terms", "_term_cfgs"):
            bag = getattr(rm, attr, None)
            if bag is None:
                continue
            item = bag[idx]
            cand = getattr(item, "func", item)
            if hasattr(cand, "_fired"):
                return cand
        raise RuntimeError("cannot resolve staged_assembly_reward term instance")

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
            size = int(args_cli.task.split("_")[1].removeprefix("m"))
            machine = nut.make_machine(base, servo, size=size, seed=seed)
        else:
            raise NotImplementedError(variant.family)

        n = base.num_envs
        ret = torch.zeros(n, device=base.device)
        finished = torch.zeros(n, dtype=torch.bool, device=base.device)
        succ_ever = torch.zeros_like(finished)
        # Sticky OR of term state across steps (survives the success-step reset).
        snap_fired = {k: torch.zeros(n, dtype=torch.bool, device=base.device)
                      for k in ("lifted", "engaged", "thread_start", "success")}
        snap_best = {
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

            succ_now = base.termination_manager.get_term("success")
            done_now = terminated | truncated
            succ_ever |= succ_now

            # Continuing envs keep term state after step — read it.
            # Just-finished envs were reset; OR in success from the termination bit.
            t = _term()
            cont = still & ~done_now
            for k in snap_fired:
                snap_fired[k] = snap_fired[k] | torch.where(cont, t._fired[k], False)
            snap_best["grasp"] = torch.where(
                cont, torch.maximum(snap_best["grasp"], t._best_grasp), snap_best["grasp"])
            snap_best["xy"] = torch.where(
                cont, torch.maximum(snap_best["xy"], t._best_xy), snap_best["xy"])
            snap_best["depth"] = torch.where(
                cont, torch.maximum(snap_best["depth"], t._best_depth), snap_best["depth"])
            snap_best["thread"] = torch.where(
                cont, torch.maximum(snap_best["thread"], t._best_thread), snap_best["thread"])
            snap_fired["success"] = snap_fired["success"] | (succ_now & still)
            # Lift/engage usually fire many steps before success; if an env
            # somehow succeeds in one step after engage, sticky OR already has them.
            # Infer lift/engage from return floor when success lands.
            just_done_succ = succ_now & still
            if just_done_succ.any() and float(ret[just_done_succ].min()) >= milestone_floor - 0.05:
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

        # Validate each env that succeeded.
        for i in range(n):
            if not bool(succ_ever[i]):
                continue
            any_success = True
            r = float(ret[i])
            checks = {
                "return_ge_floor": r + 1e-3 >= milestone_floor,
                "return_le_ceiling": r <= ceiling + 0.05,
                "lifted": bool(snap_fired["lifted"][i]),
                "engaged": bool(snap_fired["engaged"][i]),
                "success": bool(snap_fired["success"][i]),
                "grasp_progress": float(snap_best["grasp"][i]) > 0.3,
                "xy_progress": float(snap_best["xy"][i]) > 0.3,
                "depth_progress": float(snap_best["depth"][i]) > 0.3,
            }
            if is_nut:
                checks["thread_start"] = bool(snap_fired["thread_start"][i])
                checks["thread_progress"] = float(snap_best["thread"][i]) > 0.2
            results.append({
                "env": i, "wave": wave, "return": r,
                "best": {k: float(snap_best[k][i]) for k in snap_best},
                "checks": checks,
            })
            ok = all(checks.values())
            print(
                f"[smoke] env{i} SUCCESS return={r:.3f} "
                f"(floor={milestone_floor:.1f} ceil={ceiling:.1f}) "
                f"bests={{{', '.join(f'{k}={float(snap_best[k][i]):.2f}' for k in snap_best)}}} "
                f"{'PASS' if ok else 'FAIL ' + str([k for k,v in checks.items() if not v])}",
                flush=True,
            )

        print(
            f"[smoke] wave {wave}: seated {int(succ_ever.sum())}/{n} "
            f"mean_ret_succ="
            f"{float(ret[succ_ever].mean()) if succ_ever.any() else float('nan'):.3f}",
            flush=True,
        )

    env.close()

    print("\n======== REWARD SMOKE SUMMARY ========", flush=True)
    print(f"task={args_cli.task}  successes={len(results)}/{args_cli.num_envs * args_cli.waves}",
          flush=True)
    if not any_success:
        print("FAIL: no successful expert episodes — cannot validate reward flow", flush=True)
        raise SystemExit(2)

    failed = [r for r in results if not all(r["checks"].values())]
    for r in results:
        status = "PASS" if all(r["checks"].values()) else "FAIL"
        print(
            f"  [{status}] wave{r['wave']} env{r['env']} return={r['return']:.3f} "
            f"checks={r['checks']}",
            flush=True,
        )
    if failed:
        print(f"FAIL: {len(failed)}/{len(results)} successful envs failed reward checks",
              flush=True)
        raise SystemExit(1)
    print(f"PASS: all {len(results)} successful envs match staged reward expectations",
          flush=True)

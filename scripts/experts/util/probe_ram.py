"""Deterministic "ram the table" stability probe.

Drives the differential-IK arm straight down into the table for N steps (the
teleop explosion repro) while toggling the gripper, and asserts the sim stays
bounded: no NaN, joint velocities and body displacements within sane limits.
Prints the worst-case metrics + PASS/FAIL so a physics/solver change can be
validated without manual teleop.

    python scripts/experts/util/probe_ram.py --task peg_round_8mm_tight \
        --embodiment droid_differential_ik --headless

Explosion = NaN, |joint_vel| blowing past the actuator cap, or a body flung far
from where it started.
"""

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

parser = get_isaaclab_arena_cli_parser()
args_cli, _ = parser.parse_known_args()
# Physics-only probe: no cameras -> faster boot, no render dependence.
args_cli.enable_cameras = False

with SimulationAppContext(args_cli):
    import torch

    import warp as wp
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from assembly_bench.environments.assembly.assembly import AssemblyBenchEnvironment

    AssemblyBenchEnvironment.add_cli_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = False

    builder_cfg = arena_env_builder_cfg_from_argparse(args_cli)
    env = ArenaEnvBuilder(AssemblyBenchEnvironment().get_env(args_cli), builder_cfg).make_registered(
        render_mode="rgb_array")
    base = env.unwrapped
    robot = base.scene["robot"]
    env.reset()

    dim = base.action_manager.total_action_dim
    n = base.num_envs
    device = base.device
    # Diff-IK arm action = [dx, dy, dz, drx, dry, drz] (relative pose); drive
    # hard down (and a little forward) so the arm rams the table, close gripper.
    act = torch.zeros((n, dim), device=device)
    act[:, 2] = -1.0
    act[:, 0] = 0.3
    act[:, -1] = 1.0

    STEPS = 150
    # QD_LIMIT separates a bounded contact transient (~O(10) rad/s -- the fingers
    # deflecting under a hard ram) from a solver divergence (the pre-fix blow-up
    # was 1e6-1e33 rad/s). 100 sits far above any physical transient and far below
    # divergence. DISP_LIMIT catches a body being flung off the robot.
    QD_LIMIT, DISP_LIMIT = 100.0, 1.0  # rad/s, m
    jnames = list(robot.data.joint_names)
    init_body = wp.to_torch(robot.data.body_pos_w).clone()
    per_joint_max = torch.zeros(len(jnames), device=device)
    max_qd, max_disp, nan_hit = 0.0, 0.0, False
    first_diverge = None  # (step, joint) -- who blows up first

    for i in range(STEPS):
        env.step(act)
        qd = wp.to_torch(robot.data.joint_vel)
        bp = wp.to_torch(robot.data.body_pos_w)
        if torch.isnan(qd).any() or torch.isnan(bp).any():
            nan_hit = True
            print(f"[ram] step {i}: NaN detected")
            break
        qd_per_joint = qd.abs().amax(dim=0)  # max over envs, per joint
        per_joint_max = torch.maximum(per_joint_max, qd_per_joint)
        if first_diverge is None and float(qd.abs().max()) > QD_LIMIT:
            first_diverge = (i, jnames[int(qd_per_joint.argmax())])
        max_qd = max(max_qd, float(qd.abs().max()))
        max_disp = max(max_disp, float((bp - init_body).norm(dim=-1).max()))
        if i % 25 == 0:
            print(f"[ram] step {i:3d}: max|qd|={float(qd.abs().max()):6.2f} rad/s  body_disp={max_disp:.3f} m")

    exploded = nan_hit or max_qd > QD_LIMIT or max_disp > DISP_LIMIT
    print("=== RAM PROBE RESULT ===")
    print(f"  steps={STEPS} nan={nan_hit}")
    print(f"  max|joint_vel| = {max_qd:.2f} rad/s   (limit {QD_LIMIT})")
    print(f"  max_body_disp  = {max_disp:.3f} m     (limit {DISP_LIMIT})")
    if first_diverge is not None:
        print(f"  first joint past limit: step {first_diverge[0]}  '{first_diverge[1]}'")
    print("  per-joint max |vel| (rad/s):")
    order = torch.argsort(per_joint_max, descending=True)
    for j in order.tolist():
        print(f"    {jnames[j]:38s} {float(per_joint_max[j]):.3g}")
    print(f"  VERDICT: {'FAIL (exploded)' if exploded else 'PASS (bounded)'}")
    env.close()

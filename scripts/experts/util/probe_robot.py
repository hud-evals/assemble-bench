"""Probe the live env for the facts the scripted expert depends on.

Prints robot body/joint names, EE (Robotiq base_link) pose + quat convention,
jacobian shape, finger-pad geometry (for the TCP offset), part poses, obs keys,
and the action spec — so the expert is written against measured reality, not
assumptions. Run:

    python scripts/experts/util/probe_robot.py --task peg_round_M1_loose
"""

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

parser = get_isaaclab_arena_cli_parser()
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

with SimulationAppContext(args_cli):
    import torch

    import warp as wp
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from assembly_bench.environments.assembly.assembly import AssemblyBenchEnvironment

    AssemblyBenchEnvironment.add_cli_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True

    builder_cfg = arena_env_builder_cfg_from_argparse(args_cli)
    env = ArenaEnvBuilder(AssemblyBenchEnvironment().get_env(args_cli), builder_cfg).make_registered(
        render_mode="rgb_array")
    obs, _ = env.reset()
    base = env.unwrapped
    robot = base.scene["robot"]

    print("=== obs structure ===")
    for group, terms in obs.items():
        for name, val in terms.items():
            print(f"  {group}/{name}: shape={tuple(val.shape)} dtype={val.dtype}")

    print("=== action spec ===")
    am = base.action_manager
    print(f"  total_action_dim={am.total_action_dim}")
    for name, term in am._terms.items():
        print(f"  term {name}: dim={term.action_dim}")

    print("=== robot bodies ===")
    print(" ", robot.data.body_names)
    print("=== robot joints ===")
    print(" ", robot.data.joint_names)

    print("=== actuator gains (resolved: cfg override or USD-authored fallback) ===")
    for group_name, actuator in robot.actuators.items():
        model_type = "implicit" if actuator.is_implicit_model else "explicit"
        print(f"  [{group_name}] model={model_type} joints={actuator.joint_names}")
        for jname in actuator.joint_names:
            j = actuator.joint_names.index(jname)
            stiffness = float(actuator.stiffness[0, j])
            damping = float(actuator.damping[0, j])
            effort_limit = float(actuator.effort_limit[0, j])
            velocity_limit = float(actuator.velocity_limit[0, j])
            print(f"    {jname}: stiffness={stiffness:.3f} damping={damping:.3f} "
                  f"effort_limit={effort_limit:.3f} velocity_limit={velocity_limit:.3f}")
        # per-parameter resolution table: [joint_name, joint_id, usd_val, cfg_val, applied_val]
        # only populated for params where cfg was None or diverged from the USD value.
        for param, rows in actuator.joint_property_resolution_table.items():
            for jname, jid, usd_val, cfg_val, applied_val in rows:
                print(f"    resolution[{param}] {jname}[{jid}]: usd={usd_val:.3f} "
                      f"cfg={cfg_val} applied={applied_val:.3f}")

    def body_pose(bname):
        i = robot.data.body_names.index(bname)
        p = wp.to_torch(robot.data.body_pos_w)[0, i] - base.scene.env_origins[0]
        q = wp.to_torch(robot.data.body_quat_w)[0, i]
        return i, p, q

    print("=== key body poses (env-local) + quat as stored ===")
    for bname in ["panda_link0", "panda_link7", "base_link",
                  "left_inner_finger", "right_inner_finger"]:
        try:
            i, p, q = body_pose(bname)
            print(f"  {bname}[{i}]: pos=({p[0]:.4f},{p[1]:.4f},{p[2]:.4f}) "
                  f"quat=({q[0]:.4f},{q[1]:.4f},{q[2]:.4f},{q[3]:.4f})")
        except ValueError:
            print(f"  {bname}: NOT FOUND")

    # Quat convention check: panda_link0 sits at identity yaw -> identity quat.
    # xyzw stores (0,0,0,1); wxyz stores (1,0,0,0).
    _, _, q0 = body_pose("panda_link0")
    conv = "xyzw" if abs(float(q0[3]) - 1.0) < 0.1 else "wxyz"
    print(f"=== quat convention: {conv} (link0 quat={[round(float(x),3) for x in q0]}) ===")

    print("=== jacobian ===")
    jac = robot.root_physx_view.get_jacobians()
    print(f"  shape={tuple(jac.shape)} (num_envs, num_bodies-1?, 6, num_dof)")

    print("=== ee_frame sensor ===")
    ee_sensor = base.scene.sensors.get("ee_frame")
    if ee_sensor is not None:
        d = ee_sensor.data
        print(f"  target_frame_names={d.target_frame_names}")
        for j, n in enumerate(d.target_frame_names):
            p = wp.to_torch(d.target_pos_w)[0, j] - base.scene.env_origins[0]
            print(f"  {n}: pos=({p[0]:.4f},{p[1]:.4f},{p[2]:.4f})")

    print("=== part poses (env-local) ===")
    for name in ["held_part", "fixed_part", "peg_stand"]:
        if name in base.scene.rigid_objects or name in base.scene.articulations:
            a = base.scene[name]
            p = wp.to_torch(a.data.root_pos_w)[0] - base.scene.env_origins[0]
            q = wp.to_torch(a.data.root_quat_w)[0]
            print(f"  {name}: pos=({p[0]:.4f},{p[1]:.4f},{p[2]:.4f}) "
                  f"quat=({q[0]:.4f},{q[1]:.4f},{q[2]:.4f},{q[3]:.4f})")

    # Gripper linkage probe: hold the arm at its current joints and drive the
    # gripper binary action to OPEN (0) then CLOSE (1); print every finger
    # body's position both ways to see which way the pads actually move.
    hold = wp.to_torch(robot.data.joint_pos)[:, :7].clone()
    finger_bodies = [b for b in robot.data.body_names if "finger" in b or "knuckle" in b]

    def gripper_state(cmd, steps=40):
        a = torch.cat([hold, torch.full((base.num_envs, 1), cmd, device=base.device)], dim=-1)
        for _ in range(steps):
            env.step(a)
        fj = float(wp.to_torch(robot.data.joint_pos)[0, 7])
        print(f"--- gripper cmd={cmd}: finger_joint={fj:.3f}")
        for b in finger_bodies:
            i, p, _ = body_pose(b)
            print(f"    {b}: ({p[0]:.4f},{p[1]:.4f},{p[2]:.4f})")
        if ee_sensor is not None:
            d = ee_sensor.data
            for j, n in enumerate(d.target_frame_names):
                p = wp.to_torch(d.target_pos_w)[0, j] - base.scene.env_origins[0]
                print(f"    frame {n}: ({p[0]:.4f},{p[1]:.4f},{p[2]:.4f})")

    print("=== gripper linkage probe (arm held) ===")
    gripper_state(0.0)
    gripper_state(1.0)

    # Flange->pad-tip length, measured from the USD geometry (not guessed):
    # AABB of the whole Robotiq subtree; at home the tool axis is world +x, so
    # tip plane = max_x. Computed CLOSED (pads meet at the pinch point).
    from pxr import Usd, UsdGeom
    stage = base.scene.stage
    grip_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot/Gripper")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    box = cache.ComputeWorldBound(grip_prim).ComputeAlignedRange()
    mn, mx = box.GetMin(), box.GetMax()
    _, pb, _ = body_pose("base_link")
    print(f"=== Robotiq AABB (closed): x=[{mn[0]:.4f},{mx[0]:.4f}] y=[{mn[1]:.4f},{mx[1]:.4f}] "
          f"z=[{mn[2]:.4f},{mx[2]:.4f}]; base_link=({pb[0]:.4f},{pb[1]:.4f},{pb[2]:.4f}); "
          f"TOOL_LEN = {mx[0] - float(pb[0]):.4f} ===")
    gripper_state(0.0)
    box = cache.ComputeWorldBound(grip_prim).ComputeAlignedRange()
    print(f"=== Robotiq AABB (open): tip_x={box.GetMax()[0]:.4f} "
          f"TOOL_LEN_open = {box.GetMax()[0] - float(pb[0]):.4f} ===")

    env.close()

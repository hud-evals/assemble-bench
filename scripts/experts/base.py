"""Shared machinery for the scripted assembly experts.

Privileged true-pose readers (env-local, xyzw quats), a differential-IK servo
emitting the native 8-D droid_abs_joint_pos action, and a per-env phase machine
with condition-based transitions (each env advances when ITS condition holds,
so episode timing varies naturally instead of following one fixed template).
"""

import torch

import warp as wp
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import quat_apply, quat_box_minus, quat_box_plus, quat_mul


# --- privileged readers (env-local frame; this stack is xyzw end-to-end) ---------------
def pos_of(base, name):
    return wp.to_torch(base.scene[name].data.root_pos_w) - base.scene.env_origins


def quat_of(base, name):
    return wp.to_torch(base.scene[name].data.root_quat_w)


def zrot(yaw):
    """xyzw quat for a rotation of `yaw` (N,) about z."""
    h = 0.5 * yaw
    z = torch.zeros_like(h)
    return torch.stack([z, z, torch.sin(h), torch.cos(h)], dim=-1)


def rolled(quat, yaw):
    """`quat` pre-rotated by `yaw` about world z. With the gripper pointing down,
    this rolls the tool (clocks a grasped part) without tilting it."""
    return quat_mul(zrot(yaw), quat)


def home_quat(servo):
    """The DROID home EE orientation, which ALREADY points the tool straight
    down (approach axis = base_link local +x; the home quat's 90deg-about-y maps
    +x -> world -z). Grasping a vertical peg is pure translation from here -- no
    reorientation. Held constant through the whole trajectory so the wrist never
    tilts (the earlier `tool_down_quat` reorientation was correcting a phantom:
    it wrongly assumed local +z was the approach axis and swung the hand 90deg)."""
    return servo.ee_quat().clone()


class Servo:
    """(EE pose target, grip) -> 8-D absolute-joint action via differential IK.

    The servo frame is the Robotiq ``base_link`` (what the DROID obs/IK use);
    the robot root sits at the env origin with identity yaw, so env-local ==
    robot base frame. ``tcp`` exposes the live pad-tip midpoint from the
    ``ee_frame`` sensor, so grasp targets self-calibrate instead of relying on
    a guessed flange->pad offset. Grip: 0 = open, 1 = close.
    """

    def __init__(self, base):
        self.base = base
        robot = base.scene["robot"]
        self.robot = robot
        self.ee_idx = robot.data.body_names.index("base_link")
        self.jac_idx = self.ee_idx - 1          # fixed-base jacobians skip the root body
        names = list(robot.data.joint_names)
        assert names[:7] == [f"panda_joint{i}" for i in range(1, 8)], names
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            base.num_envs, base.device)
        self.ik.reset()

    def ee(self):
        return wp.to_torch(self.robot.data.body_pos_w)[:, self.ee_idx] - self.base.scene.env_origins

    def ee_quat(self):
        return wp.to_torch(self.robot.data.body_quat_w)[:, self.ee_idx]

    #: Flange (base_link) -> fingertip plane along the tool axis. MEASURED on
    #: this USD by descending a closed gripper onto the peg until first contact
    #: (run_expert --calib_toollen): 0.1717. The 2F-85 spec value (0.1628) is
    #: 9 mm short here. The ee_frame's tool_*finger frames are crank-mounted
    #: and sit near the FLANGE when open -- they are NOT the pads; never aim
    #: with them.
    TOOL_LEN = 0.1717

    def tool_axis(self):
        """Unit approach axis in the world frame. The Robotiq's approach axis is
        base_link LOCAL +X (verified: at home the fingers sit straight below the
        flange, and local +x maps to world -z). Local +z is the forward/optical
        axis, NOT the approach -- aiming with it was the root of the tilt bug."""
        q = self.ee_quat()
        x = torch.tensor([1.0, 0.0, 0.0], device=q.device).expand(q.shape[0], 3)
        return quat_apply(q, x)

    def tcp(self):
        """Fingertip-plane center (env-local): flange + TOOL_LEN along the tool axis."""
        return self.ee() + self.TOOL_LEN * self.tool_axis()

    def act(self, target_pos, target_quat, grip, *, rot_cap=0.15):
        """Absolute joint action tracking (target_pos, target_quat) for base_link.

        The orientation command steps at most `rot_cap` rad toward the target
        (shortest path), so a large initial error (the 90 deg home->tool-down
        reorientation) becomes a smooth swing instead of a PD yank. The DLS
        joint solution is emitted UNclipped: clipping it fights the solver
        near joint limits and leaves a permanent pose residual (seen as a
        2.5 deg tilt + 7 mm TCP offset that stalled the descend).
        """
        q_cur = self.ee_quat()
        delta = quat_box_minus(target_quat, q_cur)
        ang = torch.norm(delta, dim=-1, keepdim=True).clamp_min(1e-9)
        step_quat = quat_box_plus(q_cur, delta * (ang.clamp(max=rot_cap) / ang))
        self.ik.set_command(torch.cat([target_pos, step_quat], dim=-1))
        jac = wp.to_torch(self.robot.root_physx_view.get_jacobians())[:, self.jac_idx, :, :7]
        q = wp.to_torch(self.robot.data.joint_pos)[:, :7]
        q_des = self.ik.compute(self.ee(), q_cur, jac, q)
        # Diagnostics for env 0: the pose error the IK sees, the joint step it
        # answers with, and each joint's margin to its limits.
        self.dbg_pos_err = float(torch.norm(target_pos[0] - self.ee()[0]))
        self.dbg_dq = float(torch.norm(q_des[0] - q[0]))
        lim = wp.to_torch(self.robot.data.joint_pos_limits)[0, :7]
        self.dbg_lim_margin = float(torch.min(torch.minimum(q[0] - lim[:, 0], lim[:, 1] - q[0])))
        return torch.cat([q_des, grip.unsqueeze(-1)], dim=-1)


class Phase:
    """One phase of the machine, fully vectorized over envs.

    target(m) -> (pos, quat, grip) full-batch EE targets (env-local);
    done(m) -> bool mask for envs that may advance; on_enter(m, ids) runs once
    per env as it enters (capture anchors). gate/zcap shape the approach:
    descend only when xy error < gate, cap downward speed by zcap.
    """

    def __init__(self, name, target, done, *, timeout, gate=None, zcap=1.0, on_enter=None,
                 fail_on_timeout=True):
        self.name, self.target, self.done = name, target, done
        self.timeout, self.gate, self.zcap, self.on_enter = timeout, gate, zcap, on_enter
        # Timing out without `done` normally means the episode is unrecoverable
        # (missed grasp, jam): jump to the terminal phase instead of pretending.
        self.fail_on_timeout = fail_on_timeout


class Machine:
    """Per-env phase pointer + rate-limited servo loop. The caller owns env.step."""

    def __init__(self, base, servo, phases, *, pos_cap=0.02):
        self.base, self.servo, self.phases = base, servo, phases
        self.pos_cap = pos_cap                       # m per control step (15 Hz)
        N, dev = base.num_envs, base.device
        self.phase = torch.zeros(N, dtype=torch.long, device=dev)
        self.timer = torch.zeros(N, dtype=torch.long, device=dev)
        self.failed = torch.zeros(N, dtype=torch.bool, device=dev)
        self._enter(torch.arange(N, device=dev), 0)

    @property
    def finished(self):
        return self.phase >= len(self.phases) - 1

    def _enter(self, ids, k):
        if len(ids) and self.phases[k].on_enter is not None:
            self.phases[k].on_enter(self, ids)

    def restart(self, env_ids):
        """Re-enter phase 0 after IsaacLab auto-resets these slots mid-wave.

        Without this, ``hud.wrap`` opens a new trace while the expert still
        thinks the env is in the terminal hold phase -- frozen home-pose
        traces. Call once per ``done`` mask after ``env.step``.
        """
        if env_ids is None:
            return
        ids = env_ids.reshape(-1).long()
        if ids.numel() == 0:
            return
        self.phase[ids] = 0
        self.timer[ids] = 0
        self.failed[ids] = False
        self._enter(ids, 0)

    def _rate_limit(self, cur, target, gate, zcap):
        err = target - cur
        d = torch.clip(err, -self.pos_cap, self.pos_cap)
        d[:, 2] = torch.clip(d[:, 2], -zcap * self.pos_cap, zcap * self.pos_cap)
        if gate is not None:
            ok = torch.norm(err[:, :2], dim=-1) < gate
            d[:, 2] = torch.where(ok, d[:, 2], torch.zeros_like(d[:, 2]))
        return cur + d

    def action(self):
        """One control step: gather per-env targets by phase, advance pointers."""
        N, dev = self.base.num_envs, self.base.device
        pos = torch.zeros((N, 3), device=dev)
        quat = torch.zeros((N, 4), device=dev)
        grip = torch.zeros(N, device=dev)
        adv = torch.zeros(N, dtype=torch.bool, device=dev)
        bail = torch.zeros(N, dtype=torch.bool, device=dev)
        last = len(self.phases) - 1
        for k, ph in enumerate(self.phases[:-1]):
            m = self.phase == k
            if not m.any():
                continue
            p, q, g = ph.target(self)
            p = self._rate_limit(self.servo.ee(), p, ph.gate, ph.zcap)
            pos[m], quat[m], grip[m] = p[m], q[m], g[m] if torch.is_tensor(g) else g
            ok, timeout = ph.done(self), self.timer >= ph.timeout
            if ph.fail_on_timeout:
                bail |= m & timeout & ~ok
                adv |= m & ok
            else:
                adv |= m & (ok | timeout)
        m = self.phase == last
        if m.any():
            p, q, g = self.phases[last].target(self)
            pos[m], quat[m], grip[m] = p[m], q[m], g[m] if torch.is_tensor(g) else g
        self.timer += 1
        self.failed |= bail
        self.phase[bail] = last
        moved = adv | bail
        self.phase[adv] += 1
        self.timer[moved] = 0
        for k in self.phase[moved].unique().tolist():
            self._enter((moved & (self.phase == k)).nonzero().squeeze(-1), k)
        return self.servo.act(pos, quat, grip)

    def report(self):
        names = [self.phases[int(k)].name for k in self.phase]
        return {n: names.count(n) for n in dict.fromkeys(names)}

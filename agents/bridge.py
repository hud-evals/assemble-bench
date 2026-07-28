"""GymBridge subclass: openpi wire + in-process MCP tools for LLM control.

A VLA drives this env over the openpi wire (8-D joint targets at 15 Hz); an LLM
drives the *same* env through the tools in ``agents.tools``, so a tool call
steps the sim directly instead of round-tripping actions over a socket. Both
paths share the scene, the reset and the success check — only the control
surface differs, and the bridge publishes both capabilities so the agent binds
whichever it speaks.

Tools speak fingertip waypoints in millimetres, which the
``droid_differential_ik`` embodiment turns into joint targets: the LLM never
sees a joint angle. Every move is a closed-loop servo that stalls on contact,
so a blocked move reports where the arm actually ended up.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PilImage

from hud.capabilities import Capability
from hud.environment.robot import GymBridge
from hud.environment.robot.gym import action_dim_of, flatten_observation
from hud.telemetry.robot import to_numpy

# load_module(bridge.py) only puts agents/ on path; package imports need the bench root.
_BENCH_ROOT = str(Path(__file__).resolve().parent.parent)
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)

# Servo caps per 15 Hz control tick: 4 mm / 3 deg, i.e. ~6 cm/s — slow enough
# that contact stalls the arm instead of ramming a part through the table.
STEP_M = 0.004
STEP_RAD = 0.05
POS_TOL = 0.0015
ROT_TOL = 0.02
# Arena's droid_differential_ik action term halves every delta (scale=0.5), so
# commands are pre-divided and the caps above stay in real millimetres.
ACTION_SCALE = 0.5
# Flange (base_link, the IK frame) → fingertip plane, measured on this USD
# (scripts/experts/base.py Servo.TOOL_LEN).
TOOL_LEN = 0.1717
# Ticks a gripper command gets to actually open/close the fingers.
GRIP_TICKS = 15
# look(front|wrist) → Arena camera_obs keys (DROID wrist mount, not tool_cam).
CAMERAS = {"front": "camera_obs/front_cam_rgb", "wrist": "camera_obs/wrist_camera_rgb"}
# Peg presentation geometry (mm): stand top ≈ first contact from above; mid-shaft
# grasp sits above that. Asset-root z in guided poses is ~0, not a grasp height.
STAND_TOP_Z_MM = 30.0
GRASP_Z_MM = 45.0
HOVER_Z_MM = 90.0


# ── quaternion math on the wire's wxyz convention ─────────────────────────────


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions."""
    (w1, v1), (w2, v2) = (a[0], a[1:]), (b[0], b[1:])
    return np.concatenate([[w1 * w2 - v1 @ v2], w1 * v2 + w2 * v1 + np.cross(v1, v2)])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` by wxyz quaternion ``q``."""
    w, u = q[0], q[1:]
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


def rotvec_between(q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
    """Axis-angle (rad) taking ``q_from`` to ``q_to`` — the servo's rotation error."""
    err = quat_mul(q_to, np.concatenate([[q_from[0]], -q_from[1:]]))
    err = err if err[0] >= 0 else -err  # shortest way round
    sin_half = np.linalg.norm(err[1:])
    if sin_half < 1e-9:
        return np.zeros(3)
    return err[1:] / sin_half * 2 * np.arctan2(sin_half, err[0])


# ── the bridge ────────────────────────────────────────────────────────────────


class AssemblyToolBridge(GymBridge):
    """The assembly sim served twice: the openpi wire, plus MCP tools in this process.

    Tool episodes need the ``droid_differential_ik`` embodiment (the task
    template asks for it); on the joint-position embodiment the arm tools
    refuse rather than push a 7-D action into an 8-D space.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        with socket.socket() as probe:  # the tool server's port, bound in start()
            probe.bind(("127.0.0.1", 0))
            self._mcp_port = probe.getsockname()[1]
        self._mcp_task: asyncio.Task[None] | None = None
        # One LLM episode's state — the tools own these three.
        self.grip = 1.0  # binary gripper term: < 0 closes
        self.yaw = 0.0  # wrist angle relative to the reset pose
        self.last_move = ""
        self._home_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._origin = np.zeros(3)
        self._ticks = 0  # sim steps spent; the episode's real budget
        self._guided = False

    # ── lifecycle ──────────────────────────────────────────────────────────────

    async def start(self) -> None:
        await super().start()  # openpi wire + contract first
        # Bind tools to this process's bridge, then accept MCP on a free port.
        from agents import tools as tools_mod

        tools_mod.bridge = self
        self._mcp_task = asyncio.create_task(
            tools_mod.server.run_http_async(
                host="127.0.0.1", port=self._mcp_port, show_banner=False
            )
        )
        for _ in range(100):  # publish the address only once it accepts
            with contextlib.suppress(OSError):
                socket.create_connection(("127.0.0.1", self._mcp_port), 0.2).close()
                print(f"[env] tools listening on http://127.0.0.1:{self._mcp_port}/mcp", flush=True)
                return
            await asyncio.sleep(0.1)
        raise RuntimeError(f"tool server never bound port {self._mcp_port}")

    async def stop(self) -> None:
        if self._mcp_task is not None:
            self._mcp_task.cancel()
            self._mcp_task = None
        await super().stop()

    async def capabilities(self, name: str = "robot") -> list[Capability]:
        # Same episode, two control surfaces: a VLA binds the wire, an LLM the tools.
        return [
            *await super().capabilities(name),
            Capability.mcp(name="tools", url=f"http://127.0.0.1:{self._mcp_port}/mcp"),
        ]

    def reset(
        self,
        guided: bool = False,
        episode_length_s: float | None = None,
        **task_args: Any,
    ) -> str:
        """Episode reset; ``guided`` adds privileged part poses + absolute ``move_to``.

        ``episode_length_s`` stretches the sim timeout for LLM tool control only
        (VLA ``assembly`` leaves it unset → variant default, e.g. pegs at 40 s).
        """
        prompt = super().reset(**task_args)
        self._guided, self._ticks = bool(guided), 0
        self.grip, self.yaw, self.last_move = 1.0, 0.0, ""
        # The reset pose already points the tool straight down; the servo holds it.
        self._home_quat = self.read("policy/eef_quat")
        self._origin = to_numpy(self._unwrapped.scene.env_origins)[0]
        # Live property: max_episode_length = ceil(cfg.episode_length_s / step_dt).
        if episode_length_s is not None:
            self._unwrapped.cfg.episode_length_s = float(episode_length_s)
            print(
                f"[env] LLM episode_length_s → {float(episode_length_s):g}s "
                f"({int(self._unwrapped.max_episode_length)} ticks)",
                flush=True,
            )
        # list_tools must match mode: absolute move_to only when guided.
        from agents.tools import server

        if self._guided:
            server.enable(names={"move_to"}, components={"tool"})
        else:
            server.disable(names={"move_to"}, components={"tool"})
        return prompt

    # ── sim-thread reads and motion (queued via _run_on_sim by the tools) ───────

    def read(self, key: str) -> np.ndarray:
        """Slot 0's observation vector, by wire key."""
        return to_numpy(flatten_observation(self._obs)[key])[0].astype(float)

    def camera_png(self, key: str) -> bytes:
        """Slot 0's camera frame as a PNG (handles float or uint8 rgb)."""
        frame = to_numpy(flatten_observation(self._obs)[key])[0]
        if frame.dtype != np.uint8:
            frame = (
                (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                if frame.max() <= 1.0
                else frame.astype(np.uint8)
            )
        buf = io.BytesIO()
        PilImage.fromarray(frame[..., :3]).save(buf, format="png")
        return buf.getvalue()

    def read_tcp(self) -> np.ndarray:
        """Fingertip-plane center in the robot base frame (metres).

        The IK frame is the Robotiq flange; the pads sit ``TOOL_LEN`` further
        along the tool axis, which is what the agent actually aims with.
        """
        quat = self.read("policy/eef_quat")
        flange = self.read("policy/eef_pos") - self._origin
        return flange + TOOL_LEN * quat_rotate(quat, np.array([1.0, 0.0, 0.0]))

    @property
    def episode_over(self) -> bool:
        """Sticky once the env terminates or runs out of time (no more useful stepping)."""
        return bool(self._done.any())

    def check_eef(self) -> None:
        """Refuse a joint-control episode instead of pushing a 7-D action into an 8-D space."""
        if action_dim_of(self.env, batched=self.batched) != 7:
            raise ValueError(
                "this episode was built for joint control; the arm tools need "
                "embodiment='droid_differential_ik'"
            )

    def servo_pose(self, pos: np.ndarray, quat: np.ndarray, *, max_ticks: int) -> None:
        """Closed-loop servo to a fingertip pose (world/base translation + wxyz quat).

        Relative DIK only closes a fraction of each commanded delta per tick
        (PD/DLS lag) — open-loop therefore undershoots badly (~0.3×). Recomputing
        the residual every tick is what makes ``move_to`` / ``nudge`` accurate.
        """
        self.check_eef()
        for _ in range(max_ticks):
            dp = np.clip(pos - self.read_tcp(), -STEP_M, STEP_M)
            dr = np.clip(
                rotvec_between(self.read("policy/eef_quat"), quat), -STEP_RAD, STEP_RAD
            )
            if np.abs(dp).max() < POS_TOL and np.abs(dr).max() < ROT_TOL:
                return
            # Arena's action term multiplies by scale=0.5 — undo so STEP_* are real.
            self.step(np.concatenate([dp / ACTION_SCALE, dr / ACTION_SCALE, [self.grip]]))
            self._ticks += 1
            if self.episode_over:
                return

    def servo_to(self, pos: np.ndarray, yaw: float, *, max_ticks: int) -> None:
        """Step toward a fingertip waypoint; wrist yaw is relative to the reset pose."""
        # Wrist yaw on top of the (tool-down) reset orientation.
        goal_quat = quat_mul(np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]), self._home_quat)
        self.servo_pose(pos, goal_quat, max_ticks=max_ticks)

    def hold(self, ticks: int) -> None:
        """Hold the pose for ``ticks`` steps — lets the fingers close and parts settle."""
        self.check_eef()
        for _ in range(ticks):
            if self.episode_over:
                return
            self.step(np.concatenate([np.zeros(6), [self.grip]]))
            self._ticks += 1

    def nudge_delta(self, dp_m: np.ndarray, dr_rad: np.ndarray, *, max_ticks: int) -> None:
        """Displace from the current pose by ``dp_m`` / ``dr_rad`` (DOF debug).

        Translation: closed-loop on fingertip with ``dr=0``. Holding orientation
        in that loop couples into TCP through the tool lever arm (DOF ladder v2
        chaos). Rotation: open-loop only — tip motion is the observable; do not
        pin TCP (that would fight the lever-arm tip shift). Relative DIK delivers
        ~0.3× per tick, so rot budget is sized from that, not the tool's slack.
        """
        self.check_eef()
        dp_m = np.asarray(dp_m, dtype=float)
        dr_rad = np.asarray(dr_rad, dtype=float)
        translating = float(np.linalg.norm(dp_m)) > 1e-9
        rotating = float(np.linalg.norm(dr_rad)) > 1e-9
        goal_tcp = self.read_tcp() + dp_m
        # ~0.3× open-loop delivery → inflate command and (for rot-only) trim ticks.
        ol_gain = 1.0 / 0.3
        if rotating and not translating:
            max_ticks = min(
                max_ticks,
                int(np.ceil(np.linalg.norm(dr_rad) * ol_gain / STEP_RAD)) + 15,
            )
        cmd_r = (
            np.clip(dr_rad * ol_gain / max(1, max_ticks), -STEP_RAD, STEP_RAD)
            if rotating
            else np.zeros(3)
        )
        for _ in range(max_ticks):
            if self.episode_over:
                return
            if translating:
                err_p = goal_tcp - self.read_tcp()
                if np.abs(err_p).max() < POS_TOL:
                    return
                cmd_p = np.clip(err_p, -STEP_M, STEP_M)
            else:
                cmd_p = np.zeros(3)
            self.step(np.concatenate([cmd_p / ACTION_SCALE, cmd_r / ACTION_SCALE, [self.grip]]))
            self._ticks += 1

    def report(self) -> str:
        """What every tool returns: where the arm is, what it just did, and the clock."""
        pos = self.read_tcp() * 1000
        lines = [
            f"fingertips: x={pos[0]:.0f} y={pos[1]:.0f} z={pos[2]:.0f} mm, "
            f"wrist={np.degrees(self.yaw):.0f} deg, "
            f"gripper={'closed' if self.grip < 0 else 'open'}, "
            f"finger_closure={self.read('policy/gripper_pos')[0]:.2f}",
            f"budget: {self._ticks}/{int(self._unwrapped.max_episode_length)} control steps used",
        ]
        if self.last_move:
            lines.append(self.last_move)
        if self._guided:  # privileged true poses (env-local, mm)
            held, fixed = self.read("policy/held_part_pose"), self.read("policy/fixed_part_pose")
            lines.append(
                f"parts: loose part at x={held[0] * 1000:.0f} y={held[1] * 1000:.0f} "
                f"z={held[2] * 1000:.0f}, target at x={fixed[0] * 1000:.0f} "
                f"y={fixed[1] * 1000:.0f} z={fixed[2] * 1000:.0f} mm"
            )
            # Origins are roots at the table — not pad heights. Give the recipe z's.
            lines.append(
                f"heights: stand_top≈{STAND_TOP_Z_MM:.0f} mm (first contact from above), "
                f"grasp≈{GRASP_Z_MM:.0f} mm (mid-shaft), hover≈{HOVER_Z_MM:.0f} mm — "
                f"do not grasp at part z; if Δz→0 near z={STAND_TOP_Z_MM:.0f}, you are on the stand"
            )
        if self.episode_over:
            lines.append(f"episode over: {'SOLVED' if self._success.any() else 'not solved'}")
        return "\n".join(lines)

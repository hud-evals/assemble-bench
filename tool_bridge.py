"""MCP tool surface for the assembly sim, served from inside the sim process.

A VLA drives this env over the openpi wire (8-D joint targets at 15 Hz); an LLM
drives the *same* env through the tools here, so a tool call steps the sim
directly instead of round-tripping actions over a socket. Both paths share the
scene, the reset and the success check — only the control surface differs, and
the bridge publishes both capabilities so the agent binds whichever it speaks.

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
from typing import Any, Literal

import numpy as np
from fastmcp import FastMCP
from fastmcp.utilities.types import Image
from PIL import Image as PilImage

from hud.capabilities import Capability
from hud.environment.robot import GymBridge
from hud.environment.robot.gym import action_dim_of, flatten_observation
from hud.telemetry.robot import to_numpy

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
CAMERAS = {"front": "camera_obs/front_cam_rgb", "wrist": "camera_obs/wrist_camera_rgb"}

# The tool server and the one bridge in this process (bound in ``start``).
server: FastMCP = FastMCP("assembly-arm")
bridge: AssemblyToolBridge


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
        global bridge  # the tools below drive whichever bridge this process serves
        bridge = self
        self._mcp_task = asyncio.create_task(
            server.run_http_async(host="127.0.0.1", port=self._mcp_port, show_banner=False)
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

    def reset(self, guided: bool = False, **task_args: Any) -> str:
        """Episode reset; ``guided`` adds privileged part poses to every tool reply."""
        prompt = super().reset(**task_args)
        self._guided, self._ticks = bool(guided), 0
        self.grip, self.yaw, self.last_move = 1.0, 0.0, ""
        # The reset pose already points the tool straight down; the servo holds it.
        self._home_quat = self.read("policy/eef_quat")
        self._origin = to_numpy(self._unwrapped.scene.env_origins)[0]
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

    def servo_to(self, pos: np.ndarray, yaw: float, *, max_ticks: int) -> None:
        """Step the sim toward a fingertip pose, one clipped delta per control tick.

        Closed loop, so it either converges, runs out of ticks, or stalls
        against contact — the caller reports where the arm ended up.
        """
        self.check_eef()
        # Wrist yaw is applied on top of the (tool-down) reset orientation.
        goal_quat = quat_mul(np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]), self._home_quat)
        for _ in range(max_ticks):
            dp = np.clip(pos - self.read_tcp(), -STEP_M, STEP_M)
            dr = np.clip(
                rotvec_between(self.read("policy/eef_quat"), goal_quat), -STEP_RAD, STEP_RAD
            )
            if np.abs(dp).max() < POS_TOL and np.abs(dr).max() < ROT_TOL:
                return
            self.step(np.concatenate([dp / ACTION_SCALE, dr / ACTION_SCALE, [self.grip]]))
            self._ticks += 1
            if self.episode_over:
                return

    def hold(self, ticks: int) -> None:
        """Hold the pose for ``ticks`` steps — lets the fingers close and parts settle."""
        self.check_eef()
        for _ in range(ticks):
            if self.episode_over:
                return
            self.step(np.concatenate([np.zeros(6), [self.grip]]))
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
        if self.episode_over:
            lines.append(f"episode over: {'SOLVED' if self._success.any() else 'not solved'}")
        return "\n".join(lines)


# ── the tools an LLM agent sees ───────────────────────────────────────────────


@server.tool
async def look(camera: Literal["front", "wrist"] = "front") -> list[Any]:
    """Look at the workspace. Free: it costs no sim time, so look often.

    ``front`` is a fixed view of the whole table from the far side; ``wrist``
    looks down the gripper and is the one that shows whether the pads are
    actually straddling a part.
    """
    # Sim touches are queued onto the sim thread — the bridge's own mechanism.
    png = await bridge._run_on_sim(bridge.camera_png, CAMERAS[camera])
    return [await bridge._run_on_sim(bridge.report), Image(data=png, format="png")]


@server.tool
async def move_to(x_mm: float, y_mm: float, z_mm: float, yaw_deg: float | None = None) -> str:
    """Move the fingertips to a waypoint in the robot base frame, then stop.

    Millimetres: +x away from the robot, +y to its left, +z up, z=0 at the
    tabletop. ``yaw_deg`` turns the wrist (0 = the starting angle). The motion
    is closed-loop and stalls on contact, so compare the position in the reply
    against what you asked for — a gap means something is in the way. Costs sim
    time roughly in proportion to the distance travelled.
    """
    if bridge.episode_over:
        return "episode is over; no more moves.\n" + await bridge._run_on_sim(bridge.report)
    goal = np.array([x_mm, y_mm, z_mm]) / 1000
    if yaw_deg is not None:
        bridge.yaw = float(np.radians(yaw_deg))
    start = await bridge._run_on_sim(bridge.read_tcp)
    # Budget the servo by distance (plus slack to settle), never the whole episode.
    ticks = min(150, int(np.linalg.norm(goal - start) / STEP_M) + 40)
    await bridge._run_on_sim(bridge.servo_to, goal, bridge.yaw, max_ticks=ticks)
    reached = await bridge._run_on_sim(bridge.read_tcp)
    off = np.abs(goal - reached).max() * 1000
    bridge.last_move = (
        f"last move: asked for x={x_mm:.0f} y={y_mm:.0f} z={z_mm:.0f}, "
        + ("arrived" if off < 3 else f"stopped {off:.0f} mm short (blocked or out of reach)")
    )
    return await bridge._run_on_sim(bridge.report)


@server.tool
async def grasp() -> str:
    """Close the gripper on whatever is between the pads.

    ``finger_closure`` in the reply tells you what happened: 1.00 means the
    fingers closed on nothing, a value in between means they are pinching a part.
    """
    bridge.grip = -1.0  # binary term: negative closes
    await bridge._run_on_sim(bridge.hold, GRIP_TICKS)
    return await bridge._run_on_sim(bridge.report)


@server.tool
async def release() -> str:
    """Open the gripper and let the part settle."""
    bridge.grip = 1.0
    await bridge._run_on_sim(bridge.hold, GRIP_TICKS)
    return await bridge._run_on_sim(bridge.report)


@server.tool
async def get_state() -> str:
    """Report the arm pose, gripper and remaining budget without moving anything."""
    return await bridge._run_on_sim(bridge.report)

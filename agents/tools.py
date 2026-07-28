"""FastMCP tool registrations for the assembly LLM agent.

``move_to`` is gated on ``bridge._guided`` (enabled in ``reset`` when guided):
list_tools hides it in vision mode, and the body refuses if called unguided.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from fastmcp import FastMCP
from fastmcp.utilities.types import Image

from agents.bridge import CAMERAS, GRIP_TICKS, STEP_M, STEP_RAD, AssemblyToolBridge

# Bound in AssemblyToolBridge.start — tools drive whichever bridge this process serves.
server: FastMCP = FastMCP("assembly-arm")
bridge: AssemblyToolBridge


@server.tool
async def look(camera: Literal["front", "wrist"] = "front") -> list[Any]:
    """RGB snapshot + proprio. Free (no sim time) — look after every move.

    ``front``: whole table from across the robot. ``wrist``: Robotiq wrist cam
    (near the pads) — use before grasp / press to check pad centering.
    Reply text includes fingertip mm, wrist compass (deg), and grasp closure.
    """
    # Sim touches are queued onto the sim thread — the bridge's own mechanism.
    png = await bridge._run_on_sim(bridge.camera_png, CAMERAS[camera])
    return [await bridge._run_on_sim(bridge.report), Image(data=png, format="png")]


@server.tool
async def move_to(x_mm: float, y_mm: float, z_mm: float, yaw_deg: float | None = None) -> str:
    """Go to an absolute fingertip waypoint (millimetres), then stop.

    Guided mode only (privileged landmarks). This is a *target pose*, not a
    delta: ``(x_mm, y_mm, z_mm)`` is where the pads should end up in the robot
    base frame. Prefer this for long travels (home → hover, peg → hole). Once
    you are within ~20 mm, switch to ``nudge``.

    Axes (robot base, millimetres):
    - ``+x`` / ``−x``: forward / back (away from / toward the robot base)
    - ``+y`` / ``−y``: left / right (robot's left when facing forward)
    - ``+z`` / ``−z``: up / down (table ≈ 0)

    How big is a move (straight-line distance from current pose to the ask):
    - ~10 mm — small (usually use ``nudge`` instead)
    - ~100 mm — medium table move
    - ~300–600 mm — large (home ↔ workspace); fine for ``move_to``, costly on budget

    Never ask for:
    - ``z_mm < 15`` except the final insert press (table / stand collision)
    - ``z_mm > 700`` (above the useful workspace)
    - ``x_mm`` outside ~200–500 or ``y_mm`` outside ~−200–200 (off-table / unreachable)
    - a new ``move_to`` when you are already within ~20 mm — use ``nudge``

    Optional ``yaw_deg``: wrist compass (0 = reset). Motion is closed-loop and
    stalls on contact — compare reply pose to the ask; >3 mm gap means blocked.
    """
    # Defense in depth: list_tools already hides this when unguided.
    if not bridge._guided:
        return (
            "move_to is unavailable in vision mode (absolute waypoints need "
            "guided landmarks). Localize with look, then nudge.\n"
            + await bridge._run_on_sim(bridge.report)
        )
    if bridge.episode_over:
        return "episode is over; no more moves.\n" + await bridge._run_on_sim(bridge.report)
    goal = np.array([x_mm, y_mm, z_mm]) / 1000
    if yaw_deg is not None:
        bridge.yaw = float(np.radians(yaw_deg))
    start = await bridge._run_on_sim(bridge.read_tcp)
    # Relative DIK delivers ~0.3× of each open-loop tick — size the servo budget
    # from that, not from STEP_M alone (wave1 first move_to stopped ~100 mm short).
    dist = float(np.linalg.norm(goal - start))
    ticks = min(900, int(dist / (STEP_M * 0.3)) + 80)
    await bridge._run_on_sim(bridge.servo_to, goal, bridge.yaw, max_ticks=ticks)
    reached = await bridge._run_on_sim(bridge.read_tcp)
    off = np.abs(goal - reached).max() * 1000
    if off < 3:
        note = "arrived"
    elif bridge.episode_over:
        note = f"stopped {off:.0f} mm short (episode ended)"
    else:
        # Contact stall, IK limit, or tick cap — caller should re-issue or look.
        note = f"stopped {off:.0f} mm short (contact, reach limit, or tick cap)"
    bridge.last_move = f"last move: asked for x={x_mm:.0f} y={y_mm:.0f} z={z_mm:.0f}, {note}"
    return await bridge._run_on_sim(bridge.report)


# Absolute move_to starts hidden; reset(guided=True) enables it for list_tools.
server.disable(names={"move_to"}, components={"tool"})


@server.tool
async def nudge(
    dx_mm: float = 0.0,
    dy_mm: float = 0.0,
    dz_mm: float = 0.0,
    droll_deg: float = 0.0,
    dpitch_deg: float = 0.0,
    dyaw_deg: float = 0.0,
    ticks: int = 80,
) -> str:
    """Nudge the fingertips by a *relative* delta from where they are now.

    Unlike ``move_to`` (absolute target), args are offsets: ``dx_mm=+10`` means
    "10 mm forward from here", not "go to x=10". Use after a look for fine
    centering / depth — and in vision mode for all travel after localizing.

    Same axes as ``move_to`` (+x forward, +y left, +z up). Scale:
    - ~2–15 mm — typical centering / depth tweak
    - ~20–40 mm — large nudge (still OK)
    - >50 mm — split into several nudges (or use ``move_to`` when guided)

    Never ask for:
    - a nudge that would drive ``z`` below ~15 mm (except a deliberate press)
    - roll/pitch unless the tip is clearly tilted (leave at 0; prefer ``dyaw_deg``)
    - crossing >~50 mm in one call — split the travel

    Reply reports measured Δtcp so you can see ask vs got.
    """
    if bridge.episode_over:
        return "episode is over; no more moves.\n" + await bridge._run_on_sim(bridge.report)
    before = await bridge._run_on_sim(bridge.read_tcp)
    dp = np.array([dx_mm, dy_mm, dz_mm], dtype=float) / 1000.0
    dr = np.radians(np.array([droll_deg, dpitch_deg, dyaw_deg], dtype=float))
    need = max(
        int(np.linalg.norm(dp) / STEP_M) + 40,
        int(np.linalg.norm(dr) / STEP_RAD) + 40,
        int(ticks),
    )
    await bridge._run_on_sim(bridge.nudge_delta, dp, dr, max_ticks=min(400, need))
    after = await bridge._run_on_sim(bridge.read_tcp)
    d = (after - before) * 1000
    bridge.last_move = (
        f"last nudge: asked dx={dx_mm:.0f} dy={dy_mm:.0f} dz={dz_mm:.0f} mm, "
        f"dR={droll_deg:.0f} dP={dpitch_deg:.0f} dY={dyaw_deg:.0f} deg → "
        f"Δtcp x={d[0]:+.0f} y={d[1]:+.0f} z={d[2]:+.0f} mm"
    )
    return await bridge._run_on_sim(bridge.report)


@server.tool
async def grasp() -> str:
    """Close the pads on whatever is between them.

    Check ``finger_closure`` after: ~1.0 = empty (missed), ~0.3–0.9 = pinching a
    part. On a miss, open (release), look(wrist), nudge, then grasp again.
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
    """Proprio only (no camera, no motion): fingertip mm, wrist compass, grip, budget."""
    return await bridge._run_on_sim(bridge.report)

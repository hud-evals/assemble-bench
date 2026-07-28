"""Task prompts for the MCP tool surface (guided vs vision tool lists)."""


def agent_prompt(instruction: str, *, guided: bool) -> str:
    """Prompt for the MCP tool surface: same task, fingertip waypoints instead of joints."""
    # Lessons from high-level LLM-robot interfaces (Anthropic “Claude plays robotics”):
    # compass/orientation beats extra vision overlays; closed-loop self-check after every
    # move; coarse then fine; keep the plan short and re-plan from the latest reply.
    shared = (
        "You control a Franka arm with a Robotiq 2F gripper over a tabletop assembly task.\n"
        f"Task: {instruction}\n\n"
        "Frame (millimetres, robot base): +x away from the robot, +y to its left, +z up, "
        "z≈0 at the tabletop. Tools aim the fingertip plane (pads), not the flange. "
        "Wrist yaw is a compass in degrees (0 = reset); the tool points straight down — "
        "keep it that way unless the part clearly needs a yaw clock.\n\n"
    )
    common_tools = (
        "- look(front|wrist) — free (no sim time). front = whole table; wrist = near the "
        "pads (confirm straddle before grasp).\n"
        "- nudge(dx,dy,dz[,dyaw]) — relative correction (mm / deg). Prefer ≤20 mm for "
        "centering; leave roll/pitch at 0. Split travel >~50 mm into several nudges.\n"
        "- grasp / release — close or open pads. finger_closure≈1.0 means empty close; "
        "a mid value means a pinch.\n"
        "- get_state — proprio + budget, no motion.\n"
    )
    loop = (
        "\nLoop: look → act → read the reply (pose, last move, finger_closure) → look again. "
        "Re-plan from the latest state; do not dump a long open-loop script. "
        "Budget is ~80 s of sim (~1200 control steps) — short moves, no hovering delays.\n"
        "Keep z ≥ 20 mm except during grasp and the final press.\n"
    )
    if guided:
        return (
            shared
            + "Tools:\n"
            + common_tools
            + "- move_to(x,y,z[,yaw]) — absolute closed-loop waypoint (guided landmarks). "
            "Reply includes asked→got; a gap means contact, reach limit, or tick cap — "
            "do not assume you arrived. Once within ~20 mm, switch to nudge.\n"
            + loop
            + "\nGuided mode: every reply lists loose-part and target origins (mm) plus height "
            "hints (stand_top≈30 / grasp≈45 / hover≈90). Origins are asset roots at the "
            "table (z≈0) — use their XY only; use the height hints for Z.\n\n"
            "Peg insert recipe:\n"
            "1) move_to peg XY at z≈90 (hover). Re-issue only if reply is still >20 mm off.\n"
            "2) Descend to grasp z≈45 mm (mid-shaft). stand_top≈30 mm — if descent stalls "
            "there with Δz→0, you are on the block; go UP to z≈45, do not grasp at z≈30.\n"
            "3) look(wrist), nudge XY until pads straddle the peg, grasp.\n"
            "4) If finger_closure≈1.0: release, look(wrist), nudge, retry. Else lift to z≈90.\n"
            "5) move_to hole XY at z≈90, nudge XY, then lower in 10–15 mm steps into the "
            "bore; release when seated.\n"
        )
    return (
        shared
        + "Tools:\n"
        + common_tools
        + loop
        + "\nVision mode: part poses are NOT given, and absolute move_to is unavailable. "
        "Localize from look(front) / look(wrist) plus the fingertip pose in each reply, "
        "then travel with nudge. Prefer wrist for final approach.\n\n"
        "Peg insert recipe:\n"
        "1) look(front), estimate peg XY from the image and your fingertip pose; nudge "
        "in stages toward hover above the peg (z≈90).\n"
        "2) Descend toward mid-shaft (z≈45). If descent stalls near z≈30 with Δz→0, you "
        "are on the stand — go UP, do not grasp there.\n"
        "3) look(wrist), nudge XY until pads straddle the peg, grasp.\n"
        "4) If finger_closure≈1.0: release, look(wrist), nudge, retry. Else lift to z≈90.\n"
        "5) look(front), nudge toward the hole XY at z≈90, then lower in 10–15 mm steps; "
        "release when seated.\n"
    )

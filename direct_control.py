"""Inputs for the ``move_joints`` tool that ``env.py`` attaches to the sim.

The contract action is 8-D absolute joint targets (``joint_pos``), so
``DirectControl`` serves ``move_joints``.

``joint_reference`` is the first absolute target: measured arm joints, gripper
commanded open. ``contract.json`` omits part poses and expert channels, so the
tool result cannot label them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Absolute radians. gripper.open_close is 0 open, 1 closed; >0.5 closes.
# Front camera sees the table; wrist camera sits on the Robotiq.
NOTES = (
    "Joints are absolute radians. gripper.open_close is 0 open and 1 closed; "
    "values above 0.5 close. Cameras: camera_obs/front_cam_rgb (table) and "
    "camera_obs/wrist_camera_rgb (wrist)."
)

# Default pacing would close the gripper over many seconds. One tick must be
# able to cross 0.5.
GRIPPER_MAX_STEP = {"gripper.open_close": 1.0}


def joint_reference(data: dict[str, Any]) -> np.ndarray:
    """Current 8-D command: measured arm joints, gripper held open.

    ``policy/gripper_pos`` is finger opening, and its polarity is not the
    command (``>0.5`` closes). The open command is 0.
    """
    joints = np.asarray(data["policy/joint_pos"], dtype=np.float64).reshape(-1)
    if joints.size != 7:
        raise ValueError(f"policy/joint_pos must be 7 joints, got shape {joints.shape}")
    return np.concatenate([joints, [0.0]])

"""Inputs for the ``move_joints`` tool that ``env.py`` attaches to the sim.

The contract action is 8-D absolute joint targets (``joint_pos``), so
``DirectControl`` serves ``move_joints``.

``joint_reference`` is the measured 8-D state: arm joints and gripper (0 open, 1 closed). ``contract.json`` omits part poses and expert channels, so the
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
    """Current 8-D state in command units: measured arm joints and measured gripper.

    ``policy/gripper_pos`` is the finger joint over pi/4: 0 open, 1 closed, the same
    polarity as the ``gripper.open_close`` command. Reporting the real value lets a
    gripper-only call end when the fingers reach the target or stop on the part.
    A constant here never matches a close, so the tool would play out its 60 s cap and
    end the episode.
    """
    joints = np.asarray(data["policy/joint_pos"], dtype=np.float64).reshape(-1)
    if joints.size != 7:
        raise ValueError(f"policy/joint_pos must be 7 joints, got shape {joints.shape}")
    gripper = np.asarray(data["policy/gripper_pos"], dtype=np.float64).reshape(-1)
    if gripper.size != 1:
        raise ValueError(f"policy/gripper_pos must be 1 value, got shape {gripper.shape}")
    return np.concatenate([joints, np.clip(gripper, 0.0, 1.0)])

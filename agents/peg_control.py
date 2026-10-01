"""Joint-contract inputs for peg insertion. Not attached.

Two surfaces are still open, and they would not score the same:

- Direct control on this file's joint contract (``move_joints``), with no
  privileged object poses in the tool result.
- The fingertip tools in ``agents/tools.py`` (differential IK, not ready).

Stock direct control labels every non-image observation, including
``policy/held_part_pose`` and ``policy/fixed_part_pose``. Attaching it before
that result can omit those keys would leak poses. ``env.py`` does not import
this module.
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

# Fingers need about a second to finish after the command.
SETTLE_S = 1.0


def joint_reference(data: dict[str, Any]) -> np.ndarray:
    """Current 8-D command: measured arm joints, gripper held open.

    ``policy/gripper_pos`` is finger opening, and its polarity is not the
    command (``>0.5`` closes). The open command is 0.
    """
    joints = np.asarray(data["policy/joint_pos"], dtype=np.float64).reshape(-1)
    if joints.size != 7:
        raise ValueError(f"policy/joint_pos must be 7 joints, got shape {joints.shape}")
    return np.concatenate([joints, [0.0]])

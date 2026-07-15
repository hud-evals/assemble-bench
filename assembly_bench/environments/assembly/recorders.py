"""Recorder terms for teleop demo collection.

The teleop env acts in differential-IK space (6 relative EE values + gripper),
but the benchmark's canonical policy contract (pi0.5-DROID, contract.json) is
8-D absolute joint targets. ``PostStepAbsJointTargetRecorder`` records that
canonical label stream alongside the raw DIK actions: post-step, the arm's
``joint_pos_target`` buffer holds exactly the absolute target the IK solver
commanded this step (what set_joint_position_target received), and the
finger_joint target normalizes back to the 0/1 gripper convention -- so the
label is independent of the teleop device's raw gripper sign convention.

Raw ``actions`` stay in the HDF5 untouched: replay, annotation, and Mimic
generation step the DIK env with them. ``abs_joint_action`` is the training
export for the droid_abs_joint_pos contract.
"""

import torch

import warp as wp
from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

_PANDA_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
_GRIPPER_CLOSE_RAD = torch.pi / 4  # finger_joint close target (droid embodiments)


class PostStepAbsJointTargetRecorder(RecorderTerm):
    """Records the canonical 8-D [7 abs joint targets, gripper 0..1] per step."""

    def record_post_step(self):
        robot = self._env.scene["robot"]
        if not hasattr(self, "_arm_idx"):
            names = list(robot.data.joint_names)
            self._arm_idx = [names.index(j) for j in _PANDA_JOINTS]
            self._finger_idx = names.index("finger_joint")
        targets = wp.to_torch(robot.data.joint_pos_target)
        arm = targets[:, self._arm_idx]
        grip = (targets[:, self._finger_idx : self._finger_idx + 1] / _GRIPPER_CLOSE_RAD).clamp(0.0, 1.0)
        return "abs_joint_action", torch.cat([arm, grip], dim=-1)


@configclass
class PostStepAbsJointTargetRecorderCfg(RecorderTermCfg):
    """Configuration for the absolute joint-target label recorder term."""

    class_type: type[RecorderTerm] = PostStepAbsJointTargetRecorder

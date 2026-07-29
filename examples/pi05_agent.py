"""pi0.5 DROID policy for AssembleBench — self-contained, no repo-local imports.

Loads a LeRobot pi0.5 checkpoint (Hugging Face repo id or local directory) and
wires it to the benchmark's DROID contract: front + wrist RGB, 7 joint positions
plus gripper, 8-D absolute joint targets at 15 Hz.

The DROID I/O convention is not baked into the checkpoint, so this file carries
its own adapter, faithful to OpenPI's server-side ``droid_policy.py``:

  - cameras: exterior + wrist onto the model's first two slots, aspect-preserving
    resize+pad to 224 (the third slot is left absent so the policy pads + masks it)
  - state: 8-D ``[7 arm joints, gripper]``, fed raw
  - actions: arm dims come out absolute via LeRobot's AbsoluteActions
    postprocessor; the gripper is binarized here for the env's binary gripper
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, resize_with_pad_torch

from hud.agents.robot import Adapter, LeRobotModel, RobotAgent

# Final checkpoint from the writeup: pi0.5 BC (12k) + 3 rounds of code-gated
# DAgger on peg_round_8mm.
DEFAULT_CHECKPOINT = "hud-evals/pi05-AssembleBench-cgdagger-r3"

IMG = 224      # the checkpoint's camera resolution
HORIZON = 15   # open-loop action-chunk length

JOINTS_KEY = "policy/joint_pos"
GRIPPER_KEY = "policy/gripper_pos"

# Control ticks per episode at 15 Hz. Env horizons are longer; the agent stops
# here so eval wall-clock stays predictable.
STEP_CAPS = {"peg": 400, "nut": 480, "gear": 540, "debug": 300}


class DroidAdapter(Adapter):
    """DROID joint-position wiring, single or batched observations."""

    def __init__(self, *, model_image_keys: list[str], state_dim: int) -> None:
        super().__init__(model_image_keys=model_image_keys)
        self.state_dim = state_dim

    def bind(self, action_space: dict[str, Any], observation_space: dict[str, Any]) -> None:
        """DROID slots are exterior + wrist: pick the wrist camera by name and the
        first non-wrist camera as exterior (an env may expose extra exteriors)."""
        super().bind(action_space, observation_space)
        wrist = next((k for k in self.image_keys if "wrist" in k.lower()), None)
        if wrist is not None:
            exterior = next(k for k in self.image_keys if k != wrist)
            self.image_keys = [exterior, wrist]

    def _image(self, img: np.ndarray, batched: bool) -> torch.Tensor:
        t = resize_with_pad_torch(torch.from_numpy(np.ascontiguousarray(img)), IMG, IMG)
        if batched:
            return t.reshape(-1, IMG, IMG, 3).permute(0, 3, 1, 2).float() / 255.0
        return t.reshape(IMG, IMG, 3).permute(2, 0, 1).float() / 255.0

    def adapt_observation(self, obs: dict[str, Any], prompt: str) -> dict[str, Any]:
        data = obs["data"]
        joints = np.asarray(data[JOINTS_KEY], dtype=np.float32)
        gripper = np.asarray(data[GRIPPER_KEY], dtype=np.float32)
        batched = joints.ndim > 1  # [N, 7] vs [7]
        if batched:
            state = np.zeros((joints.shape[0], self.state_dim), dtype=np.float32)
            state[:, :8] = np.concatenate([joints, gripper], axis=1)
        else:
            state = np.zeros(self.state_dim, dtype=np.float32)
            state[:8] = np.concatenate([joints, gripper])
        batch: dict[str, Any] = {
            "observation.state": torch.from_numpy(state),
            "task": [prompt] * joints.shape[0] if batched else prompt,
        }
        # Env cameras in contract order (exterior, wrist) onto the model's first
        # two slots; any remaining model slot stays absent.
        for model_key, env_key in zip(self.model_image_keys, self.image_keys, strict=False):
            batch[model_key] = self._image(data[env_key], batched)
        return batch

    def adapt_chunk(self, chunk: np.ndarray, obs: dict[str, Any]) -> np.ndarray:
        """Binarize the gripper. Arm dims are already absolute (AbsoluteActions
        postprocessor), so do not add the current joints again."""
        out = np.array(chunk[:, :8], dtype=np.float32)
        out[:, 7] = (out[:, 7] > 0.5).astype(np.float32)
        return out


class PI05AssemblyAgent(RobotAgent):
    """pi0.5 rollouts on AssembleBench, with a per-family episode length."""

    adapter_cls = DroidAdapter

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        device: str | None = None,
        step_cap: int | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.step_cap = step_cap
        print(f"[agent] loading {checkpoint} ({self.device})", flush=True)
        policy = PI05Policy.from_pretrained(checkpoint).to(self.device).eval()
        pre, post = make_pre_post_processors(
            policy.config,
            checkpoint,
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )
        policy.config.n_action_steps = HORIZON
        self.model = LeRobotModel(policy, pre, post)
        self.adapter = self.adapter_cls(
            model_image_keys=list(policy.config.image_features),
            state_dim=int(policy.config.input_features["observation.state"].shape[0]),
        )

    async def __call__(self, run: Any, *, max_steps: int | None = None) -> None:
        if max_steps is None:
            family = (run.slug or "").split("_", 1)[0]
            max_steps = self.step_cap or STEP_CAPS.get(family, STEP_CAPS["peg"])
        await super().__call__(run, max_steps=max_steps)


agent = PI05AssemblyAgent

__all__ = ["DEFAULT_CHECKPOINT", "STEP_CAPS", "DroidAdapter", "PI05AssemblyAgent", "agent"]

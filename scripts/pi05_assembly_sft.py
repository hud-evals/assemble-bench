"""pi0.5 DROID agent pointing at an assembly_bench SFT checkpoint.

SFT was trained with ``--use_relative_actions``; the saved postprocessor already
runs ``absolute_actions_processor`` (enabled), so ``adapt_chunk`` only slices to
8-D and binarizes the gripper — it must NOT re-add joint position.
"""

import os

import numpy as np
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, resize_with_pad_torch

from hud.agents.robot import Adapter, LeRobotModel, RobotAgent

# Override with CHECKPOINT=/path/to/merged_sft
CHECKPOINT = os.environ.get(
    "CHECKPOINT", os.path.expanduser("~/checkpoints/pi05_assembly_bench_2_sft_20k")
)
IMAGE_SIZE = 224


class AssemblySftAdapter(Adapter):
    def __init__(self, model_image_keys: list[str], state_dim: int) -> None:
        super().__init__(model_image_keys=model_image_keys)
        self.state_dim = state_dim

    def adapt_observation(self, obs: dict, prompt: str) -> dict:
        data = obs["data"]
        state = np.zeros(self.state_dim, dtype=np.float32)
        state[:8] = np.concatenate([data["policy/joint_pos"], data["policy/gripper_pos"]])
        batch = {"observation.state": torch.from_numpy(state), "task": prompt}
        cameras = (data["camera_obs/front_cam_rgb"], data["camera_obs/wrist_camera_rgb"])
        for key, image in zip(self.model_image_keys, cameras):
            image = resize_with_pad_torch(
                torch.from_numpy(np.ascontiguousarray(image)), IMAGE_SIZE, IMAGE_SIZE
            )
            batch[key] = image.reshape(IMAGE_SIZE, IMAGE_SIZE, 3).permute(2, 0, 1).float() / 255.0
        return batch

    def adapt_chunk(self, chunk: np.ndarray, obs: dict) -> np.ndarray:
        # Postprocessor already made arm joints absolute; only binarize gripper.
        actions = np.asarray(chunk[:, :8], dtype=np.float32)
        actions[:, 7] = actions[:, 7] > 0.5
        return actions


class Pi05AssemblySftAgent(RobotAgent):
    def __init__(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[agent] loading SFT checkpoint {CHECKPOINT}", flush=True)
        policy = PI05Policy.from_pretrained(CHECKPOINT).to(device).eval()
        preprocess, postprocess = make_pre_post_processors(
            policy.config,
            CHECKPOINT,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        policy.config.n_action_steps = 15
        self.model = LeRobotModel(policy, preprocess, postprocess)
        self.adapter = AssemblySftAdapter(
            list(policy.config.image_features),
            int(policy.config.input_features["observation.state"].shape[0]),
        )


Agent = Pi05AssemblySftAgent

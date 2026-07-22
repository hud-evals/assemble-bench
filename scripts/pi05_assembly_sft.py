"""pi0.5 assembly SFT agent (checkpoint via ``CHECKPOINT=...``).

SFT was trained with ``--use_relative_actions``; the saved postprocessor already
runs ``absolute_actions_processor`` (enabled), so ``adapt_chunk`` only slices to
8-D and binarizes the gripper — it must NOT re-add joint position.

State width: ``pi05_base`` declares ``observation.state`` as 32-D (``max_state_dim``),
but assembly_bench SFT norm stats are 8-D (7 joints + gripper). Feed the
**stats** width into the preprocessor — padding to 32 here breaks normalization
(``32 vs 8``). ``pi05_prepare_state_tokenizer`` then embeds those 8 discretized
values into the prompt, matching training.
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
# Open-loop horizon via Adapter.chunk_size (model may still predict 50).
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", os.environ.get("OPEN_LOOP_HORIZON", "5")))
IMAGE_SIZE = 224
ENV_STATE_DIM = 8  # joint_pos(7) + gripper(1)


def _normalizer_state_dim(preprocess) -> int:
    """Active proprio width from saved quantile stats (not config max_state_dim)."""
    for step in getattr(preprocess, "steps", []):
        stats = getattr(step, "_tensor_stats", None) or getattr(step, "stats", None)
        if not stats:
            continue
        state_stats = stats.get("observation.state") if isinstance(stats, dict) else None
        if not state_stats:
            continue
        for key in ("q01", "q99", "mean", "std", "min", "max"):
            t = state_stats.get(key) if isinstance(state_stats, dict) else None
            if t is not None and hasattr(t, "shape") and len(t.shape) >= 1:
                return int(t.shape[-1])
    return ENV_STATE_DIM


class AssemblySftAdapter(Adapter):
    def __init__(
        self, model_image_keys: list[str], state_dim: int, *, chunk_size: int | None = None
    ) -> None:
        super().__init__(model_image_keys=model_image_keys, chunk_size=chunk_size)
        self.state_dim = state_dim

    def adapt_observation(self, obs: dict, prompt: str) -> dict:
        data = obs["data"]
        proprio = np.concatenate(
            [np.asarray(data["policy/joint_pos"], dtype=np.float32).reshape(-1),
             np.asarray(data["policy/gripper_pos"], dtype=np.float32).reshape(-1)]
        )
        state = np.zeros(self.state_dim, dtype=np.float32)
        n = min(proprio.shape[0], self.state_dim)
        state[:n] = proprio[:n]
        batch = {"observation.state": torch.from_numpy(state), "task": prompt}

        # Env has front + wrist only; pi05_base may list a third (right) camera.
        # Missing views → black frames (same as unused padded dims at train time).
        front = data["camera_obs/front_cam_rgb"]
        wrist = data["camera_obs/wrist_camera_rgb"]
        env_cams = {
            "observation.images.base_0_rgb": front,
            "observation.images.left_wrist_0_rgb": wrist,
            "observation.images.right_wrist_0_rgb": None,
            "observation.images.front": front,
            "observation.images.wrist": wrist,
        }
        for key in self.model_image_keys:
            image = env_cams.get(key)
            if image is None:
                image = np.zeros_like(front)
            image = resize_with_pad_torch(
                torch.from_numpy(np.ascontiguousarray(image)), IMAGE_SIZE, IMAGE_SIZE
            )
            batch[key] = image.reshape(IMAGE_SIZE, IMAGE_SIZE, 3).permute(2, 0, 1).float() / 255.0
        return batch

    def adapt_chunk(self, chunk: np.ndarray, obs: dict) -> np.ndarray:
        # Absolute joints already; slice to 8-D env action + binarize gripper.
        actions = np.asarray(chunk[:, :ENV_STATE_DIM], dtype=np.float32)
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
        policy.config.n_action_steps = CHUNK_SIZE  # LeRobot select_action path only
        state_dim = _normalizer_state_dim(preprocess)
        print(
            f"[agent] state_dim={state_dim} (cfg max={policy.config.input_features['observation.state'].shape[0]}) "
            f"images={list(policy.config.image_features)} chunk_size={CHUNK_SIZE}",
            flush=True,
        )
        self.model = LeRobotModel(policy, preprocess, postprocess)
        self.adapter = AssemblySftAdapter(
            list(policy.config.image_features),
            state_dim,
            chunk_size=CHUNK_SIZE,
        )


Agent = Pi05AssemblySftAgent

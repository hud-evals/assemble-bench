"""Cosmos3-Policy-DROID agent for the assembly benchmark.

Talks to a cosmos-framework OpenPI WebSocket policy server
(``python -m cosmos_framework.scripts.action_policy_server_robolab --port 8000``)
via :class:`~hud.agents.robot.RemoteModel` with ``response_key="action"``.

Packing mirrors RoboLab's ``Cosmos3Client``: wrist + half-res left/right
exteriors concatenated into one ``observation/image`` (540×640). assembly_bench
only has front + wrist, so the front cam is duplicated into both exterior
tiles (same trick as zero-filling a missing wrist on pi0.5).

Env vars:
  COSMOS_HOST / COSMOS_PORT — policy server (default localhost:8000)
  COSMOS_CHUNK_SIZE — open-loop horizon (default 32)
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from openpi_client import image_tools

from hud.agents.robot import Adapter, RemoteModel, RobotAgent
from hud.agents.robot.adapter import ActionArray

IMAGE_W = 640
IMAGE_H = 360  # per-view resize before concat (final frame is 540×640)
OPEN_LOOP_HORIZON = 32
ACTION_DIM = 8


def _as_hwc_uint8(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def pack_cosmos_image(wrist: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compose RoboLab/Cosmos concat frame: wrist on top, left|right half below."""
    wrist = image_tools.resize_with_pad(wrist, IMAGE_H, IMAGE_W)
    left = image_tools.resize_with_pad(left, IMAGE_H, IMAGE_W)
    right = image_tools.resize_with_pad(right, IMAGE_H, IMAGE_W)
    size = (IMAGE_H // 2, IMAGE_W // 2)

    def _half(img: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(np.array(img, copy=True)).permute(2, 0, 1).unsqueeze(0).float()
        t = F.interpolate(t, size=size, mode="bilinear")
        return t.squeeze(0).permute(1, 2, 0).numpy().astype(wrist.dtype)

    return np.concatenate((wrist, np.concatenate((_half(left), _half(right)), axis=1)))


class CosmosDroidAdapter(Adapter):
    """assembly_bench obs → Cosmos RoboLab request; absolute 8-D actions out."""

    def __init__(self, *, chunk_size: int = OPEN_LOOP_HORIZON) -> None:
        super().__init__(chunk_size=chunk_size)

    def adapt_observation(self, obs: dict[str, Any], prompt: str) -> dict[str, Any]:
        data = obs["data"]
        front = _as_hwc_uint8(data["camera_obs/front_cam_rgb"])
        wrist = _as_hwc_uint8(data["camera_obs/wrist_camera_rgb"])
        # No dual over-shoulder cams — duplicate the frontal exterior.
        image = pack_cosmos_image(wrist=wrist, left=front, right=front)
        joints = np.asarray(data["policy/joint_pos"], dtype=np.float32).reshape(-1)
        grip = np.asarray(data["policy/gripper_pos"], dtype=np.float32).reshape(-1)
        return {
            "observation/image": image,
            "observation/joint_position": joints[:7],
            "observation/gripper_position": grip[:1],
            "prompt": prompt,
        }

    def adapt_chunk(self, chunk: ActionArray, obs: dict[str, Any]) -> ActionArray:
        out = np.asarray(chunk, dtype=np.float32)[..., :ACTION_DIM].copy()
        out[..., -1] = (out[..., -1] > 0.5).astype(np.float32)
        return out

    def adapt_action(self, action: ActionArray, obs: dict[str, Any]) -> ActionArray:
        action = np.asarray(action, dtype=np.float32)
        if action.shape[-1] > ACTION_DIM:
            action = action[..., :ACTION_DIM]
        out = action.copy()
        out[..., -1] = (out[..., -1] > 0.5).astype(np.float32)
        return out


class CosmosDroidAgent(RobotAgent):
    def __init__(self) -> None:
        host = os.environ.get("COSMOS_HOST", "localhost")
        port = int(os.environ.get("COSMOS_PORT", "8000"))
        chunk = int(os.environ.get("COSMOS_CHUNK_SIZE", str(OPEN_LOOP_HORIZON)))
        self.model = RemoteModel(host=host, port=port, response_key="action")
        self.adapter = CosmosDroidAdapter(chunk_size=chunk)


Agent = CosmosDroidAgent

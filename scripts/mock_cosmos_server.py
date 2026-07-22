"""Minimal OpenPI-protocol mock of the Cosmos RoboLab policy server.

Validates the request shape from ``cosmos_droid.py`` and returns a hold-pose
chunk (current joints + open gripper) so the assembly env can be smoked without
loading the 16B Cosmos weights.

    conda run -n vla python scripts/mock_cosmos_server.py --port 8000
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
from openpi_client import base_policy, msgpack_numpy
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve


log = logging.getLogger("mock_cosmos")


class HoldPosePolicy(base_policy.BasePolicy):
    """Echo current joint state for ``horizon`` steps; gripper forced open."""

    def __init__(self, horizon: int = 32) -> None:
        self.horizon = horizon

    def infer(self, obs: dict) -> dict:  # type: ignore[override]
        image = np.asarray(obs["observation/image"])
        joints = np.asarray(obs["observation/joint_position"], dtype=np.float32).reshape(-1)
        grip = np.asarray(obs["observation/gripper_position"], dtype=np.float32).reshape(-1)
        prompt = obs.get("prompt", "")
        assert image.shape == (540, 640, 3), f"bad image shape {image.shape}"
        assert image.dtype == np.uint8, f"bad image dtype {image.dtype}"
        assert joints.shape == (7,), f"bad joints {joints.shape}"
        assert grip.shape == (1,), f"bad grip {grip.shape}"
        assert isinstance(prompt, str) and prompt, "missing prompt"
        row = np.concatenate([joints, np.array([0.0], dtype=np.float32)])  # open
        action = np.tile(row[None, :], (self.horizon, 1))
        log.info("infer ok prompt=%r joints=%s", prompt[:60], np.round(joints, 3).tolist())
        return {"action": action}

    def reset(self) -> None:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--horizon", type=int, default=32)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    policy = HoldPosePolicy(horizon=a.horizon)
    metadata = {"mock": "cosmos_droid", "horizon": a.horizon}
    packer = msgpack_numpy.Packer()

    def handler(websocket) -> None:  # noqa: ANN001
        websocket.send(packer.pack(metadata))
        while True:
            try:
                raw = websocket.recv()
            except ConnectionClosed:
                break
            obs = msgpack_numpy.unpackb(raw)
            websocket.send(packer.pack(policy.infer(obs)))

    log.info("mock Cosmos server on ws://%s:%d", a.host, a.port)
    with serve(handler, a.host, a.port, compression=None, max_size=None) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()

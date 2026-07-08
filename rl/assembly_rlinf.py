# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RLinf IsaacLab task wrapper for the assembly_bench (Arena) peg env.

The one required extension point that lets RLinf's PPO trainer drive the Arena
env: it reuses ``assembly_bench``'s own ``make_assembly_env`` factory (turning
the staged reward on) and repacks Arena's observation into RLinf's contract.
No task logic here -- that all lives in ``assembly_bench``.

Install into an RLinf checkout with ``rl/install_into_rlinf.sh`` (copies this to
``rlinf/envs/isaaclab/tasks/`` and registers the id in the package ``__init__``).

Obs mapping (Arena groups -> RLinf keys), grounded in ``contract.json``:
  states       = cat(policy/joint_pos[7], policy/gripper_pos[1])  # pi0.5-DROID proprio
  main_images  = camera_obs/front_cam_rgb
  wrist_images = camera_obs/wrist_camera_rgb
"""

import torch

from ..isaaclab_env import IsaaclabBaseEnv


class IsaaclabAssemblyBenchEnv(IsaaclabBaseEnv):
    """assembly_bench peg pick->insert (Franka + Robotiq, DROID, droid_jointpos)."""

    def _make_env_function(self):
        def make_env_isaaclab():
            import os

            os.environ.pop("DISPLAY", None)  # force headless; stale DISPLAY -> GLX errors

            from isaaclab.app import AppLauncher

            sim_app = AppLauncher(headless=True, enable_cameras=True).app

            # Import only after the app is up (pulls in isaaclab_arena).
            from assembly_bench.environments.assembly.assembly import make_assembly_env

            p = self.cfg.init_params
            env = make_assembly_env(task=p.task, num_envs=p.num_envs, reward="staged")
            return env.unwrapped, sim_app

        return make_env_isaaclab

    def _wrap_obs(self, obs):
        policy = obs["policy"]
        cams = obs["camera_obs"]
        states = torch.cat([policy["joint_pos"], policy["gripper_pos"]], dim=1).float()
        return {
            "main_images": cams["front_cam_rgb"],
            "wrist_images": cams["wrist_camera_rgb"],
            "states": states,
            "task_descriptions": [self.task_description] * self.num_envs,
        }

    # NOTE: no _record_metrics override -- env/success_once will over-report
    # (the staged reward is positive on a mere grasp, and part_dropped also
    # terminates). Fix when reading that metric: derive success from the named
    # `success` termination term, i.e.
    #   self.success_once |= env.termination_manager.get_term("success")

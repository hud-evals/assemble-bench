"""Teleop demo recording for assembly_bench = Arena's record_demos + canonical labels.

Arena's ``record_demos.py`` replaces ``env_cfg.recorders`` wholesale, which would
drop any benchmark recorder terms -- so this wrapper imports the upstream script
as a module (its arg parsing + AppLauncher run at import, unmodified) and patches
``create_environment_config`` to append ``PostStepAbsJointTargetRecorder``: the
8-D absolute-joint-target label stream (``abs_joint_action``) recorded alongside
the raw differential-IK ``actions``. See environments/assembly/recorders.py for
why both streams exist.

Defaults injected when absent: ``--teleop_device keyboard``,
``--num_success_steps 1`` (the task already debounces success for 3 steps;
Arena's default 10 would double-debounce).

Run (isaac env, repo root; browser client needs TCP 49100 + UDP 47998 open):

    PUBLIC_IP=$(curl -s ifconfig.me) OMNI_KIT_ACCEPT_EULA=YES \
    python scripts/teleop/record_demos.py \
      --livestream 1 --num_envs 1 --step_hz 15 \
      --dataset_file data/teleop/peg_round_8mm_dik_s001.hdf5 \
      --num_demos 10 \
      --external_environment_class_path \
        assembly_bench.environments.assembly.assembly:AssemblyBenchEnvironment \
      assembly_bench --task peg_round_8mm \
      --embodiment droid_differential_ik --teleop_device keyboard

Keyboard (click the streamed viewport first): W/S A/D Q/E translate,
Z/X T/G C/V rotate, K gripper toggle, R discard + reset.
"""

import sys

if "--teleop_device" not in sys.argv:
    sys.argv += ["--teleop_device", "keyboard"]
    print("[teleop] defaulting --teleop_device keyboard")
if "--num_success_steps" not in sys.argv:
    sys.argv += ["--num_success_steps", "1"]
    print("[teleop] defaulting --num_success_steps 1 (task already debounces 3 steps)")

# Parses CLI and launches the Isaac app at import time (module-level code).
from isaaclab_arena.scripts.imitation_learning import record_demos  # noqa: E402

# Isaac Lab imports are only safe after the app is up (the import above did that).
from assembly_bench.environments.assembly.recorders import PostStepAbsJointTargetRecorderCfg  # noqa: E402

_upstream_create_environment_config = record_demos.create_environment_config


def _create_environment_config_with_labels(output_dir: str, output_file_name: str):
    env_cfg, env_name, env_kwargs, success_term = _upstream_create_environment_config(output_dir, output_file_name)
    # Upstream just replaced env_cfg.recorders; append the canonical label term.
    env_cfg.recorders.record_post_step_abs_joint_targets = PostStepAbsJointTargetRecorderCfg()
    print("[teleop] recording abs_joint_action labels (8-D absolute joint targets)")
    return env_cfg, env_name, env_kwargs, success_term


record_demos.create_environment_config = _create_environment_config_with_labels

if __name__ == "__main__":
    record_demos.main()
    record_demos.simulation_app.close()

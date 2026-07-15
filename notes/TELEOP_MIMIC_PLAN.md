# Teleoperation → Mimic Demonstration Pipeline — Plan

Plan for generating human demonstrations on the assembly benchmark via keyboard
teleoperation streamed over Isaac Sim WebRTC, then scaling with Isaac Lab Mimic
(MimicGen-style generation). Milestones are gated: do not advance until the
acceptance criteria of the current stage pass.

Target architecture:

```
React WebRTC viewer -> Kit keyboard events -> Se3Keyboard -> differential IK
  -> Arena recorder -> HDF5 -> annotate -> Isaac Lab Mimic -> replay validation
```

Reuse (already present in the pinned IsaacLab-Arena checkout):

- `isaaclab_arena/scripts/imitation_learning/record_demos.py` — teleop + success-only HDF5 export
- `isaaclab_arena/scripts/imitation_learning/{replay_demos,annotate_demos,generate_dataset,merge_demos}.py`
- `DroidDifferentialIKKeyboardRetargeter` (`isaaclab_arena/assets/retargeter_library.py`) — keyboard is already registered against `droid_differential_ik`
- Isaac Lab `AppLauncher --livestream 1|2` (WebRTC; forwards browser keyboard/mouse into Kit)
- NVIDIA web client: `@nvidia/create-ov-web-rtc-app` scaffold + `@nvidia/ov-web-rtc` (`AppStreamer`), React components included

Known gaps in this checkout (must be filled locally):

- `AssemblyBenchEnvironment.add_cli_args()` has no `--teleop_device`, and `get_env()` passes no `teleop_device=` to `IsaacLabArenaEnvironment`
- `NISTAssemblyTask.get_mimic_env_cfg()` raises `NotImplementedError` (tasks.py ~L299)
- `DroidEmbodimentBase.mimic_env = None` — no DROID `ManagerBasedRLMimicEnv`
- No retargeter for `keyboard + droid_abs_joint_pos` (keyboard emits SE(3), not joint targets)

---

## 0. Decide the action schema (before collecting data)

Keyboard teleop through `droid_differential_ik` produces:

- 6 relative end-effector values
- 1 gripper value

The benchmark's existing policy contract (pi0.5-DROID, `contract.json`) uses:

- 7 absolute joint targets
- 1 gripper value

For the MVP, use differential IK and clearly label datasets `*_dik_*`. Before
large-scale collection, decide whether to:

1. Train on DIK actions, or
2. Record/project the controller's resulting joint targets into the canonical
   8-D DROID format.

Do not silently mix these schemas. A later alternative is a
keyboard → IK → absolute-joint bridge based on the existing scripted expert's
`Servo` (`scripts/experts/base.py`), but it adds more work.

## 1. Wire and prove teleoperation

Small required changes:

- Add `--teleop_device` to `AssemblyBenchEnvironment.add_cli_args()`.
- Resolve the selected device through Arena's `DeviceRegistry`.
- Pass it as `teleop_device=` to `IsaacLabArenaEnvironment`.
- Reject incompatible combinations such as keyboard + `droid_abs_joint_pos`.

Arena already registers keyboard against `droid_differential_ik`
(`retargeter_library.py`, `DroidDifferentialIKKeyboardRetargeter`).

First run teleop without recording:

- One environment
- 15 Hz
- Loose 8 mm round peg
- `--livestream 2`
- Stock NVIDIA browser viewer (or the desktop WebRTC client)
- Click the viewport before using keyboard controls

Acceptance criteria:

- All translation and rotation keys work.
- Gripper toggle (`K`) works.
- Reset/discard (`R`) works remotely.
- Simulation remains near real time for 10 minutes.
- You can complete at least one loose insertion.
- No stuck keyboard state after browser focus changes.

WebRTC streaming is independent of camera observations, so omit
`--enable_cameras` initially.

## 2. Record source demonstrations

After the smoke test, use Arena's recorder approximately as follows:

```bash
python isaaclab_arena/scripts/imitation_learning/record_demos.py \
  --livestream 2 \
  --device cuda:0 \
  --num_envs 1 \
  --step_hz 15 \
  --dataset_file datasets/teleop/peg_round_8mm_loose_dik_session_001.hdf5 \
  --num_demos 10 \
  --num_success_steps 1 \
  --external_environment_class_path \
  assembly_bench.environments.assembly.assembly:AssemblyBenchEnvironment \
  assembly_bench \
  --task peg_round_8mm_loose \
  --embodiment droid_differential_ik \
  --teleop_device keyboard
```

`--num_success_steps 1` is recommended because the benchmark already debounces
success for three consecutive steps (`SUCCESS_HOLD_STEPS = 3` in `tasks.py`).
A larger value adds a second debounce.

Collection practices:

- One HDF5 per session.
- Record 10–20 successes, but discard awkward or accidental recoveries.
- First collect state/action only.
- Merge sessions using Arena's `merge_demos.py`.
- Replay every source demonstration (`replay_demos.py`) before annotation.
- Record seeds, task variant, action schema and control rate in session metadata.

Only enable cameras once action recording is reliable. Native 1280×720
observations make datasets large.

## 3. Implement the Mimic contract

Two pieces are needed.

### DROID differential-IK Mimic environment

Implement the `ManagerBasedRLMimicEnv` methods for:

- Reading the DROID end-effector pose (`get_robot_eef_pose`)
- Converting relative actions to target end-effector poses (`action_to_target_eef_pose`)
- Converting target poses back to DIK actions (`target_eef_pose_to_action`)
- Extracting gripper actions (`actions_to_gripper_actions`)
- Reading object poses (`get_object_poses`)
- Later, producing automatic subtask signals (`get_subtask_term_signals`)

The existing `FrankaMimicEnv` (`isaaclab_arena/embodiments/franka/franka.py`)
is a useful reference, but do not directly alias it: its action-noise signature
is already marked deprecated, and the DROID action scaling/frame conventions
should be explicit.

Attach this Mimic class only to `DroidDifferentialIKEmbodiment`, not the DROID
base class.

### NIST assembly Mimic configuration

Replace the `NotImplementedError` in `NISTAssemblyTask.get_mimic_env_cfg()`.
Begin with three subtasks:

1. Reach and grasp
   - `object_ref="held_part"`
   - termination signal: `grasp_complete`

2. Transport and pre-align
   - `object_ref="fixed_part"`
   - termination signal: `prealign_complete`

3. Seat
   - `object_ref="fixed_part"`
   - final subtask, therefore no termination signal

Start with:

- Nearest-neighbor object selection (`nearest_neighbor_object`, `nn_k=3`)
- 3–5 interpolation steps
- Very low action noise during transport
- Zero action noise during seating
- Tight annotation-offset ranges at 15 Hz

Do not reuse one configuration unchanged across pegs, gears and nuts.

## 4. Manually annotate first

Manual annotation (`annotate_demos.py` without `--auto`) validates whether the
subtask decomposition is meaningful before writing detectors.

For each demonstration:

- Mark grasp only once the part is securely moving with the gripper.
- Mark pre-alignment once the held part is above and aligned with the target.
- Leave insertion/seating as the final segment.
- Replay the annotated episode and visually inspect every boundary.

Acceptance criteria:

- Every demonstration has exactly two ordered boundaries.
- Boundaries do not occur during failed grasps.
- Each segment is long enough to contain useful motion.
- Replaying the original actions still reaches benchmark success.

## 5. Small Mimic generation experiment

Do not immediately generate hundreds.

First experiment (`generate_dataset.py --mimic`):

- Loose round peg
- 4 parallel environments
- 20 requested trials
- No cameras
- Keep failed episodes temporarily for diagnosis (`generation_keep_failed`)
- Narrow workspace randomization
- Zero seating noise

Evaluate:

- Overall success rate
- Failure stage: grasp, transport, alignment or seating
- Contact/jam frequency
- Number of retries per accepted demonstration
- Diversity in initial poses and trajectories

Suggested gates:

- Source replay: effectively 100%
- Initial generated baseline: at least 70%
- Worth scaling: approximately 85% or better
- If failures predominantly occur after contact, stop tuning open-loop Mimic
  and move to the hybrid approach (below).

The existing closed-loop expert is already ~89% on round pegs
(`notes/EXPERT_PROGRESS.md`), so Mimic must provide meaningful diversity or
comparable throughput to justify itself.

Hybrid option (best throughput): use human demos + Mimic for reach / grasp /
transport / pre-alignment diversity, and the closed-loop scripted expert for
the contact-rich seating phase — or keep the expert as bulk generator and treat
Mimic as a measured augmentation baseline. The expert's own handoff notes
correctly flag open-loop replay as weaker under peg/bore contact.

## 6. Add automatic signals

Once manual boundaries look consistent, implement `get_subtask_term_signals`.

### `grasp_complete`

Require more than the gripper command:

- Gripper closed
- Held part lifted or displaced from its initial pose
- Held part remains near the end effector
- Condition holds for several steps
- Latch true for the rest of the episode

### `prealign_complete`

Use variant-aware geometry (reuse the seat-target math from `part_seated`):

- XY error within a pre-alignment tolerance
- Held part above the seating target
- Low enough orientation error
- For square pegs, yaw modulo π
- Part remains grasped
- Latch once true

Mimic expects clean false → true boundaries, so these signals should be
monotonic per episode.

## 7. Scale and add cameras

After state-only generation succeeds:

- Enable camera observations.
- Recheck real-time performance and storage.
- Start with 8 parallel environments; scale based on GPU memory and camera cost.
- Replay generated output through the exact `part_seated`/`hold_success`
  predicate.
- Export successes only for training, but retain a diagnostic failed set
  separately.
- Verify image/action alignment at 15 Hz before converting to LeRobot or
  another training format.

## 8. React shell last

The first custom React UI (scaffolded via
`npx @nvidia/create-ov-web-rtc-app ... sample-app --streamSource=local --framework=react`)
should remain small:

- Connect/disconnect
- Stream state and WebRTC statistics
- Current task and demonstration count
- Start/pause/reset/discard controls
- Keyboard help and focus indicator

Standard keyboard movement continues through WebRTC input forwarding
(`omni.kit.livestream.webrtc` InputHandler). Use custom WebRTC messages
(`omni.kit.livestream.messaging` / `AppStreamer.sendMessage`) only for explicit
recorder commands and status — not for every robot action. The simulation
process remains the single owner of stepping, success detection, resets, and
HDF5 writes.

Keep it on a trusted network: Isaac Sim's direct streaming endpoint has no
built-in authentication or encryption (ports: TCP 49100 signaling,
UDP 47998 media).

## Suggested schedule

| Day | Milestone |
|-----|-----------|
| 1 | Device wiring, WebRTC teleop, one successful loose insertion |
| 2 | Recorder, replay, 10–20 source demonstrations |
| 3–4 | DROID Mimic environment and task Mimic configuration |
| 5 | Manual annotation and 20-trial generation experiment |
| 6+ | Automatic boundaries, camera datasets, React UI, hybrid contact handling |

First real checkpoint: **one remotely controlled loose-peg success is recorded
to an Arena HDF5 and replays successfully.** Everything else waits until that
is proven.

# Assembly-bench demonstration data: recording spec

Everything another agent needs to reproduce the demonstration datasets: the
pipeline, how to run it, the exact fields/shapes/conventions, and the non-obvious
fixes that make the data clean. Contact-rich NIST-style assembly (peg insert,
gear mesh, nut thread) driven by a privileged scripted expert in Isaac Lab
(Arena), on the **DROID** embodiment (Franka Panda + Robotiq 2F-85).

## Pipeline

```
run_expert.py  (Isaac Sim, GPU)  ->  per-task HDF5  ->  convert_lerobot.py (vla env)  ->  LeRobot v3.0  ->  HF Hub
   success-filtered demos            data/hdf5/*.hdf5      merge + H.264 video           optional push
```

- **Record** with `scripts/experts/run_expert.py` (needs Isaac Sim + the env; run
  in the `hud-assembly-env` Docker container, interpreter `/isaac-sim/python.sh`).
- **Convert** with `scripts/experts/util/convert_lerobot.py` (needs `lerobot`+`h5py`,
  run in the `vla` conda env; no Isaac).

## How to record

```bash
# inside the hud-assembly-env container, WORKDIR /app/assembly_bench
/isaac-sim/python.sh scripts/experts/run_expert.py \
  --headless --task peg_round_M1_loose \
  --num_envs 8 --waves 40 --max_demos 50 \
  --record data/hdf5/peg_round_M1_loose.hdf5
```

- `--task`: any key in `environments/assembly/variants.py::VARIANTS`
  (`peg_{round,square}_{S,M1,M2,L}_loose`, `gear_{small,medium,large}`,
  `nut_M{8,12,16,20}`, plus `debug`).
- `--num_envs`: parallel envs (all recorded). `--waves`: reset cycles (each
  re-randomizes). `--max_demos N`: stop once N successful demos are banked.
- Only **successful** episodes are written; the recorder **appends** across waves,
  and refuses to mix tasks in one file. Delete the file to start fresh.
- Control/record rate is **15 Hz** (120 Hz sim, `decimation` 4/render_interval).

## HDF5 layout (one file per task)

```
/data                         group   attrs: task, language, source="scripted_privileged", num_demos
  demo_0                       group   attrs: success=True
    obs/
      wrist_rgb   (T,180,320,3) uint8   eye-in-hand RGB
      front_rgb   (T,180,320,3) uint8   third-person RGB
      state       (T,8)  float32
      joint_vel   (T,7)  float32
      eef_pos     (T,3)  float32
      eef_quat    (T,4)  float32
    action        (T,8)  float32
  demo_1 ...
```

`T` = per-episode length (~300–340 frames, ~20 s at 15 Hz).

## Fields & conventions (READ THIS)

| Field | Shape | Units / convention |
|-------|-------|--------------------|
| `state` | 8 | `[joint_pos(7 rad), gripper_pos(1)]` — the pi0.5-DROID proprio contract. gripper 0=open … 1=closed (`finger_joint/(pi/4)`). |
| `joint_vel` | 7 | Panda arm joint angular velocity (rad/s). **No gripper term.** |
| `eef_pos` | 3 | End-effector (`base_link`) position (m), **ROBOT BASE frame** (env-local; the recorder subtracts `env_origins` — do NOT ship raw `body_pos_w`, it carries the per-env grid offset). |
| `eef_quat` | 4 | End-effector orientation, **world frame, `wxyz`** order (offset-invariant). |
| `action` | 8 | `[7 absolute arm joint-position targets (rad), gripper]`. The **native DROID 8-D joint command** the IK expert emitted (recorded, not re-derived). gripper 1=close, 0=open. |
| images | 180×320×3 | uint8 RGB, **16:9**, rendered at 640×360 and area-downscaled to 320×180 (DROID-RLDS resolution). Two views: `wrist` (eye-in-hand) + `front` (agentview). |

- **Rate:** 15 Hz. **Robot:** Franka Panda + Robotiq 2F-85 (DROID). **Action space:**
  absolute joint position (`droid_jointpos`); differential-IK expert → stiff PD.
- **Quaternion order is `wxyz`** for `eef_quat` (world). The IsaacLab math/root
  state stack is otherwise `xyzw` — the recorded `eef_quat` is `wxyz` per the DROID
  obs helper; validate before consuming.
- **Resolution rationale:** models resize/pad themselves (pi0.5 `resize_with_pad`→224,
  GR00T pad→256→224), so store the model-agnostic 320×180. Never ship native 1280×720
  (16× the pixels, huge HDF5s) or a square squash (distorts 16:9).

## LeRobot dataset (after conversion)

```bash
# in the vla env
conda run -n vla python scripts/experts/util/convert_lerobot.py \
  --glob 'data/hdf5/*.hdf5' --repo_id <user>/<name> [--push]
```

Merges all task HDF5s into ONE multi-task LeRobot v3.0 dataset, keyed by each
episode's instruction. Features:

| key | dtype | shape |
|-----|-------|-------|
| `observation.images.front` | video (H.264, CRF 20) | (180,320,3) |
| `observation.images.wrist` | video (H.264, CRF 20) | (180,320,3) |
| `observation.state` | float32 | (8,) |
| `observation.joint_vel` | float32 | (7,) |
| `observation.eef_pos` | float32 | (3,) |
| `observation.eef_quat` | float32 | (4,) |
| `action` | float32 | (8,) |

fps 15. Task string = per-episode instruction (e.g. "pick up the 8 mm round peg
and insert it into the hole"); LeRobot dedups identical strings into `total_tasks`.

## Non-obvious things that make the data clean (keep these)

- **Release to seat.** The expert opens the gripper at the end of insert and lets
  the part settle; a gripper-held part hangs proud and never satisfies the seat check.
- **Rectangular pegs: clock the wrist yaw THROUGH the insert press**, not just align
  — the part slips in-grip once clocking stops (yaw drifts 0.5°→5-8°) and jams.
- **Teleport ghosting:** per-wave reset teleports parts, but DLAA keeps a temporal
  accumulation buffer the reset doesn't clear → a ghost of the previous episode
  survives the warmup. Fixed in the reset warmup (`tasks.py::settle_and_render`) by
  toggling AA mode (FXAA→DLAA) to hard-reset the buffer. Check wave-1+ first frames.
- **eef_pos base frame** (see table) — subtract `env_origins` at record time.
- **Success measured pre-reset**, episode length exceeds the scripted sequence, and
  per-env randomization (arm start, in-grip pose, part pose+yaw, phase timing).

## Throughput & parallelism (measured)

- **Render resolution is the main lever, not env count.** Recording is dominated by
  the RTX cameras + the 192-iter contact solver — both scale with env count, so
  adding envs gives little/no throughput win. Dropping the render 1280x720 -> 640x360
  (we store 320x180 anyway) is the real speedup and needs no quality revalidation.
- **`--num_envs 8` is the sweet spot** (on an RTX 6000 Ada, 46 GB): ~9.7 GB, fast
  steps, 2x demos/wave vs 4. **16 regressed** — at 1280x720 it hung (render/GPU
  saturation, util->1%); at 640x360 it ran but each step was ~10x slower (many envs
  pressing pegs at once overwhelms the contact solver), so wall-clock was worse.
- **Warm caches matter.** First boot ~7 min (cold Warp/PhysX/shader compile); warm
  boot ~2-4 min. Reuse the same container so caches stay warm; boot is paid once per
  task, not per wave.
- **~50 demos come from ONE boot per task** (one process, ~7-15 waves), not a reboot
  per demo. Re-boots are only the task switches.

## Env notes

- Conda: record in the container (`/isaac-sim/python.sh`); convert in `vla`.
- The GPU is the shared resource — one Isaac process at a time (two contend/hang).

# Gear Expert Handoff

## Goal

Make the privileged scripted policy in `scripts/experts/gear.py` reliably pick,
transport, and mesh all three gear sizes before collecting demonstrations.

The benchmark success check is intentionally strict:

- Gear center within 2.5 mm of the shaft center.
- Gear root less than 3.0 mm above the seat target.
- Linear speed below 0.05 m/s for three consecutive control steps.

## Repository state

- Repository: `/home/ubuntu/project/assembly_bench`
- Branch: `cursor/gear-expert-reliability-cd89`
- Base commit: `e0ad315`
- Modified file: `scripts/experts/gear.py`
- Handoff file: `gear.md`
- Changes are uncommitted.

Run `git diff -- scripts/experts/gear.py` to inspect the active policy changes.

## HUD configuration

Policy runs on this machine must use:

```bash
export HUD_API_KEY=sk-hud-dev-test-key
export HUD_API_URL=http://127.0.0.1:18000
export HUD_TELEMETRY_URL=http://127.0.0.1:18000/v2/telemetry
export HUD_WEB_URL=http://localhost:3000
```

Verify the SSH tunnel before a streamed run:

```bash
curl http://127.0.0.1:18000/health/readiness
```

Expected response:

```json
{"status":"ok"}
```

The tunnel has dropped several times during this work. A refused connection is
a tunnel failure, not a policy or HUD SDK failure.

The `hud-assembly-env` image does not include PyAV. Streamed runs currently
install it at container startup:

```bash
/isaac-sim/python.sh -m pip install -q av
```

## Initial baseline

Four parallel streamed runs were evaluated for every size before policy changes:

- Small: 0/4
- Medium: 0/4
- Large: 0/4

Jobs:

- Small: `http://localhost:3000/jobs/041d8ad9-2390-4445-9e5f-f67f7eb089eb`
- Medium: `http://localhost:3000/jobs/7c7b13e8-39ac-44e9-ad6f-32b5c4986c12`
- Large: `http://localhost:3000/jobs/81432022-ebf9-4036-a3e2-f16d03b64edf`

All traces and telemetry uploads completed. The failures were physical policy
failures, not platform failures.

## Root cause of the 0/4 baseline

The original mesh phase treated a gear as engaged once its root gap fell below
12 mm. A clashing gear naturally rests about 9-10 mm above the seat, so the
policy locked the first bad yaw phase and released the gear at the clash height.

Task success requires a gap below 3 mm. The policy and task therefore disagreed
about when mesh search was complete.

The active policy now:

- Uses a 9 mm yaw-lock gap for small/medium and 5 mm for large.
- Waits for the environment's debounced success signal before leaving mesh.
- Applies 3 mm of gentle preload during search.
- Applies 5 mm of preload only after deep engagement.
- Retries once by lifting, realigning, recapturing the in-hand offset, and
  running a second mesh search.
- Holds a deeply engaged large gear in place to settle instead of lifting it.
- Uses 1.5x wrist-yaw overtravel for small so the gear itself covers one full
  tooth pitch despite grip compliance.

These changes raised all three sizes from 0/4 to 8/8.

## Small-gear pickup failures

The first improved small run reached 2/4. Visual review found two pickup
failures:

1. A correctly grasped small gear slipped during transport:
   `http://localhost:3000/trace/c091e97a-981d-4675-bc27-b8fb948a243b`
2. The gripper oscillated during initial close, displaced the gear, and lifted
   it at an angle:
   `http://localhost:3000/trace/30c56131-627d-4ec3-8f0a-e26c15161b5b`

The original small-gear pinch was only 7 mm above the bottom of the 25 mm gear
body. The active policy raises only the small-gear grasp by 4 mm:

```python
GRASP_ROOT_Z = {"small": 0.016, "medium": 0.012, "large": 0.012}
```

This clears the tabletop and holds closer to the small gear's center of mass.
The first local validation with this change improved small from 2/4 to 3/4.

## Experiments that should not be repeated

### Lower gripper effort

The small gear was tested with:

```text
--grip_effort 1.5
```

Result: 1/4. The lower effort reduced close force but made transport and mesh
slip worse. Keep the default firm gripper drive.

### One-millimeter mesh preload

Reducing `PRESS_BIAS` from 3 mm to 1 mm produced only 1/4 on medium. It did not
develop enough contact force to finish seating. Keep 3 mm during search.

### Short mesh timeouts

Reducing first search, realignment, and retry budgets to 160/70/180 steps
produced only 1/4 on small. Restore the active 220/100/240 budgets.

Shorter policy phase timeouts also do not shorten a failed trace by themselves.
`Machine.failed` enters the terminal hold phase, but the environment does not
truncate until its 52-second episode timeout at step 779.

### Small best-gap yaw locking

Storing the yaw associated with the lowest observed gap and locking it after
two scan cycles produced 1/8. Contact response lags the commanded wrist yaw, so
the stored command is not the gear pose that caused the measured gap.

### Small 9.5 mm lock threshold

Raising the small lock threshold from 9.0 to 9.5 mm produced 2/8 on seed 1.
It locks on the ordinary tooth-clash shelf before a valid phase is found. Keep
the small threshold at 9.0 mm.

## Visually seated false negative

Trace:

```text
http://localhost:3000/trace/2294f41d-09da-4433-8453-dd691e88276c
```

The gear looked seated but recorded:

```text
xy error: 0.3 mm
seat gap: 3.2 mm
speed: approximately 0.03 m/s
```

The task requires a gap strictly below 3.0 mm, so this was a real 0.2 mm depth
miss rather than a labeling bug. The active policy uses a stronger 5 mm final
preload after engagement to finish these near-seated cases without changing the
benchmark tolerance.

## Current policy changes

The active uncommitted policy includes:

1. Small-only grasp height raised from 12 mm to 16 mm above the gear root.
2. Safe transport height reduced from 75 mm to 65 mm.
3. Hover offset reduced from 55 mm to 45 mm.
4. Hover randomization reduced from 0-25 mm to 0-10 mm.
5. Grasp dwell randomization reduced from 16-30 to 14-20 control steps.
6. Pre-mesh timeout guards reduced where the normal motion already completes
   well inside the old limits.
7. Yaw lock threshold made size-specific: 9 mm for small/medium, 5 mm for large.
8. Mesh completion changed from the raw 12 mm engagement check to confirmed
   environment success.
9. Final engaged preload increased from 3 mm to 5 mm.
10. One lift, realign, and remesh retry added with a fresh in-hand offset.
11. Small wrist-yaw search expanded by 1.5x to compensate for in-grip
    attenuation.
12. Large remesh timeout raised from 240 to 300 steps.
13. Large gears below a 6 mm gap hold pose during recovery so contact can
    settle instead of being pulled back off the shaft.
14. Final pressure remains enabled after deep yaw lock even if contact
    compression changes the original in-hand offset.

## Validation history

### Medium

- Original: 0/4
- Lock threshold plus true-seat completion: 3/4
- Added realign/remesh retry: 4/4
- Final combined policy: 8/8
- The final 8-env wave completed around step 525.
- Seed 1: 7/8
- Two-seed total: 15/16 (94%)

### Small

- Original: 0/4
- Mesh fix and retry: 2/4
- Higher grasp: 3/4 local
- Confirmed-seat streamed run: 3/4
- Over-accelerated search-budget experiment: 1/4, reverted
- Balanced streamed candidate before yaw overtravel:
  `http://localhost:3000/jobs/231456f8-af68-4a7a-92a0-516d675e7ed6`

The balanced candidate finished 3/4. The failed trace was:

```text
http://localhost:3000/trace/48d1e824-b474-4f18-ace4-8c43af20c672
```

It remained well aligned at 0.4-0.6 mm but stalled 7.7-9.6 mm above the seat,
which is the tooth-clash height. Pickup and transport succeeded.

An 8-env test initially reached 6/8. The two failures showed only 20-27 degrees
of actual gear-yaw travel against the required 36-degree small-gear pitch.
Small-only 1.5x yaw overtravel then produced 8/8, completing around step 550.

Seed 1 is less reliable because some randomized grasps attenuate yaw more
strongly:

- Seed 0: 8/8
- Seed 1: 5/8 in the two-wave run, 6/8 in an isolated repeat
- Two-seed range: 13-14/16 (81-88%)

This is suitable for success-filtered recording, but not for assuming every
generated rollout will be retained.

### Large

- Original: 0/4
- A shared 9 mm yaw lock caused 0/4 by locking on the large clash shelf.
- A 5 mm large-only lock produced 3/4.
- The remaining gear reached 3.1 mm but was lifted during retry while still
  moving.
- Holding deeply engaged large gears below 6 mm produced 4/4.
- Final combined policy: 8/8
- The final 8-env wave completed around step 400.
- Seed 1: 8/8
- Two-seed total: 16/16 (100%)

## Final readiness

The policy is ready for success-filtered data collection:

- Small: 81-88% across seeds 0-1
- Medium: 94% across seeds 0-1
- Large: 100% across seeds 0-1

Always set `--max_demos` and record until the requested number of successful
episodes is retained. Do not assume `waves * num_envs` equals the final dataset
size, especially for small.

## Run commands

### Stream four policy runs

The container must use host networking to reach the SSH tunnel:

```bash
docker run --name assembly-expert \
  --gpus all \
  --network host \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e HUD_API_KEY=sk-hud-dev-test-key \
  -e HUD_API_URL=http://127.0.0.1:18000 \
  -e HUD_TELEMETRY_URL=http://127.0.0.1:18000/v2/telemetry \
  -e HUD_WEB_URL=http://localhost:3000 \
  -e OMNI_KIT_ACCEPT_EULA=YES \
  -e PYTHONUNBUFFERED=1 \
  -v /home/ubuntu/project/assembly_bench:/app/assembly_bench \
  -w /app/assembly_bench \
  hud-assembly-env \
  bash -lc "/isaac-sim/python.sh -m pip install -q av && \
    exec /isaac-sim/python.sh scripts/experts/run_expert.py \
      --headless --task gear_small --num_envs 4 --waves 1 \
      --stream --debug_seat"
```

The development runs also mounted persistent Isaac caches from:

```text
/home/ubuntu/docker/isaac-expert-cache
```

Reuse the cache mounts from the existing `assembly-expert` container command to
avoid a cold shader and asset boot.

### Validate without streaming

Passing `--record` disables streaming unless `--stream` is also present:

```bash
/isaac-sim/python.sh scripts/experts/run_expert.py \
  --headless \
  --task gear_medium \
  --num_envs 4 \
  --waves 1 \
  --record /tmp/gear_medium_test.hdf5 \
  --debug_seat
```

## Next steps

1. Begin success-filtered recording with a small target such as 16 retained
   demonstrations per size.
2. Inspect recorded videos, action continuity, and HDF5 episode lengths before
   scaling to the full dataset.
3. Use extra waves for small because its measured retention is 81-88%.
4. Shorten failed rollouts by surfacing
   `Machine.failed` as an environment truncation or resetting failed slots.
   Reducing phase timeouts alone leaves traces running to tick 779.
5. Bake PyAV into `hud-assembly-env` so streamed runs do not install it at
   startup.

Do not weaken the task's 3 mm seat criterion to hide near misses. Finish the
last few millimeters through the policy.

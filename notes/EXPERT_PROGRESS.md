# Scripted Peg Expert — Progress & Handoff

Status as of this session. Read the source of truth first:
`/home/ubuntu/projects/.cursor/skills/data-gen/SKILL.md`.

## 1. Goal & pipeline
Build scripted **privileged** experts (read true sim poses) for the Isaac Lab Arena
assembly benchmark, generate VLA demos, and publish a LeRobot dataset.

`scripted expert (isaac6) -> HDF5 -> convert (vla) -> LeRobot dataset -> HF (lukasskellijs/assembly_bench)`

Scale by **parallel-env re-runs + per-env expert randomization**, NOT MimicGen.
Why: we have a free, closed-loop scripted source that reacts to contact every step;
mimic's open-loop replay of one authored trajectory is strictly worse at contact-rich
insertion (no correction under the peg/bore contact).

## 2. What works (proven)
- **Round-peg expert** phase machine: `hover -> descend -> grasp -> microlift -> lift -> transport -> align -> insert -> hold` (`scripts/experts/peg.py`).
- **HDF5 recorder** `scripts/experts/record.py` (success-filtered episodes).
- **LeRobot converter** `scripts/experts/convert_lerobot.py` (HDF5 -> LeRobot), round-trip load verified previously.
- Seat rate: **baseline 54%** -> insert-fix reached a **validated 81% (13/16)** in run1 this session.

## 3. Calibrated facts — DO NOT re-derive or break
- Approach axis = `base_link` local **+x** (NOT +z).
- DROID home quat **already points the tool straight down** — no reorientation phase; grasp is pure translation. Orientation held constant the whole trajectory.
- `Servo.TOOL_LEN = 0.1717` (measured; spec 0.1628 is 9 mm short here).
- Gripper: **OPEN = 0.0, CLOSE = 1.0** (canonical DROID/openpi).
- `state = [joint_pos(7), gripper_pos(1)]`; `action = [7 abs joint targets, gripper]`.
- Quaternions are **xyzw** end-to-end.
- Do NOT touch grip strength or the asset.

## 4. Current uncommitted state (be precise)
`git status` shows several dirty/untracked files. Attribution:

**Edited THIS session (expert work):**
- `scripts/experts/peg.py` — **fully mine** (Task 1 insert fix + Task 2 diversity).
- `scripts/experts/run_expert.py` — **one line only**: `seed=(args_cli.seed or 0) + wave` on the `peg.make_machine(...)` call. Everything else in its diff (`--record`/`--stream` args, `Recorder` import/calls) was pre-existing uncommitted, not mine.

**Pre-existing uncommitted (NOT touched this session):**
- `assembly_bench/environments/assembly/assembly.py`
- `rl/assembly_rlinf.py`, `rl/config/assembly_bench_ppo_openpi_pi05.yaml`
- untracked: `scripts/experts/record.py`, `scripts/experts/convert_lerobot.py`, `data/`

**Validated this session (wrist roll ±0.35 + transport live-servo + insert slip-clamp):**
`peg_round_8mm_tight` seed=0, 8×2 → **14/16 seated (87.5%)**, 14 success demos in
`data/peg_round_8mm_tight_validate.hdf5`. Wave0 8/8, wave1 6/8 (2 grasp/insert fails).

**Exact current `peg.py` behavior:**
- Insert: captures the in-hand transform `inhand = peg - ee` at `enter_insert` (peg just aligned on bore xy, pre-contact); drives a **fixed anchor `hole - inhand`** with **continuous xy correction (no gate)** + gentle `zcap`; **slip-clamp** holds z if `||peg - (ee + inhand)|| > 15 mm` (jam) instead of grinding.
- Transport/align: **live servo `ee + (hole - peg)`** (self-corrects in-grip slip in free space).
- Diversity (per-env, seeded by `seed`, active only when no calibration `aim_off`): wrist roll **±0.35 rad (~±20°)**, grasp-depth **±2.5 mm**, in-grip xy **±1 mm**, hover height **0–30 mm**, lift-height jitter (`SAFE_BASE_Z` −10/+20 mm), grasp-dwell **12–28 steps**.

## 5. Known open issue
- Residual ~12% fail rate looks like grasp miss / insert jam (wave1: 2 fails, finger open + peg still on stand). Acceptable for success-filtered bulk; revisit if seat drops below ~80%.

## 6. How to run (exact)
Envs: **`isaac6`** = sim (run experts), **`vla`** = lerobot (convert). Sim launch ~60–90 s;
8-env × 2-wave ~6 min. One `SimulationContext` per process (GPU is the shared resource).
Never `--record` together with streaming. If a run hangs: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` and `kill -9` zombie Kit/python PIDs.

**Streamed validation run (watchable):**
```bash
cd /home/ubuntu/projects/assembly_bench && \
export HUD_API_KEY=sk-hud-dev-test-key HUD_API_URL=http://localhost:8000 \
  HUD_TELEMETRY_URL=http://localhost:8000/v2/telemetry HUD_WEB_URL=http://localhost:3000 && \
OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 conda run -n isaac6 --no-capture-output \
  python scripts/experts/run_expert.py --headless --task peg_round_8mm_tight \
  --num_envs 8 --waves 2 --max_steps 340 --snap_every 20
```
`--snap_every` dumps env0 front/wrist/viewer frames to `/tmp/expert_snaps` (montage with PIL
in isaac6). NB the local HUD control plane at :8000 may be down or return 404s (best-effort;
does not block non-record runs, but its retries can slow the step loop).

**Record for dataset (NO streaming during record):**
```bash
OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 conda run -n isaac6 --no-capture-output \
  python scripts/experts/run_expert.py --headless --task peg_round_8mm_tight \
  --num_envs 8 --waves 2 --max_steps 340 --record /path/to/peg_round_8mm_tight.hdf5
```

**Convert to LeRobot (vla env):**
```bash
PYTHONUNBUFFERED=1 conda run -n vla --no-capture-output \
  python scripts/experts/convert_lerobot.py --hdf5 /path/to/peg_round_8mm_tight.hdf5 \
  --repo_id lukasskellijs/assembly_bench --fps 15
```
(Confirm exact flags by reading `convert_lerobot.py`.) Push gotchas (SKILL.md): call
`ds.finalize()` before upload; disable hf_transfer/xet; verify served bytes + `PAR1` footer;
add LeRobot dataset card; retag `v3.0`.

## 7. Done this session
1. Wrist roll dialed to ±0.35 rad; validated **14/16 (87.5%)** then bulk **57/64 (89%)**.
2. Recorded `data/peg_round_8mm_tight.hdf5` (57 success demos).
3. Converted → `data/lerobot/assembly_bench` (57 eps, 14401 frames @15 Hz). Round-trip load OK; parquet `PAR1` OK.
4. Diversity gate PASS: start-joint std≈0.087, ep-len std≈22.6, pairwise traj L2≈0.35.
5. Pushed + retagged `v3.0` → https://huggingface.co/datasets/lukasskellijs/assembly_bench
6. Diversity script: `scripts/experts/diversity_check.py`.

## 8. Reset-bug fix (this session)
**Found + fixed a dirty-reset bug that put ~84% of demos off the home start pose.**
- Symptom: only wave-0 demos started at DROID home; waves ≥1 (and post-success mid-wave
  resets) started with the arm dragged down toward the prior insert pose. Eval always
  resets to home, so those demos were off-distribution.
- Root cause: `settle_and_render` (`tasks.py`) steps physics via raw `env.sim.step()`, which
  bypasses `write_data_to_sim()`. So the home joint-position TARGET staged by
  `reset_scene_to_default(reset_joint_targets=True)` never reached PhysX; the arm teleported
  home then the stale prior-episode target dragged it back during the 50 settle steps.
- Fix: `settle_and_render` now calls `art.write_data_to_sim()` for each articulation each
  settle step, so the PD holds home. Verified: wave-1 start `tcp z` back to 0.296 (home);
  `diversity_check` `off_home` 48/57 → **0/59**.
- Secondary (flagged, NOT changed): `randomize_franka_joint_state` runs before `reset_all`,
  so `reset_scene_to_default` clobbers the intended 0.02 rad arm start noise. Train/eval
  consistent (both exact home), so low-priority; reorder if you want that variance back.
- `diversity_check.py` gained a **frame-0-near-home gate** (`HOME_JOINTS`, `HOME_TOL=0.2`).

## 9. Current dataset (regenerated post-fix)
- `lukasskellijs/assembly_bench` @ `v3.0`: **117 eps / 31736 frames @15 Hz**, 1 task
  ("pick up the 8 mm round peg…"): 59 tight (`seed=100`) + 58 loose (`seed=200`).
- Both HDF5s in `data/`; gate PASS, `off_home=0`, round-trip + `PAR1` verified.

## 10. Square (rectangular) pegs — yaw clocking (added, PAUSED mid-collection)
- `peg.py` now has a `clock=True` path (round unchanged; clock_yaw stays 0 → byte-identical).
  Closed-loop wrist-roll servo nulls live `wrap(hole_yaw - peg_yaw, pi)` (rect = 2-fold) during
  transport + align, rate-limited (`CLOCK_RATE=0.08`, no windup), gated `YAW_TOL=0.02` before
  the press, then frozen. `run_expert` enables it via `variant.rand_fixed_yaw > 0`.
- Grasp roll jitter shrinks to ±0.10 for rect (pads land on flats); clocking sets final yaw.
- Seat rate `peg_square_8mm_tight` ≈ **50–56%** (vs ~90% round): tight ~0.1 mm rect clearance
  jams on residual in-grip yaw slip during the press. Tried (all REVERTED, made it worse):
  tighter `YAW_TOL=0.008`, clocking through insert-approach, gentler rect press. Best config is
  the current one (`YAW_TOL=0.02`, standard press). Improving seat is the open lever
  (candidate: resist yaw slip in-grip / clock only pre-contact — needs a render to debug).
- **Collected so far (in `data/`):**
  - `peg_square_8mm_tight.hdf5` = **49 demos** (seed 300, 11 waves) — COMPLETE, valid.
  - `peg_square_8mm_loose.hdf5` = partial/**incomplete** (killed mid-run at ~2 waves; recorder
    never closed → discard & re-run).
- **To resume:** re-run loose bulk (`--task peg_square_8mm_loose --num_envs 8 --waves 10
  --seed 500 --record ...loose.hdf5`, overwrites the partial); optional tight top-up (~4 waves,
  new seed, separate file) to reach ~58. Then `diversity_check` both, `convert_lerobot.py
  --glob "data/peg_square_8mm_*.hdf5" --repo_id lukasskellijs/assembly_bench` (adds the square
  task alongside round), round-trip, push, retag `v3.0`.
- NOTE: square instruction is "pick up the 8 mm square peg…" (distinct from round) → new task.

## 11. Remaining steps
1. **Finish square 8 mm** (loose re-run + tight top-up → convert + push), per §10.
2. **Sweep other round pegs** (4/12/16 × loose/tight) to ~50 successes each — same expert; re-spot-check seat per size before bulk.
3. **Other geometries** for the full 29: gear (wiggle-and-press + `max_contact_impulse`), nut (thread regrip). Do NOT reuse the peg press.
4. Env-dir gotcha: `isaac6` editable installs hardcode `/home/ubuntu/projects/env/IsaacLab-Arena`; if `env/` is moved, sim runs fail with `ModuleNotFoundError: isaaclab_arena`.

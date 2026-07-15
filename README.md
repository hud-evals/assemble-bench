# assembly-bench

NIST-taskboard assembly benchmark on [Isaac Lab Arena](https://github.com/isaac-sim/IsaacLab-Arena),
built as an **external environment package** following the
[Arena-in-your-repo](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/arena_in_your_repo/external_installation.html)
pattern: Arena stays an unmodified submodule and everything here plugs in
through its registration API.

This is a port of the standalone `assembly` benchmark (DirectRLEnv, NIST
taskboard benchmarking procedure) into Arena's scene/task/registry framework —
extending Arena's example [assembly environments](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/example_workflows/example_environments.html#assembly)
(peg insert, gear mesh) into a full parametrized benchmark. A Franka on the
`table` (SeattleLab) background faces a two-station workspace: a presentation
stand holding the free part, and the fixture it must be assembled into.

## Task space

27 variants across 3 families, mirroring the source benchmark's authored task
matrix (`assembly/notes/TASK_MATRIX.md` §2):

| family | variants | goal |
|---|---|---|
| `peg_insert` | 16: `peg_{round,square}_{4,8,12,16}mm_{loose,tight}` | pick the peg off its stand bore, insert it into the matching hole |
| `gear_mesh` | 3: `gear_{small,medium,large}` | mesh the held gear onto its shaft between two fixed gears (tooth-phase alignment) |
| `nut_thread` | 8: `nut_m{4,8,12,16}_{loose,tight}` | thread the nut onto the bolt on the NIST GMC board (helical descent; a straight push jams) |

All active nut tiers (M4–M16) use the generated watertight-threaded USDs (real
ISO helix, SDF collision, baked brass/steel MDL); the unused M20 assets remain
packaged, and M16 is authored at the curated
Factory pair's reference dimensions. Per-tier seat geometry (head height,
shank, pitch) comes from the authoring tables, so success stays
FORGE-faithful across sizes.

"Square" pegs are rectangular, so they are not yaw-symmetric: the hole gets a
±0.6 rad yaw jitter per episode and the policy must clock the grasped peg to
match. Success everywhere is the seat-geometry outcome check (below), never a
proximity heuristic.

## Layout

```
assembly_bench/
├── submodules/IsaacLab-Arena           unmodified Arena (git submodule)
├── assembly_bench/                     the pip-installable package
│   ├── pyproject.toml
│   ├── environments/assembly/
│   │   ├── scene.py                    @register_asset parts (asm_* prefix)
│   │   ├── tasks.py                    seat-geometry success + NISTAssemblyTask + reset jitter
│   │   ├── variants.py                 the 27 variants as pure data (benchmark manifest)
│   │   └── assembly.py                 AssemblyBenchEnvironment (ExampleEnvironmentBase)
│   └── assets/parts/                   generated USDs (pegs, holes, gears, NIST board, YCB)
├── eval/                               batch eval: jobs config for all 27 variants
└── scripts/preview_assembly.py         renders a still + prints part poses
```

## Install

Arena needs Isaac Sim 6.x + the IsaacLab pinned in its own submodule.

```bash
git submodule update --init --recursive        # IsaacLab-Arena (+ its IsaacLab)

# into your Isaac Sim python env:
for d in submodules/IsaacLab-Arena/submodules/IsaacLab/source/isaaclab*/; do
    pip install --no-deps -e "$d"
done
pip install -e submodules/IsaacLab-Arena
pip install -e assembly_bench                  # this package
```

> In this workspace, `submodules/IsaacLab-Arena` is a symlink to
> `../bench/submodules/IsaacLab-Arena` for development convenience; replace it
> with a real git submodule when the repo is initialized.

## Run

From `submodules/IsaacLab-Arena`, using the standard Arena external-environment CLI:

```bash
python isaaclab_arena/evaluation/policy_runner.py \
    --policy_type zero_action --num_episodes 1 \
    --external_environment_class_path \
    assembly_bench.environments.assembly.assembly:AssemblyBenchEnvironment \
    assembly_bench --task peg_round_8mm_tight
```

`--task` selects the variant; `--embodiment` defaults to `droid_abs_joint_pos`
— Arena's DROID platform (Franka + Robotiq 2F-85 on the DROID stand, wrist
camera on the Robotiq base link) with absolute joint-position actions, the
pi0.5-DROID contract (8-D action: 7 joint targets + binary gripper). Any
`franka_*` embodiment also works and gets Arena's Factory-tuned high-PD arm
config. Add
`--hdr <name>` (any Arena HDR-registry entry, e.g. `carpentry_shop_robolab`)
to light the scene with an HDRI dome. To run all 27 variants as a batch, see
[`eval/README.md`](eval/README.md).

## Design notes (what the port keeps from the source benchmark)

- **Scenes are Python, not USD.** Each variant is pure data (`variants.py`):
  which held/fixed/stand/extra assets to spawn, where, and the seat geometry.
  One task class (`NISTAssemblyTask`) serves all families.
- **Asset roles.** *held* = free dynamic part; *fixed* = world-pinned fixture
  (the USDs carry their own fixed joint); *stand* = near-frictionless
  presentation bore the peg starts standing in; *extras* = kinematic context
  (flanking gears, the NIST board).
- **Success = seated + stable.** The held part's base point (root +
  `held_base_z_off`) must reach the seat target (fixed root + `seat_off`,
  rotated into the fixed frame so it tracks hole clocking) within `align_tol`
  in xy, descend past `seat_tol` in z, and be nearly at rest — held 3
  consecutive steps (`hold_success`). The contact skill lives in the policy:
  a clashing gear or cross-threaded nut can't descend, so the gap stays large.
- **Reset jitter** mirrors the source: held+stand share one xy jitter, fixed+
  extras an independent one (the nut variant pins its bolt+board dead-center),
  and rectangular holes get a yaw jitter. Friction is pinned per asset role at
  startup (slippery stand, grippy nut).
- **Contact safety.** Gear/nut assets cap the PhysX contact impulse at 1e4 so
  a tooth clash or thread jam pushes apart gently instead of exploding; sim
  settings come from Arena's `assembly_env_cfg_callback` (192 position
  iterations, 60 Hz physics), re-decimated to the benchmark's rate.
- **pi0.5/DROID-consumable I/O** (`cameras.py`). The DROID embodiment's own
  cameras, minus its two over-shoulder exterior views, plus one frontal
  exterior (`front_cam`, on the robot midline, close and low so the NIST
  parts read as 3D shapes). Everything streams at DROID-native 1280x720
  16:9 with the calibrated intrinsics (2.8 mm / 5.376x3.024 mm); the wrist
  camera is the embodiment's calibrated Robotiq mount, verified
  frame-for-frame against the source benchmark's recorded demos. Model-input
  sizing (openpi's `resize_with_pad` to 224x224) belongs to the policy
  adapter, as in the chess bench's pi0.5 eval — not the env.
  Control runs at 15 Hz (decimation 4 at 60 Hz physics, the pi0.5-DROID
  rate), and rendering happens once per policy step, so camera observations
  and recorded videos (`--record_camera_video`) are 15 fps.

### Not ported (yet)
- Dense/partial-credit reward and the wrist F/T sensor: Arena's metric surface
  is `SuccessRateMetric` + `ObjectMovedRateMetric` here.
- The visual perturbation library (machine-shop HDRIs, OSB tabletop MDL);
  `--hdr` exposes Arena's HDR registry as the background axis instead.
- Seat validation status carries over from the source benchmark: the peg
  family, `gear_medium`, and `nut_m16_loose` are seat-validated end-to-end;
  gear `small`/`large` and the generated M4–M20 nut tiers are authored and
  geometrically verified (watertight, open bore) but a blind press does not
  seat them — by design, they need a contact-search / threading policy.

## Toward LeRobot EnvHub

The environment is a standard Arena external environment, so it follows the
same path as `nvidia/isaaclab-arena-envs` on EnvHub: an `env.py` that wraps
`AssemblyBenchEnvironment` behind the generic Arena wrapper, plus an
`example_envs.yaml` entry mapping `environment: assembly_bench` and its
`--task` values. Nothing here depends on the local harness.

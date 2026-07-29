"""Arena-registered assets for the NIST assembly benchmark.

Registers every part the variants reference, under an ``asm_`` prefix so the
names never collide with Arena's own library. Benchmark USDs live in
``assets/parts/used/`` (see that folder's README). Gear base comes from
Isaac Lab Factory ``ASSET_DIR``.

Asset roles mirror the source env:
  held  -- free dynamic part (gravity on), grasped and assembled.
  fixed -- world-pinned socket / base / bolt (the curated/generated USDs carry
           their own fixed joint, so spawning them as articulations pins them).
  extras / board -- kinematic context and mesh partners (flanking gears, NIST board).
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab_tasks.direct.factory.factory_tasks_cfg import ASSET_DIR

from isaaclab_arena.assets.hdr_image_library import LibraryHDR
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.object_utils import RIGID_BODY_PROPS_HIGH_PRECISION
from isaaclab_arena.assets.register import register_asset, register_hdr

# absolute() (not resolve): HF Hub snapshots symlink into blobs/, and resolve()
# would land ASSETS_DIR on ~/.cache/huggingface/hub/assets.
ASSETS_DIR = Path(__file__).absolute().parents[2] / "assets"
PARTS_DIR = ASSETS_DIR / "parts" / "used"


@register_hdr
class MachineShopHDR(LibraryHDR):
    """The source benchmark's machine-shop backdrop (Poly Haven CC0), shipped locally."""

    name = "asm_machine_shop"
    tags = ["indoor", "workshop"]
    texture_file = str(ASSETS_DIR / "backgrounds" / "indoors" / "machine_shop_01_2k.hdr")

# Empty-joint init state for DOF-less articulations (the {".*": 0.0} default
# fails to match when the articulation has no joints).
_EMPTY_INIT = ArticulationCfg.InitialStateCfg(joint_pos={}, joint_vel={})

# Contact-impulse cap for gear/nut assets: a mis-phased tooth clash or the
# nut's fine-thread SDF bore can otherwise generate an unbounded impulse ->
# NaN blow-up. The cap must still be high enough that a gripper can actually
# clamp the part: at 1e4 the finger's clamp impulse was clipped and the pad
# sank through the gear. A low cap can only make sink-through worse, so keep it
# high (1e16) -- still finite, so the NaN guard survives a runaway SDF clash,
# but with ample headroom so a firm clamp is never clipped into the part.
GEAR_NUT_IMPULSE_CAP = 1e16

GEAR_MASS = {"small": 0.006, "medium": 0.012, "large": 0.025}


def _register(name: str, usd: str, *, mass: float | None = None, impulse_cap: float = 1e32,
              kinematic: bool = False) -> None:
    """Register one part. Free/pinned parts spawn as DOF-less articulations
    (the NIST USDs carry articulation roots); ``kinematic`` disables the
    articulation root and spawns an immovable rigid body instead."""
    spawn: dict = {
        "mass_props": sim_utils.MassPropertiesCfg(mass=mass) if mass is not None else None,
        # Contact skin. 0.001 paired with soft arm for pi0.5; expert recording
        # firms the arm via --arm_stiffness rather than widening this again.
        "collision_props": sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
    }
    if kinematic:
        spawn["rigid_props"] = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True, disable_gravity=True, max_depenetration_velocity=5.0,
            solver_position_iteration_count=192, solver_velocity_iteration_count=1,
            max_contact_impulse=impulse_cap)
        spawn["articulation_props"] = sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False)
        object_type, asset_addon = ObjectType.RIGID, {}
    else:
        # Free part. Standard high-precision rigid body (same treatment pegs use
        # and grasp cleanly with) -- the earlier gear/nut-specific throttle
        # (capped depenetration velocity, capped lin/ang velocity, extra damping)
        # was fighting the grasp, so it was dropped. Retain only two guards: a
        # bounded contact impulse (NaN safety on gear-tooth / nut-thread SDF
        # clashes) and velocity solver iterations raised 1 -> 8 to match the robot
        # so both sides resolve contact velocity (more iterations only sharpen
        # contact, never worsen it).
        spawn["rigid_props"] = RIGID_BODY_PROPS_HIGH_PRECISION.replace(
            max_contact_impulse=impulse_cap,
            solver_velocity_iteration_count=8,
        )
        object_type, asset_addon = ObjectType.ARTICULATION, {"init_state": _EMPTY_INIT}
    register_asset(type(
        name.title().replace("_", ""),
        (LibraryObject,),
        {
            "name": name,
            "tags": ["object", "assembly"],
            "usd_path": usd,
            "object_type": object_type,
            "spawn_cfg_addon": spawn,
            "asset_cfg_addon": asset_addon,
        },
    ))


# Peg family: loose-clearance pegs (50 mm long, ~1 mm chamfer mouth) + matching
# bores. The bore doubles as the presentation stand the peg starts standing in.
for _size in (4, 8, 12, 16):
    for _stem in ("round", "rect"):
        _register(f"asm_peg_{_stem}_{_size}mm_loose",
                  str(PARTS_DIR / "pegs" / f"gen_{_stem}_peg_{_size}mm_loose.usd"), mass=0.019)
        _register(f"asm_hole_{_stem}_{_size}mm",
                  str(PARTS_DIR / "pegs" / f"gen_{_stem}_hole_{_size}mm.usd"), mass=0.05)

# Gear family: re-centered gears (frame on the shaft axis) as both the free
# held part and the kinematic flanking mesh partners; base from Factory.
for _k, _m in GEAR_MASS.items():
    _register(f"asm_gear_{_k}", str(PARTS_DIR / "gears" / f"gen_gear_{_k}.usd"),
              mass=_m, impulse_cap=GEAR_NUT_IMPULSE_CAP)
    _register(f"asm_gear_{_k}_fixed", str(PARTS_DIR / "gears" / f"gen_gear_{_k}.usd"),
              mass=_m, impulse_cap=GEAR_NUT_IMPULSE_CAP, kinematic=True)
_register("asm_gear_base", f"{ASSET_DIR}/factory_gear_base.usd",
          mass=0.05, impulse_cap=GEAR_NUT_IMPULSE_CAP)

# Nut family on the NIST GMC board: IsaacGymEnvs Factory OBJs → USD
# (SDF + brass/steel MDL). Loose clearance only. M4 omitted — pad cannot grasp
# the 3.2 mm hex.
for _s in (8, 12, 16, 20):
    _register(f"asm_nut_m{_s}_loose", str(PARTS_DIR / "nuts" / f"factory_nut_m{_s}_loose.usd"),
              impulse_cap=GEAR_NUT_IMPULSE_CAP)
    _register(f"asm_bolt_m{_s}_loose", str(PARTS_DIR / "nuts" / f"factory_bolt_m{_s}_loose.usd"),
              impulse_cap=GEAR_NUT_IMPULSE_CAP)
_register("asm_nist_board", str(PARTS_DIR / "nist_gmc_base.usd"),
          mass=1.0, impulse_cap=GEAR_NUT_IMPULSE_CAP, kinematic=True)

# Smoke pick-place only (not NIST): apple → bowl. See assets/parts/debug/.
DEBUG_DIR = ASSETS_DIR / "parts" / "debug"


@register_asset
class AsmDebugApple(LibraryObject):
    """Smoke: RoboLab ``apple_01`` (~7 cm). Pick-place contact skin, not peg."""

    name = "asm_debug_apple"
    tags = ["object", "debug"]
    usd_path = str(DEBUG_DIR / "apple_01.usd")
    object_type = ObjectType.RIGID
    # Spawner ignores USD's baked 0.01 xform; this is the only scale that applies.
    scale = (0.01, 0.01, 0.01)
    spawn_cfg_addon = {
        # RoboLab pick-place skin (20 mm), not peg 1 mm — fat contact stops binary
        # grip from digging into convex-decomp hulls. Mild depenetration + lin-vel
        # cap remain as eject guards (normal grasp << 1 m/s).
        "rigid_props": RIGID_BODY_PROPS_HIGH_PRECISION.replace(
            solver_velocity_iteration_count=8,
            max_depenetration_velocity=1.0,
            max_linear_velocity=2.0,
        ),
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.15),
        "collision_props": sim_utils.CollisionPropertiesCfg(
            contact_offset=0.02, rest_offset=0.01
        ),
    }


@register_asset
class AsmDebugBowl(LibraryObject):
    """Smoke: RoboLab YCB bowl (kinematic receptacle)."""

    name = "asm_debug_bowl"
    tags = ["object", "debug"]
    usd_path = str(DEBUG_DIR / "bowl.usd")
    object_type = ObjectType.RIGID
    spawn_cfg_addon = {
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True, disable_gravity=True,
            max_depenetration_velocity=5.0,
            solver_position_iteration_count=16, solver_velocity_iteration_count=1,
        ),
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.2),
        # Match apple pick-place skin (kinematic CCD is ignored by PhysX).
        "collision_props": sim_utils.CollisionPropertiesCfg(
            contact_offset=0.02, rest_offset=0.01
        ),
    }

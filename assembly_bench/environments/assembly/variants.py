"""The benchmark manifest: NIST-taskboard assembly variants as pure data.

Ported from the source benchmark's full authored task matrix
(``assembly/notes/TASK_MATRIX.md`` §2): the 16-instance peg-insert family
(round/square x 4/8/12/16 mm x loose/tight), 3 gear-mesh sizes, and the
10-instance nut-thread family (generated M4-M20 x loose/tight) — 29 variants.
Each variant fully specifies the scene content (held / fixed / stand / extra
assets and their poses) and the seat geometry that defines success. Kept free
of Isaac imports so the CLI can list ``--task`` choices before the simulator
app launches.
"""

from dataclasses import dataclass

# Nominal tabletop height (Factory convention: parts are authored so z=0 is
# the working surface). The SeattleLabTable's collision top is recessed 3 mm
# below the visual surface, so free parts settle at -0.003 -- authored fixture
# base plates (hole/stand bores span z=[-3,0]mm) sink flush by design.
TABLE_TOP_Z = 0.0

# Workspace layout (m, env-local; z = TABLE_TOP_Z is the tabletop). Parts sit
# centered under the home gripper so they land in the wrist camera's first frame.
PEG_FIXED_POS = (0.37, 0.07, TABLE_TOP_Z)     # insertion hole / gear base
PEG_HELD_POS = (0.37, -0.07, TABLE_TOP_Z)     # presentation stand / free part
NUT_BASE_TOP = TABLE_TOP_Z + 0.009            # NIST GMC board top surface

# Gear shaft x-offsets from the gear-base root (re-centered gen_gear USDs).
GEAR_SHAFT = {"small": 0.05075, "medium": 0.02025, "large": -0.03025}


@dataclass(frozen=True)
class AssemblyVariant:
    family: str                       # "peg_insert" | "gear_mesh" | "nut_thread"
    instruction: str
    held: str                         # registered asset name of the grasped part
    fixed: str                        # registered asset name of the socket / base / bolt
    held_pos: tuple[float, float, float]
    fixed_pos: tuple[float, float, float]
    stand: str | None = None          # presentation bore the peg stands in (kinematic, slippery)
    extras: tuple[tuple[str, tuple[float, float, float]], ...] = ()  # (asset, offset from fixed)
    # Seat geometry: target = fixed_root + seat_off (in the fixed frame); the held
    # part's base point (root + held_base_z_off) must reach it within the tolerances.
    seat_off: tuple[float, float, float] = (0.0, 0.0, 0.0)
    held_base_z_off: float = 0.0
    align_tol: float = 0.0025         # xy distance to target
    seat_tol: float = 0.003           # seat gap (one-sided: gap < tol)
    # Staged-reward geometry (used only when reward_mode="staged"): the engage
    # milestone gap (near the socket/shaft/thread mouth), the depth scale for
    # continuous partial credit, and the lift-clear height off the stand.
    engage_gap: float = 0.025
    partial_socket_h: float = 0.025
    lift_clear: float = 0.03
    # Reset randomization: held(+stand) and fixed(+extras) each share one xy jitter.
    rand_xy: float = 0.05
    rand_fixed_xy: float | None = None  # None -> same as rand_xy; 0.0 pins the fixed asset
    rand_fixed_yaw: float = 0.0         # +/- yaw jitter on the fixed asset (rect-peg clocking)
    held_friction: float = 0.75
    fixed_friction: float = 0.75
    episode_length_s: float = 32.0


def _peg(size: int, geometry: str, tolerance: str) -> AssemblyVariant:
    """One peg-insert instance. 'square' pegs are rectangular (USD stem 'rect'),
    so they are not yaw-symmetric: the hole gets yaw jitter the policy must match."""
    stem = {"round": "round", "square": "rect"}[geometry]
    return AssemblyVariant(
        family="peg_insert",
        instruction=f"pick up the {size} mm {geometry} peg and insert it into the hole",
        held=f"asm_peg_{stem}_{size}mm_{tolerance}",
        fixed=f"asm_hole_{stem}_{size}mm",
        stand=f"asm_hole_{stem}_{size}mm",   # a second bore presents the peg upright
        held_pos=PEG_HELD_POS,
        fixed_pos=PEG_FIXED_POS,
        rand_fixed_yaw=0.6 if geometry == "square" else 0.0,
    )


def _gear(size: str) -> AssemblyVariant:
    """Mesh the held gear onto its shaft on the 3-shaft base; the other two gears
    flank it as kinematic mesh partners, so seating requires tooth-phase alignment."""
    flanks = tuple(
        (f"asm_gear_{k}_fixed", (GEAR_SHAFT[k], 0.0, 0.0)) for k in GEAR_SHAFT if k != size
    )
    return AssemblyVariant(
        family="gear_mesh",
        instruction=f"pick up the {size} gear and mesh it onto its shaft",
        held=f"asm_gear_{size}",
        fixed="asm_gear_base",
        held_pos=(*PEG_HELD_POS[:2], TABLE_TOP_Z - 0.005),  # gear frame z=5mm -> bottom on the table
        fixed_pos=PEG_FIXED_POS,
        extras=flanks,
        seat_off=(GEAR_SHAFT[size], 0.0, 0.0),  # the held gear's own shaft
        held_friction=1.0,          # grippy pads so the gear cannot slip out of the grasp
                                    # (was 0.4 -- too low, the gear slid through the fingers)
        engage_gap=0.02,
        partial_socket_h=0.02,
        episode_length_s=52.0,
    )


VARIANTS: dict[str, AssemblyVariant] = {}

for _size in (4, 8, 12, 16):
    for _geom in ("round", "square"):
        for _tol in ("loose", "tight"):
            VARIANTS[f"peg_{_geom}_{_size}mm_{_tol}"] = _peg(_size, _geom, _tol)

for _k in ("small", "medium", "large"):
    VARIANTS[f"gear_{_k}"] = _gear(_k)

# Nut-thread tiers. Bolt/nut dimensions (m) from the authoring script
# (gen_nutbolt_usds.py NUTBOLT table; M16 authored at the curated reference
# dims). The nut is authored at its assembled-start height, so its base sits
# head_h above the root and rests on the board at z = board_top - head_h.
NUTBOLT = {
    4: dict(head_h=0.004, shank=0.016, pitch=0.0007),
    8: dict(head_h=0.008, shank=0.018, pitch=0.00125),
    12: dict(head_h=0.012, shank=0.02, pitch=0.00175),
    16: dict(head_h=0.010, shank=0.025, pitch=0.002),
    20: dict(head_h=0.020, shank=0.045, pitch=0.0025),
}


def _nut(size: int, tolerance: str) -> AssemblyVariant:
    """Thread the nut onto its bolt on the NIST GMC board. Success is only
    reachable by helical threading (a straight push jams): nut base descends to
    head_h + shank - 1.5*pitch, within pitch*0.375 (FORGE-faithful). The bolt +
    board are pinned dead-center; only the nut jitters."""
    d = NUTBOLT[size]
    return AssemblyVariant(
        family="nut_thread",
        instruction=f"pick up the M{size} nut and thread it onto the bolt",
        held=f"asm_nut_m{size}_{tolerance}",
        fixed=f"asm_bolt_m{size}_{tolerance}",
        held_pos=(0.30, 0.10, NUT_BASE_TOP - d["head_h"]),
        fixed_pos=(0.37, 0.0, NUT_BASE_TOP),
        extras=(("asm_nist_board", (0.0, 0.0, 0.0)),),
        seat_off=(0.0, 0.0, d["head_h"] + d["shank"] - 1.5 * d["pitch"]),
        held_base_z_off=d["head_h"],
        seat_tol=0.375 * d["pitch"],
        engage_gap=0.01,           # thread mouth is close; engage = nut on the bolt tip
        rand_xy=0.03,
        rand_fixed_xy=0.0,
        held_friction=1.0,          # grippy pads so the nut survives lift/regrip cycles
        fixed_friction=0.2,
        episode_length_s=60.0,      # threading + release-regrip-rotate cycles
    )


for _size in (4, 8, 12, 16, 20):
    for _tol in ("loose", "tight"):
        VARIANTS[f"nut_m{_size}_{_tol}"] = _nut(_size, _tol)

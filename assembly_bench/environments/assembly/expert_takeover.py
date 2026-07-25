"""Sim-side CG-DAgger: latch fail → override with the peg scripted expert.

Opt-in gym wrapper. Policy (HUD) keeps stepping; for latched slots the action
is replaced by ``peg.make_machine`` / ``Servo`` so HUD video stays one unbroken
trace while recovery runs through episode end. Latch = ``policy/expert_active``.

Modes (``takeover_mode``):
  - ``grasp``  — missed pick (closed + retreating + peg on stand)
  - ``insert`` — lifted peg near mouth with no seating progress
  - ``both``   — grasp or insert

Tilt (≥30° off-stand, not seated) always latches when takeover is on — tip-in-
hand: soft tip→hole nudge near the bore then press; dropped: hover → regrasp.
Insert handoff with tip ≳10° also routes there — keep grip, tool-down, small
XY only (no wrist snap / park / drop).
"""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import torch

from assembly_bench.environments.assembly.grasp_fail import (
    FINGER_CLOSED,
    RETREAT_EPS,
    STALL_STEPS as GRASP_STALL_STEPS,
    ee_peg_xy_dist,
    grasp_fail_fire,
    grip_closed,
    peg_on_stand,
    update_stall,
)
from assembly_bench.environments.assembly.insert_fail import (
    ALIGN_HANDOFF_XY,
    PROGRESS_EPS,
    STALL_STEPS as INSERT_STALL_STEPS,
    at_seat,
    half_radius_m,
    insert_fail_fire,
    peg_diameter_mm,
    peg_hole_xy,
    seat_cost,
    update_insert_stall,
)
from assembly_bench.environments.assembly.tilt_fail import (
    STALL_STEPS as TILT_STALL_STEPS,
    TILT_COS,
    TILT_COS_INSERT,
    nearly_seated,
    peg_upright_cos,
    tilt_fail_fire,
    update_tilt_stall,
)
from assembly_bench.environments.assembly.variants import TABLE_TOP_Z, VARIANTS

_TAKEOVER_MODES = ("grasp", "insert", "both")


def _import_experts():
    """Load ``scripts/experts/{base,peg}.py`` as a package (relative imports)."""
    import importlib.util
    import types

    root = Path(__file__).resolve().parents[3]  # …/assembly_bench repo root
    experts_dir = root / "scripts" / "experts"
    pkg_name = "_assembly_bench_experts"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(experts_dir)]
        sys.modules[pkg_name] = pkg

    def _load(mod: str):
        full = f"{pkg_name}.{mod}"
        if full in sys.modules:
            return sys.modules[full]
        path = experts_dir / f"{mod}.py"
        spec = importlib.util.spec_from_file_location(full, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[full] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    return _load("base"), _load("peg")


def _as_bool_mask(x, n: int, device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.bool).view(-1)[:n]
    t = torch.as_tensor(x, device=device)
    return t.to(dtype=torch.bool).view(-1)[:n]


def _normalize_mode(mode: str) -> str:
    m = (mode or "grasp").strip().lower()
    if m not in _TAKEOVER_MODES:
        raise ValueError(f"takeover_mode={mode!r}; expected one of {_TAKEOVER_MODES}")
    return m


class ExpertTakeover(gym.Wrapper):
    """Override policy actions with the peg expert after a fail latch.

    Peg-insert variants only; other families pass through unchanged.
    Once latched, the expert keeps control until episode done (no mid-episode
    hand-back). Publishes latch via ``unwrapped._expert_takeover_active``.
    """

    def __init__(
        self,
        env,
        *,
        task: str,
        takeover_mode: str = "grasp",
        grasp_stall_steps: int = GRASP_STALL_STEPS,
        insert_stall_steps: int = INSERT_STALL_STEPS,
        tilt_stall_steps: int = TILT_STALL_STEPS,
    ):
        super().__init__(env)
        self.task = task
        self.variant = VARIANTS[task]
        self.mode = _normalize_mode(takeover_mode)
        self.grasp_stall_steps = grasp_stall_steps
        self.insert_stall_steps = insert_stall_steps
        self.tilt_stall_steps = tilt_stall_steps
        # Off-center latch: xy > half peg radius (8 mm peg → 2 mm).
        self.xy_ok = half_radius_m(peg_diameter_mm(task))
        self.enabled = self.variant.family == "peg_insert"
        self._base = None
        self._servo = None
        self._machine = None
        self._phase_idx: dict[str, int] = {}
        self.taken: torch.Tensor | None = None
        self._grasp_stall: torch.Tensor | None = None
        self._insert_stall: torch.Tensor | None = None
        self._tilt_stall: torch.Tensor | None = None
        self._prev_dist: torch.Tensor | None = None
        self._prev_cost: torch.Tensor | None = None
        if self.enabled:
            self._build_expert()
            # Slight headroom over peg default (40 s / 600 ticks) for tip recover.
            cfg = self.env.unwrapped.cfg
            target_s = 620 / 15.0  # 620 ticks @ 15 Hz
            if float(cfg.episode_length_s) < target_s:
                cfg.episode_length_s = target_s
                print(f"[takeover] episode_length_s → {target_s:g} (620 ticks @ 15 Hz)", flush=True)
            print(f"[takeover] mode={self.mode} (+tilt≥30°)", flush=True)

    def _publish_active(self):
        """ObsTerm ``expert_active`` reads this buffer on the unwrapped env."""
        if self.taken is not None:
            self.env.unwrapped._expert_takeover_active = self.taken.float()

    def _build_expert(self):
        expert_base, expert_peg = _import_experts()
        self._base = self.env.unwrapped
        self._servo = expert_base.Servo(self._base)
        clock = self.variant.rand_fixed_yaw > 0.0
        self._machine = expert_peg.make_machine(self._base, self._servo, clock=clock)
        self._phase_idx = {ph.name: i for i, ph in enumerate(self._machine.phases)}
        # Same upright bar as peg expert tip-recovery (cos of tilt from vertical).
        self._recover_upright = float(expert_peg.RECOVER_UPRIGHT)
        n, dev = self._base.num_envs, self._base.device
        self.taken = torch.zeros(n, dtype=torch.bool, device=dev)
        self._grasp_stall = torch.zeros(n, dtype=torch.long, device=dev)
        self._insert_stall = torch.zeros(n, dtype=torch.long, device=dev)
        self._tilt_stall = torch.zeros(n, dtype=torch.long, device=dev)
        self._prev_dist = torch.full((n,), float("nan"), device=dev)
        self._prev_cost = torch.full((n,), float("nan"), device=dev)
        self._publish_active()

    def _ee_peg_hole_quat(self):
        import warp as wp

        base = self._base
        ee = self._servo.ee()
        peg = wp.to_torch(base.scene["held_part"].data.root_pos_w) - base.scene.env_origins
        hole = wp.to_torch(base.scene["fixed_part"].data.root_pos_w) - base.scene.env_origins
        # Isaac root_quat_w is xyzw (same stack as peg expert / ObsTerm poses).
        quat = wp.to_torch(base.scene["held_part"].data.root_quat_w)
        finger = wp.to_torch(base.scene["robot"].data.joint_pos)[:, 7]
        return ee, peg, hole, quat, finger

    def _gate_grasp(self, action: torch.Tensor) -> torch.Tensor:
        """New grasp-fail fires this tick (not yet latched)."""
        ee, peg, _hole, _quat, finger = self._ee_peg_hole_quat()
        dist = ee_peg_xy_dist(ee, peg)
        have_prev = torch.isfinite(self._prev_dist)
        retreating = have_prev & (dist > self._prev_dist + RETREAT_EPS)
        closed = grip_closed(action, finger)
        on_stand = peg_on_stand(peg[:, 2], TABLE_TOP_Z, self.variant.lift_clear)
        self._grasp_stall = update_stall(
            self._grasp_stall, closed=closed, retreating=retreating, on_stand=on_stand)
        self._prev_dist = dist.detach()
        return grasp_fail_fire(self._grasp_stall, window=self.grasp_stall_steps) & ~self.taken

    def _gate_insert(self, action: torch.Tensor) -> torch.Tensor:
        """At-seat only: clearly off-center, or ramming with no progress."""
        _ee, peg, hole, _quat, finger = self._ee_peg_hole_quat()
        closed = grip_closed(action, finger)
        seated = at_seat(peg, hole)
        off_center = peg_hole_xy(peg, hole) > self.xy_ok
        cost = seat_cost(peg, hole)
        have_prev = torch.isfinite(self._prev_cost)
        improving = have_prev & (cost < self._prev_cost - PROGRESS_EPS)
        self._insert_stall = update_insert_stall(
            self._insert_stall,
            closed=closed,
            at_seat_now=seated,
            off_center=off_center,
            improving=improving,
        )
        self._prev_cost = cost.detach()
        return insert_fail_fire(self._insert_stall, window=self.insert_stall_steps) & ~self.taken

    def _gate_tilt(self, action: torch.Tensor) -> torch.Tensor:
        """Tip at seat (≥18°) or hard tip elsewhere (≥30°); not nearly seated."""
        _ee, peg, hole, quat, finger = self._ee_peg_hole_quat()
        closed = grip_closed(action, finger)
        seated = at_seat(peg, hole)
        cos = peg_upright_cos(quat)
        # Softer threshold while pressing at the seat (sideways jam).
        tipped = torch.where(
            closed & seated,
            cos < TILT_COS_INSERT,
            cos < TILT_COS,
        )
        # Ignore resting upright on the table / stand plane.
        above_table = peg[:, 2] > (TABLE_TOP_Z + 0.005)
        tipped = tipped & above_table
        seated = nearly_seated(peg, hole)
        self._tilt_stall = update_tilt_stall(
            self._tilt_stall, tipped=tipped, seated=seated)
        return tilt_fail_fire(self._tilt_stall, window=self.tilt_stall_steps) & ~self.taken

    def _handoff_regrasp(self, ids: torch.Tensor, reason: str):
        """Tilt-in-hand → straighten; grasp miss / dropped peg → hover regrasp."""
        if reason.startswith("grasp"):
            self._machine.restart(ids)
            print(f"[takeover] {reason} → expert (regrasp) on slots {ids.tolist()}", flush=True)
            return
        _ee, peg, _hole, _quat, finger = self._ee_peg_hole_quat()
        # Still pinching a raised peg → straighten wrist in place; else regrasp.
        closed = finger >= FINGER_CLOSED
        raised = peg[:, 2] > (TABLE_TOP_Z + 0.015)
        can_straighten = closed & raised
        to_straighten = ids[can_straighten[ids]]
        to_hover = ids[~can_straighten[ids]]
        if to_straighten.numel():
            self._machine.goto(to_straighten, self._phase_idx["straighten"])
        if to_hover.numel():
            self._machine.restart(to_hover)
        print(
            f"[takeover] {reason} → expert "
            f"straighten={to_straighten.tolist()} regrasp={to_hover.tolist()}",
            flush=True,
        )

    def _handoff_insert(self, ids: torch.Tensor):
        """Held peg → straighten if tilted; else align (center) or transport."""
        _ee, peg, hole, quat, _finger = self._ee_peg_hole_quat()
        cos = peg_upright_cos(quat)
        tipped = cos < self._recover_upright
        to_straighten = ids[tipped[ids]]
        kept = ids[~tipped[ids]]
        if to_straighten.numel():
            self._machine.goto(to_straighten, self._phase_idx["straighten"])
        xy = peg_hole_xy(peg, hole)
        # Near hole → align (center on bore) then press — never shimmy-press off-center.
        to_align = kept[xy[kept] < ALIGN_HANDOFF_XY]
        to_carry = kept[xy[kept] >= ALIGN_HANDOFF_XY]
        if to_carry.numel():
            self._machine.goto(to_carry, self._phase_idx["transport"])
        if to_align.numel():
            self._machine.goto(to_align, self._phase_idx["align"])
        print(
            f"[takeover] insert-fail → expert "
            f"straighten={to_straighten.tolist()} "
            f"transport={to_carry.tolist()} align={to_align.tolist()}",
            flush=True,
        )

    def _expert_action(self) -> torch.Tensor:
        """Advance the machine only on taken slots; freeze the rest."""
        free = ~self.taken
        ph = self._machine.phase.clone()
        tm = self._machine.timer.clone()
        failed = self._machine.failed.clone()
        fp = self._machine.failure_phase.clone()
        act = self._machine.action()
        if free.any():
            self._machine.phase[free] = ph[free]
            self._machine.timer[free] = tm[free]
            self._machine.failed[free] = failed[free]
            self._machine.failure_phase[free] = fp[free]
        return act

    def _clear_slots(self, ids: torch.Tensor):
        if ids.numel() == 0:
            return
        self.taken[ids] = False
        self._grasp_stall[ids] = 0
        self._insert_stall[ids] = 0
        self._tilt_stall[ids] = 0
        self._prev_dist[ids] = float("nan")
        self._prev_cost[ids] = float("nan")
        self._machine.restart(ids)

    def reset(self, **kwargs):
        if self.enabled:
            n = self._base.num_envs
            self._clear_slots(torch.arange(n, device=self._base.device))
            self._publish_active()
        obs, info = self.env.reset(**kwargs)
        if self.enabled:
            info = {**(info or {}), "expert_active": self.taken.detach().cpu().numpy()}
        return obs, info

    def step(self, action):
        if not self.enabled:
            return self.env.step(action)

        act = action if isinstance(action, torch.Tensor) else torch.as_tensor(
            action, device=self._base.device, dtype=torch.float32)
        if act.ndim == 1:
            act = act.unsqueeze(0)

        # Detect → latch → handoff. Latched slots keep the expert until done.
        new_grasp = (
            self._gate_grasp(act) if self.mode in ("grasp", "both")
            else torch.zeros_like(self.taken)
        )
        new_insert = (
            self._gate_insert(act) if self.mode in ("insert", "both")
            else torch.zeros_like(self.taken)
        )
        new_tilt = self._gate_tilt(act)  # always on with takeover
        # Priority: tilt (needs regrasp) > insert (held) > grasp (missed pick).
        new_insert = new_insert & ~new_tilt
        new_grasp = new_grasp & ~new_tilt & ~new_insert

        if new_tilt.any():
            ids = new_tilt.nonzero(as_tuple=False).squeeze(-1)
            self._handoff_regrasp(ids, "tilt-fail")
            self.taken = self.taken | new_tilt
        if new_grasp.any():
            ids = new_grasp.nonzero(as_tuple=False).squeeze(-1)
            self._handoff_regrasp(ids, "grasp-fail")
            self.taken = self.taken | new_grasp
        if new_insert.any():
            ids = new_insert.nonzero(as_tuple=False).squeeze(-1)
            self._handoff_insert(ids)
            self.taken = self.taken | new_insert

        if self.taken.any():
            expert = self._expert_action()
            act = torch.where(self.taken.unsqueeze(-1), expert, act)

        # Wire the applied command for HG-DAgger labels (ObsTerm executed_action).
        self.env.unwrapped._expert_executed_action = act.detach()
        self._publish_active()
        obs, rew, term, trunc, info = self.env.step(act)

        # Clear latch only on episode end (next episode starts under the policy).
        done = _as_bool_mask(term, self._base.num_envs, self._base.device) | _as_bool_mask(
            trunc, self._base.num_envs, self._base.device)
        if done.any():
            self._clear_slots(done.nonzero(as_tuple=False).squeeze(-1))
            self._publish_active()
            pol = obs.get("policy") if isinstance(obs, dict) else None
            if isinstance(pol, dict) and "expert_active" in pol:
                pol["expert_active"] = pol["expert_active"].clone()
                pol["expert_active"][done] = 0

        info = {**(info or {}), "expert_active": self.taken.detach().cpu().numpy()}
        return obs, rew, term, trunc, info

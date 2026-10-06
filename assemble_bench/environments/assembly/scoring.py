"""Staged-reward weights and the 0..1 dense episode score.

Isaac-free so the HUD server process can score an episode without booting Kit;
``rewards.py`` pays these weights per step inside the sim.
"""

# Once-fired milestones.
W_LIFT = 0.5
W_ENGAGE = 0.4
W_THREAD_START = 1.5   # nut: first meaningful turn on the bolt
W_SUCCESS = 2.0

# Continuous new-best potentials (total contribution capped by weight * 1.0).
W_GRASP_PC = 0.2
W_ALIGN_PC = 0.3
W_DEPTH_PC = 0.5
W_THREAD_PC = 0.8

# Score ceiling for an episode that never succeeds; only success scores 1.0.
PARTIAL_CAP = 0.9


def dense_score(family: str, success: bool, total_reward: float) -> float:
    """Episode score in [0, 1]: 1.0 on success, else the share of the staged reward banked.

    ``total_reward`` is the episode's accumulated staged reward; it never decreases, so
    the score only rises with progress. The share is taken against the most a
    non-succeeding episode can bank (every milestone and potential except success).
    """
    if success:
        return 1.0
    partial_max = W_LIFT + W_ENGAGE + W_GRASP_PC + W_ALIGN_PC + W_DEPTH_PC
    if family == "nut_thread":
        partial_max += W_THREAD_START + W_THREAD_PC
    return PARTIAL_CAP * min(total_reward / partial_max, 1.0)

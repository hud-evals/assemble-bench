import pytest

from assemble_bench.environments.assembly import scoring
from assemble_bench.environments.assembly.scoring import PARTIAL_CAP, dense_score

PEG_PARTIAL_MAX = (
    scoring.W_LIFT + scoring.W_ENGAGE + scoring.W_GRASP_PC + scoring.W_ALIGN_PC + scoring.W_DEPTH_PC
)
NUT_PARTIAL_MAX = PEG_PARTIAL_MAX + scoring.W_THREAD_START + scoring.W_THREAD_PC


@pytest.mark.parametrize("family", ["peg_insert", "gear_mesh", "nut_thread"])
def test_success_scores_one_whatever_was_banked(family):
    assert dense_score(family, True, 0.0) == 1.0
    assert dense_score(family, True, 100.0) == 1.0


@pytest.mark.parametrize("family", ["peg_insert", "gear_mesh", "nut_thread"])
def test_no_progress_scores_zero(family):
    assert dense_score(family, False, 0.0) == 0.0


def test_partial_credit_is_the_banked_share_of_the_non_success_maximum():
    assert dense_score("peg_insert", False, scoring.W_LIFT) == pytest.approx(
        PARTIAL_CAP * scoring.W_LIFT / PEG_PARTIAL_MAX
    )
    assert dense_score("nut_thread", False, scoring.W_LIFT) == pytest.approx(
        PARTIAL_CAP * scoring.W_LIFT / NUT_PARTIAL_MAX
    )


def test_score_rises_with_banked_reward_and_stays_below_one_without_success():
    banked = [0.0, 0.1, 0.5, 1.0, PEG_PARTIAL_MAX, PEG_PARTIAL_MAX + scoring.W_SUCCESS, 50.0]
    scores = [dense_score("peg_insert", False, r) for r in banked]
    assert scores == sorted(scores)
    assert all(0.0 <= s <= PARTIAL_CAP < 1.0 for s in scores)

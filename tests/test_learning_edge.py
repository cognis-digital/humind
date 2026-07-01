"""Edge cases, convergence, and error paths for the reinforcement layer.

ValueLearner: TD(lambda) parameter validation, eligibility-trace decay and
pruning, multi-step credit assignment, discount/lambda extremes, sign of reward.
LexiconLearner: delta-rule validation, monotone convergence to the reward sign,
bounded valence, and the overrides() confidence gate.
"""

from __future__ import annotations

import math

import pytest

from humind.learning import LexiconLearner, ValueLearner


# --- ValueLearner: parameter validation --------------------------------------
@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.1, 2.0])
def test_alpha_out_of_range_rejected(alpha):
    with pytest.raises(ValueError):
        ValueLearner(alpha=alpha)


@pytest.mark.parametrize("gamma", [-0.1, 1.01])
def test_gamma_out_of_range_rejected(gamma):
    with pytest.raises(ValueError):
        ValueLearner(gamma=gamma)


@pytest.mark.parametrize("lam", [-0.1, 1.01])
def test_lambda_out_of_range_rejected(lam):
    with pytest.raises(ValueError):
        ValueLearner(lam=lam)


def test_gamma_zero_and_one_allowed():
    assert ValueLearner(gamma=0.0).gamma == 0.0
    assert ValueLearner(gamma=1.0, lam=1.0).lam == 1.0


# --- ValueLearner: reward validation -----------------------------------------
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_reinforce_rejects_nonfinite(bad):
    v = ValueLearner()
    v.attend(["x"])
    with pytest.raises(ValueError):
        v.reinforce(bad)


def test_reinforce_rejects_non_number():
    v = ValueLearner()
    v.attend(["x"])
    with pytest.raises(TypeError):
        v.reinforce("big")


# --- ValueLearner: dynamics --------------------------------------------------
def test_reinforce_with_no_attend_is_noop():
    assert ValueLearner().reinforce(1.0) == {}


def test_attend_sets_value_zero_baseline():
    v = ValueLearner()
    v.attend(["gap"])
    assert v.value["gap"] == 0.0


def test_accumulating_trace_on_repeat_attend():
    v = ValueLearner()
    v.attend(["gap"]); v.attend(["gap"])
    # decayed once then +1 again -> trace > 1.0 for the accumulating scheme
    assert v.trace["gap"] > 1.0


def test_recent_item_gets_more_credit():
    v = ValueLearner()
    v.attend(["early"]); v.attend(["late"])
    v.reinforce(1.0)
    assert v.value["late"] > v.value["early"] > 0


def test_negative_reward_makes_value_negative():
    v = ValueLearner()
    v.attend(["bad"])
    v.reinforce(-1.0)
    assert v.value["bad"] < 0


def test_trace_prunes_when_negligible():
    v = ValueLearner(gamma=0.5, lam=0.1)             # fast trace decay
    v.attend(["stale"])
    for _ in range(20):
        v.attend(["fresh"])                          # 'stale' trace decays toward 0
    assert "stale" not in v.trace


def test_gamma_lambda_zero_means_only_current_item_credited():
    v = ValueLearner(gamma=0.0, lam=0.0)
    v.attend(["first"])
    v.attend(["second"])                             # decays 'first' trace to ~0
    updated = v.reinforce(1.0)
    assert updated["second"] > 0
    assert v.value.get("first", 0.0) == 0.0          # no credit flowed back


def test_valued_returns_sorted_top_n():
    # separate episodes: each item attended and rewarded in isolation
    v = ValueLearner()
    v.attend(["a"]); v.reinforce(1.0)
    v.trace.clear()                                   # end the episode cleanly
    v.attend(["b"]); v.reinforce(2.0)
    top = v.valued(1)
    assert len(top) == 1 and top[0][0] == "b"         # b got the larger reward


def test_valued_default_n_is_five():
    v = ValueLearner()
    for w in "abcdefg":
        v.attend([w]); v.reinforce(1.0)
    assert len(v.valued()) == 5


# --- LexiconLearner ----------------------------------------------------------
@pytest.mark.parametrize("eta", [0.0, -0.1, 1.5])
def test_lexicon_eta_validation(eta):
    with pytest.raises(ValueError):
        LexiconLearner(eta=eta)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_lexicon_update_rejects_nonfinite_reward(bad):
    with pytest.raises(ValueError):
        LexiconLearner().update(["w"], bad)


def test_lexicon_converges_toward_negative():
    lx = LexiconLearner()
    prev = 0.0
    for _ in range(20):
        lx.update(["shadowfleet"], -1.0)
        cur = lx.learned["shadowfleet"]
        assert cur <= prev                            # monotone decreasing
        prev = cur
    assert lx.learned["shadowfleet"] < -0.5


def test_lexicon_converges_toward_positive():
    lx = LexiconLearner()
    for _ in range(20):
        lx.update(["allclear"], 1.0)
    assert lx.learned["allclear"] > 0.5


def test_lexicon_valence_bounded():
    lx = LexiconLearner(eta=1.0)                      # aggressive rate
    for _ in range(50):
        lx.update(["w"], 5.0)                         # reward clamped to +1 target
    assert -1.0 <= lx.learned["w"] <= 1.0


def test_lexicon_overrides_gate_by_confidence():
    lx = LexiconLearner()
    lx.update(["weak"], 0.05)                         # nudged only slightly
    lx.update(["strong"], -1.0)
    lx.update(["strong"], -1.0)
    ov = lx.overrides()
    assert "strong" in ov and "weak" not in ov        # below 0.1 gate excluded


def test_lexicon_zero_reward_leaves_word_neutral():
    lx = LexiconLearner()
    for _ in range(5):
        lx.update(["mid"], 0.0)
    assert lx.learned["mid"] == 0.0
    assert "mid" not in lx.overrides()


def test_lexicon_multiple_words_updated_together():
    lx = LexiconLearner()
    lx.update(["one", "two", "three"], -1.0)
    assert all(lx.learned[w] < 0 for w in ("one", "two", "three"))

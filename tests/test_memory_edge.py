"""Edge cases, error paths, and dynamics for the memory model (humind.memory).

Working-memory capacity/decay/threshold validation and eviction, re-activation
ordering, episodic recency/search error paths, semantic query filters,
associative Hebbian symmetry / capping / spreading activation, and CLS
consolidation thresholds and idempotence.
"""

from __future__ import annotations

import pytest

from humind.memory import (AssociativeMemory, CognitiveMemory, EpisodicMemory,
                           SemanticMemory, WorkingMemory, WorkingItem)


# --- WorkingMemory: construction validation ----------------------------------
@pytest.mark.parametrize("cap", [0, -1, -10])
def test_working_capacity_must_be_positive(cap):
    with pytest.raises(ValueError):
        WorkingMemory(capacity=cap)


@pytest.mark.parametrize("decay", [0.0, 1.0, 1.5, -0.2])
def test_working_decay_must_forget(decay):
    with pytest.raises(ValueError):
        WorkingMemory(decay=decay)


def test_working_negative_threshold_rejected():
    with pytest.raises(ValueError):
        WorkingMemory(threshold=-0.1)


def test_working_valid_construction():
    wm = WorkingMemory(capacity=5, decay=0.5, threshold=0.1)
    assert wm.capacity == 5 and wm.decay == 0.5


# --- WorkingMemory: dynamics -------------------------------------------------
def test_capacity_eviction_keeps_most_active():
    wm = WorkingMemory(capacity=2)
    wm.add("low", 0.2); wm.add("mid", 0.6); wm.add("high", 1.0)
    contents = {i.content for i in wm.items}
    assert contents == {"mid", "high"} and "low" not in contents


def test_reactivation_bumps_and_caps_at_two():
    wm = WorkingMemory()
    wm.add("x", 1.0); wm.add("x", 1.0); wm.add("x", 1.0)
    assert len(wm.items) == 1
    assert wm.items[0].activation == 2.0            # capped


def test_reactivation_reorders_focus():
    wm = WorkingMemory(capacity=3)
    wm.add("a", 1.0); wm.add("b", 0.5)
    wm.add("b", 1.0)                                 # b now most active
    assert wm.focus(1) == ["b"]
    # list stays sorted after reactivation (no stale ordering)
    assert [i.content for i in wm.items] == ["b", "a"]


def test_new_item_min_activation_floor():
    wm = WorkingMemory()
    wm.add("y", 0.0)
    assert wm.items[0].activation == pytest.approx(0.1)


def test_tick_decays_and_prunes_below_threshold():
    wm = WorkingMemory(decay=0.5, threshold=0.3)
    wm.add("z", 0.4)                                 # 0.4 -> 0.2 < 0.3 after one tick
    wm.tick()
    assert wm.items == []


def test_tick_keeps_items_above_threshold():
    wm = WorkingMemory(decay=0.9, threshold=0.1)
    wm.add("keep", 1.0)
    wm.tick()
    assert [i.content for i in wm.items] == ["keep"]


def test_focus_returns_top_n_by_activation():
    wm = WorkingMemory()
    wm.add("a", 0.3); wm.add("b", 0.9); wm.add("c", 0.6)
    assert wm.focus(2) == ["b", "c"]


def test_focus_on_empty_is_empty():
    assert WorkingMemory().focus() == []


def test_working_item_defaults():
    it = WorkingItem("hello")
    assert it.activation == 1.0 and it.content == "hello"


# --- EpisodicMemory ----------------------------------------------------------
def _frame(text="", entities=None):
    class F:
        pass
    f = F()
    f.text = text
    f.entities = entities or []
    return f


def test_episodic_recent_orders_and_limits():
    ep = EpisodicMemory()
    for i in range(5):
        ep.record(float(i), _frame(text=f"e{i}"))
    assert [f.text for f in ep.recent(2)] == ["e3", "e4"]


def test_episodic_recent_zero_is_empty():
    ep = EpisodicMemory()
    ep.record(1.0, _frame(text="only"))
    assert ep.recent(0) == []


def test_episodic_recent_negative_raises():
    with pytest.raises(ValueError):
        EpisodicMemory().recent(-1)


def test_episodic_search_case_insensitive():
    ep = EpisodicMemory()
    ep.record(1.0, _frame(text="Vessel went DARK"))
    assert ep.search("dark") and ep.search("VESSEL")


def test_episodic_search_no_match_returns_empty():
    ep = EpisodicMemory()
    ep.record(1.0, _frame(text="all clear"))
    assert ep.search("submarine") == []


# --- SemanticMemory ----------------------------------------------------------
def test_semantic_dedupe_facts():
    sm = SemanticMemory()
    sm.assert_fact("v", "salience", "high")
    sm.assert_fact("v", "salience", "high")
    assert len(sm.facts) == 1


def test_semantic_query_by_each_field():
    sm = SemanticMemory()
    sm.assert_fact("v1", "salience", "high")
    sm.assert_fact("v2", "salience", "low")
    sm.assert_fact("v1", "flag", "sanctioned")
    assert len(sm.query(subject="v1")) == 2
    assert sm.query(obj="high") == [("v1", "salience", "high")]
    assert len(sm.query(predicate="salience")) == 2


def test_semantic_query_all_when_no_filter():
    sm = SemanticMemory()
    sm.assert_fact("a", "b", "c")
    assert len(sm.query()) == 1


def test_semantic_query_no_match():
    sm = SemanticMemory()
    sm.assert_fact("a", "b", "c")
    assert sm.query(subject="zzz") == []


# --- AssociativeMemory -------------------------------------------------------
def test_assoc_construction_rejects_nonpositive_cap():
    with pytest.raises(ValueError):
        AssociativeMemory(cap=0)


def test_assoc_symmetric_weights():
    am = AssociativeMemory()
    am.coactivate(["a", "b"])
    assert am.w["a"]["b"] == am.w["b"]["a"] == 1.0


def test_assoc_dedupes_within_a_moment():
    am = AssociativeMemory()
    am.coactivate(["a", "a", "b"])                   # 'a' repeated in one moment
    assert am.w["a"]["b"] == 1.0                     # counted once, not twice


def test_assoc_accumulates_across_moments_and_caps():
    am = AssociativeMemory(cap=3.0)
    for _ in range(10):
        am.coactivate(["a", "b"])
    assert am.w["a"]["b"] == 3.0                      # capped


def test_assoc_spread_ranks_by_strength_and_excludes_seeds():
    am = AssociativeMemory()
    am.coactivate(["v", "x"]); am.coactivate(["v", "x"]); am.coactivate(["v", "y"])
    spread = dict(am.spread(["v"]))
    assert spread["x"] > spread["y"]
    assert "v" not in spread


def test_assoc_spread_unknown_seed_empty():
    assert AssociativeMemory().spread(["ghost"]) == []


def test_assoc_associates_unknown_empty():
    assert AssociativeMemory().associates("ghost") == []


def test_assoc_associates_orders_by_weight():
    am = AssociativeMemory()
    am.coactivate(["hub", "near"]); am.coactivate(["hub", "near"])
    am.coactivate(["hub", "far"])
    assert am.associates("hub")[0] == "near"


# --- CognitiveMemory / CLS consolidation -------------------------------------
def test_consolidation_requires_min_count():
    mem = CognitiveMemory()
    mem.episodic.record(1, _frame(entities=["Solo"]))
    mem.episodic.record(2, _frame(entities=["Twice"]))
    mem.episodic.record(3, _frame(entities=["Twice"]))
    new = mem.consolidate(min_count=2)
    keys = {f[0] for f in new}
    assert "twice" in keys and "solo" not in keys


def test_consolidation_is_idempotent():
    mem = CognitiveMemory()
    mem.episodic.record(1, _frame(entities=["Dup"]))
    mem.episodic.record(2, _frame(entities=["Dup"]))
    first = mem.consolidate(min_count=2)
    second = mem.consolidate(min_count=2)             # already asserted -> nothing new
    assert first and second == []


def test_consolidation_case_folded_key():
    mem = CognitiveMemory()
    mem.episodic.record(1, _frame(entities=["Neptune-Star"]))
    mem.episodic.record(2, _frame(entities=["neptune-star"]))
    new = mem.consolidate(min_count=2)
    assert ("neptune-star", "consolidated", "2") in new


def test_cognitive_memory_default_stores_present():
    mem = CognitiveMemory()
    assert isinstance(mem.working, WorkingMemory)
    assert isinstance(mem.episodic, EpisodicMemory)
    assert isinstance(mem.semantic, SemanticMemory)
    assert isinstance(mem.assoc, AssociativeMemory)

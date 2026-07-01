"""Edge cases and integration paths for the cognitive loop (humind.mind).

Covers the perceive -> attend -> remember pipeline, predictive-coding surprise,
precision-weighted attention (urgent+surprising grabs focus), the RL reinforce
path end-to-end, value+centrality-weighted priorities, CLS consolidation via the
Mind facade, spreading-activation prediction, and the agentlex tandem
express/ingest round-trip plus its clear error when the frame/dep is missing.
"""

from __future__ import annotations

import pytest

from humind import extract
from humind.mind import Mind, _slug, _affect_label


# --- helpers -----------------------------------------------------------------
def test_slug_normalises():
    assert _slug("NEPTUNE STAR!") == "neptune-star"
    assert _slug("!!!") == "x"                       # nothing left -> placeholder


def test_affect_label_thresholds():
    assert _affect_label(-0.5) == "high"
    assert _affect_label(0.5) == "low"
    assert _affect_label(0.0) == "neutral"


# --- perception loop ---------------------------------------------------------
def test_perceive_records_episodic_and_advances_clock():
    m = Mind()
    m.perceive("first"); m.perceive("second")
    assert len(m.memory.episodic.events) == 2
    assert m._clock == 2


def test_perceive_populates_working_and_semantic():
    m = Mind("scout")
    m.perceive("URGENT: vessel NEPTUNE-STAR went dark")
    assert "NEPTUNE-STAR" in m.attention()
    assert m.memory.semantic.query(subject="neptune-star")


def test_first_input_is_maximally_surprising():
    m = Mind()
    f = m.perceive("brand new contact ORACLE-WIND appeared")
    assert f.surprise == 1.0


def test_repeat_reduces_surprise():
    m = Mind()
    m.perceive("vessel darkening near hormuz")
    second = m.perceive("vessel darkening near hormuz")
    assert second.surprise < 1.0


def test_empty_perceive_has_zero_surprise():
    m = Mind()
    f = m.perceive("")
    assert f.surprise == 0.0


def test_precision_weighting_urgent_beats_calm():
    m = Mind()
    m.perceive("Routine: vessel CALM-SEA steady on course")
    m.perceive("URGENT: vessel ALARM-BELL is a critical threat now")
    # the urgent+surprising contact should out-rank the calm one in focus
    assert "ALARM-BELL" in m.attention(1)


def test_recall_searches_episodes():
    m = Mind()
    m.perceive("the vessel went dark")
    assert m.recall("dark") and m.recall("submarine") == []


# --- reinforcement end-to-end ------------------------------------------------
def test_reinforce_learns_unknown_word_valence():
    m = Mind()
    assert extract("a shadowfleet sighting").valence == 0.0
    for _ in range(6):
        m.perceive("detected a shadowfleet maneuver")
        m.reinforce(-1.0)
    assert m.perceive("another shadowfleet contact").valence < 0


def test_reinforce_returns_credited_values():
    m = Mind()
    m.perceive("gap near strait")
    updated = m.reinforce(1.0)
    assert updated and all(isinstance(v, float) for v in updated.values())


def test_reinforce_rejects_nonfinite():
    m = Mind()
    m.perceive("something")
    with pytest.raises(ValueError):
        m.reinforce(float("inf"))


def test_reinforce_before_perceive_is_safe():
    # nothing attended and no episode -> empty credit, no crash
    assert Mind().reinforce(1.0) == {}


# --- priorities / prediction / systems --------------------------------------
def test_priorities_blend_activation_value_centrality():
    m = Mind()
    m.perceive("patrols reduces smuggling")
    m.perceive("smuggling drives darkness")
    pr = m.priorities()
    assert pr and isinstance(pr, list)


def test_predict_returns_associates_after_cooccurrence():
    m = Mind()
    m.perceive("tanker rendezvous transfer at night")
    m.perceive("tanker rendezvous transfer at night")
    assert m.predict("tanker")


def test_predict_unknown_cue_empty():
    m = Mind()
    m.perceive("vessel darkening")
    assert m.predict("nonexistent-concept") == []


def test_consolidate_via_mind_promotes_recurring():
    m = Mind()
    m.perceive("vessel NEPTUNE-STAR went dark")
    m.perceive("NEPTUNE-STAR is a high risk threat")
    new = m.consolidate(min_count=2)
    assert any(f[0] == "neptune-star" for f in new)


def test_feedback_loops_and_leverage_via_mind():
    m = Mind()
    for s in ["darkness drives escalation", "escalation drives sanctions",
              "sanctions drives smuggling", "smuggling drives darkness"]:
        m.perceive(s)
    loops = m.feedback_loops()
    assert loops
    assert m.leverage_points(1)


# --- agentlex tandem ---------------------------------------------------------
def test_express_without_perception_raises_valueerror():
    with pytest.raises(ValueError):
        Mind().express()


def test_express_and_ingest_roundtrip():
    import agentlex
    scout, command = Mind("scout"), Mind("command")
    scout.perceive("CRITICAL: vessel NEPTUNE-STAR is a high risk threat")
    msg = scout.express()
    assert isinstance(msg, agentlex.Message)
    assert msg.sender == "scout" and msg.performative == "inform"
    command.ingest(agentlex.from_wire(msg.to_wire()))
    assert "neptune-star" in command.attention()
    assert any(f[1] == "observed" for f in command.memory.semantic.query())


def test_express_uses_context_when_no_entity():
    import agentlex  # noqa: F401
    m = Mind("watch")
    m.perceive("all quiet, nothing to report")     # no capitalised entity
    msg = m.express()
    assert msg is not None                          # still expresses a frame


def test_intent_maps_to_performative():
    import agentlex  # noqa: F401
    m = Mind("q")
    f = m.perceive("Is the corridor safe?")         # query intent
    msg = m.express(f)
    assert msg.performative == "query"

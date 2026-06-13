"""Tests for humind: extraction, memory, the cognitive loop, and the agentlex tandem."""

from __future__ import annotations

from humind import extract
from humind.memory import WorkingMemory, SemanticMemory
from humind.mind import Mind


# --- extraction --------------------------------------------------------------
def test_intent_classification():
    assert extract("Is the vessel high risk?").intent == "query"
    assert extract("Please scan the corridor.").intent == "request"
    assert extract("I think we should reroute.").intent == "propose"
    assert extract("The vessel went dark.").intent == "inform"


def test_entities_and_affect():
    f = extract("CRITICAL: vessel NEPTUNE-STAR is a high risk threat")
    assert "NEPTUNE-STAR" in f.entities
    assert f.valence < 0 and f.arousal > 0      # threatening + urgent


def test_salient_terms():
    f = extract("drone drone radar acoustic drone detection detection")
    assert f.salient[0] == "drone"               # most frequent content word


# --- memory ------------------------------------------------------------------
def test_working_memory_capacity_and_decay():
    wm = WorkingMemory(capacity=3, decay=0.5, threshold=0.3)
    for c in ["a", "b", "c", "d"]:
        wm.add(c, 1.0)
    assert len(wm.items) == 3                     # capacity enforced
    wm.tick(); wm.tick()                          # activation 1.0 -> 0.5 -> 0.25 < 0.3
    assert wm.items == []                         # decayed below threshold


def test_working_memory_reactivation_and_focus():
    wm = WorkingMemory()
    wm.add("vessel", 1.0); wm.add("radar", 0.4); wm.add("vessel", 1.0)  # re-activate
    assert wm.focus(1) == ["vessel"]


def test_semantic_query():
    sm = SemanticMemory()
    sm.assert_fact("vessel-1", "salience", "high")
    assert sm.query(predicate="salience") == [("vessel-1", "salience", "high")]


# --- cognitive loop ----------------------------------------------------------
def test_mind_perceive_builds_attention_and_facts():
    m = Mind("scout")
    m.perceive("URGENT: vessel NEPTUNE-STAR went dark")
    assert "NEPTUNE-STAR" in m.attention()
    assert m.memory.semantic.query(subject="neptune-star")
    assert m.recall("dark")                       # episodic search


# --- agentlex tandem ---------------------------------------------------------
def test_tandem_express_and_ingest():
    import agentlex  # provided on PYTHONPATH locally / pip-installed in CI
    scout, command = Mind("scout"), Mind("command")
    scout.perceive("CRITICAL: vessel NEPTUNE-STAR is a high risk threat")
    msg = scout.express()
    assert isinstance(msg, agentlex.Message)
    assert msg.performative == "inform" and msg.sender == "scout"
    # round-trips through the wire and into command's memory
    heard = agentlex.from_wire(msg.to_wire())
    command.ingest(heard)
    facts = command.memory.semantic.query()
    assert any(f[1] == "observed" for f in facts)
    assert "neptune-star" in command.attention()


def test_cli_demo(capsys):
    from humind.cli import main
    assert main(["demo"]) == 0
    assert "scout says" in capsys.readouterr().out

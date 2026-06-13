"""Tests for the v0.3 cognition layers: causal extraction, negation-aware affect,
Hebbian association + spreading activation, TD(lambda) reinforcement + online valence,
predictive-coding surprise, CLS consolidation, systems-thinking loops, and the CLI."""

from __future__ import annotations

from humind import extract, causal_links
from humind.extract import affect
from humind.learning import LexiconLearner, ValueLearner
from humind.memory import AssociativeMemory, CognitiveMemory
from humind.mind import Mind
from humind.systems import CausalGraph
from humind.cli import main


# --- extraction: causal + negation ------------------------------------------
def test_causal_link_polarity():
    links = causal_links("The AIS gap leads to escalation risk")
    assert ("ais gap", "escalation risk", 1) in links
    neg = causal_links("Convoy escort reduces piracy incidents")
    assert any(c == "convoy escort" and p == -1 for c, e, p in neg)


def test_causal_reversed_connective():
    links = causal_links("Escalation happened because of the AIS gap")
    assert any(c == "ais gap" and e.startswith("escalation") for c, e, p in links)


def test_negation_flips_valence():
    assert affect("the corridor is safe")[0] > 0
    assert affect("the corridor is not safe")[0] < 0     # negation flips it


def test_extract_populates_causes_and_surprise_default():
    f = extract("sanctions pressure drives dark activity")
    assert ("sanctions pressure", "dark activity", 1) in f.causes
    assert f.surprise == 0.0                              # surprise is set by Mind, not extract


# --- associative memory ------------------------------------------------------
def test_hebbian_spreading_activation():
    am = AssociativeMemory()
    am.coactivate(["vessel", "darkening", "hormuz"])
    am.coactivate(["vessel", "darkening", "tanker"])
    spread = dict(am.spread(["vessel"]))
    assert spread["darkening"] >= spread["hormuz"]         # darkening co-occurred twice
    assert "vessel" not in spread                          # seeds excluded


# --- reinforcement learning --------------------------------------------------
def test_td_lambda_credit_assignment():
    v = ValueLearner()
    v.attend(["gap"]); v.attend(["rendezvous"]); v.reinforce(1.0)
    # both recently-attended items get positive value; the most recent (largest trace) most
    assert v.value["rendezvous"] > v.value["gap"] > 0


def test_lexicon_learns_valence_from_reward():
    lx = LexiconLearner()
    for _ in range(8):
        lx.update(["shadowfleet"], -1.0)
    assert lx.overrides()["shadowfleet"] < -0.3            # acquired a negative valence


def test_mind_reinforce_makes_unknown_word_threatening():
    m = Mind()
    assert extract("a shadowfleet sighting").valence == 0.0
    for _ in range(6):
        m.perceive("detected a shadowfleet maneuver")
        m.reinforce(-1.0)
    f = m.perceive("another shadowfleet contact")          # now scored with learned valence
    assert f.valence < 0


# --- predictive coding + prediction -----------------------------------------
def test_surprise_drops_on_repetition():
    m = Mind()
    first = m.perceive("vessel darkening near hormuz")
    second = m.perceive("vessel darkening near hormuz")
    assert first.surprise == 1.0 and second.surprise < first.surprise


def test_predict_returns_associates():
    m = Mind()
    m.perceive("tanker rendezvous transfer at night")
    m.perceive("tanker rendezvous transfer at night")
    assert m.predict()                                     # something comes to mind


# --- CLS consolidation -------------------------------------------------------
def test_consolidation_promotes_recurring_entities():
    mem = CognitiveMemory()

    class F:                                               # minimal frame stub
        def __init__(self, ents): self.entities = ents
    mem.episodic.record(1, F(["Neptune-Star"]))
    mem.episodic.record(2, F(["Neptune-Star"]))
    new = mem.consolidate(min_count=2)
    assert ("neptune-star", "consolidated", "2") in new


# --- systems thinking --------------------------------------------------------
def test_reinforcing_and_balancing_loops():
    g = CausalGraph()
    g.add_link("a", "b", 1); g.add_link("b", "a", 1)       # two +: reinforcing
    g.add_link("c", "d", 1); g.add_link("d", "c", -1)      # one -: balancing
    kinds = {tuple(sorted(lp["nodes"])): lp["kind"] for lp in g.feedback_loops()}
    assert kinds[("a", "b")] == "R" and kinds[("c", "d")] == "B"


def test_leverage_points_rank_by_centrality():
    g = CausalGraph()
    g.add_link("hub", "x", 1); g.add_link("hub", "y", 1); g.add_link("hub", "z", 1)
    assert g.leverage_points(1)[0][0] == "hub"


# --- CLI ---------------------------------------------------------------------
def test_cli_causal(capsys):
    assert main(["causal", "dark activity increases sanctions pressure",
                 "sanctions pressure drives dark activity"]) == 0
    assert "feedback loops" in capsys.readouterr().out


def test_cli_learn_and_bench(capsys):
    assert main(["learn"]) == 0
    assert main(["bench", "-n", "200"]) == 0
    assert "frames/sec" in capsys.readouterr().out

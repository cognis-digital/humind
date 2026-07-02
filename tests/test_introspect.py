"""Tests for Mind.introspect() and the `humind explain` CLI subcommand.

introspect() must return a single JSON-serialisable snapshot spanning every layer,
bound each ranked section by n, validate n, and stay in sync with the individual
accessors (focus/priorities/values/loops/leverage). The CLI wrapper must emit that
snapshot as valid JSON and exit 0.
"""

from __future__ import annotations

import json

import pytest

from humind.cli import main
from humind.mind import Mind


def _loaded(name="op"):
    m = Mind(name)
    for s in ["vessel NEPTUNE-STAR darkening near the strait",
              "an AIS gap leads to escalation",
              "escalation drives sanctions",
              "sanctions drives smuggling",
              "smuggling drives darkening"]:
        m.perceive(s)
    m.reinforce(-1.0)
    return m


def test_introspect_is_json_serialisable():
    snap = _loaded().introspect()
    dumped = json.dumps(snap)                       # must not raise
    assert json.loads(dumped) == snap               # round-trips exactly


def test_introspect_has_all_sections():
    snap = _loaded().introspect()
    for key in ("name", "clock", "focus", "priorities", "values", "predict",
                "associations", "facts", "learned_valence", "loops", "leverage",
                "surprise_last", "episodes"):
        assert key in snap


def test_introspect_reports_identity_and_counts():
    m = _loaded("scout")
    snap = m.introspect()
    assert snap["name"] == "scout"
    assert snap["episodes"] == 5 and snap["clock"] == 5


def test_introspect_bounds_ranked_sections_by_n():
    snap = _loaded().introspect(n=2)
    assert len(snap["focus"]) <= 2
    assert len(snap["priorities"]) <= 2
    assert len(snap["values"]) <= 2
    assert len(snap["leverage"]) <= 2


def test_introspect_matches_individual_accessors():
    m = _loaded()
    snap = m.introspect(n=3)
    assert snap["focus"] == m.attention(3)
    assert snap["priorities"] == m.priorities(3)
    assert [lp["nodes"] for lp in snap["loops"]] == [lp["nodes"] for lp in m.feedback_loops()]


def test_introspect_associations_keyed_by_focus():
    m = _loaded()
    snap = m.introspect(n=3)
    assert set(snap["associations"]) == set(snap["focus"])


def test_introspect_n_must_be_positive():
    with pytest.raises(ValueError):
        _loaded().introspect(n=0)
    with pytest.raises(ValueError):
        _loaded().introspect(n=-1)


def test_introspect_on_fresh_mind_is_empty_but_valid():
    snap = Mind().introspect()
    assert snap["episodes"] == 0 and snap["focus"] == []
    assert snap["surprise_last"] == 0.0
    json.dumps(snap)                                # still serialisable


def test_values_and_valence_types_are_plain():
    snap = _loaded().introspect()
    assert all(isinstance(v["value"], float) for v in snap["values"])
    assert all(isinstance(x, float) for x in snap["learned_valence"].values())


# --- CLI ---------------------------------------------------------------------
def test_cli_explain_emits_valid_json(capsys):
    rc = main(["explain", "darkness drives escalation",
               "escalation drives sanctions", "sanctions drives darkness"])
    assert rc == 0
    snap = json.loads(capsys.readouterr().out)
    assert snap["name"] == "you" and "leverage" in snap


def test_cli_explain_respects_n(capsys):
    rc = main(["explain", "-n", "2", "a drives b", "b drives c", "c drives a"])
    assert rc == 0
    snap = json.loads(capsys.readouterr().out)
    assert len(snap["priorities"]) <= 2

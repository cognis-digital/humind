"""Edge cases and error paths for the systems-thinking layer (humind.systems).

CausalGraph: link normalisation (case/whitespace/self-loops), polarity coercion,
edge dedup, isolated (effect-only) nodes, cycle detection with R/B classification,
directed-cycle deduplication (the v0.3.1 fix), max_len bounding and validation,
centrality counting, and leverage-point ranking.
"""

from __future__ import annotations

import pytest

from humind.systems import CausalGraph


# --- add_link normalisation --------------------------------------------------
def test_add_link_lowercases_and_strips():
    g = CausalGraph()
    g.add_link("  Alpha ", "BETA", 1)
    assert "alpha" in g.edges and ("beta", 1) in g.edges["alpha"]


def test_self_loop_is_ignored():
    g = CausalGraph()
    g.add_link("x", "x", 1)
    assert g.edges.get("x", []) == []


def test_empty_endpoints_ignored():
    g = CausalGraph()
    g.add_link("", "b", 1)
    g.add_link("a", "   ", 1)
    assert g.edges == {}


def test_polarity_coerced_to_plus_or_minus_one():
    g = CausalGraph()
    g.add_link("a", "b", 5)                 # any positive -> +1
    g.add_link("c", "d", -3)                # any negative -> -1
    g.add_link("e", "f", 0)                 # zero -> +1 (>= 0)
    assert ("b", 1) in g.edges["a"]
    assert ("d", -1) in g.edges["c"]
    assert ("f", 1) in g.edges["e"]


def test_duplicate_link_deduped():
    g = CausalGraph()
    g.add_link("a", "b", 1)
    g.add_link("a", "b", 1)
    assert g.edges["a"] == [("b", 1)]


def test_same_pair_different_polarity_both_kept():
    g = CausalGraph()
    g.add_link("a", "b", 1)
    g.add_link("a", "b", -1)
    assert ("b", 1) in g.edges["a"] and ("b", -1) in g.edges["a"]


def test_effect_node_registered_even_with_no_outgoing():
    g = CausalGraph()
    g.add_link("cause", "effect", 1)
    assert "effect" in g.edges and g.edges["effect"] == []


def test_nodes_sorted():
    g = CausalGraph()
    g.add_link("zeta", "alpha", 1)
    assert g.nodes() == ["alpha", "zeta"]


# --- feedback loops ----------------------------------------------------------
def test_reinforcing_loop_even_negatives():
    g = CausalGraph()
    g.add_link("a", "b", 1); g.add_link("b", "a", 1)      # 0 negatives -> R
    kinds = [lp["kind"] for lp in g.feedback_loops()]
    assert kinds and all(k == "R" for k in kinds)


def test_balancing_loop_odd_negatives():
    g = CausalGraph()
    g.add_link("a", "b", 1); g.add_link("b", "a", -1)     # 1 negative -> B
    assert g.feedback_loops()[0]["kind"] == "B"


def test_two_negatives_is_reinforcing():
    g = CausalGraph()
    g.add_link("a", "b", -1); g.add_link("b", "a", -1)    # 2 negatives -> R
    assert g.feedback_loops()[0]["kind"] == "R"


def test_no_loop_when_acyclic():
    g = CausalGraph()
    g.add_link("a", "b", 1); g.add_link("b", "c", 1)
    assert g.feedback_loops() == []


def test_directed_cycles_not_collapsed_by_node_set():
    # Same three nodes, but two genuinely different directed 3-cycles.
    g = CausalGraph()
    g.add_link("a", "b", 1); g.add_link("b", "c", 1); g.add_link("c", "a", 1)
    g.add_link("a", "c", 1); g.add_link("c", "b", 1); g.add_link("b", "a", 1)
    triangles = [lp for lp in g.feedback_loops() if len(lp["nodes"]) == 3]
    assert len(triangles) == 2                            # not collapsed to one


def test_rotation_of_same_cycle_deduped():
    g = CausalGraph()
    g.add_link("a", "b", 1); g.add_link("b", "c", 1); g.add_link("c", "a", 1)
    triangles = [lp for lp in g.feedback_loops() if len(lp["nodes"]) == 3]
    assert len(triangles) == 1                            # a-b-c == b-c-a == c-a-b


def test_max_len_bounds_search():
    g = CausalGraph()
    # a 4-cycle
    g.add_link("a", "b", 1); g.add_link("b", "c", 1)
    g.add_link("c", "d", 1); g.add_link("d", "a", 1)
    assert g.feedback_loops(max_len=3) == []              # too short to close
    assert g.feedback_loops(max_len=4)                    # long enough


def test_max_len_validation():
    with pytest.raises(ValueError):
        CausalGraph().feedback_loops(max_len=0)


# --- centrality / leverage ---------------------------------------------------
def test_centrality_counts_in_and_out():
    g = CausalGraph()
    g.add_link("hub", "a", 1); g.add_link("hub", "b", 1); g.add_link("c", "hub", 1)
    cent = g.centrality()
    assert cent["hub"] == 3                               # 2 out + 1 in
    assert cent["a"] == 1


def test_leverage_points_rank_by_centrality():
    g = CausalGraph()
    g.add_link("hub", "x", 1); g.add_link("hub", "y", 1); g.add_link("hub", "z", 1)
    lev = g.leverage_points(1)
    assert lev[0][0] == "hub"


def test_leverage_points_respects_n():
    g = CausalGraph()
    for t in "bcde":
        g.add_link("a", t, 1)
    assert len(g.leverage_points(2)) == 2


def test_empty_graph_has_no_loops_or_leverage():
    g = CausalGraph()
    assert g.feedback_loops() == []
    assert g.leverage_points() == []
    assert g.centrality() == {}

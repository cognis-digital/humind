"""Smoke tests for the demo scenarios — each must import, run, and exit 0.

The demos are self-contained (bundled inline transcript, no network), so they
double as integration tests over the public API: perception, the three memory
stores, the cognitive loop, and the agentlex tandem.
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest

DEMOS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demos")
sys.path.insert(0, DEMOS)

SCENARIOS = [
    "01_perception_pipeline",
    "02_reinforcement_learning",
    "03_memory_and_association",
    "04_systems_thinking",
    "05_two_minds_tandem",
    "06_negation_and_affect",
    "07_intent_classification",
    "08_working_memory_decay",
    "09_spreading_activation",
    "10_predictive_coding",
    "11_td_credit_assignment",
    "12_online_valence_learning",
    "13_cls_consolidation",
    "14_feedback_loops",
    "15_leverage_points",
    "16_priorities_ranking",
    "17_causal_extraction",
    "18_full_cognitive_loop",
    "19_multi_agent_broadcast",
    "20_robustness_and_errors",
    "21_introspection_snapshot",
]


@pytest.mark.parametrize("name", SCENARIOS)
def test_scenario_runs(name, capsys):
    mod = importlib.import_module(name)
    mod.main()                                  # must not raise
    assert capsys.readouterr().out.strip()      # produced narrated output


def test_run_all_exits_zero(capsys):
    run_all = importlib.import_module("run_all")
    assert run_all.main() == 0
    assert "All humind demo scenarios completed." in capsys.readouterr().out


def test_common_helpers():
    import _common
    m = _common.perceived_mind()
    assert m.attention()                        # focus is non-empty after perceiving
    assert _common.bar(2.0).startswith("#")     # full bar
    assert _common.bar(0.0).startswith(".")     # empty bar

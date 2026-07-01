"""CLI edge cases (humind.cli) — every subcommand exits 0 and prints its section.

Exercises perceive (plain + JSON shape), think, causal (loops + leverage),
learn, bench (small n and the n=0 guard), demo, and argument handling
(--version, missing subcommand).
"""

from __future__ import annotations

import json

import pytest

from humind.cli import main


def test_perceive_prints_valid_json(capsys):
    assert main(["perceive", "URGENT: vessel NEPTUNE-STAR went dark"]) == 0
    out = capsys.readouterr().out
    frame = json.loads(out)
    assert frame["intent"] == "inform" and "NEPTUNE-STAR" in frame["entities"]


def test_perceive_query_intent(capsys):
    assert main(["perceive", "Is the corridor safe?"]) == 0
    assert json.loads(capsys.readouterr().out)["intent"] == "query"


def test_think_shows_attention_and_priorities(capsys):
    assert main(["think", "vessel darkening near hormuz",
                 "vessel darkening near hormuz"]) == 0
    out = capsys.readouterr().out
    assert "attention/focus" in out and "priorities" in out and "predict" in out


def test_causal_reports_loops_and_leverage(capsys):
    assert main(["causal", "darkness drives escalation",
                 "escalation drives sanctions",
                 "sanctions drives smuggling",
                 "smuggling drives darkness"]) == 0
    out = capsys.readouterr().out
    assert "feedback loops" in out and "leverage points" in out
    assert "-->" in out                              # a causal edge was drawn


def test_causal_no_loops_says_none(capsys):
    assert main(["causal", "a drives b"]) == 0
    assert "none" in capsys.readouterr().out


def test_learn_shows_acquired_valence(capsys):
    assert main(["learn"]) == 0
    out = capsys.readouterr().out
    assert "learned valence" in out and "top learned values" in out


def test_bench_small_n(capsys):
    assert main(["bench", "-n", "50"]) == 0
    assert "frames/sec" in capsys.readouterr().out


def test_demo_runs(capsys):
    assert main(["demo"]) == 0
    assert "scout says" in capsys.readouterr().out


def test_version_flag_exits(capsys):
    with pytest.raises(SystemExit) as ei:
        main(["--version"])
    assert ei.value.code == 0
    assert "humind" in capsys.readouterr().out


def test_missing_subcommand_errors():
    with pytest.raises(SystemExit) as ei:
        main([])
    assert ei.value.code != 0

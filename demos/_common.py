"""Shared helpers for the humind demo scenarios.

Every scenario is self-contained: it builds its own fresh `Mind` (or raw
extractor) from bundled, inline sample utterances, so the demos can be run in any
order or on their own — no network, no external data, no state shared between
them. The optional agentlex tandem and LLM enrichment are imported lazily by the
scenarios that use them, and degrade to a clean skip when unavailable.
"""
from __future__ import annotations

import os
import sys

# allow `python demos/xx.py` from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from humind import Mind, extract            # noqa: E402
from humind.extract import ContextFrame     # noqa: E402

# A small, bundled "transcript" — a maritime watch room, deliberately varied in
# intent (inform / query / propose / agree) and affect (calm -> critical) so the
# cognitive facets have something to chew on. No file I/O; this *is* the fixture.
TRANSCRIPT: list[str] = [
    "Routine: vessel QUIET-DAWN is steady on course, all clear.",
    "CRITICAL: vessel NEPTUNE-STAR went dark near a high-risk corridor.",
    "URGENT: NEPTUNE-STAR is now a high risk threat, immediate action needed.",
    "Is GHOST-RUNNER inside the exclusion zone?",
    "I think we should reroute the convoy away from NEPTUNE-STAR.",
    "Agreed, reroute confirmed and the corridor is now clear.",
]

# A bundle of causal statements for the systems demo. Each connective is flanked
# by a single shared concept word, so the concepts match across statements and a
# real reinforcing loop closes (darkness -> escalation -> sanctions -> smuggling
# -> darkness), with "patrols reduces smuggling" as the balancing lever.
CAUSAL_STATEMENTS: list[str] = [
    "Darkness drives escalation.",
    "Escalation drives sanctions.",
    "Sanctions drives smuggling.",
    "Smuggling drives darkness.",
    "Patrols reduces smuggling.",
]


def rule(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def bar(value: float, width: int = 24, scale: float = 2.0) -> str:
    """A tiny text activation bar in [0, scale]."""
    n = max(0, min(width, round(width * value / scale)))
    return "#" * n + "." * (width - n)


def perceived_mind(name: str = "watch", lines: list[str] | None = None) -> Mind:
    """A Mind that has already perceived the bundled transcript (or `lines`)."""
    m = Mind(name)
    for text in (lines if lines is not None else TRANSCRIPT):
        m.perceive(text)
    return m


__all__ = ["TRANSCRIPT", "CAUSAL_STATEMENTS", "rule", "bar", "perceived_mind",
           "Mind", "extract", "ContextFrame"]

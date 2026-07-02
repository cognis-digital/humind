"""Scenario 21 - one snapshot of the whole mind (tooling / observability builders).

`Mind.introspect()` returns a single JSON-serialisable dict summarising every
layer at once: current focus, blended priorities, learned values, associative
cues, durable facts, learned valences, the systems view (loops + leverage), and
the last surprise. It is the state you would log, put on a dashboard, or hand to
another service — no per-layer poking required, and only built-in types come out.
This demo drives a Mind through a short scenario, then prints the full snapshot.
"""
import json

from _common import rule
from humind import Mind


STREAM = [
    "vessel NEPTUNE-STAR darkening near the strait",
    "an AIS gap leads to escalation",
    "escalation drives sanctions",
    "sanctions drives smuggling",
    "smuggling drives darkening",
]


def main() -> None:
    rule("INTROSPECTION SNAPSHOT  -  the whole cognitive state in one call")
    m = Mind("operator")
    for line in STREAM:
        m.perceive(line)
    m.reinforce(-1.0)                            # this thread ended badly

    snap = m.introspect(n=4)

    print("\nintrospect() returns one JSON-serialisable dict spanning every layer:\n")
    print(json.dumps(snap, indent=2))

    # It really is plain JSON — this round-trips with no custom encoder.
    restored = json.loads(json.dumps(snap))
    assert restored == snap

    print("\nA few reads off the snapshot:")
    print("   focus       :", snap["focus"])
    print("   priorities  :", snap["priorities"])
    print("   leverage    :", [lp["concept"] for lp in snap["leverage"]])
    print("   episodes    :", snap["episodes"], " last surprise:", snap["surprise_last"])

    print("\nOne call, log-ready: perception, memory, learning and systems state")
    print("in a single structure you can ship straight to a dashboard or a peer service.")


if __name__ == "__main__":
    main()

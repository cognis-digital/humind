"""Scenario 18 - the whole loop, one Mind (integrators / evaluators).

A capstone that drives a single Mind through the entire cycle on one evolving
scenario: perceive a stream (building working / episodic / associative / causal
memory), watch surprise fall as it becomes familiar, deliver an outcome and let
TD(lambda) + the delta rule learn from it, consolidate what recurred, and finally
read off blended priorities, predictions, feedback loops and leverage. It is the
integration test made narrative — every subsystem, in the order the mind uses it.
"""
from _common import rule
from humind import Mind


STREAM = [
    "vessel NEPTUNE-STAR darkening near the strait",
    "an AIS gap leads to escalation",
    "escalation drives sanctions",
    "NEPTUNE-STAR still dark, sanctions expected",
    "sanctions drives smuggling",
    "smuggling drives darkening",
]


def main() -> None:
    rule("FULL COGNITIVE LOOP  -  perceive -> learn -> consolidate -> reason")
    m = Mind("operator")

    print("\n1) PERCEIVE the stream (surprise = prediction error each step):")
    for line in STREAM:
        f = m.perceive(line)
        print(f"   surprise={f.surprise:.2f} intent={f.intent:<7} "
              f"val={f.valence:+.2f}  \"{line[:38]}\"")

    print("\n2) CURRENT FOCUS (working memory):", m.attention(4))

    print("\n3) OUTCOME: this thread ended badly (reward -1.0) — credit + learn valence:")
    updated = m.reinforce(-1.0)
    print("   TD-credited items:", len(updated))
    print("   top learned values:", [f"{k}:{v:+.2f}" for k, v in m.values.valued(3)])

    print("\n4) CONSOLIDATE recurring entities into durable semantic memory:")
    print("   ", m.consolidate(min_count=2) or "(nothing recurred >=2x)")

    print("\n5) PREDICT from the current cue (spreading activation):")
    print("   ", m.predict())

    print("\n6) SYSTEMS view — feedback loops and leverage from the causal claims:")
    for lp in m.feedback_loops():
        print(f"   [{lp['kind']}] {' -> '.join(lp['nodes'])} -> {lp['nodes'][0]}")
    print("   leverage:", m.leverage_points(3))

    print("\n7) BLENDED PRIORITIES (activation + learned value + centrality):")
    print("   ", m.priorities(5))

    print("\nOne Mind, one scenario, every subsystem — perception feeds memory,")
    print("memory feeds learning, learning and structure feed prioritised reasoning.")


if __name__ == "__main__":
    main()

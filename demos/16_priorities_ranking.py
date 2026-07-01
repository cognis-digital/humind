"""Scenario 16 - what to focus on (decision-support builders).

Raw attention (working-memory activation) is only part of the story. humind's
`priorities()` blends three signals: current **activation**, learned **value**
(what past reward says was worth attending to), and causal **centrality** (systems
leverage). This demo perceives a mixed stream, rewards one thread of context, and
shows how the value-weighted ranking diverges from raw attention — the mind
promotes the context that history says matters, not just the loudest recent line.
"""
from _common import rule
from humind import Mind


def main() -> None:
    rule("VALUE-WEIGHTED PRIORITIES  -  activation + learned value + centrality")
    m = Mind("ops")

    # Establish that an early context led to a good outcome
    m.perceive("naval patrols cleared the corridor and the convoy is safe")
    m.reinforce(+1.0)                            # 'patrols/corridor/convoy' proved valuable

    # Then some causal structure and a fresh, loud-but-unproven line
    m.perceive("darkness drives escalation")
    m.perceive("escalation drives sanctions")
    m.perceive("routine chatter about weather and fuel")

    print("\nRaw attention (recent activation only):")
    print("   ", m.attention(6))

    print("\nLearned value (TD credit from the +1.0 outcome):")
    for item, val in m.values.valued(6):
        print(f"   {item:<14} value={val:+.3f}")

    print("\nCausal centrality (systems leverage):")
    for k, d in sorted(m.systems.centrality().items(), key=lambda kv: -kv[1])[:6]:
        print(f"   {k:<14} centrality={d}")

    print("\nBlended priorities (what a rational agent should focus on):")
    print("   ", m.priorities(6))

    print("\nReading: the blend lifts proven-valuable and structurally central context")
    print("above mere recency, so 'routine chatter' does not crowd out what matters.")


if __name__ == "__main__":
    main()

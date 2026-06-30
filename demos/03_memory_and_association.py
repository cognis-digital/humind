"""Scenario 3 - students & educators.

A guided tour of humind's memory model — the classic cognitive-architecture
split, plus the two mechanisms that make it feel alive, all made concrete:

  - WORKING   - small, decaying -> the current attention (ACT-R / Miller 7+/-2).
  - EPISODIC  - a time-ordered log of *what happened, when*.
  - SEMANTIC  - durable subject-predicate-object facts.
  - ASSOCIATIVE - a **Hebbian** co-occurrence network ("fire together, wire
                  together"); cueing one concept brings associated ones to mind by
                  **spreading activation** (Collins & Loftus).

Two more emergent behaviours show up here:
  - **predictive coding** — each frame's `surprise` = the fraction of it the mind
    did *not* expect, so a repeat is unsurprising and a new contact is surprising.
  - **CLS consolidation** — entities that recur across episodes are distilled from
    the fast episodic log into durable semantic facts.

Change a line in the transcript and watch which store (and which surprise) moves.
"""
from _common import TRANSCRIPT, perceived_mind, rule


def main() -> None:
    rule("MEMORY & ASSOCIATION  -  four stores, surprise, and consolidation")
    m = perceived_mind("student")
    print(f"\nPerceived {len(TRANSCRIPT)} utterances. Now open up each store:\n")

    print("WORKING (current focus, decays each tick):")
    for c in m.attention(5):
        print(f"   * {c}")

    print("\nEPISODIC (time-ordered frames; 'surprise' = predictive-coding error):")
    for i, f in enumerate(m.memory.episodic.recent(len(TRANSCRIPT)), 1):
        print(f"   t{i}: intent={f.intent:<8} val={f.valence:+.2f} surprise={f.surprise:.2f}"
              f"  \"{f.text[:38]}\"")

    print("\nASSOCIATIVE (Hebbian): what 'neptune-star' brings to mind (spreading activation):")
    for k, w in m.memory.assoc.spread(["neptune-star"], n=5):
        print(f"   neptune-star ~ {k:<16} strength={w:.1f}")
    print("   Mind.predict(cue='neptune-star') ->", m.predict("neptune-star"))

    print("\nSEMANTIC (durable facts):")
    for s, p, o in sorted(m.memory.semantic.query())[:6]:
        print(f"   ({s}, {p}, {o})")

    print("\nCLS consolidation — entities seen >= 2 times become durable facts:")
    # NEPTUNE-STAR appears across several episodes; consolidation captures that
    newly = m.consolidate(min_count=2)
    print("   newly consolidated:", newly or "(none recurred enough)")

    print("\nPredictive coding in action — a repeat is unsurprising, a new contact is not:")
    m2 = perceived_mind("student2")
    repeat = m2.perceive("URGENT: NEPTUNE-STAR is now a high risk threat")
    fresh = m2.perceive("New contact ORACLE-WIND just appeared")
    print(f"   surprise(repeat of NEPTUNE-STAR) = {repeat.surprise:.2f}")
    print(f"   surprise(new contact ORACLE-WIND) = {fresh.surprise:.2f}")


if __name__ == "__main__":
    main()

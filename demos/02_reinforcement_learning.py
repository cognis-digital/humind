"""Scenario 2 - RL researchers.

humind doesn't just remember context — it *learns which context mattered* from
outcomes, with two textbook update rules and no tensors:

  - **TD(lambda) with eligibility traces** (Sutton & Barto). Every time an item is
    attended it leaves a decaying trace; a later `reinforce(reward)` credits each
    item in proportion to its trace, so reward flows back to the context that
    preceded the outcome — even several steps earlier. The mind *discovers* that an
    "AIS gap" reliably precedes an escalation worth attending to.
  - **Rescorla-Wagner delta rule** for affect. An unseen domain word ("shadowfleet")
    has no entry in the hand-built lexicon — but if it keeps showing up before bad
    outcomes, its learned valence is nudged negative until perception treats it as
    threatening on its own.

This demo shows both: a brand-new word *acquiring* affect, and value-weighted
priorities re-ranking attention after reward.
"""
from _common import Mind, rule


def main() -> None:
    rule("REINFORCEMENT LEARNING  -  credit the context that preceded the outcome")

    m = Mind("watch")
    word = "shadowfleet"

    print(f"\nTD(lambda): alpha={m.values.alpha} gamma={m.values.gamma} lambda={m.values.lam}\n")
    print(f"'{word}' is NOT in the built-in lexicon. Baseline affect:")
    print(f"   valence('a {word} sighting') = {Mind().perceive(f'a {word} sighting').valence:+.2f}")

    print(f"\nNow it keeps appearing right before bad outcomes (reward = -1.0):")
    for i in range(1, 7):
        m.perceive(f"detected a {word} maneuver near the strait")
        updated = m.reinforce(-1.0)               # this context led to a bad outcome
        learned = m.lexicon.learned.get(word, 0.0)
        print(f"   step {i}: learned valence('{word}') = {learned:+.3f}"
              f"   |  trace-credited items: {len(updated)}")

    print(f"\nThe word has ACQUIRED affect — perception now flags it on its own:")
    print(f"   valence('a {word} sighting') = {m.perceive(f'a {word} sighting').valence:+.2f}  (learned)")

    print("\nTD value (which attended context the reward flowed back to):")
    for item, val in m.values.valued(5):
        print(f"   {item:<16} value={val:+.3f}")

    # A positive outcome on a different context, then value-weighted priorities
    good = Mind("ops")
    good.perceive("Naval patrols cleared the corridor and the convoy is safe")
    good.reinforce(+1.0)
    good.perceive("Routine status update from the convoy")
    print("\nValue-weighted priorities blend activation + learned value + causal centrality:")
    print("   attention (raw activation):", good.attention())
    print("   priorities (value-weighted):", good.priorities())
    print("\nNothing was hand-labelled: the mind learned what to attend to from reward.")


if __name__ == "__main__":
    main()

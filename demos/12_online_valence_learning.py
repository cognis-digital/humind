"""Scenario 12 - Rescorla-Wagner valence learning (learning-theory readers).

An unseen domain word carries no affect in the built-in lexicon. But if it keeps
showing up right before bad (or good) outcomes, the **delta rule** nudges its
learned valence toward the sign of the reward, a little each time, converging but
never exceeding [-1, 1]. This demo tracks two unknown words in parallel — one that
precedes bad outcomes, one that precedes good — and prints the learning curve, so
you can watch each word *acquire* affect from experience alone.
"""
from _common import bar, rule
from humind import LexiconLearner


def main() -> None:
    rule("ONLINE VALENCE LEARNING  -  words acquire affect from outcomes (delta rule)")
    lx = LexiconLearner(eta=0.25)
    print(f"\neta (learning rate) = {lx.eta}\n")

    print("Two brand-new words, learned in parallel over 12 outcomes each:")
    print("   'shadowfleet' always precedes a BAD outcome (reward -1)")
    print("   'safeharbor'  always precedes a GOOD outcome (reward +1)\n")

    print("   step   shadowfleet                    safeharbor")
    for step in range(1, 13):
        lx.update(["shadowfleet"], -1.0)
        lx.update(["safeharbor"], +1.0)
        s = lx.learned["shadowfleet"]
        h = lx.learned["safeharbor"]
        print(f"   {step:>4}   {s:+.3f} {bar(abs(s), width=12, scale=1.0)}   "
              f"{h:+.3f} {bar(h, width=12, scale=1.0)}")

    print("\nConfidence gate: only words past |0.1| influence perception:")
    print("   overrides ->", lx.overrides())
    print("\nBoth curves are monotone and bounded — classic Rescorla-Wagner. The mind")
    print("did not need a hand-labelled lexicon entry; it learned the sign from reward.")


if __name__ == "__main__":
    main()

"""Scenario 11 - eligibility traces (RL researchers, deep dive).

The point of TD(lambda) is **temporal credit assignment**: when a reward finally
arrives, it should flow back to the context that *preceded* it — even several
steps earlier — in proportion to a decaying eligibility trace. This demo attends a
sequence of cues one step at a time, then delivers a single reward at the end, and
shows how much value each earlier cue received. The most recent cue gets the most
credit; older cues get exponentially less (governed by gamma * lambda).
"""
from _common import bar, rule
from humind import ValueLearner


def main() -> None:
    rule("TD(lambda) CREDIT ASSIGNMENT  -  reward flows back along the trace")
    v = ValueLearner(alpha=0.5, gamma=0.95, lam=0.8)
    print(f"\nalpha={v.alpha}  gamma={v.gamma}  lambda={v.lam}"
          f"   (trace decay per step = gamma*lambda = {v.gamma * v.lam:.3f})\n")

    sequence = ["ais-gap", "course-change", "rendezvous", "transfer", "escalation"]
    print("A 5-step episode is observed one cue at a time:")
    for step, cue in enumerate(sequence, 1):
        v.attend([cue])
        print(f"   step {step}: attend '{cue}'   (eligibility now on {len(v.trace)} items)")

    print("\nA single reward of +1.0 arrives at the end of the episode.")
    updated = v.reinforce(1.0)

    print("\nCredit each earlier cue received (recency-weighted by the trace):")
    for cue in sequence:
        val = updated.get(cue, 0.0)
        print(f"   {cue:<14} {bar(val, width=20, scale=0.6)} value={val:+.4f}")

    print("\nReading: 'escalation' (most recent) is credited most; 'ais-gap' (earliest)")
    print("least — but it is credited nonzero, so the mind can still learn that an")
    print("early precursor mattered. That backward flow is what a flat bandit lacks.")


if __name__ == "__main__":
    main()

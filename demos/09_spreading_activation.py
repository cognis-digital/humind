"""Scenario 9 - associative recall (memory researchers).

'Cells that fire together wire together.' humind's associative memory is a
**Hebbian co-occurrence network**: concepts attended in the same moment get their
pairwise link strengthened (capped). Cueing one concept then brings the others to
mind by **spreading activation** (Collins & Loftus), ranked by link strength. This
demo builds the network from a short transcript and probes what different cues
recall — and shows that repeated co-occurrence wins.
"""
from _common import perceived_mind, rule


EXTRA = [
    "tanker rendezvous under darkness near the strait",
    "tanker rendezvous under darkness, AIS gap observed",
    "tanker rendezvous, ship-to-ship transfer at night",
    "patrol boat sighted near the strait at dawn",
]


def main() -> None:
    rule("SPREADING ACTIVATION  -  cue one concept, associated ones come to mind")
    m = perceived_mind("watch", lines=EXTRA)

    print("\nThe Hebbian network learned co-occurrences from 4 utterances.")
    print("Probe it with cues (strength = how often they fired together):\n")

    for cue in ("tanker", "rendezvous", "strait", "darkness"):
        assoc = m.memory.assoc.spread([cue], n=4)
        line = ", ".join(f"{k}({w:.0f})" for k, w in assoc) or "(nothing yet)"
        print(f"   cue '{cue:<10}' ~> {line}")

    print("\nMind.predict() uses the same mechanism on the current focus:")
    print("   predict(cue='tanker') ->", m.predict("tanker"))

    print("\nRepetition matters: 'rendezvous' co-occurred with 'tanker' three times,")
    print("so it ranks above 'strait', which appeared with it only once. The link")
    print("weights are inspectable in m.memory.assoc.w — no black box.")


if __name__ == "__main__":
    main()

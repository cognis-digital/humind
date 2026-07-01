"""Scenario 17 - prose to structure (NLP / knowledge-graph builders).

Before there can be a causal-loop diagram, causal *language* has to be parsed into
(cause, effect, polarity) triples. humind recognises a set of connectives —
positive ('leads to', 'drives', 'escalates'), negative ('reduces', 'mitigates',
'prevents'), and reversed ('because of', 'due to', which put the effect first) —
and normalises them all to a consistent cause->effect direction. This demo runs a
batch of sentences and prints the triple each one yields, including the tricky
reversed and negative cases.
"""
from _common import extract, rule


SENTENCES = [
    "an AIS gap leads to escalation risk",
    "sanctions pressure drives dark activity",
    "convoy escort reduces piracy incidents",
    "patrols mitigate smuggling activity",
    "escalation happened because of the AIS gap",       # reversed connective
    "the shortage was driven by export controls",       # reversed connective
    "the weather is calm today",                        # no causal claim -> nothing
]


def main() -> None:
    rule("CAUSAL EXTRACTION  -  prose -> (cause, effect, polarity) triples")
    print("\nEach sentence is scanned for causal connectives and normalised:\n")
    for s in SENTENCES:
        triples = extract(s).causes
        if triples:
            for c, e, pol in triples:
                sign = "+" if pol > 0 else "-"
                print(f"   \"{s}\"")
                print(f"      -> {c}  --({sign})-->  {e}")
        else:
            print(f"   \"{s}\"")
            print(f"      -> (no causal claim detected)")

    print("\nPositive connectives (leads to / drives / escalates ...) give +1;")
    print("negative ones (reduces / mitigates / prevents ...) give -1; reversed ones")
    print("(because of / driven by / due to) are flipped so cause always precedes")
    print("effect. These triples are exactly what the systems-thinking layer consumes.")


if __name__ == "__main__":
    main()

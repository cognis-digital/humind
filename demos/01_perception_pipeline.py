"""Scenario 1 - ML engineers.

The first thing any context engine has to do is *perceive*: turn a raw
utterance into a structured frame. humind's perception stage is a transparent,
stdlib pipeline — entities, the speech-act **intent**, **affect** (valence +
arousal, negation-aware), the **salient** terms, and **causal links** — so a
downstream model or agent can see *why* each field came out the way it did, not
just trust a black box. This demo runs the bundled transcript through
`extract()` and shows the frame field by field.
"""
from _common import TRANSCRIPT, extract, rule


def main() -> None:
    rule("PERCEPTION PIPELINE  -  one transparent frame per utterance")
    print("\nutterance  ->  intent | valence/arousal | entities | salient | causes\n")

    for text in TRANSCRIPT:
        f = extract(text)
        ents = ", ".join(f.entities) or "-"
        sal = ", ".join(f.salient[:3]) or "-"
        flag = "  <- threatening" if f.valence < -0.3 else "  <- urgent" if f.arousal >= 0.5 else ""
        print(f"  \"{text[:46]:<46}\"")
        print(f"     intent={f.intent:<8} valence={f.valence:+.2f} arousal={f.arousal:.2f}"
              f"  ent=[{ents}]  salient=[{sal}]{flag}")

    # negation awareness — the same valence word flips on a negator
    print("\nNegation-aware affect (within the prior two tokens):")
    for s in ("the corridor is safe", "the corridor is not safe"):
        print(f"   valence(\"{s}\") = {extract(s).valence:+.2f}")

    # causal language is parsed into (cause -> effect, polarity) triples
    print("\nCausal links are parsed out too (feeds the systems-thinking layer):")
    for c, e, pol in extract("An AIS gap leads to escalation risk").causes:
        print(f"   {c}  --({'+' if pol > 0 else '-'})-->  {e}")

    print("\nEvery field is derived by an inspectable lexicon/heuristic (humind/extract.py):")
    print("  intent  <- speech-act cues (?, please, I think, agreed, no ...)")
    print("  affect  <- a small valence lexicon (negation-flipped) + an arousal word set")
    print("  salient <- content-word frequency, stopwords removed")
    print("  causes  <- causal connectives (leads to / drives / reduces ...)")
    print("No network, no weights to download - the frame is reproducible and explainable.")


if __name__ == "__main__":
    main()

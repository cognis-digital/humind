"""Scenario 6 - affect & negation (linguists / analysts).

Affect is not a bag-of-words sum: humind is **negation-aware**. A valence word
flips sign when a negator sits within the prior two tokens ("safe" -> +, "not
safe" -> -), and it stays put when the negator is out of range. This demo walks a
set of minimal pairs so you can see exactly where the flip does and does not fire,
and how arousal (urgency) is counted separately from valence (good/bad).
"""
from _common import extract, rule


PAIRS = [
    ("the corridor is safe", "the corridor is not safe"),
    ("the situation is clear", "the situation is never clear"),
    ("this is good", "this is not good"),
    ("the vessel is at high risk", "the vessel is not at high risk"),
]


def main() -> None:
    rule("AFFECT & NEGATION  -  a valence word flips within a 2-token window")
    print("\nMinimal pairs (positive form  vs  negated form):\n")
    for pos, neg in PAIRS:
        vp, vn = extract(pos).valence, extract(neg).valence
        print(f"   {vp:+.2f}  \"{pos}\"")
        print(f"   {vn:+.2f}  \"{neg}\"   <- sign {'flipped' if vp * vn < 0 else 'held'}\n")

    print("Window boundary — a negator 3+ tokens away does NOT reach the word:")
    for s in ("not safe", "not really quite very safe"):
        print(f"   valence(\"{s}\") = {extract(s).valence:+.2f}")

    print("\nArousal is independent of valence — urgency word count / 3, capped at 1.0:")
    for s in ("routine status update",
              "urgent alert now",
              "urgent critical danger threat alarm emergency"):
        f = extract(s)
        print(f"   arousal={f.arousal:.2f} valence={f.valence:+.2f}  \"{s}\"")

    print("\nWhy it matters: 'NOT a high risk threat' must not read as threatening.")
    print("The rule is transparent (humind/extract.py:_NEGATORS) — no model to trust.")


if __name__ == "__main__":
    main()

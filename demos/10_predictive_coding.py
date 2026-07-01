"""Scenario 10 - prediction error (computational-neuroscience readers).

A predictive mind is surprised only by what it did not expect. After each frame,
humind forms an **expectation** for the next input (what it just saw, plus what
that brings to mind via spreading activation). The next frame's **surprise** is
the fraction of its concepts that were *not* expected — a running prediction
error. This demo streams a small watch log and prints the surprise of each line,
so you can watch familiar reports go quiet and a genuinely new contact spike.
"""
from _common import bar, rule
from humind import Mind


STREAM = [
    "vessel NEPTUNE-STAR darkening near the strait",
    "vessel NEPTUNE-STAR darkening near the strait",       # exact repeat -> low surprise
    "NEPTUNE-STAR still dark near the strait",              # mostly familiar
    "new contact ORACLE-WIND just appeared offshore",       # novel -> spike
    "ORACLE-WIND darkening near the strait",               # now partly expected
]


def main() -> None:
    rule("PREDICTIVE CODING  -  surprise = the fraction you did NOT expect")
    m = Mind("watch")
    print("\nEach line's surprise is prediction error against the running expectation:\n")
    print("   surprise                        utterance")
    for line in STREAM:
        s = m.perceive(line).surprise
        print(f"   {s:.2f} {bar(s, width=16, scale=1.0)}  \"{line[:40]}\"")

    print("\nReading: the exact repeat is unsurprising; the brand-new contact")
    print("ORACLE-WIND spikes; then, once it is associated with 'strait/darkening',")
    print("its follow-up is only partly novel. Surprise is a natural salience signal —")
    print("a high-surprise + high-arousal frame is exactly what grabs attention.")


if __name__ == "__main__":
    main()

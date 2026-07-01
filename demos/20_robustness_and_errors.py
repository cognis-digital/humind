"""Scenario 20 - graceful failure (operators / integrators).

Software that ingests free text and outcomes has to fail loudly on programmer
error and quietly on missing optional infrastructure. This demo shows both sides
of humind's error contract: malformed or out-of-range inputs raise a clear,
specific exception (a str where text is required, a non-forgetting decay, a
non-finite reward, an out-of-range learning rate), while optional LLM enrichment
against an unreachable backend is skipped without disturbing the transparent core
frame. No traceback escapes — every case is caught and explained.
"""
from _common import rule
from humind import (Mind, WorkingMemory, ValueLearner, LexiconLearner, extract)
from humind.extract import salient


def expect_error(label, fn):
    try:
        fn()
        print(f"   [MISS] {label}: no error raised")
    except (TypeError, ValueError) as e:
        print(f"   [ok]  {label}: {type(e).__name__}: {e}")


def main() -> None:
    rule("ROBUSTNESS & ERRORS  -  loud on bugs, quiet on missing infrastructure")

    print("\nA) Programmer errors raise clear, specific exceptions:")
    expect_error("extract(non-str)", lambda: extract(1234))
    expect_error("salient(k<0)", lambda: salient("some text", k=-1))
    expect_error("WorkingMemory(capacity=0)", lambda: WorkingMemory(capacity=0))
    expect_error("WorkingMemory(decay=1.0)", lambda: WorkingMemory(decay=1.0))
    expect_error("ValueLearner(alpha=0)", lambda: ValueLearner(alpha=0.0))
    expect_error("reinforce(NaN)", lambda: (lambda v: (v.attend(["x"]), v.reinforce(float("nan"))))(ValueLearner()))
    expect_error("LexiconLearner(eta=2)", lambda: LexiconLearner(eta=2.0))
    expect_error("Mind.express() with nothing perceived", lambda: Mind().express())

    print("\nB) Missing optional infrastructure degrades gracefully (no exception):")
    m = Mind("scout")
    frame = m.perceive("vessel NEPTUNE-STAR went dark",
                       enrich=True, endpoint="http://127.0.0.1:1")   # unreachable
    print(f"   enrichment skipped, core intact: entities={frame.entities} notes={frame.notes!r}")

    print("\nC) Odd-but-valid input is handled, not crashed:")
    for weird in ("", "   ", "!!!???", "12345", "ALLCAPS SHOUTING NOW"):
        f = extract(weird)
        print(f"   extract({weird!r:<22}) -> intent={f.intent} val={f.valence:+.2f} ent={f.entities}")

    print("\nThe contract: bugs fail fast with a message you can act on; absent")
    print("optional backends never take the transparent core down with them.")


if __name__ == "__main__":
    main()

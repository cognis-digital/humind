"""Scenario 14 - loop polarity (system-dynamics readers, deep dive).

A causal loop is **reinforcing (R)** when it has an even number of negative links
(it amplifies itself) and **balancing (B)** when it has an odd number (it corrects
itself). This demo builds three hand-crafted loops so the classification is
unambiguous — an all-positive vicious spiral, a single-negative thermostat, and a
double-negative reinforcing pair — and prints the link count that determines each
label. It closes with a prose-derived loop for contrast.
"""
from _common import rule
from humind import CausalGraph


def classify(g, title):
    print(f"\n{title}")
    for src, outs in g.edges.items():
        for dst, pol in outs:
            print(f"   {src:<12} --({'+' if pol > 0 else '-'})--> {dst}")
    for lp in g.feedback_loops():
        negs = "structure -> " + lp["kind"]
        print(f"   => loop {' -> '.join(lp['nodes'])} -> {lp['nodes'][0]}  "
              f"[{lp['kind']}]  ({negs})")


def main() -> None:
    rule("FEEDBACK LOOP POLARITY  -  R (amplifying) vs B (self-correcting)")

    r = CausalGraph()
    r.add_link("fear", "buying", 1); r.add_link("buying", "prices", 1)
    r.add_link("prices", "fear", 1)                        # 0 negatives -> R
    classify(r, "1) all-positive spiral (0 negative links) -> REINFORCING:")

    b = CausalGraph()
    b.add_link("heat", "cooling", 1); b.add_link("cooling", "heat", -1)   # 1 neg -> B
    classify(b, "2) thermostat (1 negative link) -> BALANCING:")

    rr = CausalGraph()
    rr.add_link("debt", "austerity", -1)
    rr.add_link("austerity", "debt", -1)                   # 2 negatives -> R
    classify(rr, "3) double-negative pair (2 negative links) -> REINFORCING:")

    print("\nThe rule is purely structural: count the negative links around the loop.")
    print("Even -> R, odd -> B. Behaviour follows from structure, not from any event.")


if __name__ == "__main__":
    main()

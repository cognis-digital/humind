"""Scenario 4 - systems builders (systems thinking).

humind reads causal language and assembles a **causal-loop diagram** (CLD): nodes
are concepts, edges carry a **polarity** (+ same-direction, - opposite). On top of
that structure it does the two core moves of system dynamics:

  - **feedback_loops()** finds cycles and labels each **R** (reinforcing — an even
    number of negative links, self-amplifying) or **B** (balancing — odd number,
    self-correcting). Behaviour comes from loop structure, not isolated events
    (Forrester / Sterman).
  - **leverage_points()** ranks concepts by causal centrality — the most connected
    structural nodes are where intervention has the most reach (Meadows).

This demo feeds the mind a handful of plain-English causal statements and lets the
structure — a vicious reinforcing loop and the lever that breaks it — fall out.
"""
from _common import CAUSAL_STATEMENTS, Mind, rule


def main() -> None:
    rule("SYSTEMS THINKING  -  causal-loop diagram, feedback loops, leverage")
    m = Mind("analyst")

    print("\nPerceive plain-English causal statements:")
    for s in CAUSAL_STATEMENTS:
        m.perceive(s)
        print(f"   \"{s}\"")

    print("\nCAUSAL-LOOP DIAGRAM (cause --(polarity)--> effect):")
    for src, outs in m.systems.edges.items():
        for dst, pol in outs:
            print(f"   {src:<22} --({'+' if pol > 0 else '-'})-->  {dst}")

    print("\nFEEDBACK LOOPS (R = reinforcing / self-amplifying, B = balancing):")
    loops = m.feedback_loops()
    if loops:
        for lp in loops:
            arrow = " -> ".join(lp["nodes"]) + " -> " + lp["nodes"][0]
            tag = "reinforcing (vicious/virtuous)" if lp["kind"] == "R" else "balancing (self-correcting)"
            print(f"   [{lp['kind']}] {arrow}    <- {tag}")
    else:
        print("   (no closed loops found)")

    print("\nLEVERAGE POINTS (highest causal centrality = where to intervene):")
    for concept, degree in m.leverage_points(3):
        print(f"   {concept:<22} centrality={degree}")

    print("\nReading: the reinforcing loop (darkness -> escalation -> sanctions ->")
    print("smuggling -> darkness) is self-amplifying; 'patrols reduces smuggling' is")
    print("the balancing lever, and 'smuggling' ranks top leverage because it's where")
    print("that lever attaches. The structure - not any single event - explains the")
    print("behaviour, and it was derived from prose.")


if __name__ == "__main__":
    main()

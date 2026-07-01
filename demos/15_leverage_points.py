"""Scenario 15 - where to intervene (policy / analysis readers).

Meadows: the highest-leverage place to intervene in a system is usually its most
structurally connected node, not the loudest symptom. humind ranks concepts by
causal **centrality** (in-degree + out-degree in the causal-loop diagram) to
surface those points. This demo assembles a small causal model of a maritime
smuggling dynamic from prose, then ranks its leverage points and shows why the hub
concept — the one the balancing lever attaches to — comes out on top.
"""
from _common import rule
from humind import Mind


MODEL = [
    "darkness drives escalation",
    "escalation drives sanctions",
    "sanctions drives smuggling",
    "smuggling drives darkness",
    "patrols reduces smuggling",
    "intelligence reduces smuggling",
]


def main() -> None:
    rule("LEVERAGE POINTS  -  intervene where the structure is most connected")
    m = Mind("analyst")
    for s in MODEL:
        m.perceive(s)

    print("\nCausal-loop diagram derived from prose:")
    for src, outs in m.systems.edges.items():
        for dst, pol in outs:
            print(f"   {src:<14} --({'+' if pol > 0 else '-'})--> {dst}")

    print("\nCentrality (in + out degree) per concept:")
    for concept, deg in sorted(m.systems.centrality().items(), key=lambda kv: -kv[1]):
        print(f"   {concept:<14} {'*' * deg} {deg}")

    print("\nTop leverage points:")
    for concept, deg in m.leverage_points(3):
        print(f"   {concept:<14} centrality={deg}")

    print("\nReading: 'smuggling' tops the ranking — it closes the vicious loop AND is")
    print("where both balancing levers (patrols, intelligence) attach, so an")
    print("intervention there reaches the most of the system. The symptom you SEE")
    print("(darkness) is not necessarily where you should PUSH.")


if __name__ == "__main__":
    main()

"""Scenario 19 - many minds, one channel (multi-agent-systems builders).

The tandem scales past two agents. Here several scout minds each perceive their
own local report and **express** it as an agentlex message; a single command mind
**ingests** the whole broadcast, folding every observation into one shared
semantic memory and attention focus. Then agentlex derives which contacts to
escalate from the merged facts. It shows humind as the per-agent understanding and
agentlex as the shared symbolic bus — grounded facts in, inference out.

Requires the agentlex tandem dependency; degrades to a clean skip without it.
"""
from _common import Mind, rule


REPORTS = [
    ("scout-a", "CRITICAL: vessel NEPTUNE-STAR is a high risk threat", True),
    ("scout-b", "Routine: vessel QUIET-DAWN is steady and of low concern", False),
    ("scout-c", "CRITICAL: vessel GHOST-RUNNER is a high risk threat", True),
    ("scout-d", "Routine: tender PALE-MOON is steady and of low concern", False),
]


def main() -> None:
    rule("MULTI-AGENT BROADCAST  -  many scouts express, one command ingests")
    try:
        import agentlex
        from agentlex import parse_term
    except ImportError:
        print("\nagentlex not installed - skipping (core is unaffected).")
        print("Install:  pip install git+https://github.com/cognis-digital/agentlex.git")
        return

    command = Mind("command")
    kb = agentlex.KnowledgeBase()

    print("\nEach scout perceives locally and broadcasts an agentlex message:\n")
    for who, text, sanctioned in REPORTS:
        scout = Mind(who)
        msg = scout.express(scout.perceive(text))
        print(f"   {who} -> {msg.to_wire()}")
        command.ingest(agentlex.from_wire(msg.to_wire()))   # merge into shared memory
        kb.assert_fact(msg.content)
        if sanctioned:
            kb.assert_fact(parse_term(f"sanctioned({msg.content.args[0]})"))

    print("\nCommand's merged semantic memory (all scouts folded in):")
    for f in sorted(command.memory.semantic.query()):
        print(f"   {f[0]} {f[1]} {f[2]}")
    print("\nCommand's shared attention focus:", command.attention(5))

    kb.add_rule(parse_term("escalate(?v)"),
                [parse_term("observed(?v, high)"), parse_term("sanctioned(?v)")])
    derived = kb.infer()
    print("\n[agentlex] escalate :- observed(?v, high), sanctioned(?v):")
    print("   ", [str(d) for d in derived] or "nothing to escalate")

    print("\nFour independent minds contributed grounded facts to one channel, and a")
    print("single rule turned the merged picture into a decision. That is the tandem")
    print("at fleet scale: distributed understanding, centralised inference.")


if __name__ == "__main__":
    main()

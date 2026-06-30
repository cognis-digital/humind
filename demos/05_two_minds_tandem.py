"""Scenario 5 - systems builders (multi-agent).

The tandem: `humind` is the *understanding*, `agentlex` is the *language* two
minds speak. A scout perceives free text, distils it to a frame, and **expresses**
it as a precise agentlex symbolic message; a command mind **ingests** it back into
its own memory across the wire. Then an agentlex `KnowledgeBase` **derives** which
contacts to escalate from a rule — humind supplies the grounded facts, agentlex
supplies the inference. An optional private-LLM brief is added when a backend is
reachable, and skipped gracefully otherwise.

Requires the agentlex tandem dependency (installed in CI / locally). If it is not
importable, the demo says so and exits cleanly — the cognitive core is unaffected.
"""
from _common import Mind, rule

# bundled mini watchlist: (name, severity, sanctioned) — the inline fixture
WATCHLIST = [
    ("NEPTUNE-STAR", "high", True),
    ("QUIET-DAWN", "low", False),
    ("GHOST-RUNNER", "high", False),
]


def main() -> None:
    rule("TWO MINDS, ONE LANGUAGE  -  humind understands, agentlex reasons")
    try:
        import agentlex
        from agentlex import parse_term
    except ImportError:
        print("\nagentlex is not installed - skipping the tandem demo (core is unaffected).")
        print("Install it:  pip install git+https://github.com/cognis-digital/agentlex.git")
        return

    # 1) one mind expresses, another ingests — unifiable symbols, not prose
    scout, command = Mind("scout"), Mind("command")
    report = "CRITICAL: vessel NEPTUNE-STAR went dark and is a high risk threat"
    print(f"\nscout perceives free text:\n   \"{report}\"")
    frame = scout.perceive(report)
    msg = scout.express(frame)
    print(f"   -> EXPRESS (agentlex): {msg.to_wire()}")
    command.ingest(agentlex.from_wire(msg.to_wire()))
    print("   -> command INGESTS; its semantic memory now holds:",
          sorted(command.memory.semantic.query()))
    print(f"   -> command focus: {command.attention()}")

    # 2) the analyst builds a knowledge base and lets agentlex DERIVE escalations
    analyst = Mind("analyst")
    kb = agentlex.KnowledgeBase()
    print("\nanalyst expresses each contact; agentlex remembers the facts:")
    for name, sev, sanctioned in WATCHLIST:
        word = "high risk threat" if sev == "high" else "steady and of low concern"
        sentence = f"{'CRITICAL' if sev == 'high' else 'Routine'}: vessel {name} is {word}"
        m = analyst.express(analyst.perceive(sentence))
        print(f"   {m.to_wire()}")
        kb.assert_fact(m.content)
        if sanctioned:
            kb.assert_fact(parse_term(f"sanctioned({m.content.args[0]})"))

    kb.add_rule(parse_term("escalate(?v)"),
                [parse_term("observed(?v, high)"), parse_term("sanctioned(?v)")])
    derived = kb.infer()
    print("\n[agentlex] derived under  escalate :- observed(?v, high), sanctioned(?v):")
    print(f"   {[str(d) for d in derived] or 'nothing to escalate'}")

    # 3) optional private-LLM brief — skipped gracefully with no backend
    try:
        from humind import addins
        where = addins.discover()
        if where:
            base, models = where
            names = ", ".join(n for n, _, _ in WATCHLIST)
            brief = addins.interpret("One-line watch-officer brief for: " + names,
                                     base, models[0] if models else "default")
            print(f"\n[optional LLM brief] {brief}")
        else:
            print("\n[optional LLM brief] no backend reachable - skipped (core ran fully).")
    except Exception as exc:
        print(f"\n[optional LLM brief] unavailable ({exc}) - skipped (core ran fully).")

    print("\nTwo agents shared facts with zero free-text ambiguity, and a rule turned")
    print("them into a decision. That's the tandem: understanding produces language;")
    print("language drives inference.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Executable Cognis interop demo - the INTEROP map, running.

Pipeline (each step a real repo):
  maritimeint  -> a grey-fleet watchlist of vessels (real if maritimeint is installed,
                 else a small inline sample so this always runs)
  humind       -> perceive each finding, express it as a symbolic message
  agentlex     -> ingest the messages into a KnowledgeBase, then a forward-chaining
                 rule derives which vessels to ESCALATE (high-risk AND sanctioned)
  edgemesh     -> (optional) route a one-line natural-language brief through your fleet

Run:  python examples/cluster_demo.py
Needs: humind + agentlex (humind's deps). maritimeint + an edgemesh/fleet endpoint are
optional - the demo degrades gracefully without them.
"""

from __future__ import annotations

import agentlex
from agentlex import KnowledgeBase, Symbol, parse_term
from humind.mind import Mind


def watchlist() -> list[dict]:
    """Real maritimeint watchlist if available, else a representative sample."""
    try:
        import os
        from maritimeint.core import load_messages
        from maritimeint.locate import locate
        from maritimeint.sanctions import load_sanctions
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # try the maritimeint demo data if co-located
        for base in (os.path.join(root, "mi"), os.environ.get("MARITIMEINT_DIR", "")):
            ais = os.path.join(base, "demos", "ais_sample.json")
            if base and os.path.exists(ais):
                msgs = load_messages(ais)
                static = {m.mmsi: {"name": m.name} for m in msgs}
                res = locate(msgs, sanctions=load_sanctions(os.path.join(base, "demos", "sanctions_sample.json")),
                             static=static)
                return [{"name": v["name"] or v["mmsi"], "tier": v["tier"],
                         "sanctioned": v["sanctioned"]} for v in res["watchlist"]]
    except Exception:
        pass
    # inline fallback (mirrors maritimeint's sample scenario)
    return [
        {"name": "NEPTUNE-STAR", "tier": "HIGH", "sanctioned": True},
        {"name": "QUIET-DAWN", "tier": "MEDIUM", "sanctioned": False},
        {"name": "GHOST-RUNNER", "tier": "MEDIUM", "sanctioned": False},
    ]


def main() -> int:
    wl = watchlist()
    print(f"[maritimeint] watchlist: {[v['name'] for v in wl]}\n")

    analyst = Mind("analyst")
    kb = KnowledgeBase()

    for v in wl:
        sev = "high risk threat" if v["tier"] == "HIGH" else "of interest"
        sentence = f"{'CRITICAL' if v['tier'] == 'HIGH' else 'Note'}: vessel {v['name']} is {sev}"
        frame = analyst.perceive(sentence)                 # humind: understand
        msg = analyst.express(frame)                       # humind -> agentlex message
        print(f"[humind->agentlex] {msg.to_wire()}")
        kb.assert_fact(msg.content)                        # agentlex: remember the fact
        if v["sanctioned"]:
            kb.assert_fact(parse_term(f"sanctioned({msg.content.args[0]})"))

    # agentlex reasoning: high-risk AND sanctioned => escalate
    kb.add_rule(parse_term("escalate(?v)"),
                [parse_term("observed(?v, high)"), parse_term("sanctioned(?v)")])
    derived = kb.infer()
    print(f"\n[agentlex] escalations derived: {[str(d) for d in derived] or 'none'}")

    # optional edgemesh-routed brief
    try:
        from humind import addins
        where = addins.discover()
        if where:
            base, models = where
            brief = addins.interpret(
                "Summarize for a watch officer: " + ", ".join(
                    f"{v['name']} ({v['tier']}{', sanctioned' if v['sanctioned'] else ''})" for v in wl),
                base, models[0] if models else "default")
            print(f"\n[edgemesh->LLM brief] {brief}")
        else:
            print("\n[edgemesh] no model backend reachable - skipping the LLM brief (core ran fully)")
    except Exception as exc:
        print(f"\n[edgemesh] enrichment unavailable ({exc}) - core ran fully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

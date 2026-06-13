"""humind CLI — perceive text, inspect attention/memory, and run the agentlex tandem.

    humind perceive "URGENT: vessel NEPTUNE STAR went dark near Hormuz"
    humind think "report A" "follow-up B" ...     # perceive a sequence, show focus
    humind demo                                    # two minds converse via agentlex
"""

from __future__ import annotations

import argparse
import json
import sys

from humind import __version__
from humind.extract import extract
from humind.mind import Mind


def _demo() -> int:
    scout, command = Mind("scout"), Mind("command")
    scout.perceive("CRITICAL: vessel NEPTUNE-STAR went dark near a high-risk corridor")
    msg = scout.express()                          # understanding -> agentlex
    print("scout says   :", msg.to_wire())
    command.ingest(msg)                            # agentlex -> command's memory
    facts = command.memory.semantic.query()
    print("command learned:", [f"{s} {p} {o}" for s, p, o in facts])
    print("command focus :", command.attention())
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="humind", description=__doc__.splitlines()[0])
    p.add_argument("--version", action="version", version=f"humind {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("perceive"); sp.add_argument("text")
    sp.add_argument("--ai", action="store_true", help="enrich with an LLM if a backend is reachable")
    sp.add_argument("--endpoint", default=None, help="OpenAI-compatible base URL (edgemesh/fleet)")
    sp.add_argument("--model", default=None)
    st = sub.add_parser("think"); st.add_argument("text", nargs="+")
    sc = sub.add_parser("causal"); sc.add_argument("text", nargs="+")
    sub.add_parser("learn")
    sb = sub.add_parser("bench"); sb.add_argument("-n", type=int, default=20000)
    sub.add_parser("demo")
    args = p.parse_args(argv)

    if args.cmd == "perceive":
        if args.ai or args.endpoint:
            frame = Mind("you").perceive(args.text, enrich=True, endpoint=args.endpoint, model=args.model)
            print(json.dumps(frame.to_dict(), indent=2))
        else:
            print(json.dumps(extract(args.text).to_dict(), indent=2))
        return 0
    if args.cmd == "think":
        mind = Mind("you")
        for t in args.text:
            mind.perceive(t)
        print("attention/focus :", mind.attention())
        print("priorities      :", mind.priorities())      # value + centrality weighted
        print("predict (assoc) :", mind.predict())          # spreading activation
        print("facts           :", [f"{s} {pr} {o}" for s, pr, o in mind.memory.semantic.query()])
        return 0
    if args.cmd == "causal":
        mind = Mind("you")
        for t in args.text:
            mind.perceive(t)
        print("causal links :")
        for src, outs in mind.systems.edges.items():
            for dst, pol in outs:
                print(f"   {src}  --({'+' if pol > 0 else '-'})-->  {dst}")
        loops = mind.feedback_loops()
        print("feedback loops :", [f"{lp['kind']}:{' -> '.join(lp['nodes'])}" for lp in loops] or "none")
        print("leverage points:", mind.leverage_points())
        return 0
    if args.cmd == "learn":
        return _learn()
    if args.cmd == "bench":
        return _bench(args.n)
    if args.cmd == "demo":
        try:
            return _demo()
        except ImportError as exc:
            print(f"error: {exc}", file=sys.stderr); return 1
    return 0


def _learn() -> int:
    """Show RL credit assignment: an unseen word acquires affect from outcomes."""
    mind = Mind("you")
    word = "shadowfleet"
    print(f"before : valence('{word}') =", extract(f"a {word} sighting").valence)
    for _ in range(6):
        mind.perceive(f"detected a {word} maneuver near the strait")
        mind.reinforce(-1.0)                                # this context led to a bad outcome
    learned = mind.lexicon.overrides().get(word, 0.0)
    print(f"after  : learned valence('{word}') = {learned}  (negative = threatening)")
    print("top learned values :", mind.values.valued(3))
    return 0


def _bench(n: int) -> int:
    import time
    text = "URGENT: vessel NEPTUNE-STAR went dark; the AIS gap leads to escalation risk."
    t0 = time.perf_counter()
    for _ in range(n):
        extract(text)
    dt = time.perf_counter() - t0
    print(f"extract x{n}: {dt:.3f}s  ->  {n / dt:,.0f} frames/sec  ({dt / n * 1e6:.1f} us/frame)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

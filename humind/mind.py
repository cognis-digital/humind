"""The cognitive loop — perceive → attend → remember → express.

`Mind` ties extraction and memory together, and **interoperates with `agentlex`**:
it turns understood context into a symbolic agentlex message (`express`) and folds a
received agentlex message back into memory (`ingest`). That's the tandem — `humind`
is the understanding, `agentlex` is the language two minds speak.

agentlex is a soft dependency: the cognitive core works without it; `express`/`ingest`
raise a clear message if it isn't installed.
"""

from __future__ import annotations

import re

from humind.extract import ContextFrame, extract
from humind.learning import LexiconLearner, ValueLearner
from humind.memory import CognitiveMemory
from humind.systems import CausalGraph

# intent -> agentlex performative (they line up by design)
_PERFORMATIVE = {"inform": "inform", "request": "request", "query": "query",
                 "propose": "propose", "agree": "agree", "refuse": "refuse"}


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9-]", "-", s).strip("-").lower() or "x"


def _affect_label(valence: float) -> str:
    return "high" if valence <= -0.3 else "low" if valence >= 0.3 else "neutral"


class Mind:
    def __init__(self, name: str = "agent") -> None:
        self.name = name
        self.memory = CognitiveMemory()
        self.values = ValueLearner()               # RL: which context is worth attending
        self.lexicon = LexiconLearner()            # online affect learning
        self.systems = CausalGraph()               # systems-thinking causal-loop model
        self._expectation: set[str] = set()        # predictive-coding prior for next input
        self._clock = 0

    # --- perception ----------------------------------------------------------
    def perceive(self, text: str, *, enrich: bool = False, endpoint: str | None = None,
                 model: str | None = None) -> ContextFrame:
        frame = extract(text, self.lexicon.overrides())   # affect reflects learned valence
        if enrich or endpoint:                     # optional LLM augmentation (graceful)
            try:
                from humind import addins
                where = (endpoint, [model or "default"]) if endpoint else addins.discover()
                if where:
                    base, models = where
                    frame.notes = addins.interpret(text, base, model or (models or ["default"])[0])
            except Exception:
                pass                               # core is unaffected if enrichment fails

        # predictive coding: surprise = how much of this input we did NOT expect
        cur = list(dict.fromkeys([e.lower() for e in frame.entities] + frame.salient))
        novel = [c for c in cur if c not in self._expectation]
        frame.surprise = round(len(novel) / len(cur), 3) if cur else 0.0

        self._clock += 1
        self.memory.episodic.record(self._clock, frame)
        # precision-weighting: urgent AND surprising input grabs the most attention
        weight = 1.0 + frame.arousal + frame.surprise
        for e in frame.entities:
            self.memory.working.add(e, weight)
            self.memory.semantic.assert_fact(_slug(e), "salience", _affect_label(frame.valence))
        for s in frame.salient:
            self.memory.working.add(s, 0.5 * weight)
        self.memory.working.tick()

        # Hebbian association, RL eligibility, and the causal-loop model
        self.memory.assoc.coactivate(cur, strength=1.0)
        self.values.attend([e.lower() for e in frame.entities] or frame.salient[:3])
        for cause, effect, pol in frame.causes:
            self.systems.add_link(cause, effect, pol)
        # expectation for the NEXT input: what we just saw, plus what it brings to mind
        self._expectation = set(cur) | {k for k, _ in self.memory.assoc.spread(cur, n=8)}
        return frame

    def attention(self, n: int = 3) -> list[str]:
        return self.memory.working.focus(n)

    def recall(self, term: str) -> list[ContextFrame]:
        return self.memory.episodic.search(term)

    # --- learning, prediction, systems-thinking ------------------------------
    def predict(self, cue: str | None = None, n: int = 5) -> list[str]:
        """What the current cue brings to mind (spreading activation). Cues from the
        single most-active item by default, so its associates aren't excluded."""
        seeds = [cue] if cue else self.attention(1)
        return [k for k, _ in self.memory.assoc.spread(seeds, n)]

    def reinforce(self, reward: float) -> dict[str, float]:
        """Outcome feedback: credit recently-attended context (TD-lambda) and learn the
        affective valence of the latest episode's salient terms (delta rule). Positive
        reward = that context was worth attending to."""
        updated = self.values.reinforce(reward)
        recent = self.memory.episodic.recent(1)
        if recent and recent[0].salient:
            self.lexicon.update(recent[0].salient, reward)
        return updated

    def priorities(self, n: int = 5) -> list[str]:
        """Value-weighted attention: blend working-memory activation, learned value, and
        causal centrality (systems leverage). What a rational agent should focus on."""
        cent = self.systems.centrality()
        scored: dict[str, float] = {}
        for it in self.memory.working.items:
            key = it.content.lower()
            scored[it.content] = (it.activation
                                  + self.values.value.get(key, 0.0)
                                  + 0.3 * cent.get(key, 0))
        return [k for k, _ in sorted(scored.items(), key=lambda kv: kv[1], reverse=True)[:n]]

    def consolidate(self, min_count: int = 2) -> list[tuple[str, str, str]]:
        """Move recurring episodic entities into durable semantic memory (CLS)."""
        return self.memory.consolidate(min_count)

    def leverage_points(self, n: int = 3) -> list[tuple[str, int]]:
        """Highest-centrality concepts in the causal-loop model (where to intervene)."""
        return self.systems.leverage_points(n)

    def feedback_loops(self, max_len: int = 6) -> list[dict]:
        """Reinforcing (R) / balancing (B) loops discovered in the causal model."""
        return self.systems.feedback_loops(max_len)

    # --- introspection -------------------------------------------------------
    def introspect(self, n: int = 5) -> dict:
        """A single JSON-serialisable snapshot of the whole cognitive state.

        One call surfaces what the mind is currently doing across every layer —
        focus, blended priorities, learned values, durable facts, associative
        cues, and the systems-thinking view (loops + leverage). Handy for logging,
        dashboards, or explaining *why* the mind is prioritising what it is. The
        result contains only built-in types (str/float/int/list/dict), so it
        drops straight into `json.dumps` or a `--format json` pipe.

        `n` bounds each ranked list; pass `n <= 0` for a raw error.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        focus = self.attention(n)
        return {
            "name": self.name,
            "clock": self._clock,
            "focus": focus,
            "priorities": self.priorities(n),
            "values": [{"item": k, "value": round(v, 4)} for k, v in self.values.valued(n)],
            "predict": self.predict(n=n),
            "associations": {c: self.memory.assoc.associates(c.lower(), n) for c in focus},
            "facts": [{"subject": s, "predicate": p, "object": o}
                      for s, p, o in sorted(self.memory.semantic.query())],
            "learned_valence": dict(sorted(self.lexicon.overrides().items())),
            "loops": [{"kind": lp["kind"], "nodes": lp["nodes"]}
                      for lp in self.feedback_loops()],
            "leverage": [{"concept": c, "centrality": d} for c, d in self.leverage_points(n)],
            "surprise_last": round(
                self.memory.episodic.recent(1)[0].surprise, 3
            ) if self.memory.episodic.events else 0.0,
            "episodes": len(self.memory.episodic.events),
        }

    # --- language (agentlex tandem) -----------------------------------------
    def express(self, frame: ContextFrame | None = None):
        """Turn understood context into an agentlex Message."""
        agentlex = _require_agentlex()
        frame = frame or (self.memory.episodic.recent(1) or [None])[0]
        if frame is None:
            raise ValueError("nothing perceived yet to express")
        ent = _slug(frame.entities[0]) if frame.entities else "context"
        content = agentlex.Compound("observed", (agentlex.Symbol(ent),
                                                  agentlex.Symbol(_affect_label(frame.valence))))
        return agentlex.Message(_PERFORMATIVE.get(frame.intent, "inform"),
                                self.name, "broadcast", content)

    def ingest(self, message) -> None:
        """Fold a received agentlex Message into memory."""
        agentlex = _require_agentlex()
        self._clock += 1
        c = message.content
        if isinstance(c, agentlex.Compound) and c.args:
            subj = str(c.args[0])
            obj = str(c.args[1]) if len(c.args) > 1 else ""
            self.memory.semantic.assert_fact(subj, c.functor, obj)
            self.memory.working.add(subj, 1.0)
        # also log the heard utterance episodically
        self.memory.episodic.record(self._clock, extract(message.to_wire()))


def _require_agentlex():
    try:
        import agentlex
        return agentlex
    except ImportError as exc:  # pragma: no cover
        raise ImportError("the agentlex tandem needs the 'agentlex' package: "
                          "pip install git+https://github.com/cognis-digital/agentlex.git") from exc

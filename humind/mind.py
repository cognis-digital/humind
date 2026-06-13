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
from humind.memory import CognitiveMemory

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
        self._clock = 0

    # --- perception ----------------------------------------------------------
    def perceive(self, text: str) -> ContextFrame:
        frame = extract(text)
        self._clock += 1
        self.memory.episodic.record(self._clock, frame)
        weight = 1.0 + frame.arousal               # urgent input grabs more attention
        for e in frame.entities:
            self.memory.working.add(e, weight)
            self.memory.semantic.assert_fact(_slug(e), "salience", _affect_label(frame.valence))
        for s in frame.salient:
            self.memory.working.add(s, 0.5 * weight)
        self.memory.working.tick()
        return frame

    def attention(self, n: int = 3) -> list[str]:
        return self.memory.working.focus(n)

    def recall(self, term: str) -> list[ContextFrame]:
        return self.memory.episodic.search(term)

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

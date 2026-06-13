"""Memory model for humind — three stores, loosely after cognitive architectures.

Inspired by (not a literal model of) ACT-R activation/decay, Miller's 7±2 working-
memory limit, and Global Workspace Theory's notion of a current 'focus':

  - **WorkingMemory** — small, capacity-bounded, items carry an *activation* that
    decays each tick; the most-active items are the current focus/attention.
  - **EpisodicMemory** — a time-ordered log of perceived frames (what happened, when).
  - **SemanticMemory** — durable (subject, predicate, object) facts distilled from
    experience; this is the natural place to bridge to a durable backend like
    `engram` / `hermes` / `memorybank`.

Pure standard library.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WorkingItem:
    content: str
    activation: float = 1.0


class WorkingMemory:
    """Capacity-bounded, decaying short-term store (the attention buffer)."""

    def __init__(self, capacity: int = 7, decay: float = 0.8, threshold: float = 0.2) -> None:
        self.capacity = capacity
        self.decay = decay
        self.threshold = threshold
        self.items: list[WorkingItem] = []

    def add(self, content: str, salience: float = 1.0) -> None:
        for it in self.items:                       # re-activate if already present
            if it.content == content:
                it.activation = min(2.0, it.activation + salience)
                return
        self.items.append(WorkingItem(content, max(0.1, salience)))
        self.items.sort(key=lambda i: i.activation, reverse=True)
        del self.items[self.capacity:]              # evict least-active beyond capacity

    def tick(self) -> None:
        for it in self.items:
            it.activation *= self.decay
        self.items = [i for i in self.items if i.activation >= self.threshold]

    def focus(self, n: int = 3) -> list[str]:
        return [i.content for i in sorted(self.items, key=lambda i: i.activation, reverse=True)[:n]]


class EpisodicMemory:
    def __init__(self) -> None:
        self.events: list[tuple[float, object]] = []

    def record(self, ts: float, frame) -> None:
        self.events.append((ts, frame))

    def recent(self, n: int = 5) -> list:
        return [f for _, f in self.events[-n:]]

    def search(self, term: str) -> list:
        t = term.lower()
        return [f for _, f in self.events if t in getattr(f, "text", "").lower()]


class SemanticMemory:
    def __init__(self) -> None:
        self.facts: set[tuple[str, str, str]] = set()

    def assert_fact(self, subject: str, predicate: str, obj: str) -> None:
        self.facts.add((subject, predicate, obj))

    def query(self, subject: str | None = None, predicate: str | None = None,
              obj: str | None = None) -> list[tuple[str, str, str]]:
        return [f for f in self.facts
                if (subject is None or f[0] == subject)
                and (predicate is None or f[1] == predicate)
                and (obj is None or f[2] == obj)]


@dataclass
class CognitiveMemory:
    working: WorkingMemory = field(default_factory=WorkingMemory)
    episodic: EpisodicMemory = field(default_factory=EpisodicMemory)
    semantic: SemanticMemory = field(default_factory=SemanticMemory)

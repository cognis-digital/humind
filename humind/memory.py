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
        # Validate up front: capacity <= 0 would silently wipe every item on `add`
        # (del self.items[capacity:]), and decay >= 1 never forgets — both are footguns.
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1) so activation forgets, got {decay}")
        if threshold < 0.0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        self.capacity = capacity
        self.decay = decay
        self.threshold = threshold
        self.items: list[WorkingItem] = []

    def add(self, content: str, salience: float = 1.0) -> None:
        for it in self.items:                       # re-activate if already present
            if it.content == content:
                it.activation = min(2.0, it.activation + salience)
                break
        else:
            self.items.append(WorkingItem(content, max(0.1, salience)))
        self.items.sort(key=lambda i: i.activation, reverse=True)  # keep order consistent
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
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        return [f for _, f in self.events[-n:]] if n else []

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


class AssociativeMemory:
    """Hebbian co-occurrence network with spreading activation.

    'Cells that fire together wire together' — items attended in the same moment have
    their pairwise association strengthened (capped). `spread()` propagates activation
    one hop from a set of seeds, so cueing one concept brings associated ones to mind
    (spreading activation, after Collins & Loftus / ACT-R)."""

    def __init__(self, cap: float = 5.0) -> None:
        if cap <= 0:
            raise ValueError(f"cap must be > 0, got {cap}")
        self.w: dict[str, dict[str, float]] = {}
        self.cap = cap

    def coactivate(self, items, strength: float = 1.0) -> None:
        items = list(dict.fromkeys(items))               # de-dupe, keep order
        for i, a in enumerate(items):
            row = self.w.setdefault(a, {})
            for b in items[i + 1:]:
                row[b] = min(self.cap, row.get(b, 0.0) + strength)
                back = self.w.setdefault(b, {})
                back[a] = min(self.cap, back.get(a, 0.0) + strength)

    def spread(self, seeds, n: int = 5) -> list[tuple[str, float]]:
        act: dict[str, float] = {}
        for s in seeds:
            for nbr, weight in self.w.get(s, {}).items():
                if nbr not in seeds:
                    act[nbr] = act.get(nbr, 0.0) + weight
        return sorted(act.items(), key=lambda kv: kv[1], reverse=True)[:n]

    def associates(self, item: str, n: int = 5) -> list[str]:
        return [k for k, _ in sorted(self.w.get(item, {}).items(),
                                     key=lambda kv: kv[1], reverse=True)[:n]]


@dataclass
class CognitiveMemory:
    working: WorkingMemory = field(default_factory=WorkingMemory)
    episodic: EpisodicMemory = field(default_factory=EpisodicMemory)
    semantic: SemanticMemory = field(default_factory=SemanticMemory)
    assoc: AssociativeMemory = field(default_factory=AssociativeMemory)

    def consolidate(self, min_count: int = 2) -> list[tuple[str, str, str]]:
        """Complementary Learning Systems: distil entities that recur across episodes
        into durable semantic facts (fast hippocampal log -> slow neocortical store).
        Returns the facts newly consolidated."""
        counts: dict[str, int] = {}
        for _, frame in self.episodic.events:
            for e in getattr(frame, "entities", []):
                key = e.lower()
                counts[key] = counts.get(key, 0) + 1
        new = []
        for ent, c in counts.items():
            if c >= min_count:
                fact = (ent, "consolidated", str(c))
                if fact not in self.semantic.facts:
                    self.semantic.assert_fact(*fact)
                    new.append(fact)
        return new

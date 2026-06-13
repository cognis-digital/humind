"""Reinforcement layer — humind learns *which context mattered* from outcomes.

Two well-studied mechanisms, implemented transparently and stdlib-only:

  - **ValueLearner** — temporal-difference credit assignment with **eligibility
    traces** (TD(lambda), Sutton & Barto). Each time an item is attended it leaves a
    decaying trace; a later `reinforce(reward)` updates the *value* of every item in
    proportion to its trace, so credit flows back to the context that preceded the
    outcome — even several steps later. This is how the engine discovers that, say,
    "AIS gap" reliably precedes an escalation worth attending to.

  - **LexiconLearner** — online valence learning by the **Rescorla-Wagner / delta
    rule**: words occurring in a reinforced episode have their affective valence nudged
    toward the sign of the reward. New domain words thus *acquire* affect from
    experience instead of needing a hand-built lexicon.

Both are bounded, explainable, and converge with a fixed learning rate. No tensors,
no training loop — just the update equations.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ValueLearner:
    """TD(lambda) value with accumulating eligibility traces."""
    alpha: float = 0.2          # learning rate
    gamma: float = 0.95         # discount
    lam: float = 0.8            # trace decay (eligibility)
    value: dict[str, float] = field(default_factory=dict)
    trace: dict[str, float] = field(default_factory=dict)

    def attend(self, items) -> None:
        """Mark items as attended this step; decay then bump their eligibility."""
        for k in list(self.trace):                       # decay all existing traces
            self.trace[k] *= self.gamma * self.lam
            if self.trace[k] < 1e-4:
                del self.trace[k]
        for it in items:
            self.trace[it] = self.trace.get(it, 0.0) + 1.0   # accumulating trace
            self.value.setdefault(it, 0.0)

    def reinforce(self, reward: float) -> dict[str, float]:
        """Apply a reward; credit flows to every item by its current trace."""
        updated = {}
        for it, e in self.trace.items():
            self.value[it] = self.value.get(it, 0.0) + self.alpha * reward * e
            updated[it] = self.value[it]
        return updated

    def valued(self, n: int = 5) -> list[tuple[str, float]]:
        return sorted(self.value.items(), key=lambda kv: kv[1], reverse=True)[:n]


@dataclass
class LexiconLearner:
    """Online affective-valence learning (delta rule, valence in [-1, 1])."""
    eta: float = 0.15
    learned: dict[str, float] = field(default_factory=dict)

    def update(self, words, reward: float) -> None:
        target = max(-1.0, min(1.0, reward))            # reward sign = desired valence
        for w in words:
            cur = self.learned.get(w, 0.0)
            self.learned[w] = round(cur + self.eta * (target - cur), 4)

    def overrides(self) -> dict[str, float]:
        """Learned words confident enough to influence extraction."""
        return {w: v for w, v in self.learned.items() if abs(v) >= 0.1}

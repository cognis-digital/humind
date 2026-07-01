"""Systems-thinking layer — turn understood context into a causal-loop model.

Builds a **causal-loop diagram** (CLD) from causal statements: nodes are concepts,
edges carry a **polarity** (+ same-direction, - opposite). On top of it:

  - **feedback_loops()** — finds cycles and classifies each as **reinforcing** (R, an
    even number of negative links -> a self-amplifying loop) or **balancing** (B, odd
    number -> a self-correcting loop). This is the core move of system dynamics
    (Forrester / Sterman): behavior comes from loop structure, not isolated events.
  - **leverage_points()** — ranks concepts by causal **centrality** (Meadows: the
    highest-leverage places to intervene are the most connected structural nodes).

Pure standard library, deterministic, explainable — you can read off *why* a loop is
reinforcing.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CausalGraph:
    # concept -> list of (target_concept, polarity in {+1, -1})
    edges: dict[str, list[tuple[str, int]]] = field(default_factory=dict)
    _seen: set = field(default_factory=set)

    def add_link(self, cause: str, effect: str, polarity: int = 1) -> None:
        cause, effect = cause.strip().lower(), effect.strip().lower()
        if not cause or not effect or cause == effect:
            return
        key = (cause, effect, polarity)
        if key in self._seen:
            return
        self._seen.add(key)
        self.edges.setdefault(cause, []).append((effect, 1 if polarity >= 0 else -1))
        self.edges.setdefault(effect, self.edges.get(effect, []))

    def nodes(self) -> list[str]:
        return sorted(self.edges)

    def feedback_loops(self, max_len: int = 6) -> list[dict]:
        """Return cycles as {nodes, polarity:+1/-1, kind:R|B}. Dedupes rotations.

        Deduplication is by the *directed* cycle (canonical rotation), not merely the
        set of nodes — so two loops over the same concepts but in different directions,
        or with different link polarities, are reported separately rather than collapsed.
        """
        if max_len < 1:
            raise ValueError(f"max_len must be >= 1, got {max_len}")
        loops: list[dict] = []
        seen: set = set()

        def _canon(cycle: list[str]) -> tuple[str, ...]:
            # rotate so the lexicographically smallest node leads; keeps direction
            i = min(range(len(cycle)), key=lambda j: cycle[j])
            return tuple(cycle[i:] + cycle[:i])

        def dfs(start, node, path, sign):
            if len(path) > max_len:
                return
            for nxt, pol in self.edges.get(node, []):
                if nxt == start and len(path) >= 1:
                    cycle = path[:]
                    canon = _canon(cycle)
                    if canon not in seen:
                        seen.add(canon)
                        prod = sign * pol
                        loops.append({"nodes": cycle, "polarity": prod,
                                      "kind": "R" if prod > 0 else "B"})
                elif nxt not in path:
                    dfs(start, nxt, path + [nxt], sign * pol)

        for n in self.edges:
            dfs(n, n, [n], 1)
        return loops

    def centrality(self) -> dict[str, int]:
        deg: dict[str, int] = {n: 0 for n in self.edges}
        for src, outs in self.edges.items():
            for dst, _ in outs:
                deg[src] = deg.get(src, 0) + 1
                deg[dst] = deg.get(dst, 0) + 1
        return deg

    def leverage_points(self, n: int = 3) -> list[tuple[str, int]]:
        return sorted(self.centrality().items(), key=lambda kv: kv[1], reverse=True)[:n]

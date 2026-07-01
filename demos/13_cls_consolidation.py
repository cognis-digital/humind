"""Scenario 13 - complementary learning systems (memory-systems readers).

Fast, episodic memory logs everything; slow, semantic memory keeps only what
recurs. humind's **CLS consolidation** distils entities that appear across
multiple episodes (the fast hippocampal log) into durable semantic facts (the
slow neocortical store). This demo streams a watch log where one contact recurs
and others are one-offs, then runs consolidation and shows exactly which entities
crossed the recurrence threshold into durable memory.
"""
from _common import rule
from humind import Mind


LOG = [
    "vessel NEPTUNE-STAR went dark near the strait",
    "one-off sighting of DRIFTWOOD far offshore",
    "NEPTUNE-STAR reappeared, still dark",
    "NEPTUNE-STAR now inside the exclusion zone",
    "brief contact with PALE-MOON, then lost",
]


def main() -> None:
    rule("CLS CONSOLIDATION  -  recurring episodes become durable semantic facts")
    m = Mind("archivist")
    print("\nStream a watch log into episodic memory:\n")
    for line in LOG:
        m.perceive(line)
        ents = ", ".join(m.memory.episodic.events[-1][1].entities) or "-"
        print(f"   logged  [{ents}]  \"{line[:44]}\"")

    print(f"\nEpisodic memory now holds {len(m.memory.episodic.events)} frames.")
    print("Entity recurrence across those frames:")
    counts = {}
    for _, f in m.memory.episodic.events:
        for e in f.entities:
            counts[e.lower()] = counts.get(e.lower(), 0) + 1
    for ent, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        mark = "  <- recurs" if c >= 2 else ""
        print(f"   {ent:<16} seen {c}x{mark}")

    print("\nRun consolidation (min_count=2): only recurring entities are promoted.")
    new = m.consolidate(min_count=2)
    print("   newly consolidated facts:", new or "(none recurred enough)")

    print("\nConsolidation is idempotent — running it again promotes nothing new:")
    print("   second pass:", m.consolidate(min_count=2) or "(nothing new)")


if __name__ == "__main__":
    main()
